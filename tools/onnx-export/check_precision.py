"""Compare a reduced-precision export against its own fp32 graph, on the CUDA provider.

The export's acceptance check, run on the GPU host right after the export (in the
exporter image, `--gpus all`). It answers one question per model dir: does
model.fp16.onnx compute what model.onnx computes, on the provider production uses?
It is NOT the retrieval gate -- the reranker's is rerank_ab.py against the torch fp32
reference (gate1-runbook §5.5), the embedder's the labelled 14.2 question set.

    python tools/onnx-export/check_precision.py --kind embed  --model-dir /work/bge-m3-onnx \
        --fixture /work/fixture.json --json /work/check-embed.json
    python tools/onnx-export/check_precision.py --kind rerank --model-dir /work/bge-reranker-onnx \
        --fixture /work/fixture.json --json /work/check-rerank.json

Texts are the Gate 1 fixture's (tools/eval/dump_rerank_fixture.py): 8 real queries
with their 40 RRF candidates -- real chunk lengths, real vocabulary.

NO CPU FALLBACK: only CUDAExecutionProvider is requested, and the script fails unless
the session reports it. A CUDA/cuDNN mismatch otherwise makes onnxruntime run on CPU
with a warning, and an fp16 graph on CPU is a different (and far slower) computation.

Before that, FIDELITY: model.onnx against the HF model itself (torch fp32 on CPU, at
the revision in meta.json, from the exporter's HF cache) on the first query's texts --
proof the graph computes the model, which no fp16-vs-fp32 comparison can give.

It also checks PADDING INVARIANCE of every variant: the first query's texts scored
one at a time must match the same texts scored as one padded batch. An export traced
on an unpadded probe can bake "no mask" into the graph (FINDINGS.md finding 2); both
precisions would then agree with each other and still be wrong on every padded batch.

Bars (fixed here, not flags, so a run cannot quietly loosen them):
  both    fidelity: ONNX fp32 vs torch fp32 -- embed dense cosine >= 0.9999, rerank
          max |logit delta| <= 0.001
  embed   dense cosine fp16 vs fp32 >= 0.999 on every text; sparse weight max |delta|
          reported (it feeds a dot product, so it is judged by the question set);
          padding: solo vs batched dense cosine >= 0.999
  rerank  max |logit delta| <= 0.05 -- the bar Gate 1 set for "same ranking numerics"
          (torch fp16 vs fp32 measured 0.0095); padding: solo vs batched max |logit
          delta| <= 0.05. Top-6 agreement is REPORTED, not judged: fixture candidates
          tie (duplicate chunks score identically), so a 0.002 shift can reorder rank 6
          (FINDINGS.md finding 4). Whether a reordering matters is rerank_ab.py's call,
          with the fp32 judge over the deep pool.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

EMBED_COS_MIN = 0.999
FIDELITY_COS_MIN = 0.9999
FIDELITY_DELTA_MAX = 0.001
RERANK_DELTA_MAX = 0.05
TOP_K = 6
VARIANTS = {"fp32": "model.onnx", "fp16": "model.fp16.onnx"}


def session(path: str):
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # the CUDA/cuDNN libraries from the nvidia-* wheels
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(path, sess_options=so, providers=["CUDAExecutionProvider"])
    got = s.get_providers()
    if not got or got[0] != "CUDAExecutionProvider":
        raise SystemExit(f"{path}: session providers are {got}, not CUDA -- refusing to "
                         "compare (onnxruntime fell back to CPU; check the driver, "
                         "--gpus, and the CUDA/cuDNN majors onnxruntime-gpu expects)")
    return s


def tokenizer(model_dir: str, max_length: int):
    from tokenizers import Tokenizer

    t = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    t.enable_truncation(max_length=max_length)
    t.enable_padding(pad_id=t.token_to_id("<pad>"), pad_token="<pad>")
    return t


def feeds(encs, names) -> dict:
    f = {"input_ids": np.array([e.ids for e in encs], dtype=np.int64),
         "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64)}
    return {k: v for k, v in f.items() if k in names}


def timed(s, feed) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    out = s.run(None, feed)[0]
    return np.asarray(out, dtype=np.float32), time.perf_counter() - t0


def padding_embed(s, tok, names, fx) -> float:
    """Min cosine between each text embedded alone and inside one padded batch."""
    texts = [c["text"] for c in fx["queries"][0]["candidates"][:8]]
    cls = timed(s, feeds(tok.encode_batch(texts), names))[0][:, 0]
    worst = 1.0
    for i, t in enumerate(texts):
        one = timed(s, feeds([tok.encode(t)], names))[0][0, 0]
        worst = min(worst, float(one @ cls[i] / np.linalg.norm(one) / np.linalg.norm(cls[i])))
    return worst


def padding_rerank(s, tok, names, fx) -> float:
    """Max |logit delta| between each pair scored alone and inside one padded batch."""
    q = fx["queries"][0]
    pairs = [(q["query"], c["text"]) for c in q["candidates"][:8]]
    batch = timed(s, feeds(tok.encode_batch(pairs), names))[0].reshape(-1)
    solo = [timed(s, feeds([tok.encode(*p)], names))[0].reshape(-1)[0] for p in pairs]
    return float(np.max(np.abs(batch - np.array(solo))))


def fidelity(kind: str, model_dir: str, meta: dict, fx: dict) -> float:
    """ONNX fp32 (CUDA) vs the HF model in torch fp32 (CPU), on the first query's texts:
    embed -> min dense cosine, rerank -> max |logit delta|."""
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    q = fx["queries"][0]
    texts = [c["text"] for c in q["candidates"][:8]]
    s = session(os.path.join(model_dir, VARIANTS["fp32"]))
    names = {i.name for i in s.get_inputs()}
    hf = AutoTokenizer.from_pretrained(meta["model_id"], revision=meta["revision"])
    with torch.no_grad():
        if kind == "embed":
            tok = tokenizer(model_dir, 8192)
            onx = timed(s, feeds(tok.encode_batch(texts), names))[0][:, 0]
            m = AutoModel.from_pretrained(meta["model_id"], revision=meta["revision"],
                                          dtype=torch.float32).eval()
            ref = m(**hf(texts, padding=True, return_tensors="pt")).last_hidden_state[:, 0]
            ref = ref.numpy()
            cos = np.sum(onx * ref, 1) / np.linalg.norm(onx, axis=1) / np.linalg.norm(ref, axis=1)
            return float(cos.min())
        max_length = fx["meta"]["baseline"]["max_length"]
        tok = tokenizer(model_dir, max_length)
        onx = timed(s, feeds(tok.encode_batch([(q["query"], t) for t in texts]), names))[0]
        m = AutoModelForSequenceClassification.from_pretrained(
            meta["model_id"], revision=meta["revision"], dtype=torch.float32).eval()
        ref = m(**hf([q["query"]] * len(texts), texts, padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt")).logits.numpy()
        return float(np.max(np.abs(onx.reshape(-1) - ref.reshape(-1))))


def check_embed(model_dir: str, fx: dict, batch: int) -> dict:
    texts = sorted({q["query"] for q in fx["queries"]}
                   | {c["text"] for q in fx["queries"] for c in q["candidates"]}, key=len)
    tok = tokenizer(model_dir, 8192)
    sp = np.load(os.path.join(model_dir, "sparse_linear.npz"))
    res, secs, pad = {}, {}, {}
    for v, f in VARIANTS.items():
        s = session(os.path.join(model_dir, f))
        names = {i.name for i in s.get_inputs()}
        s.run(None, feeds(tok.encode_batch(texts[:2]), names))  # warm-up: CUDA init
        dense, sparse, secs[v] = [], [], 0.0
        for i in range(0, len(texts), batch):
            encs = tok.encode_batch(texts[i:i + batch])
            h, dt = timed(s, feeds(encs, names))
            secs[v] += dt
            cls = h[:, 0]
            dense.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
            w = np.maximum(h @ sp["weight"].T + sp["bias"], 0)[..., 0]
            mask = np.array([e.attention_mask for e in encs], dtype=bool)
            sparse.extend(w[j][mask[j]] for j in range(len(encs)))
        res[v] = (np.concatenate(dense), sparse)
        pad[v] = padding_embed(s, tok, names, fx)
        del s
    cos = np.sum(res["fp32"][0] * res["fp16"][0], axis=1)
    sp_delta = max(float(np.max(np.abs(a - b))) for a, b in zip(res["fp32"][1], res["fp16"][1]))
    sp_max = max(float(np.max(a)) for a in res["fp32"][1])
    ok = bool(cos.min() >= EMBED_COS_MIN) and all(c >= EMBED_COS_MIN for c in pad.values())
    return {"texts": len(texts), "batch": batch,
            "dense_cos_min": float(cos.min()), "dense_cos_mean": float(cos.mean()),
            "sparse_weight_max_delta": sp_delta, "sparse_weight_max": sp_max,
            "padding_cos_min": pad,
            "seconds": secs,
            "bar": f"dense_cos_min and padding_cos_min >= {EMBED_COS_MIN}", "pass": ok}


def check_rerank(model_dir: str, fx: dict) -> dict:
    max_length = fx["meta"]["baseline"]["max_length"]
    tok = tokenizer(model_dir, max_length)
    scores, secs, pad = {}, {}, {}
    for v, f in VARIANTS.items():
        s = session(os.path.join(model_dir, f))
        names = {i.name for i in s.get_inputs()}
        q0 = fx["queries"][0]
        s.run(None, feeds(tok.encode_batch([(q0["query"], "warm-up")]), names))
        scores[v], secs[v] = [], []
        for q in fx["queries"]:
            encs = tok.encode_batch([(q["query"], c["text"]) for c in q["candidates"]])
            out, dt = timed(s, feeds(encs, names))
            scores[v].append(out.reshape(-1))
            secs[v].append(dt)
        pad[v] = padding_rerank(s, tok, names, fx)
        del s
    per_query = []
    for q, a, b in zip(fx["queries"], scores["fp32"], scores["fp16"]):
        top_a = list(np.argsort(-a)[:TOP_K])
        top_b = list(np.argsort(-b)[:TOP_K])
        per_query.append({"query": q["query"][:60], "max_delta": float(np.max(np.abs(a - b))),
                          "top6_identical": top_a == top_b})
    worst = max(p["max_delta"] for p in per_query)
    same = sum(p["top6_identical"] for p in per_query)
    ok = worst <= RERANK_DELTA_MAX and all(d <= RERANK_DELTA_MAX for d in pad.values())
    return {"queries": per_query, "max_delta": worst,
            "top6_identical": f"{same}/{len(per_query)}", "padding_max_delta": pad,
            "median_s_per_query": {v: statistics.median(t) for v, t in secs.items()},
            "bar": f"max_delta and padding_max_delta <= {RERANK_DELTA_MAX}",
            "pass": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["embed", "rerank"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--fixture", required=True, help="the Gate 1 rerank fixture.json")
    ap.add_argument("--batch", type=int, default=16, help="embed: texts per run")
    ap.add_argument("--json", dest="json_out", required=True)
    args = ap.parse_args()

    for f in VARIANTS.values():
        if not os.path.exists(os.path.join(args.model_dir, f)):
            raise SystemExit(f"{args.model_dir}/{f} not found")
    fx = json.load(open(args.fixture))
    import onnxruntime as ort

    meta = json.load(open(os.path.join(args.model_dir, "meta.json")))
    fid = fidelity(args.kind, args.model_dir, meta, fx)
    fid_ok = (fid >= FIDELITY_COS_MIN if args.kind == "embed" else fid <= FIDELITY_DELTA_MAX)
    res = (check_embed(args.model_dir, fx, args.batch) if args.kind == "embed"
           else check_rerank(args.model_dir, fx))
    res["fidelity_vs_torch_fp32"] = fid
    res["bar"] = (f"fidelity {'cos >=' if args.kind == 'embed' else '|delta| <='} "
                  f"{FIDELITY_COS_MIN if args.kind == 'embed' else FIDELITY_DELTA_MAX}; "
                  + res["bar"])
    res["pass"] = bool(res["pass"] and fid_ok)
    res = {"kind": args.kind, "model_id": meta["model_id"], "revision": meta["revision"],
           "files": {f: meta["files"][f]["sha256"] for f in VARIANTS.values()},
           "onnxruntime": ort.__version__, "provider": "CUDAExecutionProvider",
           "fixture_created_utc": fx["meta"]["created_utc"], **res}
    with open(args.json_out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "queries"}, indent=2))
    print(f"{args.kind}: {'PASS' if res['pass'] else 'FAIL'} ({res['bar']})")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

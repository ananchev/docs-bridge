"""A/B one reranker against the recorded baseline, on the §2.2 fixture.

Serves three jobs with one comparison, because they ask the identical question:
  * Gate 1 §2.3 — GPU probe:            --backend torch --device cuda --precision fp16
  * Gate 1 §3   — smaller reranker:     --backend onnx --model-dir /opt/bge-reranker-base-onnx
  * Gate 2 §5.2 — reranker A/B:         the same two, on the upgraded toolchain
  * packaging (plan §8 3b item 1):      --backend docs-bridge, inside the server image

The fixture (tools/eval/dump_rerank_fixture.py) carries the queries, the real RRF
candidate passages and the CPU-INT8 baseline scores, so this script needs NO
Qdrant, NO corpus and NO docs-bridge config — it runs on a bare GPU box with only
torch + transformers installed:

    python tools/eval/rerank_ab.py --fixture fixture.json \
        --backend torch --model-id BAAI/bge-reranker-v2-m3 \
        --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
        --device cuda --precision fp16 --json gpu-probe.json

The ONNX backend loads the dir's file for --precision (model.int8.onnx, model.onnx,
model.fp16.onnx; --onnx-file for another) and with --device cuda requests the CUDA
provider ONLY, refusing to run if the session fell back to CPU.

The ONNX path deliberately re-implements OnnxReranker's ~15 lines instead of
importing it, for two reasons: a probe must not require an image rebuild to try a
model the image does not carry, and the GPU host has no docs_bridge install. The
one intentional difference is `token_type_ids`, taken from the tokenizer encoding
rather than hardcoded zeros — identical for XLM-R (v2-m3 and bge-reranker-base are
both XLM-R, type_vocab_size=1), but correct for a BERT-family reranker too.
Prove faithfulness before trusting any comparison:

    --backend onnx --model-dir <the baked baseline dir> --selftest

which must reproduce the fixture's baseline column to float noise.

THE BAR (plan §2.4): top-3 ordered-identical and top-6 Jaccard >= 0.83 against
the REFERENCE. Since 2026-09-28 the reference is the unquantized model — a saved
fp32 run passed as `--reference notes/gate1/gpu-fp32.json` — not the fixture's
CPU-INT8 column: INT8 sits up to 1.03 logits from fp32, so fp32 itself fails an INT8
bar and no more precise scorer could ever pass it. Without `--reference` the INT8
column is still used (legacy, and what the ONNX `--selftest` must reproduce).

A disagreement is adjudicated by the reference acting as a FIXED judge over the
deeper `oracle_pool`: the judge's global-best top-k there, restricted to chunks in
the reranked set, must all be kept by the candidate. Differences confined to chunks
the judge would not rank globally best are accepted. The judge must never be the
candidate itself — grading each scorer against its own deep top-k (the method before
2026-09-28) lets each recall its own gold at 1.00 and turns every set change into a
"regression". The judge needs the reference's pool scores, which a run stores only
when made with `--score-pool`; a reference without them cannot adjudicate, so any
disagreement then fails.

Any saved run can be re-judged offline, with no model and no GPU:

    python tools/eval/rerank_ab.py --fixture fixture.json --backend replay \\
        --replay notes/gate1/gpu-fp16.json --reference notes/gate1/gpu-fp32.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time


# --- scorers ------------------------------------------------------------------
class OnnxScorer:
    """Faithful stand-in for docs_bridge.rerank.OnnxReranker (see module docstring)."""

    kind = "onnx"

    # the model dir layout of tools/onnx-export/_common.py
    FILES = {"int8": "model.int8.onnx", "fp32": "model.onnx", "fp16": "model.fp16.onnx"}

    def __init__(self, model_dir: str, precision: str, max_length: int,
                 threads: int | None, device: str, onnx_file: str | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if precision not in self.FILES:
            raise SystemExit(f"--backend onnx has no {precision} model file "
                             f"({', '.join(self.FILES)})")
        onnx_file = onnx_file or self.FILES[precision]
        path = os.path.join(model_dir, onnx_file)
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found")
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # CUDA ONLY, no CPU fallback list: a CUDA/cuDNN mismatch would otherwise run
        # the whole graph on CPU with a warning, and time and score the wrong thing.
        if device == "cuda" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # CUDA/cuDNN from the nvidia-* wheels
        providers = ["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, sess_options=so, providers=providers)
        self.providers = self.session.get_providers()
        if self.providers[0] != providers[0]:
            raise SystemExit(f"--device {device}: the session got {self.providers}, not "
                             f"{providers[0]} -- refusing to measure a CPU fallback")
        self._inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = self.tokenizer.token_to_id("<pad>")
        if pad_id is None:
            pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=0 if pad_id is None else pad_id,
                                      pad_token="<pad>" if pad_id else "[PAD]")
        self._len_tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self._len_tok.enable_truncation(max_length=max_length)
        meta_path = os.path.join(model_dir, "meta.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.desc = (f"onnx {precision} {onnx_file} on {self.providers[0]} "
                     f"({self.meta.get('model_id', model_dir)}"
                     f"@{str(self.meta.get('revision', '?'))[:12]})")

    def lengths(self, query: str, passages: list[str]) -> list[int]:
        return [len(e.ids) for e in
                self._len_tok.encode_batch([(query, p) for p in passages])]

    def score(self, query: str, passages: list[str]) -> list[float]:
        import numpy as np

        encs = self.tokenizer.encode_batch([(query, p) for p in passages])
        feeds = {
            "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encs], dtype=np.int64),
        }
        feeds = {k: v for k, v in feeds.items() if k in self._inputs}
        logits = self.session.run(None, feeds)[0]
        return np.asarray(logits, dtype=np.float32).reshape(-1).tolist()


class ServerScorer:
    """The production class itself, docs_bridge.rerank.OnnxReranker, bucketing
    included -- for the packaging check: run INSIDE the server image, it proves the
    image's onnxruntime / CUDA libraries / model file score like the reference, which
    the re-implemented OnnxScorer cannot (it runs wherever this script runs)."""

    kind = "docs-bridge"

    def __init__(self, model_dir: str, device: str, precision: str, bucket: int,
                 max_length: int, threads: int | None) -> None:
        from docs_bridge.rerank import OnnxReranker

        self.rr = OnnxReranker(model_dir, device=device, precision=precision,
                               bucket=bucket, max_length=max_length, threads=threads)
        self.providers = self.rr.session.get_providers()
        meta = json.load(open(os.path.join(model_dir, "meta.json")))
        self.desc = (f"docs_bridge.rerank.OnnxReranker {precision} on {self.providers[0]}"
                     f"{f' bucket={bucket}' if bucket else ''} "
                     f"({meta['model_id']}@{str(meta['revision'])[:12]})")

    def lengths(self, query: str, passages: list[str]) -> list[int]:
        raise SystemExit("--backend docs-bridge buckets inside OnnxReranker; "
                         "pass --bucket, it is handed to the class, not wrapped")

    def score(self, query: str, passages: list[str]) -> list[float]:
        return self.rr.score(query, passages)


class TorchScorer:
    """§2.3's probe: the official cross-encoder weights, fp16 on CUDA. Tests physics,
    not packaging — no ONNX export, no CUDA/ORT build, nothing to rebuild."""

    kind = "torch"

    def __init__(self, model_id: str, revision: str, device: str, precision: str,
                 max_length: int, threads: int | None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda but torch.cuda.is_available() is False "
                             "— check the driver, the container's --device flag, and "
                             "that this is a CUDA build of torch")
        if threads and device == "cpu":
            torch.set_num_threads(threads)
        self.device = device
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        m = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        if precision == "fp16":
            m = m.half()
        elif precision == "bf16":
            m = m.bfloat16()
        self.model = m.to(device).eval()
        gpu = (torch.cuda.get_device_name(0) if device == "cuda"
               and torch.cuda.is_available() else platform.processor() or platform.machine())
        self.desc = f"torch {precision} on {device} [{gpu}] ({model_id}@{revision[:12]})"

    def lengths(self, query: str, passages: list[str]) -> list[int]:
        enc = self.tok([query] * len(passages), passages, padding=False,
                       truncation=True, max_length=self.max_length)
        return [len(x) for x in enc["input_ids"]]

    def _sync(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def score(self, query: str, passages: list[str]) -> list[float]:
        with self.torch.no_grad():
            enc = self.tok([query] * len(passages), passages, padding=True,
                           truncation=True, max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            self._sync()
            return logits.reshape(-1).float().cpu().tolist()


class ReplayScorer:
    """Replays the per-pair scores of a saved --json run, so a run can be re-judged
    against a different reference without its model or its hardware. Its timings are
    the saved ones — replaying measures nothing."""

    kind = "replay"

    def __init__(self, path: str, fx: dict) -> None:
        run = load_run(path, fx)
        self.desc = f"replay of {path}: {run['candidate']}"
        self.saved = {}
        self.timings = {}
        for r in run["queries"]:
            self.saved[(r["query"], len(r["scores"]))] = r["scores"]
            if r.get("pool_scores"):
                self.saved[(r["query"], len(r["pool_scores"]))] = r["pool_scores"]
            self.timings[r["query"]] = r["median_s"]

    def score(self, query: str, passages: list[str]) -> list[float]:
        try:
            return self.saved[(query, len(passages))]
        except KeyError:
            raise SystemExit(f"replayed run has no scores for {len(passages)} passages of "
                             f"{query[:50]!r} (pool scores need a --score-pool run)")


def load_run(path: str, fx: dict) -> dict:
    """A saved --json run, checked to line up with this fixture pair for pair."""
    run = json.load(open(path))
    if run.get("fixture_meta") != fx["meta"]:
        raise SystemExit(f"{path} was made on a different fixture — its scores do not "
                         f"line up with this one")
    rq = run.get("queries") or []
    if len(rq) != len(fx["queries"]) or any(
            r["query"] != q["query"] or len(r.get("scores") or []) != len(q["candidates"])
            for r, q in zip(rq, fx["queries"])):
        raise SystemExit(f"{path} has no per-pair scores for this fixture (runs before "
                         f"2026-09-28 dropped them) — re-run it with this version")
    return run


class Bucketed:
    """Wraps a scorer to length-sort and score in sub-batches of `bucket`, restoring
    the caller's order. Padding is mathematically inert, so for a float model this
    is a pure speed change; under dynamic INT8 the per-tensor activation scales are
    computed over whatever shares a batch, so it can move scores — which is why it
    goes through the same adjudication as any other candidate reranker rather than
    being assumed free."""

    def __init__(self, inner, bucket: int) -> None:
        self.inner, self.bucket = inner, bucket
        self.kind = inner.kind
        self.desc = f"{inner.desc} + length-bucketed sub-batches of {bucket}"

    def score(self, query: str, passages: list[str]) -> list[float]:
        lens = self.inner.lengths(query, passages)
        order = sorted(range(len(passages)), key=lambda i: lens[i])
        out = [0.0] * len(passages)
        for i in range(0, len(order), self.bucket):
            idx = order[i:i + self.bucket]
            for j, sc in zip(idx, self.inner.score(query, [passages[j] for j in idx])):
                out[j] = sc
        return out


# --- ranking helpers ----------------------------------------------------------
def rank_ids(entries: list[dict], scores: list[float], k: int) -> list:
    """search.py's selection, verbatim: stable sort by score desc, keep k — so equal
    scores keep RRF order, exactly as production breaks ties."""
    order = sorted(range(len(entries)), key=lambda i: scores[i], reverse=True)[:k]
    return [entries[i]["id"] for i in order]


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default=os.environ.get("AB_FIXTURE", "fixture.json"))
    ap.add_argument("--backend", choices=["onnx", "docs-bridge", "torch", "replay"],
                    default="onnx")
    ap.add_argument("--model-dir", help="onnx backend: the exported model dir")
    ap.add_argument("--model-id", default="BAAI/bge-reranker-v2-m3",
                    help="torch backend: HF model id")
    ap.add_argument("--revision",
                    help="torch backend, REQUIRED: the HF commit (40 hex) -- the one the "
                         "ONNX export it is compared with was made from")
    ap.add_argument("--onnx-file",
                    help="onnx backend: a model file other than the precision's default "
                         "(e.g. an optimised fp16 graph); --precision still labels it")
    ap.add_argument("--replay", help="replay backend: a saved --json run to re-judge")
    ap.add_argument("--reference",
                    help="a saved --json run to judge against (plan §2.4: the fp32 run, "
                         "notes/gate1/gpu-fp32.json). Default: the fixture's CPU-INT8 "
                         "column (legacy)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--precision", choices=["int8", "fp32", "fp16", "bf16"],
                    default="int8")
    ap.add_argument("--threads", type=int, default=0, help="0 = every core")
    ap.add_argument("--max-length", type=int, default=0, help="0 = the fixture's")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per query")
    ap.add_argument("--jaccard-min", type=float, default=0.83)
    ap.add_argument("--bucket", type=int, default=0,
                    help="length-sort the candidates and score in sub-batches of "
                         "this size (0 = one batch, today's behaviour). Cuts padding "
                         "waste; under INT8 it can also move scores, so it is judged "
                         "on the same bar as a new model.")
    ap.add_argument("--score-pool", action="store_true",
                    help="also score the deep oracle pool and store it in --json, so "
                         "this run can serve as a --reference judge (the reference "
                         "run needs it; candidates do not)")
    ap.add_argument("--selftest", action="store_true",
                    help="assert this scorer reproduces the reference's scores")
    ap.add_argument("--json", dest="json_out", help="write the full result as JSON")
    args = ap.parse_args()

    fx = json.load(open(args.fixture))
    meta = fx["meta"]
    k = meta["default_k"]
    max_length = args.max_length or meta["baseline"]["max_length"]

    if args.backend == "onnx":
        model_dir = args.model_dir or meta["baseline"]["model_dir"]
        scorer = OnnxScorer(model_dir, args.precision, max_length=max_length,
                            threads=args.threads or None, device=args.device,
                            onnx_file=args.onnx_file)
    elif args.backend == "docs-bridge":
        if not args.model_dir:
            raise SystemExit("--backend docs-bridge needs --model-dir")
        scorer = ServerScorer(args.model_dir, args.device, args.precision, args.bucket,
                              max_length, args.threads or None)
    elif args.backend == "torch":
        if not (args.revision and len(args.revision) == 40):
            raise SystemExit("--backend torch needs --revision <40-hex HF commit>")
        scorer = TorchScorer(args.model_id, args.revision, args.device, args.precision,
                             max_length, args.threads or None)
    else:
        if not args.replay:
            raise SystemExit("--backend replay needs --replay <run.json>")
        if args.bucket:
            raise SystemExit("--bucket re-batches live scoring; a replay has none")
        scorer = ReplayScorer(args.replay, fx)

    if args.bucket and args.backend != "docs-bridge":
        scorer = Bucketed(scorer, args.bucket)

    # the reference: its production scores set the bar, its pool scores make the judge
    if args.reference:
        ref = load_run(args.reference, fx)
        ref_label = f"{args.reference}: {ref['candidate']}"
        ref_scores = [r["scores"] for r in ref["queries"]]
        ref_pool = [r.get("pool_scores") for r in ref["queries"]]
    else:
        ref_label = (f"fixture baseline column: {meta['baseline']['kind']} "
                     f"({meta['baseline']['model_meta'].get('model_id', '?')}) — legacy; "
                     f"plan §2.4 now judges against fp32 (--reference)")
        ref_scores = [[c["baseline_score"] for c in q["candidates"]] for q in fx["queries"]]
        ref_pool = [[c["baseline_score"] for c in q.get("oracle_pool") or []] or None
                    for q in fx["queries"]]

    print(f"\nrerank A/B | fixture={args.fixture}")
    print(f"  CPU baseline (timing): {meta['baseline']['kind']} on {meta['host']} "
          f"[{meta['machine']}, {meta['cpu_count']} cores], "
          f"{meta['timing']['rerank_ms_per_pair']:.0f} ms/pair")
    print(f"  reference + judge:     {ref_label}")
    print(f"  candidate:             {scorer.desc}")
    print(f"  top_n={meta['top_n']} k={k} max_length={max_length} "
          f"oracle_pool={meta['oracle_pool']} queries={len(fx['queries'])}")
    print(f"  bar: top-{min(3, k)} ordered-identical AND top-{k} Jaccard >= "
          f"{args.jaccard_min} vs the reference\n")

    # warm-up on the first query's batch: model load, kernel autotune, thread pool
    if scorer.kind != "replay":
        warm = fx["queries"][0]["candidates"][:8]
        t0 = time.perf_counter()
        scorer.score(fx["queries"][0]["query"], [c["text"] for c in warm])
        print(f"  warm-up (8 pairs, includes any lazy init): "
              f"{time.perf_counter() - t0:.2f} s\n")

    hdr = (f"{'query':<42}{'s':>7}{'ms/pr':>7}{'spdup':>7}{'top3':>6}"
           f"{'jacc':>6}{'judge':>7}{'maxS':>8}")
    print(hdr)
    print("-" * len(hdr))

    results, fails, flags = [], [], []
    for qi, q in enumerate(fx["queries"]):
        cands = q["candidates"]
        texts = [c["text"] for c in cands]
        ref_q = ref_scores[qi]

        if scorer.kind == "replay":
            new = scorer.score(q["query"], texts)
            med = scorer.timings[q["query"]]      # the replayed run's own timing
        else:
            samples = []
            for _ in range(max(1, args.repeats)):
                t0 = time.perf_counter()
                new = scorer.score(q["query"], texts)
                samples.append(time.perf_counter() - t0)
            med = statistics.median(samples)

        ref_top = rank_ids(cands, ref_q, k)
        new_top = rank_ids(cands, new, k)
        top3 = ref_top[:3] == new_top[:3]
        jac = jaccard(ref_top, new_top)

        # the judge: the REFERENCE's global-best top-k over the deep pool, restricted
        # to chunks the reranked set can hold. Fixed across candidates — never the
        # candidate's own view, which would grade it against itself.
        pool = q.get("oracle_pool") or []
        oracle = {}
        if pool and ref_pool[qi]:
            gold = rank_ids(pool, ref_pool[qi], k)
            prod_ids = {c["id"] for c in cands}
            gold_in = [g for g in gold if g in prod_ids]
            oracle = {
                "judge_gold": gold, "judge_gold_in_prod": gold_in,
                "candidate_keeps": [g for g in gold_in if g in new_top],
                "reference_keeps": [g for g in gold_in if g in ref_top],
            }

        row = {
            "query": q["query"], "control": q["control"], "pairs": len(cands),
            "median_s": med, "ms_per_pair": med / len(cands) * 1000.0,
            "speedup_vs_baseline": q["rerank_s"] / med if med else None,
            "top3_identical": top3, "jaccard": jac,
            "reference_top": ref_top, "candidate_top": new_top,
            "reference_max": max(ref_q), "candidate_max": max(new),
            "scores": new, "oracle": oracle,
        }
        if args.score_pool and pool:
            row["pool_scores"] = scorer.score(q["query"], [c["text"] for c in pool])
        results.append(row)

        judge = (f"{len(oracle['candidate_keeps'])}/{len(oracle['judge_gold_in_prod'])}"
                 if oracle else "n/a")
        print(f"{q['query'][:41]:<42}{med:>7.2f}{row['ms_per_pair']:>7.1f}"
              f"{(row['speedup_vs_baseline'] or 0):>6.1f}x"
              f"{('yes' if top3 else 'NO'):>6}{jac:>6.2f}{judge:>7}"
              f"{row['candidate_max']:>8.2f}")

        if q["control"]:
            # The known-negative must stay negative, or the score sign no longer
            # separates real answers from abstention noise on this hardware/model.
            # Only a sign the REFERENCE got right and this scorer lost is a failure;
            # a reference that was already positive is a fact about the fixture (the
            # corpus really does hold something bread-adjacent), so it is a warning.
            if row["candidate_max"] >= 0 > row["reference_max"]:
                fails.append(f"control query scored POSITIVE ({row['candidate_max']:+.2f}) "
                             f"where the reference was {row['reference_max']:+.2f} — this "
                             f"scorer no longer separates answers from abstention")
            elif row["candidate_max"] >= 0:
                print(f"  note: control is positive for BOTH reference "
                      f"({row['reference_max']:+.2f}) and candidate "
                      f"({row['candidate_max']:+.2f}) — inspect the fixture, not the scorer")
            continue
        if not top3 or jac < args.jaccard_min:
            flags.append(row)

    # --- verdict --------------------------------------------------------------
    real = [r for r in results if not r["control"]]
    med_speedup = statistics.median([r["speedup_vs_baseline"] for r in real
                                     if r["speedup_vs_baseline"]])
    med_ms = statistics.median([r["ms_per_pair"] for r in real])
    print(f"\nspeed: {med_ms:.1f} ms/pair median ({med_speedup:.1f}x the CPU baseline's "
          f"{meta['timing']['rerank_ms_per_pair']:.0f} ms/pair)"
          f"{' — replayed, not measured' if scorer.kind == 'replay' else ''}")
    print(f"       => {med_ms * meta['top_n'] / 1000:.2f} s per search at "
          f"top_n={meta['top_n']}, "
          f"{med_ms * 60 / 1000:.2f} s at multi_top_n=60")

    if args.selftest:
        worst = 0.0
        for r, want in zip(results, ref_scores):
            for got, w in zip(r["scores"], want):
                worst = max(worst, abs(got - w))
        ok = worst < 1e-3
        print(f"\nSELFTEST: max |candidate - reference| = {worst:.2e} "
              f"=> {'FAITHFUL' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"selftest mismatch {worst:.2e} — this scorer does not "
                         f"reproduce the reference; no comparison below is trustworthy")

    print("")
    if flags:
        print(f"DIFFERS from the reference on {len(flags)}/{len(real)} queries — "
              f"adjudicating with the reference as a fixed judge over the deep pool:")
        for r in flags:
            o = r["oracle"]
            print(f"\n  {r['query'][:70]}")
            print(f"    top-3 identical: {r['top3_identical']}  jaccard: {r['jaccard']:.2f}")
            if not o:
                print("    no judge: the reference has no pool scores (make it with "
                      "--score-pool) — cannot adjudicate")
                fails.append(f"unadjudicated difference on: {r['query'][:60]}")
                continue
            n = len(o["judge_gold_in_prod"])
            print(f"    judge's global-best top-{k}: {n} of them in the reranked "
                  f"{r['pairs']}; candidate keeps {len(o['candidate_keeps'])}/{n}, "
                  f"reference keeps {len(o['reference_keeps'])}/{n}")
            lost = [g for g in o["judge_gold_in_prod"] if g not in o["candidate_keeps"]]
            if lost:
                print(f"    -> REGRESSION (reject): drops the judge's global-best {lost}")
                fails.append(f"judge-confirmed regression on: {r['query'][:60]}")
            else:
                print("    -> ACCEPT: differs only on chunks the judge does not rank "
                      "globally best")
    else:
        print(f"MATCHES the reference on all {len(real)} real queries "
              f"(top-3 identical, top-{k} Jaccard >= {args.jaccard_min}).")

    verdict = "PASS" if not fails else "FAIL"
    print(f"\nVERDICT: {verdict}")
    for f in fails:
        print(f"  - {f}")

    if args.json_out:
        json.dump({
            "candidate": scorer.desc, "backend": args.backend,
            "device": args.device, "precision": args.precision,
            "model_dir": getattr(args, "model_dir", None), "model_id": args.model_id,
            "revision": args.revision, "onnx_file": args.onnx_file,
            "bucket": args.bucket, "reference": ref_label,
            "host": platform.node(), "machine": platform.machine(),
            "fixture_meta": meta, "verdict": verdict, "failures": fails,
            "median_ms_per_pair": med_ms, "median_speedup": med_speedup,
            # per-pair scores kept (8 x 40 floats; + the pool with --score-pool) so a
            # run can be replayed, diffed against another, or serve as the reference
            "queries": results,
        }, open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

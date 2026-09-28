"""Embed parity across a runtime upgrade — the blocking gate for the Mac mini.

`teamcenter` holds ~137k vectors embedded by the OLD runtime (onnxruntime 1.20.1,
tokenizers 0.19.1). The Mac mini keeps that collection, so after the 2026-09
toolchain upgrade its server embeds queries with the NEW runtime (onnxruntime 1.30,
tokenizers 0.23) against doc vectors made by the old one. This gate proves the two
runtimes put text in the same place. The model bytes are frozen (images/models/), so
only the runtime can differ — and this tool checks that too, rather than assume it.

Two steps. `dump` runs INSIDE an image (old, then new), against the live collection:

    sudo podman run --rm -i --network docs-bridge-net \\
      -v /data/docker/docs-bridge/config/config.yaml:/config/config.yaml:ro,z \\
      -v ~/phase-a/out:/out:z -v ~/phase-a/src/notes/gate2:/q:ro,z \\
      --entrypoint python <IMAGE> - dump --collection teamcenter --n 400 \\
      --queries /q/questions-teamcenter.yaml --out /out/embed-old.json \\
      < tools/eval/embed_upgrade_parity.py

(the new image adds `--ids-from /out/embed-old.json` so both embed the SAME chunks),
then `compare` runs anywhere with numpy — the new image is simplest:

    ... --entrypoint python <NEW IMAGE> - compare \\
        --old /out/embed-old.json --new /out/embed-new.json --json /out/parity.json

WHAT IS MEASURED, and the bar (PASS needs all of them):
  1. SAME MODEL — sha256 of the four model files identical in both images.
  2. SAME TOKENS — token ids identical for every text (reported; a mismatch is the
     usual cause of any drift below, so it is shown first).
  3. OLD vs NEW runtime, same texts, same batching (1 text per call, like a query):
     dense cosine >= 0.999 for EVERY text, sparse Jaccard >= 0.95 on average — the
     bar the INT8 adoption was cleared with (stack-upgrade-strategy.md §3).
  4. NO NEW DRIFT vs the STORED vectors: re-embedding never reproduces a stored
     vector exactly even on the old runtime — ingest embedded in batches, and
     dynamic INT8 scales depend on what shares a batch (Gate 1 test 4). So the old
     runtime's own distance to the stored vectors is the baseline, and the new
     runtime must not be further: mean dense cosine within 1e-4 of the old one's,
     and its worst text within 1e-3 of the old one's worst.
Questions (--queries) take part in 1-3; they have no stored vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

MODEL_FILES = ("model.int8.onnx", "sparse_linear.npz", "tokenizer.json", "meta.json")
DENSE_MIN = 0.999            # old vs new, per text
SPARSE_JACCARD_MIN = 0.95    # old vs new, mean
STORED_MEAN_TOL = 1e-4       # new-vs-stored mean may trail old-vs-stored by this
STORED_WORST_TOL = 1e-3      # ... and its worst text by this


# --- dump (inside an image) -----------------------------------------------------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_queries(path: str | None) -> list[str]:
    if not path:
        return []
    import yaml

    raw = yaml.safe_load(open(path)) or {}
    return [q["question"] for q in raw.get("questions", []) if q.get("question")]


def dump(args) -> int:
    from importlib.metadata import version

    from qdrant_client import QdrantClient, models

    from docs_bridge import config
    from docs_bridge.embed_onnx import OnnxEmbedder

    cfg = config.load(args.config)
    client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port, timeout=120)

    if args.ids_from:
        ids = [c["id"] for c in json.load(open(args.ids_from))["chunks"]]
        points = client.retrieve(args.collection, ids=ids, with_payload=True,
                                 with_vectors=True)
        by_id = {str(p.id): p for p in points}
        points = [by_id[i] for i in ids if i in by_id]
    else:
        # scroll returns points in id order; ids are uuid5(chunk_id), so the first n
        # are a pseudo-random spread across documents, and the order is stable.
        points, _ = client.scroll(args.collection, limit=args.n, with_payload=True,
                                  with_vectors=True)
    points = [p for p in points if (p.payload or {}).get("text", "").strip()]
    print(f"[dump] {len(points)} chunks from {args.collection}", file=sys.stderr)

    emb = OnnxEmbedder(cfg.onnx_model_dir, use_int8=cfg.embedding_int8)
    model = {f: _sha256(os.path.join(cfg.onnx_model_dir, f)) for f in MODEL_FILES
             if os.path.exists(os.path.join(cfg.onnx_model_dir, f))}

    def embed_one(text: str) -> dict:
        ids = emb.tokenizer.encode(text).ids
        d, s = emb.encode([text])          # one text per call: no batch effects
        return {
            "token_ids_sha1": hashlib.sha1(str(ids).encode()).hexdigest(),
            "n_tokens": len(ids),
            "dense": d[0],
            "sparse": {"indices": s[0].indices, "values": s[0].values},
        }

    chunks = []
    for i, p in enumerate(points):
        vec = p.vector or {}
        stored_sparse = vec.get("sparse")
        row = {"id": str(p.id), "doc_id": p.payload.get("doc_id"),
               "text": p.payload["text"],
               "stored_dense": vec.get("dense"),
               "stored_sparse": ({"indices": list(stored_sparse.indices),
                                  "values": list(stored_sparse.values)}
                                 if isinstance(stored_sparse, models.SparseVector)
                                 else None)}
        row.update(embed_one(p.payload["text"]))
        chunks.append(row)
        if (i + 1) % 50 == 0:
            print(f"[dump]   {i + 1}/{len(points)}", file=sys.stderr)
    queries = [{"text": q, **embed_one(q)} for q in _load_queries(args.queries)]

    out = {
        "runtime": {
            "python": platform.python_version(), "machine": platform.machine(),
            **{pkg: version(pkg) for pkg in ("onnxruntime", "tokenizers", "numpy",
                                             "qdrant-client")},
        },
        "model_dir": cfg.onnx_model_dir, "int8": cfg.embedding_int8, "model": model,
        "collection": args.collection, "chunks": chunks, "queries": queries,
    }
    json.dump(out, open(args.out, "w"))
    print(f"[dump] wrote {args.out}: {len(chunks)} chunks, {len(queries)} questions "
          f"| onnxruntime {out['runtime']['onnxruntime']}, "
          f"tokenizers {out['runtime']['tokenizers']}", file=sys.stderr)
    return 0


# --- compare (anywhere with numpy) ----------------------------------------------
def _cos(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _jaccard(a: dict | None, b: dict | None) -> float | None:
    if a is None or b is None:
        return None
    sa, sb = set(a["indices"]), set(b["indices"])
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def _stats(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": sum(xs) / len(xs), "min": min(xs)}


def compare(args) -> int:
    old, new = json.load(open(args.old)), json.load(open(args.new))
    fails: list[str] = []

    print("embed upgrade parity")
    for side, d in (("old", old), ("new", new)):
        r = d["runtime"]
        print(f"  {side}: onnxruntime {r['onnxruntime']}, tokenizers {r['tokenizers']}, "
              f"numpy {r['numpy']}, python {r['python']} [{r['machine']}]")

    # 1. same model bytes
    same_model = old["model"] == new["model"] and bool(old["model"])
    print(f"\n1. SAME MODEL: {'yes' if same_model else 'NO'}")
    for f in MODEL_FILES:
        o, n = old["model"].get(f), new["model"].get(f)
        print(f"     {f:<18} {'=' if o == n else '≠'} {(n or 'missing')[:16]}")
    if not same_model:
        fails.append("model files differ — this compares two MODELS, not two runtimes")

    old_c = {c["id"]: c for c in old["chunks"]}
    pairs = [(old_c[c["id"]], c) for c in new["chunks"] if c["id"] in old_c]
    if len(pairs) != len(old["chunks"]):
        fails.append(f"only {len(pairs)}/{len(old['chunks'])} chunks line up — "
                     f"run the new dump with --ids-from the old one")
    qpairs = [(o, n) for o, n in zip(old["queries"], new["queries"])
              if o["text"] == n["text"]]
    allpairs = pairs + qpairs

    # 2. same tokens
    tok_diff = [n.get("id", n["text"][:50]) for o, n in allpairs
                if o["token_ids_sha1"] != n["token_ids_sha1"]]
    print(f"\n2. SAME TOKENS: {len(allpairs) - len(tok_diff)}/{len(allpairs)} texts "
          f"tokenize identically")
    for t in tok_diff[:5]:
        print(f"     differs: {t}")

    # 3. old vs new runtime
    dense = [_cos(o["dense"], n["dense"]) for o, n in allpairs]
    jac = [_jaccard(o["sparse"], n["sparse"]) for o, n in allpairs]
    ds, js = _stats(dense), _stats(jac)
    worst = sorted(zip(dense, allpairs), key=lambda t: t[0])[:3]
    print(f"\n3. OLD vs NEW runtime ({len(pairs)} chunks + {len(qpairs)} questions, "
          f"1 text per call)")
    print(f"     dense cosine  mean {ds['mean']:.6f}  min {ds['min']:.6f}   "
          f"(bar: every text >= {DENSE_MIN})")
    print(f"     sparse Jacc.  mean {js['mean']:.4f}  min {js['min']:.4f}     "
          f"(bar: mean >= {SPARSE_JACCARD_MIN})")
    for c, (o, n) in worst:
        print(f"     worst: {c:.6f}  {(n.get('doc_id') or 'question')}: {n['text'][:60]!r}")
    if ds["min"] < DENSE_MIN:
        fails.append(f"old-vs-new dense cosine min {ds['min']:.6f} < {DENSE_MIN}")
    if js["mean"] < SPARSE_JACCARD_MIN:
        fails.append(f"old-vs-new sparse Jaccard mean {js['mean']:.4f} < {SPARSE_JACCARD_MIN}")

    # 4. drift vs the stored vectors: the old runtime is the baseline
    so = _stats([_cos(o["dense"], o["stored_dense"]) for o, _ in pairs if o["stored_dense"]])
    sn = _stats([_cos(n["dense"], n["stored_dense"]) for _, n in pairs if n["stored_dense"]])
    jo = _stats([_jaccard(o["sparse"], o["stored_sparse"]) for o, _ in pairs])
    jn = _stats([_jaccard(n["sparse"], n["stored_sparse"]) for _, n in pairs])
    print(f"\n4. vs the STORED vectors (baseline = the old runtime's own distance)")
    print(f"     dense  old->stored mean {so['mean']:.6f} min {so['min']:.6f} | "
          f"new->stored mean {sn['mean']:.6f} min {sn['min']:.6f}")
    if jo.get("n") and jn.get("n"):
        print(f"     sparse old->stored mean {jo['mean']:.4f}          | "
              f"new->stored mean {jn['mean']:.4f}")
    if sn["mean"] < so["mean"] - STORED_MEAN_TOL:
        fails.append(f"new runtime drifts further from the stored vectors: mean "
                     f"{sn['mean']:.6f} vs old {so['mean']:.6f}")
    if sn["min"] < so["min"] - STORED_WORST_TOL:
        fails.append(f"new runtime's worst stored match {sn['min']:.6f} vs old "
                     f"{so['min']:.6f}")

    verdict = "PASS" if not fails else "FAIL"
    print(f"\nVERDICT: {verdict}")
    for f in fails:
        print(f"  - {f}")
    if args.json_out:
        json.dump({"verdict": verdict, "failures": fails,
                   "runtime": {"old": old["runtime"], "new": new["runtime"]},
                   "same_model": same_model, "model": new["model"],
                   "token_mismatches": tok_diff,
                   "old_vs_new": {"dense": ds, "sparse_jaccard": js},
                   "vs_stored": {"old_dense": so, "new_dense": sn,
                                 "old_sparse_jaccard": jo, "new_sparse_jaccard": jn},
                   "n_chunks": len(pairs), "n_questions": len(qpairs)},
                  open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")
    return 0 if verdict == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="embed a sample inside an image")
    d.add_argument("--config", default="/config/config.yaml")
    d.add_argument("--collection", default="teamcenter")
    d.add_argument("--n", type=int, default=400)
    d.add_argument("--ids-from", help="a previous dump: embed exactly its chunks")
    d.add_argument("--queries", help="questions YAML (notes/gate2/questions-*.yaml)")
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="old dump vs new dump")
    c.add_argument("--old", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--json", dest="json_out")
    args = ap.parse_args()
    return dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())

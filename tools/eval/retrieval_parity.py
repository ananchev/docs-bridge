"""Gate 2 - retrieval parity: does the ONNX stack return the same results as the
FlagEmbedding stack? This is the user-facing quality gate.

Method (content-agnostic, no hand-written questions needed): sample chunk texts and
use each as a query. Embed with FlagEmbedding and search the FlagEmbedding collection
(`--ref`, the ground-truth index); embed the SAME text with ONNX and search the ONNX
collection (`--cand`); compare the top-k chunk_id sets (excluding the query's own
chunk). High overlap = the backends rank the corpus the same way → the swap is
invisible to search. Dense and sparse are measured separately (they test different
signals). Optionally also run a file of real questions (--queries).

Also runs a self-match sanity check: a chunk embedded by ONNX must retrieve itself at
rank 1 with score ~1.0 in the ONNX collection.

    python tools/retrieval_parity.py --config /config/config.yaml \
        --ref aig --cand aig_onnx --model-dir /data/cache/bge-m3-onnx --int8 \
        --n 100 --k 10

Both models are loaded one at a time (FlagEmbedding first, released, then ONNX) to
keep RAM low. Thresholds are advisory (printed PASS/WARN), not hard exits — you and
the assistant read the numbers together with the speed result to decide adoption.
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
from qdrant_client import models as qm

from docs_bridge import config, qdrant_io
from docs_bridge.embed import Embedder
from docs_bridge.embed_onnx import OnnxEmbedder

DENSE_OVERLAP_OK = 0.70
SPARSE_OVERLAP_OK = 0.60


def _sample(client, collection: str, n: int) -> list[tuple[str, str]]:
    pts, _ = client.scroll(collection, limit=n, with_payload=True, with_vectors=False)
    return [
        (p.payload["chunk_id"], p.payload["text"])
        for p in pts
        if (p.payload or {}).get("text", "").strip()
    ]


def _dense_topk(client, col, vec, k, exclude) -> list[str]:
    res = client.search(
        col, query_vector=qm.NamedVector(name="dense", vector=vec),
        limit=k + 1, with_payload=["chunk_id"],
    )
    out = [r.payload["chunk_id"] for r in res if r.payload["chunk_id"] != exclude]
    return out[:k]


def _sparse_topk(client, col, sv, k, exclude) -> list[str]:
    res = client.search(
        col,
        query_vector=qm.NamedSparseVector(
            name="sparse", vector=qm.SparseVector(indices=sv.indices, values=sv.values)
        ),
        limit=k + 1, with_payload=["chunk_id"],
    )
    out = [r.payload["chunk_id"] for r in res if r.payload["chunk_id"] != exclude]
    return out[:k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="/config/config.yaml")
    ap.add_argument("--ref", required=True, help="ground-truth collection (FlagEmbedding)")
    ap.add_argument("--cand", required=True, help="candidate collection (ONNX)")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--n", type=int, default=100, help="number of chunk-as-query probes")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", help="optional newline-delimited real questions file")
    args = ap.parse_args()

    cfg = config.load(args.config)
    client = qdrant_io.connect(cfg)

    probes = _sample(client, cfg.subject(args.ref).collection, args.n)
    chunk_ids = [c for c, _ in probes]
    texts = [t for _, t in probes]
    extra_q = []
    if args.queries:
        extra_q = [ln.strip() for ln in open(args.queries) if ln.strip()]
    print(f"{len(texts)} chunk-as-query probes + {len(extra_q)} real questions; k={args.k}")

    ref_col = cfg.subject(args.ref).collection

    # --- FlagEmbedding side (ground truth) -> search ref collection -----------
    print("FlagEmbedding: embedding probes + searching ref ...")
    flag = Embedder(cfg.embedding_model)
    fd, fs = flag.encode(texts)
    ref_dense = [_dense_topk(client, ref_col, fd[i], args.k, chunk_ids[i])
                 for i in range(len(texts))]
    ref_sparse = [_sparse_topk(client, ref_col, fs[i], args.k, chunk_ids[i])
                  for i in range(len(texts))]
    qd, qs = (flag.encode(extra_q) if extra_q else ([], []))
    ref_q_dense = [_dense_topk(client, ref_col, qd[i], args.k, None) for i in range(len(extra_q))]
    # CONTROL self-match: torch query of a chunk finds itself at rank 1 in the torch
    # `aig` index. Same 20 probes / same logic as the ONNX self-match below, so the two
    # are directly comparable. If this is also ~15/20, the imperfection is the corpus +
    # Qdrant scalar quant (shared by both backends), NOT INT8 embedding loss.
    ref_self_hits = 0
    for i in range(min(20, len(texts))):
        top = _dense_topk(client, ref_col, fd[i], 1, None)
        if top and top[0] == chunk_ids[i]:
            ref_self_hits += 1
    del flag
    gc.collect()

    # --- ONNX side -> search cand collection ----------------------------------
    print("ONNX: embedding probes + searching cand ...")
    onnx = OnnxEmbedder(args.model_dir, use_int8=args.int8)
    od, os_ = onnx.encode(texts)
    cand_dense = [_dense_topk(client, args.cand, od[i], args.k, chunk_ids[i])
                  for i in range(len(texts))]
    cand_sparse = [_sparse_topk(client, args.cand, os_[i], args.k, chunk_ids[i])
                   for i in range(len(texts))]
    oqd, _ = (onnx.encode(extra_q) if extra_q else ([], []))
    cand_q_dense = [_dense_topk(client, args.cand, oqd[i], args.k, None) for i in range(len(extra_q))]

    # self-match sanity: ONNX query of a chunk finds itself at rank 1 in cand
    self_hits = 0
    for i in range(min(20, len(texts))):
        top = _dense_topk(client, args.cand, od[i], 1, None)
        if top and top[0] == chunk_ids[i]:
            self_hits += 1
    del onnx
    gc.collect()

    def overlap(a: list[list[str]], b: list[list[str]]) -> float:
        vals = [len(set(x) & set(y)) / args.k for x, y in zip(a, b)]
        return float(np.mean(vals)) if vals else float("nan")

    d_ov = overlap(ref_dense, cand_dense)
    s_ov = overlap(ref_sparse, cand_sparse)
    q_ov = overlap(ref_q_dense, cand_q_dense) if extra_q else float("nan")

    print("\n=== GATE 2: retrieval parity (ONNX vs FlagEmbedding) ===")
    print(f"  dense  overlap@{args.k} : {d_ov:.3f}  "
          f"{'PASS' if d_ov >= DENSE_OVERLAP_OK else 'WARN'} (>= {DENSE_OVERLAP_OK})")
    print(f"  sparse overlap@{args.k} : {s_ov:.3f}  "
          f"{'PASS' if s_ov >= SPARSE_OVERLAP_OK else 'WARN'} (>= {SPARSE_OVERLAP_OK})")
    if extra_q:
        print(f"  real-question dense overlap@{args.k}: {q_ov:.3f}")
    print(f"  self-match (ONNX  chunk -> rank1 in cand): {self_hits}/20")
    print(f"  self-match (torch chunk -> rank1 in ref ): {ref_self_hits}/20  "
          f"<- CONTROL: if ~= the ONNX line, 15/20 is corpus+Qdrant, not INT8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

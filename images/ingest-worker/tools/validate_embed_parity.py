"""Gate 1 - vector parity: does the ONNX (and INT8) backend reproduce FlagEmbedding?

The FlagEmbedding/torch path is GROUND TRUTH. This samples real chunk texts from a
populated Qdrant collection, embeds them with FlagEmbedding and with the ONNX
backend (fp32, then INT8 if present), and reports objective agreement with hard
thresholds — no human judgement needed.

    python tools/validate_embed_parity.py --config /config/config.yaml \
        --subject aig --model-dir /data/cache/bge-m3-onnx --n 200

DENSE  : cosine(flag, onnx) per text. fp32 must be ~1.0 (proves correct conversion);
         INT8 is measured (expect ~0.99).
SPARSE : Jaccard of the activated token-id sets + Pearson of weights on shared ids.

Exits non-zero if the fp32 gate fails (conversion is wrong). INT8 is reported, not
gated here — its real test is Gate 2 (retrieval) plus the speed win.
"""

from __future__ import annotations

import argparse
import gc

import numpy as np

from ingest_worker import config, qdrant_io
from ingest_worker.embed import Embedder, SparseVec
from ingest_worker.embed_onnx import OnnxEmbedder

# fp32 ONNX must essentially equal torch; below this the conversion is broken.
FP32_DENSE_MIN = 0.999
FP32_SPARSE_JACCARD_MIN = 0.95


def _sample_texts(client, collection: str, n: int) -> list[str]:
    points, _ = client.scroll(
        collection_name=collection, limit=n, with_payload=True, with_vectors=False
    )
    texts = [p.payload.get("text", "") for p in points]
    return [t for t in texts if t.strip()]


def _encode_all(emb, texts: list[str], batch: int = 16):
    dense: list[list[float]] = []
    sparse: list[SparseVec] = []
    for i in range(0, len(texts), batch):
        d, s = emb.encode(texts[i : i + batch])
        dense.extend(d)
        sparse.extend(s)
    return np.array(dense, dtype=np.float32), sparse


def _dense_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # vectors are L2-normalized on both sides -> cosine is the row-wise dot product
    return np.sum(a * b, axis=1)


def _sparse_metrics(fa: list[SparseVec], fb: list[SparseVec]):
    jac, corr = [], []
    for a, b in zip(fa, fb):
        sa, sb = set(a.indices), set(b.indices)
        union = sa | sb
        jac.append(len(sa & sb) / len(union) if union else 1.0)
        shared = sa & sb
        if len(shared) >= 2:
            da = dict(zip(a.indices, a.values))
            db = dict(zip(b.indices, b.values))
            va = np.array([da[i] for i in shared])
            vb = np.array([db[i] for i in shared])
            if va.std() > 0 and vb.std() > 0:
                corr.append(float(np.corrcoef(va, vb)[0, 1]))
    return float(np.mean(jac)), (float(np.mean(corr)) if corr else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="/config/config.yaml")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    cfg = config.load(args.config)
    collection = cfg.subject(args.subject).collection
    client = qdrant_io.connect(cfg)

    texts = _sample_texts(client, collection, args.n)
    print(f"sampled {len(texts)} chunk texts from '{collection}'")
    if not texts:
        print("no texts found — is the collection populated?")
        return 2

    # Ground truth first, then release torch before loading ONNX sessions.
    print("embedding with FlagEmbedding (ground truth) ...")
    flag = Embedder(cfg.embedding_model)
    fd, fs = _encode_all(flag, texts)
    del flag
    gc.collect()

    rc = 0

    print("embedding with ONNX fp32 ...")
    onnx32 = OnnxEmbedder(args.model_dir, use_int8=False)
    od, os32 = _encode_all(onnx32, texts)
    cos = _dense_cosine(fd, od)
    jac, corr = _sparse_metrics(fs, os32)
    print(f"\n=== ONNX-fp32 vs FlagEmbedding (n={len(texts)}) ===")
    print(f"  dense cosine : min={cos.min():.5f} mean={cos.mean():.5f} "
          f"p5={np.percentile(cos,5):.5f}")
    print(f"  sparse Jaccard: mean={jac:.4f}   weight Pearson: mean={corr:.4f}")
    fp32_pass = cos.min() >= FP32_DENSE_MIN and jac >= FP32_SPARSE_JACCARD_MIN
    print(f"  GATE 1 (fp32): {'PASS' if fp32_pass else 'FAIL'} "
          f"(need dense_min>={FP32_DENSE_MIN}, jaccard>={FP32_SPARSE_JACCARD_MIN})")
    if not fp32_pass:
        rc = 1
    del onnx32
    gc.collect()

    import os
    if os.path.exists(os.path.join(args.model_dir, "model.int8.onnx")):
        print("\nembedding with ONNX INT8 ...")
        onnx8 = OnnxEmbedder(args.model_dir, use_int8=True)
        od8, os8 = _encode_all(onnx8, texts)
        cos8 = _dense_cosine(fd, od8)
        jac8, corr8 = _sparse_metrics(fs, os8)
        print(f"\n=== ONNX-INT8 vs FlagEmbedding (n={len(texts)}) ===")
        print(f"  dense cosine : min={cos8.min():.5f} mean={cos8.mean():.5f} "
              f"p5={np.percentile(cos8,5):.5f}")
        print(f"  sparse Jaccard: mean={jac8:.4f}   weight Pearson: mean={corr8:.4f}")
        print("  (INT8 is reported, not gated here — Gate 2 retrieval + speed decide)")

    print(f"\nexit {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

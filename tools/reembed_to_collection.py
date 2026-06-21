"""Re-embed an existing collection's chunks with the ONNX backend into a NEW
collection — the substrate for Gate 2 (retrieval parity) and the embed-speed A/B.

It reads chunk texts straight from a populated source collection's payloads (every
point already carries `text` + the full metadata), re-embeds them with OnnxEmbedder,
and upserts to a target collection using the SAME point ids/payload. So:
  * the FlagEmbedding source collection (e.g. `aig`) is never touched — it stays the
    frozen ground truth;
  * no re-parse and no manifest are involved;
  * point ids line up 1:1 across collections, so retrieval results are directly
    comparable by chunk_id.

    python tools/reembed_to_collection.py --config /config/config.yaml \
        --source aig --target aig_onnx --model-dir /data/cache/bge-m3-onnx --int8 --fresh
"""

from __future__ import annotations

import argparse
import time

from docs_bridge import config, qdrant_io
from docs_bridge.embed_onnx import OnnxEmbedder
from docs_bridge.models import Chunk

_FIELDS = (
    "doc_id", "subject", "source_path", "chunk_index",
    "text", "section_path", "content_hash", "last_updated",
)


def _chunk_from_payload(p: dict) -> Chunk:
    return Chunk(**{k: p.get(k) for k in _FIELDS})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="/config/config.yaml")
    ap.add_argument("--source", required=True, help="source subject name (e.g. aig)")
    ap.add_argument("--target", required=True, help="target collection (e.g. aig_onnx)")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--int8", action="store_true", help="use the INT8 model (else fp32)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fresh", action="store_true", help="drop target first")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after ~N chunks (speed probe); 0 = whole collection")
    args = ap.parse_args()

    cfg = config.load(args.config)
    src = cfg.subject(args.source).collection
    client = qdrant_io.connect(cfg)

    if args.fresh and client.collection_exists(args.target):
        print(f"dropping existing target collection {args.target}")
        client.delete_collection(args.target)
    qdrant_io.ensure_collection(client, cfg, args.target)  # same dim/quant as the app

    emb = OnnxEmbedder(args.model_dir, use_int8=args.int8)
    print(f"re-embedding {src} -> {args.target} (int8={args.int8}, batch={args.batch})")

    done = 0
    t0 = time.time()
    offset = None
    buf: list[Chunk] = []

    def flush() -> None:
        nonlocal done
        if not buf:
            return
        dense, sparse = emb.encode([c.text for c in buf])
        qdrant_io.upsert(client, args.target, buf, dense, sparse)
        done += len(buf)
        dt = time.time() - t0
        print(f"  {done}  {dt/done:.3f}s/chunk  elapsed={dt:.0f}s", flush=True)
        buf.clear()

    stop = False
    while not stop:
        points, offset = client.scroll(
            collection_name=src, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in points:
            if not (p.payload or {}).get("text", "").strip():
                continue
            buf.append(_chunk_from_payload(p.payload))
            if len(buf) >= args.batch:
                flush()
                if args.limit and done >= args.limit:
                    stop = True
                    break
        if offset is None:
            break
    flush()

    dt = time.time() - t0
    print(f"DONE {done} chunks -> {args.target} in {dt:.0f}s "
          f"({dt/max(done,1):.3f}s/chunk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

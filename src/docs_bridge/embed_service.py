"""The embedder service: bge-m3 on one device at one precision, over HTTP.

EC2's embedder is its own unit (plan §8 3b; stack-upgrade-strategy.md §7 item 1):
this process, in its own image with its own pinned onnxruntime, is the ONLY place
the model runs. The ingest worker embeds chunks through it and the server embeds
queries through it (embed.ServiceEmbedder), so query and document vectors come
from one runtime by construction, and either caller can be rebuilt or upgraded
without touching the vectors. It can move to another instance: callers know it
only by `embed.url`.

Surface (no auth -- like Qdrant, it sits on the private network only; plan §7):
  GET  /healthz          {"status": "ok"} once the model is loaded and checked
  GET  /v1/fingerprint   {"id", "fingerprint"} -- what decides the vectors
                         (OnnxEmbedder.fingerprint); the worker stamps it on the
                         collection, so a different embedder is refused there
  POST /v1/embed         {"texts": [...]} -> {"fingerprint_id", "dense", "sparse"}

Every flag is required -- model dir, device, precision, address -- so the unit a
collection was stamped with is stated in the deploy, never picked by omission.
Startup hashes the model files (fingerprint) and runs one embed, so a missing
file, a second onnxruntime build or a CUDA provider that did not load stops the
service before it listens.

Encodes run one at a time (a lock): OnnxEmbedder's TOKEN_BUDGET bounds the memory
of ONE run, and an ingest batch and a query sharing the GPU would double it. A
query arriving during an ingest batch waits for that batch (~0.1-1 s on the T4).
"""

from __future__ import annotations

import argparse
import logging
import threading

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .embed import fingerprint_id
from .embed_onnx import OnnxEmbedder
from .ort_session import PRECISIONS, check_precision

log = logging.getLogger(__name__)


def build_app(embedder: OnnxEmbedder) -> Starlette:
    fp = embedder.fingerprint
    fid = fingerprint_id(fp)
    lock = threading.Lock()

    def encode(texts: list[str]):
        with lock:
            return embedder.encode(texts)

    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def fingerprint(_: Request) -> JSONResponse:
        return JSONResponse({"id": fid, "fingerprint": fp})

    async def embed(request: Request) -> JSONResponse:
        try:
            texts = (await request.json())["texts"]
        except Exception:
            return JSONResponse({"error": 'body must be {"texts": [...]}'}, status_code=400)
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return JSONResponse({"error": "texts must be a list of strings"}, status_code=400)
        if not texts:
            return JSONResponse({"fingerprint_id": fid, "dense": [], "sparse": []})
        try:
            dense, sparse = await run_in_threadpool(encode, texts)
        except Exception:
            log.exception("embed failed (%d texts)", len(texts))
            return JSONResponse({"error": "embed failed"}, status_code=500)
        return JSONResponse({
            "fingerprint_id": fid,
            "dense": dense,
            "sparse": [{"indices": s.indices, "values": s.values} for s in sparse],
        })

    return Starlette(routes=[
        Route("/healthz", healthz, methods=["GET"]),
        Route("/v1/fingerprint", fingerprint, methods=["GET"]),
        Route("/v1/embed", embed, methods=["POST"]),
    ])


def serve(argv: list[str] | None = None) -> None:
    """Console-script entrypoint (`docs-bridge-embedder`)."""
    p = argparse.ArgumentParser(prog="docs-bridge-embedder", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, help="bge-m3 model dir (models image)")
    p.add_argument("--device", required=True, choices=sorted(PRECISIONS))
    p.add_argument("--precision", required=True, choices=["fp16", "fp32", "int8"])
    p.add_argument("--host", required=True)
    p.add_argument("--port", required=True, type=int)
    args = p.parse_args(argv)
    try:
        check_precision("embedder", args.device, args.precision, "the command line")
    except ValueError as e:
        p.error(str(e))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    embedder = OnnxEmbedder(args.model_dir, device=args.device, precision=args.precision)
    log.info("hashing the model files (fingerprint)")
    fp = embedder.fingerprint
    log.info("fingerprint %s: %s", fingerprint_id(fp), fp)
    embedder.encode(["startup check"])  # pays CUDA's one-off initialisation, too
    app = build_app(embedder)

    import uvicorn

    log.info("embedder service on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    serve()

# docs-bridge server — hybrid search -> rerank -> MCP/REST (design §9/§15).
#
# Build CONTEXT is the repo root (shares the docs_bridge package + tools/ with the
# ingest-worker image), so build with:
#     docker build -f images/docs-bridge/Dockerfile -t docs-bridge:latest .
#
# aarch64 Linux only (Pi 5 + M2 Asahi); build once, run on both (§1). Build on the
# M2 (16GB) and distribute the saved image — the 8GB Pi cannot run the INT8 quantize
# step (OOM), same as the ingest-worker.
#
# The server holds TWO baked INT8 models resident: the bge-m3 embedder (for the
# query vector — the SAME model ingest used, so vectors match) and the bge-reranker
# cross-encoder. Both are exported in a discarded `exporter` stage; the runtime
# image copies in only the INT8 files. No torch / no Docling at runtime.
FROM python:3.11-slim-bookworm AS base

# onnxruntime needs libgomp (OpenMP). No libgl/Docling deps — the server never parses.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/data/cache \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
# .[server] = the shared core + FastMCP + uvicorn. Lean: no torch, no Docling.
RUN pip install --no-cache-dir '.[server]'

# --- exporter: build-time self-export of BOTH INT8 models ---------------------
# Discarded stage. .[ingest,export] reproduces the exact torch+transformers env that
# produced the validated bge-m3 INT8 model, plus onnx. Each model is exported then
# quantized in SEPARATE processes so the torch (~2.3GB) and quantize (~4-6x model)
# peaks never stack — the 8GB Pi rule, harmless on the M2.
FROM base AS exporter
COPY tools ./tools
RUN pip install --no-cache-dir '.[ingest,export]'
RUN python tools/export_bge_m3_onnx.py --out /opt/bge-m3-onnx
RUN python tools/export_bge_m3_onnx.py --out /opt/bge-m3-onnx --quantize-only
RUN python tools/export_bge_reranker_onnx.py --out /opt/bge-reranker-onnx
RUN python tools/export_bge_reranker_onnx.py --out /opt/bge-reranker-onnx --quantize-only

# --- runtime: base + ONLY the INT8 files the two loaders read -----------------
FROM base AS runtime
# bge-m3 embedder (OnnxEmbedder): int8 encoder + sparse head + tokenizer + meta.
COPY --from=exporter /opt/bge-m3-onnx/model.int8.onnx \
                     /opt/bge-m3-onnx/sparse_linear.npz \
                     /opt/bge-m3-onnx/tokenizer.json \
                     /opt/bge-m3-onnx/meta.json \
                     /opt/bge-m3-onnx/
# bge-reranker cross-encoder (OnnxReranker): int8 model + tokenizer + meta.
COPY --from=exporter /opt/bge-reranker-onnx/model.int8.onnx \
                     /opt/bge-reranker-onnx/tokenizer.json \
                     /opt/bge-reranker-onnx/meta.json \
                     /opt/bge-reranker-onnx/

EXPOSE 8080
ENTRYPOINT ["docs-bridge-server"]

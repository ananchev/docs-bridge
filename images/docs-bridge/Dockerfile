# docs-bridge server — hybrid search -> rerank -> MCP/REST (design §9/§15).
#
# Build CONTEXT is the repo root (shares the docs_bridge package with the
# ingest-worker image), so build with (vhost2):
#     sudo podman build -f images/docs-bridge/Dockerfile \
#         -t localhost/docs-bridge:<tag> .
# Always tag fully qualified (localhost/...): the image deployed until 2026-09 was
# tagged from a bare name and landed under docker.io/library/, one `podman pull`
# away from being replaced by a registry image.
#
# Two targets, one Dockerfile: linux/arm64 (the Mac mini, vhost2) and linux/amd64
# (the EC2 host). TARGETARCH picks the matching lock. The CUDA reranker for the GPU
# target is plan §6.3 and not built here yet — this image is CPU on both arches.
#
# The server holds TWO INT8 models resident: the bge-m3 embedder (for the query
# vector — the SAME bytes ingest used, so vectors match) and the bge-reranker
# cross-encoder. Neither is exported here any more: both come from the frozen models
# image (images/models/Dockerfile). No torch / no Docling at runtime.
ARG MODELS_IMAGE=localhost/docs-bridge-models:frozen-2026-09
FROM ${MODELS_IMAGE} AS models

FROM python:3.11-slim-bookworm AS base

# onnxruntime needs libgomp (OpenMP). No libgl/Docling deps — the server never parses.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/data/cache \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- deps: the full locked set for this arch (hashes checked) -------------------
FROM base AS deps
ARG TARGETARCH
COPY images/docs-bridge/requirements-${TARGETARCH}.lock ./requirements.lock
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# --- runtime: deps + ONLY the INT8 files the two loaders read + the app ----------
FROM deps AS runtime
# bge-m3 embedder (OnnxEmbedder): int8 encoder + sparse head + tokenizer + meta.
COPY --from=models /opt/bge-m3-onnx/model.int8.onnx \
                   /opt/bge-m3-onnx/sparse_linear.npz \
                   /opt/bge-m3-onnx/tokenizer.json \
                   /opt/bge-m3-onnx/meta.json \
                   /opt/bge-m3-onnx/
# bge-reranker cross-encoder (OnnxReranker): int8 model + tokenizer + meta.
COPY --from=models /opt/bge-reranker-onnx/model.int8.onnx \
                   /opt/bge-reranker-onnx/tokenizer.json \
                   /opt/bge-reranker-onnx/meta.json \
                   /opt/bge-reranker-onnx/
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8080
ENTRYPOINT ["docs-bridge-server"]

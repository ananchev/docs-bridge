# docs-bridge server — hybrid search -> rerank -> MCP/REST (design §9/§15).
#
# Build CONTEXT is the repo root (shares the docs_bridge package with the
# ingest-worker image), e.g. on vhost2:
#     sudo podman build -f images/docs-bridge/Dockerfile --target runtime-cpu \
#         --build-arg MODELS_IMAGE=localhost/docs-bridge-models:frozen-2026-09 \
#         -t localhost/docs-bridge:<tag> .
# Always tag fully qualified (localhost/...): the image deployed until 2026-09 was
# tagged from a bare name and landed under docker.io/library/, one `podman pull`
# away from being replaced by a registry image.
#
# Two TARGETS, chosen explicitly (`--target`; a build without one fails):
#   runtime-cpu  INT8 embedder + reranker on CPU, linux/arm64 (the Mac mini) or
#                linux/amd64; TARGETARCH picks the lock. MODELS_IMAGE: frozen-2026-09.
#   runtime-gpu  the fp16 reranker on CUDA, linux/amd64 (EC2), from a GPU models tag:
#       docker build -f images/docs-bridge/Dockerfile --target runtime-gpu \
#           --build-arg MODELS_IMAGE=localhost/docs-bridge-models:gpu-2026-10 \
#           -t localhost/docs-bridge:gpu-<commit> .
#     and run with `--gpus all`; the server refuses to start without the CUDA provider.
#
# Models are never exported here: they come from a models image (images/models/),
# named by the REQUIRED build arg MODELS_IMAGE -- no default, so a build cannot pick
# up the wrong precision's models by omission. No torch / no Docling at runtime.
ARG MODELS_IMAGE
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

# --- deps: the full locked set (hashes checked), one per onnxruntime build --------
FROM base AS deps-cpu
ARG TARGETARCH
COPY images/docs-bridge/requirements-${TARGETARCH}.lock ./requirements.lock
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# onnxruntime-gpu INSTEAD of onnxruntime, with CUDA 13 / cuDNN 9 as nvidia-* wheels
# (no nvidia/cuda base image: tools/onnx-export/FINDINGS.md finding 1). x86_64 only.
FROM base AS deps-gpu
ARG TARGETARCH
RUN [ "$TARGETARCH" = amd64 ] || { echo "runtime-gpu is linux/amd64 only" >&2; exit 1; }
COPY images/docs-bridge/requirements-amd64-gpu.lock ./requirements.lock
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# --- runtime-cpu: the INT8 line (Mac mini) — both models in-process ----------------
FROM deps-cpu AS runtime-cpu
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

# --- runtime-gpu: EC2 — the fp16 reranker on CUDA ----------------------------------
# MODELS_IMAGE must be a GPU models tag (images/models/gpu.Dockerfile). No embedder:
# EC2 embeds queries through the embedder service (plan §8 3b), so this image holds
# no bge-m3 at all, and a search fails loudly until the service is wired in.
FROM deps-gpu AS runtime-gpu
COPY --from=models /opt/bge-reranker-onnx/model.fp16.onnx \
                   /opt/bge-reranker-onnx/tokenizer.json \
                   /opt/bge-reranker-onnx/meta.json \
                   /opt/bge-reranker-onnx/
COPY --from=models /opt/docs-bridge-models.json /opt/docs-bridge-models.json
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
EXPOSE 8080
ENTRYPOINT ["docs-bridge-server"]

# --- no default: a build without --target stops here -------------------------------
FROM base AS choose-a-target
RUN echo "build with --target runtime-cpu or --target runtime-gpu" >&2; exit 1

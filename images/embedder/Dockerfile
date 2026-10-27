# embedder — bge-m3 on the GPU behind HTTP (src/docs_bridge/embed_service.py).
#
# EC2's embedder unit (plan §8 3b): the frozen model plus its own pinned
# onnxruntime-gpu and nothing else -- no Qdrant client, no Docling, no torch -- so no
# other dependency bump can reach the vectors. The worker and the server call it by
# address (embed.url). Build CONTEXT is the repo root:
#     docker build -f images/embedder/Dockerfile \
#         --build-arg MODELS_IMAGE=localhost/docs-bridge-models:gpu-2026-10 \
#         -t localhost/docs-bridge-embedder:gpu-<commit> .
# and run with `--gpus all` and every flag of docs-bridge-embedder (README.md).
#
# MODELS_IMAGE has no default: a build cannot pick up the wrong model by omission.
# It must be a GPU models tag (images/models/gpu.Dockerfile). linux/amd64 only.
ARG MODELS_IMAGE
FROM ${MODELS_IMAGE} AS models

FROM python:3.11-slim-bookworm AS deps
ARG TARGETARCH
RUN [ "$TARGETARCH" = amd64 ] || { echo "the embedder image is linux/amd64 only" >&2; exit 1; }
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
# onnxruntime-gpu with CUDA 13 / cuDNN 9 as nvidia-* wheels (no nvidia/cuda base
# image: tools/onnx-export/FINDINGS.md finding 1).
COPY images/embedder/requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

FROM deps AS runtime
# bge-m3 at BOTH GPU precisions until the 14.2 question set decides between them
# (tools/onnx-export/FINDINGS.md finding 7); the service runs the one --precision
# names. fp32 is the graph plus its 2.3 GB external-data file.
COPY --from=models /opt/bge-m3-onnx/model.onnx \
                   /opt/bge-m3-onnx/model.onnx.data \
                   /opt/bge-m3-onnx/model.fp16.onnx \
                   /opt/bge-m3-onnx/sparse_linear.npz \
                   /opt/bge-m3-onnx/tokenizer.json \
                   /opt/bge-m3-onnx/meta.json \
                   /opt/bge-m3-onnx/
COPY --from=models /opt/docs-bridge-models.json /opt/docs-bridge-models.json
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
# Not root: the model files are 0644 (tools/onnx-export/FINDINGS.md finding 3).
USER nobody
EXPOSE 8090
ENTRYPOINT ["docs-bridge-embedder"]

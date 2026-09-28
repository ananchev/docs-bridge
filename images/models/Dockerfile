# docs-bridge-models — the frozen ONNX model artifacts both runtime images copy in.
#
# WHY THIS EXISTS. Until 2026-09 each image re-exported bge-m3 (and the server also
# bge-reranker-v2-m3) at build time, from whatever torch / transformers / onnx /
# onnxruntime the build installed, pulling the HF weights at `main`. That made the
# model a by-product of the dependency set: bumping any of those — as the 2026-09
# toolchain upgrade does — would silently re-export a *different* model.int8.onnx
# under the 137k `teamcenter` vectors the old one produced. The model is now an
# artifact with its own identity, and the runtime images never build one.
#
# THIS BUILD takes the bytes out of the images that are deployed right now, so it is
# not a re-export at all:
#   bge-m3             <- the INGEST-WORKER image (it embedded the stored vectors)
#   bge-reranker-v2-m3 <- the SERVER image (the INT8 reranker Gate 1 measured)
# Build it once, before the old images are replaced, on vhost2:
#
#   sudo podman build -f images/models/Dockerfile \
#       --build-arg WORKER_IMAGE=localhost/ingest-worker:latest \
#       --build-arg SERVER_IMAGE=docker.io/library/docs-bridge:latest \
#       -t localhost/docs-bridge-models:frozen-2026-09 .
#
# and keep it (`podman save` it next to the Qdrant snapshots). A future NEW model
# (e.g. the fp16 reranker for the GPU target, plan §6.3) is made with
# tools/onnx-export/ and published as a new tag of this image — never exported
# inside a runtime build again.
ARG WORKER_IMAGE=localhost/ingest-worker:latest
ARG SERVER_IMAGE=docker.io/library/docs-bridge:latest
FROM ${WORKER_IMAGE} AS worker
FROM ${SERVER_IMAGE} AS server

FROM scratch
COPY --from=worker /opt/bge-m3-onnx/ /opt/bge-m3-onnx/
COPY --from=server /opt/bge-reranker-onnx/ /opt/bge-reranker-onnx/

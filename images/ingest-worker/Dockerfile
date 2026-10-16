# ingest-worker — two-pass Docling -> BGE-M3 -> Qdrant ingestion.
#
# Build CONTEXT is the repo root (the docs_bridge package is shared with the
# docs-bridge server image), so build with (vhost2):
#     sudo podman build -f images/ingest-worker/Dockerfile \
#         -t localhost/ingest-worker:<tag> .
# Always tag fully qualified (localhost/...): a bare name is normalised to
# docker.io/library/, one `podman pull` away from being replaced by a registry image.
#
# linux/arm64 only: the worker runs on vhost2 (M2 Pro, Asahi), which ingests BOTH
# corpora — the Mac mini's `teamcenter` and the EC2 work archive (plan §1.1).
#
# LAYER ORDER keeps an app-code change a SECONDS rebuild: the heavy dependency
# install is keyed on the lock file only, the model copy on the models image only,
# and src/ is copied and installed (--no-deps) in the very last layer.
#
# The BGE-M3 INT8 model is NOT exported here any more. It comes from the frozen
# models image (images/models/Dockerfile) — the exact bytes that embedded the stored
# `teamcenter` vectors — so a dependency bump can never change the model under them.
ARG MODELS_IMAGE=localhost/docs-bridge-models:frozen-2026-09
FROM ${MODELS_IMAGE} AS models

FROM python:3.11-slim-bookworm AS base

# Docling pulls PDFs via pypdfium2 and rasterises; RapidOCR uses OpenCV. These cover
# their native deps. Torch/onnxruntime wheels are self-contained on arm64.
#
# FONTS are not optional. Docling's layout model looks at a RENDERED image of each
# page, and many PDFs do not embed the base-14 fonts — the Teamcenter manuals set every
# code sample in non-embedded Courier. With no font on the system (the slim image has
# none) pdfium draws those blocks badly, the layout model sees pictures / formulas /
# checkboxes instead of code, and their text is dropped from the chunks: measured
# 2026-10-15, ~70% of command lines lost from the 2512 manuals (`taxonomy -f=...`).
# fonts-urw-base35 = the metric-compatible substitutes for Courier/Helvetica/Times
# (+ Symbol, ZapfDingbats); fonts-liberation2 = the same for Arial / Times New Roman /
# Courier New, common in Office-made PDFs. tools/phase-a/1-build.sh checks for them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        fontconfig \
        fonts-urw-base35 \
        fonts-liberation2 \
    && rm -rf /var/lib/apt/lists/*

# HOME on the mounted data volume so the heavy model caches (HuggingFace + Docling
# layout/table artifacts, the chunker's tokenizer) persist across container recreates
# and are not re-downloaded on every run. RapidOCR's ONNX models ship inside its wheel.
ENV HOME=/data/cache \
    HF_HOME=/data/cache/huggingface \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- deps: the full locked set (hashes checked), keyed ONLY on the lock ---------
# torch is the +cpu wheel from the PyTorch index (see images/lock.sh for why).
FROM base AS deps
COPY images/ingest-worker/requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock \
        --extra-index-url https://download.pytorch.org/whl/cpu

# --- runtime: deps + the frozen model + the app source ---------------------------
# Copy the four files OnnxEmbedder loads, NOT the whole dir (the frozen image holds
# only those, but being explicit keeps a stray fp32 model.onnx + its ~2.2 GB external
# data blob out of the image if a future models tag carries one).
FROM deps AS runtime
COPY --from=models /opt/bge-m3-onnx/model.int8.onnx \
                   /opt/bge-m3-onnx/sparse_linear.npz \
                   /opt/bge-m3-onnx/tokenizer.json \
                   /opt/bge-m3-onnx/meta.json \
                   /opt/bge-m3-onnx/
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# Args are the CLI: e.g. `podman run ... ingest-worker sync --subject all`
ENTRYPOINT ["ingest"]
CMD ["sync", "--subject", "all"]

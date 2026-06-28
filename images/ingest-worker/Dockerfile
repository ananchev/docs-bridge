# ingest-worker — two-pass Docling -> BGE-M3 -> Qdrant ingestion.
#
# Build CONTEXT is the repo root (the docs_bridge package + tools/ are shared with
# the docs-bridge server image), so build with:
#     docker build -f images/ingest-worker/Dockerfile -t ingest-worker:latest .
#
# Both target hosts (Pi 5 Rocky 10, M2 Asahi Fedora) are aarch64 Linux, so this
# builds once and runs on both — no multi-arch (design §1). Build on the M2 (16GB)
# and distribute: the 8GB Pi cannot run the INT8 quantize step (OOM).
#
# LAYER ORDER is tuned so a change to src/ (app code) rebuilds in SECONDS: the heavy
# .[ingest] dependency install and the multi-minute BGE-M3 ONNX export are keyed on
# pyproject.toml + the export tool ONLY, never on src/. Application source is copied
# and reinstalled (--no-deps) in the very last stage, so only that small layer plus
# the (cached) model copy re-run when app code changes.
#
# Multi-stage: the default embed backend is ONNX/INT8 (validated 2026-06-21, 4.5x
# faster than FlagEmbedding/torch at on-par retrieval quality). The `exporter` stage
# self-exports BAAI/bge-m3 -> INT8 ONNX at build time; the final image copies in ONLY
# the ~600 MB INT8 model (the 2.2 GB fp32 model.onnx is dropped). The model is host-
# invariant so baking it does not violate the config-only portability model (design
# §6) — only host-VARIABLE values stay in the mounted config.
FROM python:3.11-slim-bookworm AS base

# Docling pulls PDFs via pypdfium2 and may rasterise; these cover the common
# native deps. Torch/transformers wheels are self-contained on arm64.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HOME on the mounted data volume so the heavy model caches (HuggingFace +
# Docling layout/OCR artifacts) persist across container recreates and are not
# re-downloaded on every run. containers-at-home mounts /data (with :Z).
ENV HOME=/data/cache \
    HF_HOME=/data/cache/huggingface \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- deps: install the heavy .[ingest] dependency set keyed ONLY on pyproject.toml.
# A STUB package satisfies the local-project install (setuptools finds src/docs_bridge)
# without copying the real src/, so editing app code busts neither this layer nor the
# export stage that derives from it. The real package is installed --no-deps in the
# runtime stage. pyproject pins all deps as a set (§1), so this layer is stable.
FROM base AS deps
COPY pyproject.toml ./
RUN mkdir -p src/docs_bridge && touch src/docs_bridge/__init__.py \
    && pip install --no-cache-dir '.[ingest]'

# --- exporter: build-time self-export of the INT8 ONNX model -----------------
# Discarded stage: pulls BAAI/bge-m3 once (network at build) and exports the encoder
# with torch.onnx + dynamic INT8. Depends on `deps` + the .[export] extra (onnx) + the
# export tool — but NOT on src/, so app-code changes leave it fully cached. Nothing
# from this stage reaches the final image except the four INT8 files copied below.
FROM deps AS exporter
RUN pip install --no-cache-dir '.[export]'      # adds onnx; the stub satisfies '.'
COPY tools/onnx-export ./tools/onnx-export
# Two separate RUNs on purpose: the torch export (~2.3GB resident) and the INT8
# quantize (loads the 2.3GB fp32 ONNX) must NOT peak at the same time, or the 8GB
# Pi OOM-kills the build (exit 137). Splitting into two processes keeps each peak
# well under 8GB; the M2 (16GB) is unaffected. Output is byte-identical either way.
RUN python tools/onnx-export/export_bge_m3_onnx.py --out /opt/bge-m3-onnx
RUN python tools/onnx-export/export_bge_m3_onnx.py --out /opt/bge-m3-onnx --quantize-only

# --- runtime: deps + the real app source + ONLY the INT8 runtime files -------
# FROM deps (NOT exporter) so the discarded onnx export dep doesn't bloat the image.
# Copy the real src over the stub and force-reinstall --no-deps (deps already present;
# the version is unchanged so a plain install would no-op and ship the stub). This is
# the ONLY layer app-code changes invalidate. Copy the four files OnnxEmbedder loads,
# NOT the whole dir: the fp32 model.onnx is >2GB so ONNX export writes its weights to a
# separate external-data blob, and a plain `rm model.onnx` would leave that ~2.2GB blob
# behind (it bloated the image to 4.9GB). model.int8.onnx is <2GB = self-contained.
FROM deps AS runtime
COPY src ./src
RUN pip install --no-cache-dir --no-deps --force-reinstall .
COPY --from=exporter /opt/bge-m3-onnx/model.int8.onnx \
                     /opt/bge-m3-onnx/sparse_linear.npz \
                     /opt/bge-m3-onnx/tokenizer.json \
                     /opt/bge-m3-onnx/meta.json \
                     /opt/bge-m3-onnx/

# Args are the CLI: e.g. `docker run ... ingest-worker sync --subject all`
ENTRYPOINT ["ingest"]
CMD ["sync", "--subject", "all"]

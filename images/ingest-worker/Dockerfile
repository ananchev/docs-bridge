# ingest-worker — two-pass Docling -> BGE-M3 -> Qdrant ingestion.
#
# Both target hosts (Pi 5 Rocky 10, M2 Asahi Fedora) are aarch64 Linux, so this
# builds once and runs on both — no multi-arch (design §1). Build on the Pi first.
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
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# --- exporter: build-time self-export of the INT8 ONNX model -----------------
# Discarded stage: pulls BAAI/bge-m3 once (network at build), exports the encoder
# with torch.onnx + dynamic INT8, then drops the fp32 model.onnx. None of the
# build-time download or the fp32 graph reaches the final image.
FROM base AS exporter
COPY tools ./tools
RUN pip install --no-cache-dir '.[export]'
RUN python tools/export_bge_m3_onnx.py --out /opt/bge-m3-onnx --quantize \
    && rm -f /opt/bge-m3-onnx/model.onnx

# --- runtime: base + the baked INT8 model ------------------------------------
FROM base AS runtime
COPY --from=exporter /opt/bge-m3-onnx /opt/bge-m3-onnx

# Args are the CLI: e.g. `docker run ... ingest-worker sync --subject all`
ENTRYPOINT ["ingest"]
CMD ["sync", "--subject", "all"]

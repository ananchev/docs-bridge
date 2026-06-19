# ingest-worker — two-pass Docling -> BGE-M3 -> Qdrant ingestion.
#
# Both target hosts (Pi 5 Rocky 10, M2 Asahi Fedora) are aarch64 Linux, so this
# builds once and runs on both — no multi-arch (design §1). Build on the Pi first.
FROM python:3.11-slim-bookworm

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

# Args are the CLI: e.g. `docker run ... ingest-worker sync --subject all`
ENTRYPOINT ["ingest"]
CMD ["sync", "--subject", "all"]

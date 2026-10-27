# ingest-worker

Two-pass, hash-delta, self-updating ingestion for docs-bridge. Parses a corpus
with **Docling**, embeds with **BGE-M3** (dense + sparse in one pass), and upserts
into **Qdrant** — one collection per subject. Full walkthrough in
[`../../docs/ingestion.md`](../../docs/ingestion.md). What was found building it:
[FINDINGS.md](FINDINGS.md).

## What it does

```
scan disk → diff vs SQLite manifest → classify each doc:
  new      (hash unseen)       → parse + embed + upsert
  changed  (hash differs)      → delete old points + re-parse + re-embed
  deleted  (gone from disk)    → delete points + forget
  unchanged                    → skip (no embed cost)
```

**Two-pass** = parse-all-then-embed-all, staging chunks to SQLite in between, so
Docling and BGE-M3 are never resident together (peak memory is bounded by the
larger model stack, not their sum).

## Targets

There is no default target; a build without `--target` fails.

| Target | Arch | Parses on | Embedder | Config |
|---|---|---|---|---|
| `runtime-gpu` | amd64 only (EC2) | CUDA — run with `--gpus all` | none in the image — chunks go to the embedder service (`images/embedder/`) | `parse: {device: cuda}`, `embed: {mode: service, url: ...}` |
| `runtime-service` | amd64 or arm64 | CPU | the embedder service | `parse: {device: cpu}`, `embed: {mode: service, url: ...}` |
| `runtime-cpu` | arm64 (Mac mini) or amd64 | CPU | INT8 bge-m3 in-process, from the **required** build arg `MODELS_IMAGE` (`frozen-2026-09`) | `parse: {device: cpu}`, `embed: {mode: local, device: cpu, precision: int8, model_dir: /opt/bge-m3-onnx}` |

`TARGETARCH` picks the lock, `requirements-<arch>.lock` (`images/lock.sh`: extras
`app` + `ingest` + `cpu`; torch from the PyTorch CPU index). `runtime-gpu` has its
own, `requirements-amd64-gpu.lock`: the amd64 lock with torch's CUDA 13 build from
PyPI and its `nvidia-*` libraries, nothing else changed (`lock.sh` checks it). All
targets carry the CPU onnxruntime: RapidOCR runs on it.

Only Docling's layout and table models move to the GPU. The PDF backend
(docling-parse) and the chunker stay on the CPU.

```bash
docker build -f images/ingest-worker/Dockerfile --target runtime-gpu \
    -t localhost/ingest-worker:gpu-<commit> .
```

## Run

```bash
# all configured subjects
docker run --rm \
  -v /data:/data:Z \
  -v /path/to/config.yaml:/config/config.yaml:ro \
  localhost/ingest-worker:<tag> sync --subject all

# one subject, verbose (idempotency check: a no-change re-sync prints 0/0/0)
docker run --rm ... ingest-worker sync --subject aig --verbose
```

Config shape: [`../../config.example.yaml`](../../config.example.yaml), documented
in [`../../docs/configuration.md`](../../docs/configuration.md). The real config is
templated and mounted by the operator (deployment) repo — this image bakes in
nothing host-specific.

## Layout

| File | Role |
|---|---|
| `cli.py` | `sync --subject <name\|all>` |
| `config.py` | load `/config/config.yaml` |
| `parse.py` | pass 1 — Docling → structure-aware chunks (owns the Docling import) |
| `embed.py` / `embed_onnx.py` | pass 2 — BGE-M3 dense+sparse, through the embedder service or in-process (`embed.mode`) |
| `manifest.py` | SQLite: persistent `docs` manifest + `staged_chunks` scratch |
| `qdrant_io.py` | collection create, idempotent upsert, doc-scoped delete |
| `sync.py` | hash-delta + two-pass orchestration; checks the collection's embedder stamp before PASS 2 deletes anything |

## Notes

- Volumes: corpora under `/data/docs/<subject>`, manifest at
  `/data/state/manifest.sqlite`, model caches under `/data/cache` (so they
  survive recreates).
- Library API pins matter — the Docling chunker (`parse.py`'s API notes), and
  qdrant-client sparse models are pinned in `pyproject.toml`; re-verify on bump.
- Collection layout (named `dense` + `sparse` vectors, `doc_id`/`subject` payload
  indexes) **must** match the docs-bridge query side.

# ingest-worker

Two-pass, hash-delta, self-updating ingestion for docs-bridge. Parses a corpus
with **Docling**, embeds with **BGE-M3** (dense + sparse in one pass), and upserts
into **Qdrant** — one collection per subject. Full walkthrough in
[`../../docs/ingestion.md`](../../docs/ingestion.md).

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

## Run

```bash
# all configured subjects
docker run --rm \
  -v /data:/data:Z \
  -v /path/to/config.yaml:/config/config.yaml:ro \
  ingest-worker sync --subject all

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
| `embed.py` / `embed_onnx.py` | pass 2 — BGE-M3 dense+sparse; ONNX/INT8 by default, FlagEmbedding/torch fallback |
| `manifest.py` | SQLite: persistent `docs` manifest + `staged_chunks` scratch |
| `qdrant_io.py` | collection create, idempotent upsert, doc-scoped delete |
| `sync.py` | hash-delta + two-pass orchestration |

## Notes

- Volumes: corpora under `/data/docs/<subject>`, manifest at
  `/data/state/manifest.sqlite`, model caches under `/data/cache` (so they
  survive recreates).
- Library API pins matter — Docling chunker, FlagEmbedding `encode` kwargs, and
  qdrant-client sparse models are pinned in `pyproject.toml`; re-verify on bump.
- Collection layout (named `dense` + `sparse` vectors, `doc_id`/`subject` payload
  indexes) **must** match the docs-bridge query side.

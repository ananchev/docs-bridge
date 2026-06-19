# ingest-worker

Two-pass, hash-delta, self-updating ingestion for docs-bridge. Parses a corpus
with **Docling**, embeds with **BGE-M3** (dense + sparse in one pass), and upserts
into **Qdrant** — one collection per subject. See the stack design in
[`../../docs-bridge-ansible-design.md`](../../docs-bridge-ansible-design.md) §8.

## What it does

```
scan disk → diff vs SQLite manifest → classify each doc:
  new      (hash unseen)       → parse + embed + upsert
  changed  (hash differs)      → delete old points + re-parse + re-embed
  deleted  (gone from disk)    → delete points + forget
  unchanged                    → skip (no embed cost)
```

**Two-pass** = parse-all-then-embed-all, staging chunks to SQLite in between, so
Docling and BGE-M3 are never resident together (the 8 GB Pi budget, design §1/§8).

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

Config shape: [`config.example.yaml`](./config.example.yaml). The real config is
templated and mounted by **containers-at-home** (design §6) — this image bakes in
nothing host-specific.

## Layout

| File | Role |
|---|---|
| `cli.py` | `sync --subject <name\|all>` |
| `config.py` | load `/config/config.yaml` |
| `parse.py` | pass 1 — Docling → structure-aware chunks (owns the Docling import) |
| `embed.py` | pass 2 — BGE-M3 dense+sparse (owns the FlagEmbedding import) |
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

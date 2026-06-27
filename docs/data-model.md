# Data model

docs-bridge keeps state in two places: **Qdrant** (the vectors and their payloads)
and a **SQLite manifest** (change-detection state and per-run staging). Source:
`qdrant_io.py`, `manifest.py`, `models.py`.

## Qdrant collections

One collection per subject (the subject's `collection` name in config). Each
collection is created on first ingest by `qdrant_io.ensure_collection()` with:

| Vector | Name | Config |
|---|---|---|
| Dense | `dense` | size = `embedding_dim` (1024), distance = cosine, `on_disk` per config |
| Sparse | `sparse` | Qdrant sparse vector |

Plus two keyword payload indexes:

- **`doc_id`** — makes doc-scoped deletes cheap (used on every change/delete).
- **`subject`** — available for query-time filtering.

Optional **scalar INT8 quantization** is enabled when `qdrant.quantization: scalar`
(keeps quantized vectors in RAM); `none` stores full-precision vectors.

### Point ids

A point's id is a deterministic **UUIDv5** of the human-readable chunk id
`"{doc_id}:{chunk_index}"` under a fixed namespace. Because the id is a pure
function of the chunk's identity, re-ingesting a document **overwrites** its points
instead of duplicating them — this is what makes upserts idempotent. The readable
`chunk_id` is also kept in the payload.

### Payload

Every point carries the full chunk so retrieval can cite and return text without a
second lookup:

| Field | Meaning |
|---|---|
| `chunk_id` | `"{doc_id}:{chunk_index}"`, human-readable |
| `doc_id` | path relative to the subject dir (posix) — the stable document id |
| `subject` | the subject this chunk belongs to |
| `source_path` | absolute path of the source file (citation) |
| `section_path` | joined heading trail, e.g. `Installation > Prerequisites` (citation) |
| `chunk_index` | position of the chunk within the document |
| `content_hash` | the **document's** SHA-256 (ties all of a doc's chunks together) |
| `last_updated` | source file mtime, ISO-8601 UTC (citation) |
| `text` | the chunk text |

> `content_hash` is the document hash, not the chunk hash. It is what hash-delta
> detection compares against the manifest, and what lets a doc's chunks be deleted
> together by `doc_id`.

## SQLite manifest

One SQLite file at `manifest_path` (default `/data/state/manifest.sqlite`) holds
two tables.

### `docs` — persistent change-detection state

One row per ingested document.

```sql
CREATE TABLE docs (
    doc_id       TEXT PRIMARY KEY,
    subject      TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    content_hash TEXT NOT NULL,   -- compared against the on-disk hash each run
    last_updated TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_docs_subject ON docs(subject);
```

On each sync the worker reads `docs` for the subject and compares hashes to
classify files as new / changed / deleted / unchanged
([ingestion.md](ingestion.md#1-scan-and-classify)).

### `staged_chunks` — per-run scratch

Written by parse (pass 1), drained by embed (pass 2), cleared at the end of the run.

```sql
CREATE TABLE staged_chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    subject      TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    section_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    text         TEXT NOT NULL
);
CREATE INDEX idx_staged_subject ON staged_chunks(subject);
```

Staging to disk (rather than holding chunks in memory) is what lets pass 1 release
Docling before pass 2 loads the embedder — the two heavy model stacks are never
co-resident. Batches are drained ordered by `(doc_id, chunk_index)` so a document's
chunks stay contiguous.

## What is authoritative

- **Qdrant** is the searchable index. Lose it and you re-ingest.
- **The manifest** is the change-detection ledger. Lose it and the next sync sees
  every file as *new* and re-ingests the whole corpus (correct, just slow).
- **The source files on disk** are the only irreplaceable input — back those up,
  plus the Qdrant collections and the manifest if you want to avoid a full re-ingest
  after a restore.

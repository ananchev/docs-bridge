# Ingestion

The ingest-worker is a **one-shot CLI**. Each run performs an incremental,
hash-delta sync of one or more subjects and then exits. Schedule it (systemd timer,
cron, k8s CronJob) for nightly updates.

```bash
ingest sync --subject all          # every configured subject
ingest sync --subject teamcenter   # one subject
ingest sync --subject all -v       # verbose (DEBUG) logging
```

Deployed form:

```bash
docker run --rm \
  -v ./config.yaml:/config/config.yaml:ro -v ./data:/data \
  ingest-worker:latest sync --subject all
```

The CLI always exits `0`; the printed per-subject summary line is the signal a
scheduler reads. Source: `cli.py`, `sync.py`.

## The sync algorithm

For each subject, `sync.sync_subject()` runs:

### 1. Scan and classify

`parse.scan()` walks the subject directory recursively and builds a map of
`doc_id → path` for every file whose suffix is supported and that passes the
include/exclude filters (see [below](#file-filters)). The `doc_id` is the file's
path **relative to the subject dir** (posix) — stable across runs and readable in
citations.

Each on-disk file is SHA-256 hashed, then compared against the manifest's known
state to classify it:

| Class | Condition |
|---|---|
| **new** | `doc_id` not in the manifest |
| **changed** | hash differs from the manifest |
| **deleted** | in the manifest but no longer on disk |
| **unchanged** | hash matches — skipped entirely |

If nothing is new, changed, or deleted, the run is a **no-op**: it logs the summary
and returns without parsing, embedding, or touching Qdrant. That is the idempotency
guarantee — a re-sync of an unchanged corpus reports `0 / 0 / 0`.

### 2. Pass 1 — parse (Docling)

Only **new + changed** documents are parsed. For each one, `parse.Parser`:

- Converts the file with Docling (`DocumentConverter`). OCR is **off by default**
  (it is the dominant CPU cost and is needless for digital PDFs); table-structure
  detection stays on because tables carry real content. Both are config-toggled.
- Chunks the document with Docling's `HybridChunker`, which merges undersized
  blocks and splits oversized ones toward the configured token budget.
- Emits a `Chunk` per non-empty piece, carrying its `section_path` (the joined
  heading trail, e.g. `Installation > Prerequisites`), the document content hash,
  and the file's mtime as `last_updated`.

Each chunk is **staged to SQLite** (`staged_chunks` table) rather than held in
memory. Once every document is parsed, the parser is dropped and `gc.collect()` is
called so **Docling is fully gone before the embedder loads**. This two-pass split
is the whole reason staging to disk exists: it bounds peak memory by the larger of
the two model stacks instead of their sum.

### 3. Pass 2 — embed and upsert

First, stale Qdrant points are removed:

- **changed** docs: their old points are deleted (by `doc_id` filter) before the
  new ones are inserted — this is also what makes a doc *shrinking* (fewer chunks)
  safe, with no need to track the old per-chunk id set.
- **deleted** docs: their points are deleted and their manifest row is forgotten.

Then the embedder is loaded (`get_embedder()` — ONNX/INT8 by default) and the
staged chunks are drained in batches of `ingest.batch_size`. Each batch is embedded
to dense + sparse vectors and upserted to the subject's collection. Chunks are
drained ordered by `(doc_id, chunk_index)` so a document's chunks stay together.

Finally the staging table is cleared and a `SyncStats` summary is returned:

```
[teamcenter] 2 new / 1 changed / 0 deleted / 47 unchanged (318 chunks embedded)
```

## The manifest and staging

Both tables live in one SQLite file (`manifest_path`, default
`/data/state/manifest.sqlite`). See [data-model.md](data-model.md#sqlite-manifest)
for the schema.

- **`docs`** — persistent. One row per document with its `content_hash`. This is
  what hash-delta detection compares against across runs.
- **`staged_chunks`** — scratch, per run. Written by pass 1, drained by pass 2,
  cleared at the end. Persisting it on disk is what lets the two passes hand off
  without being co-resident in memory.

## Idempotency and deletes

Point ids in Qdrant are a deterministic UUIDv5 of `"{doc_id}:{chunk_index}"`, so
re-ingesting a document **overwrites** its points rather than duplicating them. A
document's lifecycle:

- **edited** → old points deleted by `doc_id`, then re-inserted (handles growth,
  shrink, and content change uniformly).
- **renamed** → a delete of the old `doc_id` + an add of the new one (correct,
  because the `doc_id` is the path).
- **removed** → points deleted, manifest row dropped.

## File filters

By default every file with a supported suffix (`.pdf .html .htm .md .docx .pptx`)
under a subject dir is ingested. You can narrow this with glob filters at two
levels — **global** (top-level `include`/`exclude` in config) and **per-subject**
(on a subject entry). The two levels are unioned:

- **`include`** is a whitelist: if *any* include pattern exists (at either level),
  a file must match one of them.
- **`exclude`** is a blacklist and always wins.

Patterns are matched against the `doc_id` (the posix path relative to the subject
dir) with `fnmatch`, so `*` spans `/`. This is how, for example, a Doxygen HTML
tree can be dropped in as-is with its navigation/index pages excluded. See the
worked example in [`config.example.yaml`](../config.example.yaml) and
[configuration.md](configuration.md).

## Known limitation: interrupted pass 2

If a run is killed **during pass 2** (embedding), the staged chunks and the
provisionally-written manifest rows can be left in a split state that a plain
re-run does not cleanly reconcile. There is no built-in resume; the practical
recovery is to re-stage (re-run the parse for the affected docs) or embed the
remaining staged chunks out of band. Treat a clean exit as the unit of work.

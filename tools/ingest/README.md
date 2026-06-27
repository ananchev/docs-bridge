# tools/ingest

Corpus ingestion helpers — get source documents from the Nextcloud archive into
the docs-bridge `teamcenter` corpus on the runtime host, and run the worker.

Two source shapes are supported, distinguished by **how you tag in Nextcloud**:

- **PDFs (and other single files)** → tag the *files* `to_ingest`; copied byte-for-byte.
- **Doxygen API doc sets** → tag the *folder* `to_ingest`; converted HTML→Markdown
  first (`doxy2md.py`), then copied. Tagging the folder avoids tagging ~hundreds of files.

Both flows share the `to_ingest` tag, the destination, and the prune step. The
copy queries split cleanly by mimetype (files vs directory), so they never overlap.

## Hosts

- **Nextcloud host** (has the archive + the `nextcloud-postgres` container): runs
  `doxy2md.py`, `ingest-copy*.sh`, `ingest-prune.sh`, `ingest-flip-tag.sh`.
  Needs `~/doxy2md.py` + a venv (`~/doxy2md-venv` with `beautifulsoup4`, `lxml`),
  and key-based SSH to the runtime host.
- **Runtime host** (Qdrant + the ingest-worker image): runs `run-ingest.sh`.

Set `NC_USER` to your Nextcloud account (it builds the archive path); default is a
placeholder.

## Scripts

| Script | Host | Purpose |
|---|---|---|
| `doxy2md.py` | NC | Doxygen HTML → clean Markdown preprocessor. Standalone (never imports `docs_bridge`); mirrors the input tree, one `.md` per class/group page, `#`/`##` headings → citation section paths. Called by `ingest-copy-doxygen.sh`; run by hand only to spot-check output. |
| `ingest-copy.sh` | NC | Copy every **file** tagged `to_ingest` to the runtime drop dir, mirroring the NC structure (path = doc id). Ends with a prune. |
| `ingest-copy-doxygen.sh` | NC | For every **folder** tagged `to_ingest`: convert its Doxygen HTML via `doxy2md.py`, then `rsync --delete` the `.md` per folder (drops stale pages on regen). Ends with a prune. |
| `ingest-prune.sh` | NC | Reconcile the whole target to the tagged set: delete anything tagged neither `to_ingest` nor `ingested`. Dry-run by default; `--apply` to delete. Never walks the archive (tag query + a `find` of the small target). |
| `ingest-flip-tag.sh` | NC | After ingest + verify, move `to_ingest` → `ingested` (the "done" marker). Covers files and folders. |
| `run-ingest.sh` | runtime | One-shot worker `sync` with wall-clock timing; logs to `~/ingest-logs/`. Built for `screen`/`tmux`. |

## Sample invocations

```bash
# --- on the Nextcloud host ---
# 0. one-time: venv for the converter
python3 -m venv ~/doxy2md-venv && ~/doxy2md-venv/bin/pip install beautifulsoup4 lxml

# spot-check the converter by hand (optional)
~/doxy2md-venv/bin/python ~/doxy2md.py /path/to/doxygen/html /tmp/out

# preview what the prune would remove (safe, read-only)
NC_USER=youruser ~/ingest-prune.sh

# copy a tagged doxygen set (convert + copy + prune)
NC_USER=youruser ~/ingest-copy-doxygen.sh
# copy tagged PDFs instead
NC_USER=youruser ~/ingest-copy.sh

# --- on the runtime host ---
screen -S ingest
./run-ingest.sh            # subject defaults to 'all'
# detach: Ctrl-a d   reattach: screen -r ingest

# --- back on the Nextcloud host, after verifying ---
~/ingest-flip-tag.sh
```

## End-to-end sequence

1. Tag the source in Nextcloud (`to_ingest`): folders for Doxygen sets, files for PDFs.
2. Run the matching copy script(s) — `ingest-copy-doxygen.sh` and/or `ingest-copy.sh`.
   (Each ends with `ingest-prune.sh --apply`, so untagging in Nextcloud removes a doc
   on the next copy — no flag needed.)
3. On the runtime host: `run-ingest.sh` → verify retrieval.
4. `ingest-flip-tag.sh` to mark the batch `ingested`. Run **all** copy scripts you
   need *before* flipping — one flip clears file- and folder-tags together.

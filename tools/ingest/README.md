# tools/ingest

Corpus ingestion helpers — get source documents from the Nextcloud archive into
a docs-bridge corpus (subject `mysubject` in the examples) on the runtime host, and
run the worker.

Three source shapes, all **Nextcloud-tag-driven** with the same `to_ingest` tag:

- **PDFs (and other single files)** → tag the *files* `to_ingest`; copied byte-for-byte.
- **Doxygen API doc sets** → tag the *folder* `to_ingest`; converted HTML→Markdown
  first (`doxy2md.py`), then copied. Tagging the folder avoids tagging ~hundreds of files.
- **Javadoc sets** (a multi-module Java API reference) → tag the *folder*
  `to_ingest`; converted via `javadoc2md.py`, then copied. Tag a few module folders for
  a scoped corpus, or the single top-level parent folder for the whole reference (the
  converter dedups repeated base classes by FQN within a run). The same copy script also
  converts a Teamcenter SOA `libs` jar dir (→ jar dependency/copy index) and a `wsdls`
  dir (→ per-service wire contracts) when those folders are tagged.

All three share the `to_ingest` tag, the destination subject, and the prune step.
The **file** flow and the two **folder** flows split by mimetype, so they never overlap.
The two folder flows share the dir query but the converters are mutually exclusive by
content (each emits nothing for the other's pages, leaving no staging dir to copy), so
running both over the same tag is safe — each picks up only its own folders.

## Hosts

- **Nextcloud host** (has the archive + the `nextcloud-postgres` container): runs
  `doxy2md.py`, `javadoc2md.py`, `soa-libs-index.py`, `wsdl2md.py`, `ingest-copy*.sh`,
  `nextcloud-import.sh`, `ingest-prune.sh`, `ingest-flip-tag.sh`. Needs those scripts in
  `~` + a venv (`~/doxy2md-venv` with `beautifulsoup4`, `lxml`), and key-based SSH to the
  runtime host.
- **Runtime host** (Qdrant + the ingest-worker image): runs `run-ingest.sh`.

Set `NC_USER` to your Nextcloud account (it builds the archive path); default is a
placeholder.

## Scripts

| Script | Host | Purpose |
|---|---|---|
| `doxy2md.py` | NC | Doxygen HTML → clean Markdown preprocessor. Standalone (never imports `docs_bridge`); mirrors the input tree, one `.md` per class/group page, `#`/`##` headings → citation section paths. Called by `ingest-copy-doxygen.sh`; run by hand only to spot-check output. |
| `javadoc2md.py` | NC | Javadoc HTML → clean Markdown preprocessor. Standalone sibling of `doxy2md.py`; auto-detects modern (JDK 11+) and legacy (JDK 8) Javadoc, keeps class/interface/enum + package-summary pages (drops use/index/tree/search/help), `# <FQN>` / `## <member>` headings → citation section paths, with signature + params/returns/throws. Dedups repeated FQNs within a run. Called by `ingest-copy-javadoc.sh`. |
| `soa-libs-index.py` | NC | Teamcenter SOA client `libs` jar dir → one `soa-libs-index.md`: package→jar map + transitive OSGi `Require-Bundle` closure + runtime baseline, so an LLM can advise which jars to copy for a trimmed client. Stdlib-only; emits nothing if the folder has no jars. Called by `ingest-copy-javadoc.sh`. |
| `wsdl2md.py` | NC | SOAP WSDL/XSD → per-service Markdown wire contract (endpoint, namespaces, operations + soapAction, request/response message types, faults). Standalone (needs `lxml`); one `.md` per `.wsdl`, emits nothing if the folder has no WSDLs. Called by `ingest-copy-javadoc.sh`. |
| `ingest-copy.sh` | NC | Copy every **file** tagged `to_ingest` to the runtime drop dir, mirroring the NC structure (path = doc id). Ends with a prune. |
| `ingest-copy-doxygen.sh` | NC | For every **folder** tagged `to_ingest`: convert its Doxygen HTML via `doxy2md.py`, then `rsync --delete` the `.md` per folder (drops stale pages on regen). Ends with a prune. |
| `ingest-copy-javadoc.sh` | NC | Twin of `ingest-copy-doxygen.sh` for **folders** tagged `to_ingest`: runs `javadoc2md.py`, `soa-libs-index.py`, and `wsdl2md.py` over each (only the matching one emits), then `rsync --delete` the `.md` per folder. Ends with a prune. Run alongside the doxygen one — they don't clash. |
| `ingest-prune.sh` | NC | Reconcile the whole target to the tagged set: delete anything tagged neither `to_ingest` nor `ingested`. Dry-run by default; `--apply` to delete. Never walks the archive (tag query + a `find` of the small target). |
| `ingest-flip-tag.sh` | NC | After ingest + verify, move `to_ingest` → `ingested` (the "done" marker). Covers files and folders. |
| `nextcloud-import.sh` | NC | Register files dropped onto the data dir over SSH: chown to the web user (33:33) + normalise perms, `occ files:scan` the subtree, then `occ fulltextsearch:index`. Needs root (chown) + docker access. Run before tagging. |
| `run-ingest.sh` | runtime | One-shot worker `sync` with wall-clock timing; logs to `~/ingest-logs/`. Built for `screen`/`tmux`. |

## Sample invocations

```bash
# --- on the Nextcloud host ---
# 0. one-time: venv for the converter
python3 -m venv ~/doxy2md-venv && ~/doxy2md-venv/bin/pip install beautifulsoup4 lxml

# spot-check the converter by hand (optional)
~/doxy2md-venv/bin/python ~/doxy2md.py /path/to/doxygen/html /tmp/out

# register files dropped onto the data dir over SSH, then tag them in Nextcloud
NC_USER=youruser sudo -E ~/nextcloud-import.sh "Some/Folder"

# preview what the prune would remove (safe, read-only)
NC_USER=youruser ~/ingest-prune.sh

# copy tagged API-doc folders (convert + copy + prune) — run BOTH; each picks its own
NC_USER=youruser ~/ingest-copy-doxygen.sh
NC_USER=youruser ~/ingest-copy-javadoc.sh
# copy tagged PDFs instead
NC_USER=youruser ~/ingest-copy.sh

# spot-check a converter by hand (optional)
~/doxy2md-venv/bin/python ~/javadoc2md.py /path/to/<Module> /tmp/out

# --- on the runtime host ---
screen -S ingest
./run-ingest.sh            # subject defaults to 'all'
# detach: Ctrl-a d   reattach: screen -r ingest

# --- back on the Nextcloud host, after verifying ---
~/ingest-flip-tag.sh
```

## End-to-end sequence

1. Tag the source in Nextcloud (`to_ingest`): folders for Doxygen / Javadoc sets,
   files for PDFs.
2. Run the matching copy script(s) — `ingest-copy-doxygen.sh`, `ingest-copy-javadoc.sh`,
   and/or `ingest-copy.sh`. (Each ends with `ingest-prune.sh --apply`, so untagging in
   Nextcloud removes a doc on the next copy — no flag needed.)
3. On the runtime host: `run-ingest.sh` → verify retrieval.
4. `ingest-flip-tag.sh` to mark the batch `ingested`. Run **all** copy scripts you
   need *before* flipping — one flip clears file- and folder-tags together.

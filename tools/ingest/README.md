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

Set `SUBJECT` to the destination corpus (the drop dir is `docs/<subject>`); default is
the `mysubject` placeholder. The copy scripts pass it through to `ingest-prune.sh`, so
one run only ever touches that corpus — the other corpora on the host are untouched:

```bash
SUBJECT=polarion NC_USER=youruser ~/ingest-copy.sh
```

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
| `resume-embed.sh` | runtime | Recovery for a `sync` killed/crashed mid-Pass-2 (embed). Drains the already-staged chunks straight to embed→Qdrant (no re-parse), since a plain re-`sync` no-ops (see resume gap below). Idempotent + re-runnable; retries each upsert with reconnect/backoff so a Qdrant restart doesn't abort the drain. Logs to `~/ingest-logs/resume_*.log`; built for `screen`/`tmux`. |
| `verify-corpus.py` | runtime | Reconcile the manifest against what Qdrant actually holds: per doc, recorded `chunk_count` vs real point count. Names the docs that are MISSING (manifest says done, index has nothing), PARTIAL, over-count, or ORPHAN. Stdlib-only, read-only. Exit 1 on drift, so it can gate a scheduled run. `--write-broken FILE` feeds `reingest-docs.py`. |
| `reingest-docs.py` | runtime | Force named docs to be re-ingested: deletes their manifest rows (+ any staged rows) so the next `sync` classifies them NEW. The repair for a doc whose manifest row is a lie — hash-delta would otherwise skip it forever. Dry-run by default, `--apply` to act; backs the manifest up first; `--purge-points` also clears stale points. Needs write access to the manifest (root-owned → `sudo`). |

## Resume after an interrupted Pass 2

The worker's two-pass `sync` records each doc in the manifest **during Pass 1**
(parse), *before* its chunks are embedded in Pass 2. So if Pass 2 is interrupted
(e.g. Qdrant disconnects mid-upsert), those docs look *unchanged* on the next run.
**`sync` now handles this itself:** `sync_subject` drains any leftover `staged_chunks`
*before* it classifies, so simply **re-running `./run-ingest.sh <subject>` resumes an
interrupted embed** (no re-parse) and then proceeds normally. Idempotent because point
ids are a deterministic UUIDv5 of `chunk_id`, so re-upserts overwrite the partial points
rather than duplicate them. `qdrant_io.upsert` also retries with backoff on a Qdrant
disconnect, so a brief Qdrant restart no longer aborts the run.

`resume-embed.sh` remains as an **explicit Pass-2-only** drain (same logic, standalone)
for when you want to run just the embed step without invoking the full `sync` — handy
to confirm/benchmark a recovery, or if you ever need to drain staged chunks without a
disk scan.

## When the manifest outlives the chunks (silent drift)

The auto-drain above closes the gap going forward, but it cannot repair damage already
done. If Pass 2 was interrupted *and* a later run cleared `staged_chunks`, the doc keeps
its Pass-1 manifest row — right hash, right `chunk_count` — while Qdrant holds none of
its points. Hash-delta then classifies it **unchanged** on every future run, so the doc
is permanently absent from the index and no `sync` output ever mentions it. Manifest
totals look healthy; retrieval is quietly blind to that document.

`verify-corpus.py` is the only thing that sees this, because it asks Qdrant instead of
trusting the manifest. Repair is two steps:

```bash
./verify-corpus.py --subject mysubject --write-broken /tmp/broken.txt   # detect
sudo ./reingest-docs.py --subject mysubject --docs-from /tmp/broken.txt # preview
sudo ./reingest-docs.py --subject mysubject --docs-from /tmp/broken.txt --apply
./run-ingest.sh mysubject                                              # re-parse + embed
./verify-corpus.py --subject mysubject                                 # confirm clean
```

Worth running after any ingest that did not exit 0 — and worth running on a schedule,
since the whole point of this failure mode is that nothing else reports it.

## Podman short names: fully qualify the image

Podman defaults to `short-name-mode = "enforcing"`. An unqualified image ref that is
**not** in local storage sends podman down the *pull* path, where it wants to prompt for
which registry to use; with no TTY it exits 125 with `short-name resolution enforced but
cannot prompt without a TTY`. From a `screen` session you get the prompt and never
notice the trap. From a systemd timer the run dies in under a second, every night,
visible only in journald.

So `run-ingest.sh` (and the operator's timer unit) reference the worker
**fully qualified** — `localhost/ingest-worker:latest`. `localhost/` is podman's
namespace for images that were built rather than pulled, so the ref can never be read as
a remote one. Note that building via the Docker-compat API with a bare
`ingest-worker:latest` stores it as `docker.io/library/ingest-worker:latest`, which
*looks* remote and reintroduces the ambiguity — tag the build `localhost/...` too.

Check a suspect timer with `journalctl -u <unit> | grep -c short-name` before assuming
the ingest merely had nothing to do.

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
SUBJECT=mysubject NC_USER=youruser ~/ingest-prune.sh

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

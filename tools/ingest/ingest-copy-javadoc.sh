#!/usr/bin/env bash
# ingest-copy-javadoc.sh — for every FOLDER tagged 'to_ingest' in Nextcloud,
# convert its JDK Javadoc HTML to clean Markdown (javadoc2md.py, in a venv) and copy
# the .md tree to the docs-bridge 'mysubject' drop dir on runtime-host, MIRRORING the
# Nextcloud folder structure (relative path = doc id) — the Javadoc TWIN of
# ingest-copy-doxygen.sh. Same 'to_ingest' tag, same directory-only selection (the
# exact complement of ingest-copy.sh's file selection), same prune + flip.
#
# It shares the dir-tag query with ingest-copy-doxygen.sh, but the two never clash:
# the converters are mutually exclusive by content (doxy2md emits nothing for a
# Javadoc folder; javadoc2md emits nothing for a Doxygen folder), and a folder that
# produced no .md leaves no staging dir, so the copy loop skips it. Run BOTH copy
# scripts before flipping (the shared flip clears all folder-tags together).
#
# WHAT TO TAG — your call:
#   * a few individual MODULE folders -> a lean, scoped corpus; OR
#   * the single top-level PARENT folder that holds all modules -> the WHOLE reference
#     in one run. javadoc2md dedups repeated base classes by FQN within a run, so the
#     whole-set option does NOT store one copy of the shared base/runtime classes per
#     module — it keeps one. Either way everything lands under the tagged subtree, so
#     the prune protects it.
#
# Prereqs: ~/javadoc2md.py + ~/doxy2md-venv (beautifulsoup4, lxml); key-based SSH to
# $DEST_SSH; dest writable by that user.
set -euo pipefail

TAG_NAME="to_ingest"
PG_CONTAINER="nextcloud-postgres"
PG_USER="nextcloud"
PG_DB="nextcloud"
NC_USER="${NC_USER:-youruser}"   # Nextcloud account that owns the tagged folders
SRC_ROOT="/path/to/nextcloud/data/${NC_USER}/files"
DEST_SSH="runtime-host"
DEST_DIR="/data/docs-bridge-payload/docs/mysubject/"
VENV_PY="$HOME/doxy2md-venv/bin/python"
JAVADOC2MD="$HOME/javadoc2md.py"

q() { docker exec -i "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -t -A "$@"; }

tag_id="$(q -c "SELECT id FROM oc_systemtag WHERE name='${TAG_NAME}' LIMIT 1;")"
[[ -n "$tag_id" ]] || { echo "Tag '${TAG_NAME}' not found. Create it in the UI first." >&2; exit 1; }

# Only DIRECTORY objects carrying the tag (same query as ingest-copy-doxygen.sh; the
# exact complement of ingest-copy.sh's mimetype<>'httpd/unix-directory').
list="$(mktemp)"; stage="$(mktemp -d)"
trap 'rm -rf "$list" "$stage"' EXIT
q -c "
  SELECT regexp_replace(fc.path,'^files/','')
  FROM oc_systemtag_object_mapping m
  JOIN oc_filecache fc ON fc.fileid = m.objectid::bigint
  JOIN oc_mimetypes mt ON mt.id = fc.mimetype
  WHERE m.systemtagid=${tag_id} AND m.objecttype='files'
    AND mt.mimetype='httpd/unix-directory'
  ORDER BY fc.path;" > "$list"

count="$(grep -c . "$list" || true)"
[[ "$count" -gt 0 ]] || { echo "No folders tagged '${TAG_NAME}'. Nothing to do."; exit 0; }
echo ">> ${count} folder(s) tagged '${TAG_NAME}'."

# Convert each tagged folder into the staging tree at the SAME relative path, so the
# staged layout mirrors Nextcloud (and therefore the doc ids on runtime-host). A folder
# that holds no Javadoc converts to nothing -> no staging dir -> skipped below.
while IFS= read -r rel; do
  [[ -n "$rel" ]] || continue
  src="${SRC_ROOT}/${rel}"
  [[ -d "$src" ]] || { echo "  ! not a directory, skipping: $rel" >&2; continue; }
  echo "  - converting: $rel"
  "$VENV_PY" "$JAVADOC2MD" "$src" "${stage}/${rel}"
done < "$list"

# Mirror each converted set under DEST_DIR (relative path = doc id), per folder with
# --delete so a regenerated set that DROPPED pages doesn't leave stale .md. Scoped to
# the folder's own subtree, so other corpora are never touched. Folders that produced
# no .md (e.g. a Doxygen folder, or a non-doc folder) have no staging dir -> skipped.
# -s (--protect-args) keeps spaces in folder names intact across the remote shell.
echo ">> Copying converted Markdown -> ${DEST_SSH}:${DEST_DIR} (mirroring NC structure)"
while IFS= read -r rel; do
  [[ -n "$rel" && -d "${stage}/${rel}" ]] || continue
  ssh "$DEST_SSH" "mkdir -p \"${DEST_DIR}${rel}\""
  rsync -avh -s --delete --info=progress2 "${stage}/${rel}/" "${DEST_SSH}:${DEST_DIR}${rel}/"
done < "$list"

# Reconcile the whole target: drop anything no longer tagged (to_ingest ∪ ingested).
~/ingest-prune.sh --apply

echo ">> Done. Tag left as '${TAG_NAME}'. After ingest + verify: ~/ingest-flip-tag.sh"

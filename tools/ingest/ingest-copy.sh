#!/usr/bin/env bash
# ingest-copy.sh — copy every file tagged 'to_ingest' in Nextcloud to the
# docs-bridge 'mysubject' drop dir on runtime-host, MIRRORING the Nextcloud folder
# structure. In docs-bridge the relative path IS the doc id, so the folder
# layout keeps every doc unique — the same basename in different folders or
# product versions never collides. Idempotent (rsync delta). Tag left untouched.
#
# Prereqs: key-based SSH to $DEST_SSH; dest writable by that user.
set -euo pipefail

TAG_NAME="to_ingest"
PG_CONTAINER="nextcloud-postgres"
PG_USER="nextcloud"
PG_DB="nextcloud"
NC_USER="${NC_USER:-youruser}"   # Nextcloud account that owns the tagged files
SRC_ROOT="/path/to/nextcloud/data/${NC_USER}/files"
DEST_SSH="runtime-host"
DEST_DIR="/data/docs-bridge-payload/docs/mysubject/"

q() { docker exec -i "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -t -A "$@"; }

tag_id="$(q -c "SELECT id FROM oc_systemtag WHERE name='${TAG_NAME}' LIMIT 1;")"
[[ -n "$tag_id" ]] || { echo "Tag '${TAG_NAME}' not found." >&2; exit 1; }

list="$(mktemp)"; trap 'rm -f "$list"' EXIT
q -c "
  SELECT regexp_replace(fc.path,'^files/','')
  FROM oc_systemtag_object_mapping m
  JOIN oc_filecache fc ON fc.fileid = m.objectid::bigint
  JOIN oc_mimetypes mt ON mt.id = fc.mimetype
  WHERE m.systemtagid=${tag_id} AND m.objecttype='files'
    AND mt.mimetype<>'httpd/unix-directory'
  ORDER BY fc.path;" > "$list"

count="$(grep -c . "$list" || true)"
[[ "$count" -gt 0 ]] || { echo "No files tagged '${TAG_NAME}'. Nothing to copy."; exit 0; }
echo ">> ${count} file(s) tagged '${TAG_NAME}' -> ${DEST_SSH}:${DEST_DIR} (mirroring NC structure)"

# --files-from recreates the relative tree under DEST_DIR (path = doc id).
rsync -avh --info=progress2 --files-from="$list" "${SRC_ROOT}/" "${DEST_SSH}:${DEST_DIR}"

# Reconcile the whole target: drop anything no longer tagged (to_ingest ∪ ingested).
~/ingest-prune.sh --apply

echo ">> Copy done. Tag left as '${TAG_NAME}'. After ingest + verify: ~/ingest-flip-tag.sh"

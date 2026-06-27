#!/usr/bin/env bash
# ingest-flip-tag.sh — move every file currently tagged 'to_ingest' to
# 'ingested' (visual "done" marker). Run AFTER copy + ingest + verification.
# Idempotent: files already 'ingested' are de-duped, then the old tag removed.
set -euo pipefail

FROM_TAG="to_ingest"
TO_TAG="ingested"
PG_CONTAINER="nextcloud-postgres"
PG_USER="nextcloud"
PG_DB="nextcloud"

q() { docker exec -i "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -t -A "$@"; }

from_id="$(q -c "SELECT id FROM oc_systemtag WHERE name='${FROM_TAG}' LIMIT 1;")"
to_id="$(q   -c "SELECT id FROM oc_systemtag WHERE name='${TO_TAG}'   LIMIT 1;")"
[[ -n "$from_id" ]] || { echo "Tag '${FROM_TAG}' not found." >&2; exit 1; }
[[ -n "$to_id"   ]] || { echo "Tag '${TO_TAG}' not found. Create it in the UI first." >&2; exit 1; }

n="$(q -c "SELECT count(*) FROM oc_systemtag_object_mapping WHERE systemtagid=${from_id} AND objecttype='files';")"
[[ "$n" -gt 0 ]] || { echo "No files tagged '${FROM_TAG}'. Nothing to flip."; exit 0; }

q -c "BEGIN;
  INSERT INTO oc_systemtag_object_mapping (objectid, objecttype, systemtagid)
    SELECT objectid, objecttype, ${to_id}
    FROM oc_systemtag_object_mapping
    WHERE systemtagid=${from_id} AND objecttype='files'
  ON CONFLICT DO NOTHING;
  DELETE FROM oc_systemtag_object_mapping
    WHERE systemtagid=${from_id} AND objecttype='files';
COMMIT;" >/dev/null

echo ">> Flipped ${n} file(s): '${FROM_TAG}' -> '${TO_TAG}'."

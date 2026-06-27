#!/usr/bin/env bash
# ingest-prune.sh — reconcile the vhost2 'teamcenter' drop dir to the Nextcloud
# desired set. Any file there whose source is tagged NEITHER 'to_ingest' NOR
# 'ingested' is an orphan and gets removed. Tagged FILES are matched exactly;
# tagged FOLDERS protect their whole subtree (covers the .md that the doxygen
# copy generates under a tagged folder). Reconciles the ENTIRE target every run,
# regardless of which copy script called it — so "untag in Nextcloud" == "gone
# from the corpus on the next copy", no flip flag needed for deletion.
#
# Cheap: ONE tag query + a `find` of the (small) target. It never walks the
# Nextcloud archive — the desired set is pure tag metadata.
#
# DRY-RUN by default (prints what it would delete). Pass --apply to delete.
set -euo pipefail

APPLY=0; [[ "${1:-}" == "--apply" ]] && APPLY=1

PG_CONTAINER="nextcloud-postgres"
PG_USER="nextcloud"
PG_DB="nextcloud"
DEST_SSH="vhost2"
DEST_DIR="/data/docs-bridge-payload/docs/teamcenter"   # no trailing slash

q() { docker exec -i "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -t -A "$@"; }

ids="$(q -c "SELECT id FROM oc_systemtag WHERE name IN ('to_ingest','ingested');" | paste -sd, -)"
[[ -n "$ids" ]] || { echo "Neither 'to_ingest' nor 'ingested' tag exists." >&2; exit 1; }

desired_files="$(mktemp)"; desired_dirs="$(mktemp)"; actual="$(mktemp)"
trap 'rm -f "$desired_files" "$desired_dirs" "$actual"' EXIT

# protected exact files (file-tag flow)
q -c "
  SELECT regexp_replace(fc.path,'^files/','')
  FROM oc_systemtag_object_mapping m
  JOIN oc_filecache fc ON fc.fileid=m.objectid::bigint
  JOIN oc_mimetypes mt ON mt.id=fc.mimetype
  WHERE m.systemtagid IN (${ids}) AND m.objecttype='files'
    AND mt.mimetype<>'httpd/unix-directory'
  ORDER BY 1;" > "$desired_files"

# protected subtrees (folder-tag flow, e.g. the doxygen sets)
q -c "
  SELECT regexp_replace(fc.path,'^files/','')
  FROM oc_systemtag_object_mapping m
  JOIN oc_filecache fc ON fc.fileid=m.objectid::bigint
  JOIN oc_mimetypes mt ON mt.id=fc.mimetype
  WHERE m.systemtagid IN (${ids}) AND m.objecttype='files'
    AND mt.mimetype='httpd/unix-directory'
  ORDER BY 1;" > "$desired_dirs"

# actual files on the target, as paths relative to DEST_DIR (= doc ids)
ssh "$DEST_SSH" "find '$DEST_DIR' -type f -printf '%P\n' 2>/dev/null" | sort > "$actual"

# orphans = actual not matched exactly by a tagged file and not under a tagged dir
mapfile -t orphans < <(awk -v DF="$desired_files" -v DD="$desired_dirs" '
  BEGIN {
    while ((getline l < DF) > 0) if (l != "") files[l]=1
    n=0; while ((getline l < DD) > 0) if (l != "") dirs[++n]=l
  }
  $0 == "" { next }
  { p=$0
    if (p in files) next
    for (i=1;i<=n;i++) if (index(p, dirs[i] "/") == 1) next
    print p }
' "$actual")

count="${#orphans[@]}"
if [[ "$count" -eq 0 ]]; then
  echo "Target matches the tagged set ('to_ingest' ∪ 'ingested'). Nothing to prune."
  exit 0
fi

if [[ "$APPLY" -eq 0 ]]; then
  echo "DRY-RUN: ${count} orphan file(s) under ${DEST_SSH}:${DEST_DIR}"
  echo "         (source tagged NEITHER 'to_ingest' NOR 'ingested'):"
  printf '  - %s\n' "${orphans[@]}"
  echo ">> Re-run with --apply to delete them (and empty dirs)."
  exit 0
fi

printf '%s\n' "${orphans[@]}" \
  | ssh "$DEST_SSH" "cd '$DEST_DIR' && xargs -d '\n' -r rm -f -- && find . -mindepth 1 -type d -empty -delete"
echo ">> Pruned ${count} orphan file(s) from ${DEST_SSH}:${DEST_DIR}."

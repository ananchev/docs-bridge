#!/usr/bin/env bash
# nextcloud-import.sh — make a manually-added folder visible to Nextcloud.
#
# When files are dropped straight onto the data dir over SSH (not via Nextcloud), they
# (a) end up owned by the SSH user instead of the web user, and (b) are unknown to
# Nextcloud until its file cache is rescanned. This script fixes both for a given
# subtree under <user>/files:
#   1. chown -> the web user (uid:gid 33:33) and normalise perms to match the rest of
#      the data tree (dirs 0775, files 0664), then restore +x on helper *.sh / *.py so
#      the tooling living alongside the docs stays runnable.
#   2. occ files:scan --path=<user>/files/<rel> inside the nextcloud-app container, so
#      the new files appear in the web UI / API (and become taggable for ingest).
#   3. occ fulltextsearch:index so the new content is searchable (and reachable via the
#      FTS-backed MCP search). Note: this (re)indexes all FTS providers, not just <rel>,
#      so it can take a while.
#
# Usage — the path may be RELATIVE to <user>/files OR an ABSOLUTE path under the data
# dir; both resolve to the same place:
#   NC_USER=youruser sudo -E ./nextcloud-import.sh "Some/Folder"
#   NC_USER=youruser sudo -E ./nextcloud-import.sh "/path/to/nextcloud/data/youruser/files/Some/Folder"
#
# chown/chmod need root; occ runs via docker (the invoking user must be able to reach
# the docker socket). Re-runnable.
set -euo pipefail

NC_CONTAINER="${NC_CONTAINER:-nextcloud-app}"   # the Nextcloud app container
NC_DATA="${NC_DATA:-/path/to/nextcloud/data}"
NC_USER="${NC_USER:-youruser}"                  # Nextcloud account that owns the files
WEB_UID="${WEB_UID:-33}"; WEB_GID="${WEB_GID:-33}"

BASE="$NC_DATA/$NC_USER/files"
arg="${1:?usage: [NC_USER=...] sudo -E $0 <path relative to <user>/files | absolute path>}"
arg="${arg%/}"                              # drop any trailing slash
case "$arg" in
  "$BASE"/*) REL="${arg#"$BASE"/}" ;;       # absolute, under <user>/files
  "$BASE")   REL="" ;;                       # the files root itself
  /*) echo "!! Absolute path is not under $BASE :" >&2
      echo "   $arg" >&2; exit 1 ;;
  *) REL="$arg" ;;                           # already relative to <user>/files
esac

HOST_PATH="$BASE${REL:+/$REL}"
[ -e "$HOST_PATH" ] || { echo "!! Path not found: $HOST_PATH" >&2; exit 1; }

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo ">> 1/3  Ownership ${WEB_UID}:${WEB_GID} + permissions on:"
echo "        $HOST_PATH"
$SUDO chown -R "${WEB_UID}:${WEB_GID}" "$HOST_PATH"
$SUDO find "$HOST_PATH" -type d -exec chmod 0775 {} +
$SUDO find "$HOST_PATH" -type f -exec chmod 0664 {} +
# keep the helper scripts that live with the docs executable
$SUDO find "$HOST_PATH" -type f \( -name '*.sh' -o -name '*.py' \) -exec chmod 0775 {} +

SCAN_PATH="${NC_USER}/files${REL:+/$REL}"
echo ">> 2/3  occ files:scan  (${SCAN_PATH})"
docker exec -u "${WEB_UID}:${WEB_GID}" "$NC_CONTAINER" php occ files:scan --path="${SCAN_PATH}"

# 3) Full-text index so the new content is searchable (and picked up by the FTS-backed
#    MCP search). This (re)indexes all FTS providers, so it can run a while.
echo ">> 3/3  occ fulltextsearch:index"
docker exec -u www-data "$NC_CONTAINER" php occ fulltextsearch:index

echo ">> Done. '${REL:-<files root>}' is owned by the web user, registered in Nextcloud, and full-text indexed."

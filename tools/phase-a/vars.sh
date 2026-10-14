# Shared settings for the Phase A step scripts (sourced, not run).
NEW=2026.10-macmini                                   # tag for the new images
OLD_WORKER=localhost/ingest-worker:latest             # deployed; embedded teamcenter
OLD_SERVER=docker.io/library/docs-bridge:latest       # deployed; note the namespace
MODELS=localhost/docs-bridge-models:frozen-2026-09
CFG=/data/docker/docs-bridge/config/config.yaml       # the live config (old stack)
P=/data/docs-bridge-payload                           # host side of /data
A=$HOME/phase-a                                       # everything for this run
SRC=$A/src                                            # the repo, rsynced from mac-mini
OUT=$A/out                                            # results; comes back to notes/phase-a/
NEW_CFG=$A/config-rebuild.yaml                        # the rebuild's config (step 5 writes it)
NEW_MANIFEST=manifest-subject-keyed.sqlite            # under $P/state
SUBJECTS="teamcenter-whats-new teamcenter-2512"       # rebuild order: small one first
mkdir -p "$OUT"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '\033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
FAILED=0
finish() {
  if [ "$FAILED" = 0 ]; then printf '\n\033[32mStep passed.\033[0m %s\n' "$*"
  else printf '\n\033[31mStep FAILED — stop here and send me %s\033[0m\n' "$OUT"; exit 1; fi
}

# A worker run with the live mounts (same as the systemd unit) and the rebuild config.
worker() {
  sudo podman run --rm --network docs-bridge-net \
    -v $P/docs:/data/docs:ro,z -v $P/state:/data/state:z -v $P/cache:/data/cache:z \
    -v $NEW_CFG:/config/config.yaml:ro,z "$@"
}
# Print the `collections` stamp table of a manifest file.
stamps() {
  sudo python3 -c 'import sqlite3,sys
for r in sqlite3.connect(sys.argv[1]).execute("SELECT collection, subject, embedder FROM collections"): print(*r)' "$1"
}
points() {  # $1 collection -> "status count"
  curl -s localhost:6333/collections/$1 | python3 -c \
    'import json,sys; r=json.load(sys.stdin).get("result") or {}; print(r.get("status","missing"), r.get("points_count",0))'
}

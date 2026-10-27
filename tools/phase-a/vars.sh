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
NEW_CFG=${NEW_CFG:-$A/config-rebuild.yaml}           # the rebuild's config (step 5 writes it);
                                                      # after the switch-over 10 runs on $CFG
NEW_MANIFEST=manifest-subject-keyed.sqlite            # under $P/state
SUBJECTS="teamcenter-whats-new teamcenter-2512"       # rebuild order: small one first
mkdir -p "$OUT"

# Ask for the sudo password ONCE, up front, and keep it fresh for the whole script —
# a long step (an ingest) otherwise outlives sudo's 5-minute cache and the next sudo
# sits at a password prompt hidden behind a pipe.
sudo -v || exit 1
( while kill -0 $$ 2>/dev/null; do sudo -n true 2>/dev/null; sleep 50; done ) &
SUDO_KEEPALIVE=$!
trap 'kill $SUDO_KEEPALIVE 2>/dev/null' EXIT

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '\033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
FAILED=0
finish() {
  if [ "$FAILED" = 0 ]; then printf '\n\033[32mStep passed.\033[0m %s\n' "$*"
  else printf '\n\033[31mStep FAILED — stop here and send me %s\033[0m\n' "$OUT"; exit 1; fi
}

# A worker run with the live mounts (same as the systemd unit) and the rebuild config.
# Memory cap for every worker container: a runaway parse then dies alone (exit 137)
# instead of pushing the host into swap and the kernel OOM killer (2026-10-15 01:56).
WORKER_MEM=7g
WORKER_LIMITS="--memory $WORKER_MEM --memory-swap $WORKER_MEM"
worker() {
  sudo podman run --rm --network docs-bridge-net $WORKER_LIMITS \
    -v $P/docs:/data/docs:ro,z -v $P/state:/data/state:z -v $P/cache:/data/cache:z \
    -v $NEW_CFG:/config/config.yaml:ro,z "$@"
}
# Print the `collections` stamp table of a manifest file.
stamps() {
  sudo python3 -c 'import sqlite3,sys
db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
for r in db.execute("SELECT collection, subject, embedder FROM collections"): print(*r)' "$1"
}
# Start a detached rebuild of one subject: capped, NOT verbose (a verbose run logs
# every PDF page; podman stopped recording at ~31 MB and hid the end of the last run).
start_rebuild() {  # $1 subject
  sudo podman rm -f rebuild-$1 >/dev/null 2>&1
  sudo podman run -d --name rebuild-$1 --network docs-bridge-net $WORKER_LIMITS \
    -v $P/docs:/data/docs:ro,z -v $P/state:/data/state:z -v $P/cache:/data/cache:z \
    -v $NEW_CFG:/config/config.yaml:ro,z \
    localhost/ingest-worker:$NEW sync --subject $1 >/dev/null
}
# Log RAM every 10 min while container $1 runs, into file $2.
ram_logger() {
  sudo nohup sh -c 'while podman ps -q --filter name='$1' | grep -q .; do
    date; free -m | sed -n 2,3p
    podman stats --no-stream --format "{{.Name}} {{.MemUsage}}"
    sleep 600
  done; date; free -m; podman stats --no-stream' > $2 2>&1 &
}
points() {  # $1 collection -> "status count"
  curl -s localhost:6333/collections/$1 | python3 -c \
    'import json,sys; r=json.load(sys.stdin).get("result") or {}; print(r.get("status","missing"), r.get("points_count",0))'
}

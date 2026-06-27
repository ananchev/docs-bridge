#!/usr/bin/env bash
# run-ingest.sh — one-shot docs-bridge ingest with wall-clock timing.
# Logs to ~/ingest-logs/ingest_<startISO>.log (start moment in the filename) and
# prints total elapsed time at the end. Built to run inside screen/tmux:
#
#     screen -S ingest
#     ./run-ingest.sh [subject]        # subject defaults to 'all'
#     # detach: Ctrl-a d   reattach: screen -r ingest
#
# The ingest-worker image + docs-bridge-net are ROOTFUL podman, so the container
# is launched via sudo (you'll be prompted once). The log stays owned by you.
set -uo pipefail

SUBJECT="${1:-all}"
DATA_ROOT="/data/docs-bridge-payload"
CONFIG="/data/docker/docs-bridge/config/config.yaml"
NET="docs-bridge-net"
IMAGE="ingest-worker:latest"

start_iso="$(date +%Y%m%dT%H%M%S%z)"          # ISO-8601 basic, no colons (fs-safe)
logdir="$HOME/ingest-logs"; mkdir -p "$logdir"
log="$logdir/ingest_${start_iso}.log"

{
  echo "docs-bridge ingest"
  echo "subject : ${SUBJECT}"
  echo "started : $(date --iso-8601=seconds)"
  echo "log     : ${log}"
  echo "--------------------------------------------------"
} | tee "$log"

start=$(date +%s)
sudo docker run --rm \
  --network "$NET" \
  -v "${DATA_ROOT}/docs:/data/docs:ro,z" \
  -v "${DATA_ROOT}/state:/data/state:z" \
  -v "${DATA_ROOT}/cache:/data/cache:z" \
  -v "${CONFIG}:/config/config.yaml:ro,z" \
  "$IMAGE" sync --subject "$SUBJECT" ${VERBOSE:+--verbose} 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
end=$(date +%s); el=$((end - start))

{
  echo "--------------------------------------------------"
  echo "finished: $(date --iso-8601=seconds)"
  echo "exit    : ${status}"
  printf 'elapsed : %dh%02dm%02ds (%ds)\n' $((el/3600)) $(((el%3600)/60)) $((el%60)) "$el"
} | tee -a "$log"
exit "$status"

#!/usr/bin/env bash
# Step 10 (after the switch-over): ingest a NEW subject into the live stack, e.g.
#   NEW_CFG=/data/docker/docs-bridge/config/config.yaml ./10-ingest-subject.sh teamcenter-14.2
# Unlike 9 it does NOT edit any config: the subject comes from the operator contract
# (containers-at-home applications/docs-bridge.yml), rendered with
# `./run-playbook.sh vhost2 docs-bridge ... --tags prep` — prep only: `schedule` would
# re-enable the timer, and the running server keeps its subject list until restarted,
# so nobody is routed to the half-built pool meanwhile.
#
# Put the docs FLAT in $P/docs/<subject> first (PDFs at the top, no What's New guides —
# those belong in teamcenter-whats-new). Disables the nightly timer (stopped is not
# enough: a reboot would bring it back mid-ingest), offers to stop the live server
# (10g cap instead of 7g), and starts the ingest detached. Progress / checks:
#   NEW_CFG=$CFG ./6-check-rebuild.sh <subject>
set -uo pipefail
. "$(dirname "$0")/vars.sh"
S=${1:?usage: NEW_CFG=$CFG $0 <subject>}; DIR=$P/docs/$S

[ "$NEW_CFG" = "$CFG" ] || { bad "run with NEW_CFG=$CFG — after the switch-over the live config is the one"; finish; }

say "$S in the live config"
pair=$(worker --entrypoint python localhost/ingest-worker:$NEW -c '
import sys
from docs_bridge import config
s = config.load("/config/config.yaml").subject(sys.argv[1]); print(s.dir, s.collection)' "$S" 2>&1 | tail -1)
case "$pair" in /data/docs/*) ok "dir and collection: $pair" ;;
  *) bad "not in $CFG ($pair) — add it to the contract and deploy --tags prep first"; finish ;; esac

say "the $S documents"
[ -d "$DIR" ] || { bad "$DIR does not exist"; finish; }
n=$(find "$DIR" -type f | wc -l); pdfs=$(find "$DIR" -maxdepth 1 -iname '*.pdf' | wc -l)
echo "   $n files, $pdfs PDFs at the top level"
[ "$n" -gt 0 ] || { bad "$DIR is empty"; finish; }
[ -d "$DIR/Siemens" ] && { bad "found $DIR/Siemens — lay it out flat like teamcenter-2512"; finish; }
wn=$(find "$DIR" -iname '*what*s*new*' | head -5)
[ -z "$wn" ] || { bad "What's New files belong in teamcenter-whats-new, not here:"; echo "$wn"; finish; }
ok "layout looks right"

if sudo podman ps -q --filter name=rebuild- | grep -q .; then
  bad "a rebuild is already running — ./6-check-rebuild.sh shows its progress"; finish
fi

say "nightly timer: disable (not just stop) until this subject is checked"
sudo systemctl disable --now docs-bridge-ingest.timer 2>&1 | grep -v '^Removed' || true
systemctl is-active -q docs-bridge-ingest.service && { bad "a nightly run is in flight — wait, then re-run"; finish; }
ok "timer: $(systemctl is-enabled docs-bridge-ingest.timer 2>&1), $(systemctl is-active docs-bridge-ingest.timer 2>&1)"

say "live server"
if sudo podman ps --format '{{.Names}}' | grep -qx docs-bridge; then
  read -r -p "Stop the live docs-bridge server for this ingest? (no search meanwhile) [y/N] " ans
  [[ "$ans" =~ ^[Yy] ]] && { sudo podman stop docs-bridge >/dev/null && ok "stopped"; }
else ok "already stopped"; fi
if ! sudo podman ps --format '{{.Names}}' | grep -qx docs-bridge; then
  WORKER_MEM=10g; WORKER_LIMITS="--memory $WORKER_MEM --memory-swap $WORKER_MEM"
fi

cnt=$(worker --entrypoint python localhost/ingest-worker:$NEW -c '
import sys
from docs_bridge import config
from docs_bridge.parse import scan
cfg = config.load("/config/config.yaml"); print(len(scan(cfg.subject(sys.argv[1]), cfg)))' "$S")
echo "   the worker will ingest $cnt documents" | tee $OUT/10-$S-files.txt

say "starting $S (detached, capped at $WORKER_MEM)"
start_rebuild $S && ok "running as container rebuild-$S" || { bad "did not start"; finish; }
ram_logger rebuild-$S $OUT/10-$S-ram.txt
finish "Progress any time: NEW_CFG=$CFG ./6-check-rebuild.sh $S"

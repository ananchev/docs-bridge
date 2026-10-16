#!/usr/bin/env bash
# Step 9: ingest Teamcenter 2606 into the rebuild (before the switch-over, so the live
# server never offers a half-built pool). Put the docs FLAT in
# $P/docs/teamcenter-2606 first, laid out like teamcenter-2512 (PDFs at the top,
# `SOA Reference/`, `AIG APIs Documentation/` if shipped) and WITHOUT What's New guides
# (those belong in teamcenter-whats-new).
#
# Disables the nightly timer (stopped is not enough: a reboot would bring it back and
# run the OLD worker at 03:30), offers to stop the live server, adds the subject to
# $NEW_CFG (the same entry as the operator contract), and starts the ingest detached:
# memory-capped, batch 32 with the embedder's token budget. Progress / checks:
#   ./6-check-rebuild.sh teamcenter-2606
set -uo pipefail
. "$(dirname "$0")/vars.sh"
S=teamcenter-2606; DIR=$P/docs/$S

say "the 2606 documents"
[ -d "$DIR" ] || { bad "$DIR does not exist — move the 2606 docs there first"; finish; }
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

say "nightly timer: disable (not just stop) until the switch-over re-enables it"
sudo systemctl disable --now docs-bridge-ingest.timer 2>&1 | grep -v '^Removed' || true
systemctl is-active -q docs-bridge-ingest.service && { bad "an old ingest run is in flight — wait, then re-run"; finish; }
ok "timer: $(systemctl is-enabled docs-bridge-ingest.timer 2>&1), $(systemctl is-active docs-bridge-ingest.timer 2>&1)"

say "live server"
if sudo podman ps --format '{{.Names}}' | grep -qx docs-bridge; then
  read -r -p "Stop the live docs-bridge server for this ingest? (no search meanwhile) [y/N] " ans
  [[ "$ans" =~ ^[Yy] ]] && { sudo podman stop docs-bridge >/dev/null && ok "stopped"; }
else ok "already stopped"; fi
if ! sudo podman ps --format '{{.Names}}' | grep -qx docs-bridge; then
  WORKER_MEM=10g; WORKER_LIMITS="--memory $WORKER_MEM --memory-swap $WORKER_MEM"
fi

say "add $S to $NEW_CFG (idempotent)"
sudo podman run --rm -i -v $NEW_CFG:/in.yaml:ro,z --entrypoint python localhost/ingest-worker:$NEW - \
  > $NEW_CFG.tmp <<'PY'
import yaml
c = yaml.safe_load(open("/in.yaml"))
c["subjects"] = [s for s in c["subjects"] if s["name"] != "teamcenter-2606"] + [{
    "name": "teamcenter-2606", "dir": "/data/docs/teamcenter-2606", "collection": "teamcenter_2606",
    "description": (
        "Siemens Teamcenter PLM, release 2606 — installation, configuration, "
        "administration, Active Workspace, BMIDE data modelling, server/client APIs, "
        "SOA & ITK development and customization (incl. the SOA API reference), and "
        "the Active Integration Gateway (T4S/T4EA/T4O) for SAP and "
        "enterprise-application integration (AIG runs on and is configured through "
        "Teamcenter, so AIG install/config topics live here). Use for how Teamcenter "
        "2606 works; for what changed between releases use teamcenter-whats-new.")}]
print(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
PY
[ -s $NEW_CFG.tmp ] && mv $NEW_CFG.tmp $NEW_CFG && ok "subjects: $(grep -c '^- name:' $NEW_CFG)" \
  || { bad "could not update the config"; finish; }
cnt=$(worker --entrypoint python localhost/ingest-worker:$NEW -c '
from docs_bridge import config
from docs_bridge.parse import scan
cfg = config.load("/config/config.yaml"); print(len(scan(cfg.subject("teamcenter-2606"), cfg)))')
echo "   the worker will ingest $cnt documents" | tee $OUT/9-2606-files.txt

say "starting $S (detached, capped at $WORKER_MEM)"
start_rebuild $S && ok "running as container rebuild-$S" || { bad "did not start"; finish; }
ram_logger rebuild-$S $OUT/9-ram.txt
finish "Progress any time: ./6-check-rebuild.sh $S — when done it checks all three subjects."

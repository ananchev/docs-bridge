#!/usr/bin/env bash
# Step 5b: resume the teamcenter-2512 rebuild after it was killed (2026-10-15 01:56:
# kernel OOM while docling 2.130 parsed one PDF for 2 h, growing to 8.2 GB).
#
# A resumed sync first EMBEDS everything the parse pass staged (docling is not loaded
# then), then parses only the documents not yet done. The PDF that blew up is put in
# QUARANTINE (excluded from the subject in $NEW_CFG) so the rest can finish; it is
# re-admitted once diag-parse.sh has shown what to fix. Detached, memory-capped, not
# verbose. Embedding ~112k chunks takes ~7-10 h; live queries are slower meanwhile.
set -uo pipefail
. "$(dirname "$0")/vars.sh"
# Documents to keep OUT of teamcenter-2512 (authoritative: the config's exclude list is
# set to exactly this). Empty since 2026-10-15: the AIG Basic Configuration Guide is back
# in — its 01:56 OOM happened with the host out of swap and the live server at 7 GB, and
# the second OOM was the embed batch size, not the PDF. If it still breaks the 10g cap,
# put it back here.
QUARANTINE=()

if sudo podman ps -q --filter name=rebuild-teamcenter-2512 | grep -q .; then
  bad "a rebuild is already running — ./6-check-rebuild.sh shows its progress"; finish
fi
systemctl is-active -q docs-bridge-ingest.timer && { bad "the nightly timer is active again — stop it first"; finish; }

say "config: embed batch 32 (memory); quarantined docs: ${#QUARANTINE[@]}"
[ ${#QUARANTINE[@]} = 0 ] || printf '  %s\n' "${QUARANTINE[@]}"
sudo podman run --rm -i -v $NEW_CFG:/in.yaml:ro,z --entrypoint python localhost/ingest-worker:$NEW - \
  "${QUARANTINE[@]}" > $NEW_CFG.tmp <<'PY'
import sys, yaml
c = yaml.safe_load(open("/in.yaml"))
s = next(s for s in c["subjects"] if s["name"] == "teamcenter-2512")
if sys.argv[1:]:
    s["exclude"] = sorted(sys.argv[1:])
else:
    s.pop("exclude", None)
# 2026-10-15: batch 64 peaks at 7.6 GB on the corpus' longest chunk (587 tokens; the
# embedder pads a batch to its longest, attention ~ batch x len^2); batch 32 peaks at
# 3.7 GB and is slightly faster (0.154 vs ~0.175 s/chunk) — measured on vhost2.
c.setdefault("ingest", {})["batch_size"] = 32
print(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
PY
[ -s $NEW_CFG.tmp ] && mv $NEW_CFG.tmp $NEW_CFG && ok "added to $NEW_CFG" || { bad "could not update the config"; finish; }

say "the live server has grown overnight"
sudo podman stats --no-stream --format '{{.Name}} {{.MemUsage}}' docs-bridge
free -m | sed -n 1,3p
echo "  s = STOP it until the rebuild and the gate are done (frees all of it; no search meanwhile)"
echo "  r = restart it (frees what it grew; a few seconds of downtime)"
echo "  n = leave it"
read -r -p "Live docs-bridge server: [s/r/n] " ans
case "$ans" in
  [Ss]*) sudo podman stop docs-bridge >/dev/null && ok "stopped — start it again with: sudo podman start docs-bridge" \
           || bad "stop failed — check: sudo podman ps -a" ;;
  [Rr]*) sudo podman restart docs-bridge >/dev/null && ok "restarted" || bad "restart failed — check: sudo podman ps -a" ;;
esac
sleep 5; free -m | sed -n 2,3p

# With the live server stopped there is ~12 GB free: give the worker 10 GB, so a big PDF
# parse has room (it reached 8.2 GB on 2026-10-15 01:56). With the server up, keep 7 GB.
if ! sudo podman ps --format '{{.Names}}' | grep -qx docs-bridge; then
  WORKER_MEM=10g; WORKER_LIMITS="--memory $WORKER_MEM --memory-swap $WORKER_MEM"
fi
say "resuming teamcenter-2512 (detached, capped at $WORKER_MEM)"
start_rebuild teamcenter-2512 && ok "running as container rebuild-teamcenter-2512" || { bad "did not start"; finish; }
ram_logger rebuild-teamcenter-2512 $OUT/5b-ram.txt
ok "RAM log: $OUT/5b-ram.txt"
finish "Progress any time: ./6-check-rebuild.sh — when it says done, it runs the checks."

#!/usr/bin/env bash
# Step 5 (runbook B.2-B.4): pause the nightly ingest, ingest the small what's-new
# subject in the foreground (~10-20 min, doubles as the speed probe), then start the
# big 2512 ingest DETACHED (~13 h) with a RAM logger, and return. Start it in the
# evening: live queries are slower while it runs. Safe to close SSH after it returns.
set -uo pipefail
. "$(dirname "$0")/vars.sh"
[ -s "$NEW_CFG" ] || { bad "no $NEW_CFG — run ./4-config.sh first"; finish; }

say "pausing the nightly ingest timer"
systemctl list-timers --all | grep docs-bridge > $OUT/5-timer-before.txt
sudo systemctl stop docs-bridge-ingest.timer
if systemctl is-active -q docs-bridge-ingest.service; then
  bad "an old ingest run is in flight — wait for it to finish, then re-run this script"; finish
fi
ok "timer stopped (undo: sudo systemctl start docs-bridge-ingest.timer)"

say "teamcenter-whats-new (foreground, ~10-20 min)"
t0=$(date +%s)
worker localhost/ingest-worker:$NEW sync --subject teamcenter-whats-new --verbose \
  > $OUT/5-whats-new.txt 2>&1 && ok "done" || { bad "see $OUT/5-whats-new.txt"; finish; }
dt=$(( $(date +%s) - t0 ))
line=$(grep '^\[teamcenter-whats-new\]' $OUT/5-whats-new.txt | tail -1); echo "$line"
chunks=$(sed -n 's/.*(\([0-9]*\) chunks embedded).*/\1/p' <<< "$line")
echo "$dt s for $chunks chunks = $(python3 -c "print(round($dt/max($chunks,1),2))") s/chunk incl. parse" \
  | tee -a $OUT/5-whats-new.txt

say "starting teamcenter-2512 detached"
sudo podman rm -f rebuild-teamcenter-2512 >/dev/null 2>&1
sudo podman run -d --name rebuild-teamcenter-2512 --network docs-bridge-net \
  -v $P/docs:/data/docs:ro,z -v $P/state:/data/state:z -v $P/cache:/data/cache:z \
  -v $NEW_CFG:/config/config.yaml:ro,z \
  localhost/ingest-worker:$NEW sync --subject teamcenter-2512 --verbose >/dev/null \
  && ok "running as container rebuild-teamcenter-2512" || { bad "did not start"; finish; }

say "RAM logger (every 10 min until the rebuild exits)"
sudo nohup sh -c 'while podman ps -q --filter name=rebuild-teamcenter-2512 | grep -q .; do
  date; free -m | sed -n 2p
  podman stats --no-stream --format "{{.Name}} {{.MemUsage}}"
  sleep 600
done; date; free -m; podman stats --no-stream' > $OUT/5-ram.txt 2>&1 &
ok "logging to $OUT/5-ram.txt"

finish "Watch it (optional): sudo podman logs -f --tail 5 rebuild-teamcenter-2512   (Ctrl-C detaches)
Tomorrow: ./6-check-rebuild.sh"

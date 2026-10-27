#!/usr/bin/env bash
# Step 6 (runbook B.6): after a detached rebuild run — is everything complete and
# consistent? Re-sync must be a no-op, manifest must match Qdrant, one embedder stamp
# per collection, collections indexed. Checks EVERY subject in $NEW_CFG. ~10 min.
# Re-run it as often as you like; while the run is going it only shows progress.
#   ./6-check-rebuild.sh                  # the teamcenter-2512 run (5 / 5b / 8)
#   ./6-check-rebuild.sh teamcenter-2606  # the 2606 run (9)
set -uo pipefail
. "$(dirname "$0")/vars.sh"
cd "$SRC"
RUN=${1:-teamcenter-2512}
COLL=${RUN//-/_}

if sudo podman ps -q --filter name=rebuild-$RUN | grep -q .; then
  staged=$(sudo python3 -c 'import sqlite3, sys
db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
print(db.execute("SELECT count(*) FROM staged_chunks WHERE subject = ?", (sys.argv[2],)).fetchone()[0])' \
    $P/state/$NEW_MANIFEST $RUN)
  echo "Still running: $COLL has $(points $COLL | cut -d' ' -f2) points; $staged chunks staged."
  echo "(parsing first: staged rises; then embedding: points rise towards it)"
  free -m | sed -n 1,3p
  exit 0
fi

say "how the $RUN run ended"
sudo podman logs rebuild-$RUN > $OUT/6-rebuild-$RUN.txt 2>&1
code=$(sudo podman inspect rebuild-$RUN --format '{{.State.ExitCode}}' 2>/dev/null)
grep "^\[$RUN\]" $OUT/6-rebuild-$RUN.txt | tail -1
if [ "$code" = 0 ]; then ok "exit 0"; sudo podman rm rebuild-$RUN >/dev/null
elif [ -z "$code" ]; then echo "(no rebuild-$RUN container — already checked; re-running the checks)"
else
  bad "exit code $code (137 = hit the memory cap) — send me $OUT; re-running its start script resumes it"; finish
fi

# every subject:collection pair of the rebuild config
PAIRS=$(sudo podman run --rm -v $NEW_CFG:/config/config.yaml:ro,z --entrypoint python \
  localhost/ingest-worker:$NEW -c 'from docs_bridge import config
print(" ".join(f"{s.name}:{s.collection}" for s in config.load("/config/config.yaml").subjects))')
N=$(wc -w <<< "$PAIRS")
echo "subjects: $PAIRS"

say "1. a second sync must change nothing"
for pair in $PAIRS; do worker localhost/ingest-worker:$NEW sync --subject ${pair%%:*} 2>&1 | grep '^\['; done \
  | tee $OUT/6-rerun.txt
[ "$(grep -c ' 0 new / 0 changed / 0 deleted' $OUT/6-rerun.txt)" = "$N" ] && ok "0/0/0 for all $N" || bad "not a no-op"

say "2. manifest vs Qdrant, every doc"
for pair in $PAIRS; do
  sudo python3 tools/ingest/verify-corpus.py --manifest $P/state/$NEW_MANIFEST \
    --subject ${pair%%:*} --collection ${pair##*:} --quiet \
    && ok "${pair%%:*} matches" || bad "${pair%%:*} has drift"
done 2>&1 | tee $OUT/6-verify.txt
grep -q FAIL $OUT/6-verify.txt && FAILED=1

say "3. embedder stamps"
stamps $P/state/$NEW_MANIFEST | tee $OUT/6-stamps.txt
[ "$(cut -d' ' -f3- $OUT/6-stamps.txt | sort -u | wc -l)" = 1 ] && [ "$(wc -l < $OUT/6-stamps.txt)" = "$N" ] \
  && ok "$N collections, one embedder" || bad "stamps differ or are missing"

say "4. indexed?"
for pair in $PAIRS; do c=${pair##*:}; echo "$c $(points $c)"; done | tee $OUT/6-counts.txt
grep -qv ' green ' $OUT/6-counts.txt \
  && bad "not green yet — Qdrant is still indexing; wait 15 min and re-run" || ok "green"
sudo podman ps --format '{{.Names}}' | grep -qx docs-bridge \
  || echo "NOTE: the live server is stopped — start it after the switch-over (it serves the OLD index until then)"
if [ "$RUN" = teamcenter-2512 ]; then finish "Next: ./7-gate.sh"
elif [ "$NEW_CFG" = "$CFG" ]; then
  finish "Next: sudo podman restart docs-bridge (or start) — it reads subjects at start — then sudo systemctl enable --now docs-bridge-ingest.timer"
else finish "Next: the switch-over (with me)"; fi

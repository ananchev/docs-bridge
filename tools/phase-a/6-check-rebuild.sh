#!/usr/bin/env bash
# Step 6 (runbook B.6): after the overnight run — is the rebuild complete and
# consistent? Re-sync must be a no-op, manifest must match Qdrant, one embedder stamp
# per collection, collections indexed. ~10 min. Re-run it as often as you like.
set -uo pipefail
. "$(dirname "$0")/vars.sh"
cd "$SRC"

if sudo podman ps -q --filter name=rebuild-teamcenter-2512 | grep -q .; then
  staged=$(sudo python3 -c 'import sqlite3
db = sqlite3.connect("file:'$P/state/$NEW_MANIFEST'?mode=ro", uri=True)
print(db.execute("SELECT count(*) FROM staged_chunks WHERE subject = ?", ("teamcenter-2512",)).fetchone()[0])')
  echo "Still running: teamcenter_2512 has $(points teamcenter_2512 | cut -d' ' -f2) points; $staged chunks staged."
  echo "(while it embeds, points rise towards the staged count; then it parses the rest)"
  free -m | sed -n 1,3p
  exit 0
fi

say "how the overnight run ended"
sudo podman logs rebuild-teamcenter-2512 > $OUT/6-rebuild-2512.txt 2>&1
code=$(sudo podman inspect rebuild-teamcenter-2512 --format '{{.State.ExitCode}}' 2>/dev/null)
grep '^\[teamcenter-2512\]' $OUT/6-rebuild-2512.txt | tail -1
if [ "$code" = 0 ]; then ok "exit 0"; sudo podman rm rebuild-teamcenter-2512 >/dev/null
else
  bad "exit code $code (137 = hit the memory cap) — send me $OUT; ./5b-resume.sh resumes it"; finish
fi
grep -iE 'out of memory|killed' $OUT/5*-ram.txt && bad "memory trouble in the RAM logs"

say "1. a second sync must change nothing"
for s in $SUBJECTS; do worker localhost/ingest-worker:$NEW sync --subject $s 2>&1 | grep '^\['; done \
  | tee $OUT/6-rerun.txt
[ "$(grep -c ' 0 new / 0 changed / 0 deleted' $OUT/6-rerun.txt)" = 2 ] && ok "0/0/0 for both" || bad "not a no-op"

say "2. manifest vs Qdrant, every doc"
for pair in teamcenter-2512:teamcenter_2512 teamcenter-whats-new:teamcenter_whats_new; do
  sudo python3 tools/ingest/verify-corpus.py --manifest $P/state/$NEW_MANIFEST \
    --subject ${pair%%:*} --collection ${pair##*:} --quiet \
    && ok "${pair%%:*} matches" || bad "${pair%%:*} has drift"
done 2>&1 | tee $OUT/6-verify.txt
grep -q FAIL $OUT/6-verify.txt && FAILED=1

say "3. embedder stamps"
stamps $P/state/$NEW_MANIFEST | tee $OUT/6-stamps.txt
[ "$(cut -d' ' -f3- $OUT/6-stamps.txt | sort -u | wc -l)" = 1 ] && [ "$(wc -l < $OUT/6-stamps.txt)" = 2 ] \
  && ok "two collections, one embedder" || bad "stamps differ or are missing"

say "4. indexed?"
for c in teamcenter_2512 teamcenter_whats_new teamcenter; do echo "$c $(points $c)"; done | tee $OUT/6-counts.txt
grep -E '^teamcenter_(2512|whats_new) ' $OUT/6-counts.txt | grep -qv ' green ' \
  && bad "not green yet — Qdrant is still indexing; wait 15 min and re-run" || ok "green"
finish "Next: ./7-gate.sh (after the labels are verified)"

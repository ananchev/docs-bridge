#!/usr/bin/env bash
# Step 7 (runbook step 6): THE GATE. The 45 questions through the real search, old
# image + old `teamcenter` vs new image + teamcenter-2512 & teamcenter-whats-new
# (together the same documents). ~25 min; one image at a time (RAM).
set -uo pipefail
. "$(dirname "$0")/vars.sh"
cd "$SRC"
Q=notes/gate2/questions-teamcenter.yaml

drafts=$(grep -c 'status: draft' $Q || true)
[ "$drafts" = 0 ] || echo "NOTE: $drafts questions still have draft labels — the verdict is only as good as the labels."

say "gate config (same rerank budget on both sides) and flattened labels"
sudo podman run --rm -i -v $NEW_CFG:/in.yaml:ro,z --entrypoint python localhost/docs-bridge:$NEW - \
  > $A/config-gate.yaml <<'PY'
import yaml
c = yaml.safe_load(open("/in.yaml"))
c.setdefault("rerank", {})["multi_top_n"] = c["rerank"].get("top_n", 30)
print(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
PY
sed 's#doc: "Siemens/Teamcenter 2512/#doc: "#' $Q > notes/gate2/questions-teamcenter-flat.yaml
[ "$(grep -c 'doc: "Siemens' notes/gate2/questions-teamcenter-flat.yaml)" = 0 ] && ok "labels flattened" || bad "labels"

rab_run() {  # $1 image  $2 label  $3 config  $4 subject(s)  $5 questions file
  sudo podman run --rm -i --network docs-bridge-net \
    -v $3:/config/config.yaml:ro,z -v $OUT:/out:z -v $SRC/notes/gate2:/q:ro,z \
    --entrypoint python $1 - run --questions /q/$5 \
    --subject $4 --label $2 --out /out/7-retrieval-$2.json \
    < tools/eval/retrieval_ab.py > $OUT/7-retrieval-$2.txt 2>&1
  tail -6 $OUT/7-retrieval-$2.txt
}
say "old: live image, old collection (~12 min)"
rab_run $OLD_SERVER old $CFG teamcenter questions-teamcenter.yaml
say "new: new image, rebuilt collections (~12 min)"
rab_run localhost/docs-bridge:$NEW new $A/config-gate.yaml teamcenter-2512,teamcenter-whats-new \
  questions-teamcenter-flat.yaml

say "compare"
python3 tools/eval/retrieval_ab.py compare --old $OUT/7-retrieval-old.json \
  --new $OUT/7-retrieval-new.json --json $OUT/7-compare.json | tee $OUT/7-compare.txt
{ date; free -m; } >> $OUT/5-ram.txt
grep -q 'VERDICT: PASS' $OUT/7-compare.txt && ok "PASS" \
  || bad "not PASS — a MISS may be a stale label (docling 2.130 parsed the text differently); I check each"
echo "(ignore the top-k overlap figure: the re-parse changed every chunk id)"
finish "Send me $OUT (rsync -a --exclude '*.tar' vhost2:phase-a/out/ notes/phase-a/) — then the switch-over."

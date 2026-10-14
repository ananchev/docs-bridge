#!/usr/bin/env bash
# Step 3 (runbook step 7): a small real ingest into SCRATCH collections with the new
# worker — docling 2.130 on teamcenter files, the embedder stamp, and the identity fix
# (two subjects reading the SAME dir must both re-sync as 0/0/0). ~15 min.
# Touches nothing live: own dir, own manifest, own collections (deleted at the end).
set -uo pipefail
. "$(dirname "$0")/vars.sh"
S=$A/smoke

say "picking sample files from teamcenter-2512 (3 smallest PDFs >300 KB, 1 SOA page)"
rm -rf $S && mkdir -p $S/docs $S/state $S/cache
find $P/docs/teamcenter-2512 -maxdepth 1 -name '*.pdf' -size +300k -printf '%s\t%p\n' \
  | sort -n | head -3 | cut -f2- | while read -r f; do cp --reflink=always "$f" $S/docs/; done
find "$P/docs/teamcenter-2512/SOA Reference" -name '*.md' -size +4k | head -1 \
  | while read -r f; do cp --reflink=always "$f" $S/docs/; done
ls -la $S/docs | tee $OUT/3-smoke-files.txt

sudo podman run --rm -i -v $CFG:/in.yaml:ro,z --entrypoint python localhost/ingest-worker:$NEW - \
  > $S/config.yaml <<'EOF'
import yaml
c = yaml.safe_load(open("/in.yaml"))
c["subjects"] = [
    {"name": "smoke-a", "dir": "/data/docs/smoke", "collection": "phase_a_smoke_a"},
    {"name": "smoke-b", "dir": "/data/docs/smoke", "collection": "phase_a_smoke_b"},
]
c["manifest_path"] = "/data/state/manifest-smoke.sqlite"
c["include"], c["exclude"] = [], []
print(yaml.safe_dump(c, sort_keys=False))
EOF
smoke() {
  sudo podman run --rm --network docs-bridge-net \
    -v $S/docs:/data/docs/smoke:ro,z -v $S/state:/data/state:z \
    -v $S/cache:/data/cache:z -v $S/config.yaml:/config/config.yaml:ro,z \
    localhost/ingest-worker:$NEW sync --subject all "$@"
}

say "first sync (parse + embed both subjects; the first run downloads docling's models)"
smoke --verbose > $OUT/3-smoke-ingest.txt 2>&1 && ok "sync exited cleanly" || bad "sync failed — see $OUT/3-smoke-ingest.txt"
grep '^\[' $OUT/3-smoke-ingest.txt

say "second sync — must change nothing, for BOTH subjects"
smoke 2>&1 | grep '^\[' | tee $OUT/3-smoke-rerun.txt
n=$(grep -c ' 0 new / 0 changed / 0 deleted' $OUT/3-smoke-rerun.txt)
[ "$n" = 2 ] && ok "0/0/0 twice (identity fix works)" || bad "expected two 0/0/0 lines"

say "embedder stamps"
stamps $S/state/manifest-smoke.sqlite | tee $OUT/3-smoke-stamps.txt
[ "$(wc -l < $OUT/3-smoke-stamps.txt)" = 2 ] && grep -q '"onnxruntime": "1.30.0"' $OUT/3-smoke-stamps.txt \
  && ok "two collections stamped, onnxruntime 1.30.0" || bad "stamps missing or unexpected"

say "a look at the parsed text (saved for me; skim it if you like)"
curl -s -X POST localhost:6333/collections/phase_a_smoke_a/points/scroll \
  -H 'Content-Type: application/json' \
  -d '{"limit":400,"with_payload":["doc_id","section_path","text"]}' > $OUT/3-smoke-chunks.json
python3 -c '
import json; pts = json.load(open("'$OUT'/3-smoke-chunks.json"))["result"]["points"]
print(len(pts), "chunks; three samples:")
for p in pts[::max(1, len(pts)//3)][:3]:
    pl = p["payload"]; print("---", pl["doc_id"], "|", pl["section_path"]); print(pl["text"][:300])'

curl -s -X DELETE localhost:6333/collections/phase_a_smoke_a > /dev/null
curl -s -X DELETE localhost:6333/collections/phase_a_smoke_b > /dev/null
finish "Next: the labels (step 8 in the runbook), then ./4-config.sh"

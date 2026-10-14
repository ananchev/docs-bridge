# Phase A runbook — the Mac mini rebuild on the new toolchain (vhost2)

> **Cold start — where this stands (2026-10-14).** Branch **`mac-mini`** (the old
> `main`, renamed 2026-10-14) is the Mac mini's line; `multi-user-rollout` carries the
> EC2 work and later merges `mac-mini` for the shared fixes. `mac-mini` forked at
> `4458105` (the toolchain upgrade), then gained: subject-qualified identity (manifest
> keys and point ids), the per-collection **embedder stamp**, merging identical
> passages across pools in `search`, and this runbook.
>
> **Decision (2026-10-14, user):** no re-embed of the old collection (the former step
> R), no parity with the stored vectors. The Mac mini **rebuilds from scratch**:
> docling 2.130 re-parses every file, the frozen bge-m3 INT8 model embeds on the new
> runtime, into a **new manifest** and **new collections**, laid out as three subjects:
>
> | Subject | Collection | Source |
> |---|---|---|
> | `teamcenter-2512` | `teamcenter_2512` | `/data/docs/teamcenter`, minus the what's-new docs |
> | `teamcenter-whats-new` | `teamcenter_whats_new` | `/data/docs/teamcenter`, only the what's-new docs |
> | `teamcenter-2606` | `teamcenter_2606` | `/data/docs/teamcenter-2606` (new) |
> | `teamcenter-aig`, `power-query` | `aig_docling130`, `pq_docling130` | unchanged dirs |
>
> Two subjects share the old dir and split it with include/exclude, so no file moves
> and the old stack keeps reading that dir unchanged. The other subjects (`teamcenter-aig`,
> `power-query`) are rebuilt too, into `<collection>_docling130`: on the new manifest
> their files are all "new", and with subject-qualified point ids a sync into their
> old collections would add a second copy of every point. Steps 0–3 were run on
> 2026-09-29 for the previous plan; the models image (step 2) stands, the **images
> must be rebuilt** (step 3) because they predate the identity change.
>
> **Inputs still needed from the user:** the globs that pick out the what's-new docs
> (`WHATSNEW` below), the 2606 files staged on vhost2, and the three descriptions.

Nothing here touches the running stack until the switch-over: every image gets a new
tag, `:latest` is never retagged, the live server keeps serving the old `teamcenter`
collection, and the old manifest is never opened by the new worker (it refuses
old-schema manifests anyway — `LegacyManifestError`). The one change to the running
host is that the ingest timer is **paused** from step B.2 until the switch-over.

What changed in the repo, and why it matters for these steps:

| Change | Consequence here |
|---|---|
| docling 2.15.1 → 2.130.0; onnxruntime 1.20.1 → 1.30.0; tokenizers 0.19.1 → 0.23.2; numpy 1.26 → 2.4.6; torch 2.5.1 → 2.14.0+cpu | every file is re-parsed and re-embedded; chunks differ from the old collection, so the gate (step 6) compares by **document**, not chunk |
| **Models are no longer exported at build.** Both images copy them from `localhost/docs-bridge-models` | the step-2 image must exist on the host before any build |
| Full transitive **locks** per image (`images/lock.sh`) | the embedder runtime is provably identical in worker and server |
| Manifest keys and point ids include the **subject** | two subjects with the same layout (2512 / 2606) can't overwrite each other's rows; the old manifest file can't be reused |
| Each collection is **stamped** with the embedder fingerprint (model file hashes + onnxruntime/tokenizers versions) | a later worker with a different embedder is refused instead of silently mixing vectors (`EmbedderMismatchError`) |
| `search` across several pools folds identical passages into one result (`subjects`, `also_in`) | a both-versions search isn't half duplicates |

Root podman needs `sudo` on vhost2, so these are yours to run. Every step writes into
`~/phase-a/out/`; bring that directory back into the repo's `notes/phase-a/` when
done and I will write the results up.

Throughout:

```bash
NEW=2026.10-macmini                                   # tag for the new images
OLD_WORKER=localhost/ingest-worker:latest             # deployed; embedded teamcenter
OLD_SERVER=docker.io/library/docs-bridge:latest       # deployed; note the namespace
MODELS=localhost/docs-bridge-models:frozen-2026-09
CFG=/data/docker/docs-bridge/config/config.yaml       # the live config (old stack)
NEW_CFG=~/phase-a/config-rebuild.yaml                 # step B.1 writes it
P=/data/docs-bridge-payload                           # host side of /data
```

These are plain shell variables: a new shell or a reconnect loses them. Keep the
block in `~/phase-a/vars.sh` and `source` it after every login.

---

## Step 0 — stage the source (workstation → vhost2, 1 min)

From the repo root on the workstation, on `mac-mini`:

```bash
git switch mac-mini && git pull
git rev-parse --short HEAD | tee /dev/stderr          # note it in the results; step 3 builds this
rsync -a --delete --exclude .git ./ vhost2:phase-a/src/
```

## Step 2 — the frozen models (done 2026-09-29; check it is still there)

```bash
sudo podman image exists $MODELS && echo present
ls -la ~/phase-a/out/docs-bridge-models-frozen-2026-09.tar 2>/dev/null \
  || echo "tar moved — check it is next to the Qdrant snapshots"
```

If the image is gone, `sudo podman load -i <the tar>`. Never rebuild it from the
deployed images now — that was only valid before anything replaced them.

## Step 3 — build the new images from this commit (~5 min)

```bash
cd ~/phase-a/src
sudo podman build -f images/docs-bridge/Dockerfile   -t localhost/docs-bridge:$NEW   . 2>&1 | tail -3
sudo podman build -f images/ingest-worker/Dockerfile -t localhost/ingest-worker:$NEW . 2>&1 | tail -3
for img in localhost/docs-bridge:$NEW localhost/ingest-worker:$NEW; do
  echo "== $img"
  sudo podman run --rm --entrypoint sh $img -c \
    'sha256sum /opt/bge-m3-onnx/*;
     python -c "from importlib.metadata import version as v; print({p: v(p) for p in [\"onnxruntime\",\"tokenizers\",\"numpy\",\"qdrant-client\"]})";
     python -c "import docs_bridge.manifest as m; print(\"identity:\", hasattr(m, \"EmbedderMismatchError\"))";
     pip list 2>/dev/null | grep -ci nvidia'
done 2>&1 | tee ~/phase-a/out/03-new-images.txt
```

**Expect** the same bge-m3 hashes as the 2026-09-29 step 0 in both,
`onnxruntime 1.30.0, tokenizers 0.23.2, numpy 2.4.6` in both, `identity: True` (the
image carries this commit's code), and `0` nvidia packages.

## Step 4 — embed parity: dropped

It compared the new runtime against the 137k stored vectors. Nothing reuses them
now, so there is nothing to be in parity with. (Its 2026-09-29 result — cosine ~0.987
from the runtime alone — is why the fingerprint includes the runtime versions.)

## Step 5 — the reranker's numerics, old vs new runtime (~8 min)

Same INT8 reranker bytes, new onnxruntime. Two runs on the Gate 1 fixture:

```bash
cd ~/phase-a/src
rab() {  # $1 output name  [$2...] extra args
  local name=$1; shift
  sudo podman run --rm -i -v ~/gate1/out:/fx:ro,z -v ~/phase-a/out:/out:z \
    -v ~/phase-a/src/notes/gate1:/ref:ro,z \
    --entrypoint python localhost/docs-bridge:$NEW - \
    --fixture /fx/fixture.json --backend onnx --model-dir /opt/bge-reranker-onnx \
    --threads 6 --repeats 1 --json /out/$name.json "$@" \
    < tools/eval/rerank_ab.py 2>&1 | tee ~/phase-a/out/$name.txt
}
rab 05-rerank-vs-old-int8 --selftest                       # new runtime vs the old INT8 column
rab 05-rerank-vs-fp32 --reference /ref/gpu-fp32.json       # distance to the reference model
```

**Expect** the first to print `SELFTEST … FAITHFUL` (max |Δ| < 1e-3 against the
scores the old runtime produced) — if so the second is a formality. The second
reports how far the INT8 reranker sits from fp32 on the new runtime; the bar (plan
§5.2) is **no worse than on the old one**: max |Δ| ≤ 1.03, top-6 ordered-identical
≥ 2/8. `--threads 6` because C = 6 on this host (Gate 1).

## Step 7 — smoke-ingest real files into SCRATCH collections (~15 min)

Proves the new worker end to end (parse → embed → stamp → upsert) and the identity
fix on the real stack, without touching the live manifest or collections. Put a
handful of files in a scratch dir — **a few teamcenter PDFs and one SOA HTML page**
(the corpus being rebuilt), plus one deck with speaker notes and one scanned PDF if
you have them. Two smoke subjects read **the same dir**: before this commit their
identical doc_ids would have fought over one manifest row and re-embedded on every
run; now the re-run must be 0/0/0 for both.

```bash
mkdir -p ~/phase-a/smoke/docs ~/phase-a/smoke/state ~/phase-a/smoke/cache
cp <your files> ~/phase-a/smoke/docs/
sudo podman run --rm -i -v $CFG:/in.yaml:ro,z --entrypoint python localhost/ingest-worker:$NEW - \
  > ~/phase-a/smoke/config.yaml <<'EOF'
import yaml
c = yaml.safe_load(open("/in.yaml"))
c["subjects"] = [
    {"name": "smoke-a", "dir": "/data/docs/smoke", "collection": "phase_a_smoke_a"},
    {"name": "smoke-b", "dir": "/data/docs/smoke", "collection": "phase_a_smoke_b"},
]
c["manifest_path"] = "/data/state/manifest-smoke.sqlite"
c["include"], c["exclude"] = [], []
c.setdefault("parse", {})["ocr"] = True
print(yaml.safe_dump(c, sort_keys=False))
EOF
grep -nE 'manifest_path|smoke|ocr' ~/phase-a/smoke/config.yaml
smoke() {
  sudo podman run --rm --network docs-bridge-net \
    -v ~/phase-a/smoke/docs:/data/docs/smoke:ro,z -v ~/phase-a/smoke/state:/data/state:z \
    -v ~/phase-a/smoke/cache:/data/cache:z -v ~/phase-a/smoke/config.yaml:/config/config.yaml:ro,z \
    localhost/ingest-worker:$NEW sync --subject all "$@"
}
smoke --verbose 2>&1 | tee ~/phase-a/out/07-smoke-ingest.txt
smoke 2>&1 | tail -2 | tee ~/phase-a/out/07-smoke-rerun.txt      # expect 0/0/0 for BOTH subjects
stamps() {  # $1 manifest file: one row per collection, the embedder that filled it
  sudo python3 -c 'import sqlite3,sys
for r in sqlite3.connect(sys.argv[1]).execute("SELECT collection, subject, embedder FROM collections"): print(*r)' "$1"
}
stamps ~/phase-a/smoke/state/manifest-smoke.sqlite | tee ~/phase-a/out/07-smoke-stamps.txt
# what landed — speaker notes and figure captions should be visible in the text
curl -s -X POST localhost:6333/collections/phase_a_smoke_a/points/scroll \
  -H 'Content-Type: application/json' \
  -d '{"limit":200,"with_payload":["doc_id","section_path","text"]}' \
  > ~/phase-a/out/07-smoke-chunks.json
grep -o 'Speaker notes:[^"]\{0,80\}\|Figure:[^"]\{0,80\}' ~/phase-a/out/07-smoke-chunks.json | head
# clean up
curl -s -X DELETE localhost:6333/collections/phase_a_smoke_a
curl -s -X DELETE localhost:6333/collections/phase_a_smoke_b
```

**Expect** a clean exit; the re-run `0 new / 0 changed / 0 deleted` for **both**
smoke subjects; two stamp rows naming `onnxruntime 1.30.0` with the same file hashes;
readable text from the teamcenter PDFs and the SOA page (look at a few chunks — this
is the first time docling 2.130 sees this corpus). The first run downloads docling's
new layout models into the scratch cache (a few hundred MB).

## Step 8 — verify the question labels (you, ~45 min, BEFORE step 6)

`notes/gate2/questions-teamcenter.yaml`: 42 questions + 3 controls, each label
checked by script to match a real chunk (0 misses, 2026-09-28) — but drafted by me
from random chunks. For each: read the `evidence`, decide whether that chunk really
answers the question as a Teamcenter user would mean it, fix `expect` if not, and
set `status: verified` (or `drop`). Step 6's verdict is only as good as these.

The labels match by `doc_id` (unchanged by the rebuild: the path relative to the
subject dir) plus optional `section` / `contains` substrings, which **were taken from
the old parse**. Prefer short, distinctive `contains` strings — docling 2.130 may
break lines, ligatures or table text differently, and a long substring is the first
thing to miss for reasons that are not retrieval.

---

## Step B — the rebuild (two nights)

### B.0 — baseline the old collection (2 min)

The gate compares against it, so know its state:

```bash
cd ~/phase-a/src
sudo python3 tools/ingest/verify-corpus.py --subject teamcenter --manifest $P/state/manifest.sqlite \
  2>&1 | tee ~/phase-a/out/B0-verify-old.txt          # expect exit 0, no drifted docs
curl -s localhost:6333/collections/teamcenter | python3 -c \
  'import json,sys; r=json.load(sys.stdin)["result"]; print(r["status"], r["points_count"])' \
  | tee ~/phase-a/out/B0-count-old.txt
```

### B.1 — stage 2606 and write the new config (10 min)

Copy the 2606 documentation to `$P/docs/teamcenter-2606/`, owned like the existing
`teamcenter` dir. Then derive the new config from the live one — same Qdrant,
chunking, batch and server settings; new subjects; new manifest. **Fill in
`WHATSNEW`** (fnmatch globs against the path relative to `/data/docs/teamcenter`;
`*` spans `/`) and the three descriptions — the descriptions are what an LLM reads to
choose a subject, so name the version in each:

```bash
sudo podman run --rm -i -v $CFG:/in.yaml:ro,z --entrypoint python localhost/ingest-worker:$NEW - \
  > $NEW_CFG <<'EOF'
import yaml
WHATSNEW = ["<glob>", "<glob>"]                       # <-- fill in
c = yaml.safe_load(open("/in.yaml"))
old = [s for s in c["subjects"] if s["name"] == "teamcenter"]
assert len(old) == 1, "expected exactly one teamcenter subject"
others = [s for s in c["subjects"] if s["name"] != "teamcenter"]
for s in others:                                      # rebuilt into new collections too
    s["collection"] += "_docling130"
c["subjects"] = [
    {"name": "teamcenter-2512", "dir": old[0]["dir"], "collection": "teamcenter_2512",
     "exclude": WHATSNEW,
     "description": "Siemens Teamcenter 2512 product documentation ..."},          # <-- edit
    {"name": "teamcenter-whats-new", "dir": old[0]["dir"], "collection": "teamcenter_whats_new",
     "include": WHATSNEW,
     "description": "What's new / release notes across Teamcenter versions ..."},  # <-- edit
    {"name": "teamcenter-2606", "dir": "/data/docs/teamcenter-2606", "collection": "teamcenter_2606",
     "description": "Siemens Teamcenter 2606 product documentation ..."},          # <-- edit
] + others
c["manifest_path"] = "/data/state/manifest-subject-keyed.sqlite"
print(yaml.safe_dump(c, sort_keys=False))
EOF
grep -nE 'name:|collection:|manifest_path|include|exclude' $NEW_CFG
```

`others` are the remaining subjects (aig, pq): same names and dirs, new collection
names. They are not gated (the question set is teamcenter's); B.6 verifies them.

Check the split before spending a night on it — the what's-new subject must get
exactly the files you mean, and 2512 everything else:

```bash
sudo podman run --rm -v $P/docs:/data/docs:ro,z -v $NEW_CFG:/config/config.yaml:ro,z \
  --entrypoint python localhost/ingest-worker:$NEW -c '
from docs_bridge import config
from docs_bridge.parse import scan
cfg = config.load("/config/config.yaml")
for n in [s.name for s in cfg.subjects]:
    found = scan(cfg.subject(n), cfg)
    print(n, len(found))
    if n == "teamcenter-whats-new":
        print("  " + "\n  ".join(sorted(found)))
' 2>&1 | tee ~/phase-a/out/B1-split.txt
```

**Expect** `teamcenter-2512` + `teamcenter-whats-new` = the old file count (B.0's
manifest rows), and the listed what's-new files to be exactly the intended ones.

### B.2 — pause the nightly ingest

The old worker must not compete for the cores, and the old collection should stay
fixed as the gate's reference:

```bash
systemctl list-timers --all | grep docs-bridge | tee ~/phase-a/out/B2-timer.txt
sudo systemctl stop docs-bridge-ingest.timer
systemctl is-active docs-bridge-ingest.service        # must not say "active" (a run in flight)
```

To undo (the plan is abandoned): `sudo systemctl start docs-bridge-ingest.timer`.

### B.3 — night 1: 2512 + what's-new, detached

A full teamcenter ingest was ~13 h on the old stack; 2606 is probably similar, so
it gets its own night. `-d` keeps the run alive if SSH drops. Live queries get slower
while it runs (same cores), so start in the evening:

```bash
rebuild() {  # $1 subject
  sudo podman run -d --name rebuild-$1 --network docs-bridge-net \
    -v $P/docs:/data/docs:ro,z -v $P/state:/data/state:z -v $P/cache:/data/cache:z \
    -v $NEW_CFG:/config/config.yaml:ro,z \
    localhost/ingest-worker:$NEW sync --subject $1 --verbose
}
rebuild teamcenter-whats-new                          # small: doubles as the speed probe
sudo podman logs -f rebuild-teamcenter-whats-new      # wait for its summary line
for s in teamcenter-aig power-query; do               # the small ones, one at a time
  rebuild $s && sudo podman wait rebuild-$s && sudo podman logs rebuild-$s 2>&1 | tail -1
done
rebuild teamcenter-2512
sudo podman logs -f rebuild-teamcenter-2512           # Ctrl-C detaches; the run goes on
```

Same mounts as the systemd unit, so the run sees what the nightly ingest will. The
cache is the live one: docling 2.130's models download into it alongside the old
ones. `embedded N / M` debug lines give the rate — if the what's-new run is far
slower per chunk than ~0.34 s, bring its log back before committing the night.

**If a run dies** (OOM, reboot): `sudo podman rm rebuild-<subject>` and start the
same command again. It drains whatever was staged, then hash-delta skips every file
already done — no work is repeated beyond the batch in flight.

### B.4 — log RAM while it runs

Qdrant now grows a second copy of teamcenter next to the old one (and a third
corpus on night 2), plus the worker and the live server, in 16 GB:

```bash
sudo -v
sudo nohup sh -c 'while podman ps -q --filter name=rebuild- | grep -q .; do
  date; free -m | sed -n 2p
  podman stats --no-stream --format "{{.Name}} {{.MemUsage}}"
  sleep 600
done' > ~/phase-a/out/B4-ram-$(date +%m%d).txt 2>&1 &
```

In the morning: `available` should never have approached 0, and no container should
have restarted (`sudo podman ps -a`). **A RAM problem is the one finding that would
change the plan** (e.g. snapshot and drop the old collection before night 2, accepting
a restore-based rollback).

### B.5 — night 2: 2606

```bash
sudo podman rm rebuild-teamcenter-whats-new rebuild-teamcenter-aig rebuild-power-query \
  rebuild-teamcenter-2512
rebuild teamcenter-2606
sudo podman logs -f rebuild-teamcenter-2606
```

(and B.4's RAM loop again).

### B.6 — check the rebuild (10 min), after both nights

```bash
sudo podman rm rebuild-teamcenter-2606
cd ~/phase-a/src
# every subject:collection pair of the new config
PAIRS=$(sudo podman run --rm -v $NEW_CFG:/config/config.yaml:ro,z --entrypoint python \
  localhost/ingest-worker:$NEW -c 'from docs_bridge import config
print(" ".join(f"{s.name}:{s.collection}" for s in config.load("/config/config.yaml").subjects))')
echo $PAIRS | tee ~/phase-a/out/B6-pairs.txt
# 1. a re-run is a no-op for every subject (hash-delta + the identity fix)
sudo podman run --rm --network docs-bridge-net \
  -v $P/docs:/data/docs:ro,z -v $P/state:/data/state:z -v $P/cache:/data/cache:z \
  -v $NEW_CFG:/config/config.yaml:ro,z localhost/ingest-worker:$NEW sync --subject all \
  2>&1 | grep '^\[' | tee ~/phase-a/out/B6-rerun.txt
# 2. manifest vs Qdrant, per collection (names differ from subjects: pass both)
for pair in $PAIRS; do
  sudo python3 tools/ingest/verify-corpus.py --manifest $P/state/manifest-subject-keyed.sqlite \
    --subject ${pair%%:*} --collection ${pair##*:} --quiet
done 2>&1 | tee ~/phase-a/out/B6-verify.txt
# 3. one embedder stamp per collection, all identical (stamps() from step 7)
stamps $P/state/manifest-subject-keyed.sqlite | tee ~/phase-a/out/B6-stamps.txt
# 4. indexed and counted
for pair in $PAIRS; do c=${pair##*:}
  curl -s localhost:6333/collections/$c | python3 -c \
    "import json,sys; r=json.load(sys.stdin)['result']; print('$c', r['status'], r['points_count'])"
done | tee ~/phase-a/out/B6-counts.txt
```

**Expect** `0 new / 0 changed / 0 deleted` for every subject, every verify exit 0,
one stamp row per collection, all with the same embedder, and `green` everywhere (wait for green before step
6 — a yellow collection measures the indexer, not retrieval). Don't trust a 0/0/0
without the verify: the manifest records intent, Qdrant records fact (the August
37-PDF gap).

The timer stays paused until the switch-over.

## Step 6 — THE GATE: old image + old collection vs new image + rebuilt collections (~25 min)

**Run it after B.6 and step 8.** The old side searches `teamcenter`; the new side
searches `teamcenter-2512,teamcenter-whats-new` — together exactly the old document
set. A two-pool search uses `rerank.multi_top_n` as its candidate budget; the gate
config sets it equal to `top_n`, so both sides rerank the same number of candidates
and the comparison isn't tilted towards the new side:

```bash
sudo podman run --rm -i -v $NEW_CFG:/in.yaml:ro,z --entrypoint python localhost/docs-bridge:$NEW - \
  > ~/phase-a/config-gate.yaml <<'PY'
import yaml
c = yaml.safe_load(open("/in.yaml"))
c.setdefault("rerank", {})["multi_top_n"] = c["rerank"].get("top_n", 30)
print(yaml.safe_dump(c, sort_keys=False))
PY
grep -n 'top_n' ~/phase-a/config-gate.yaml
```

Then, **one image at a time** (each holds ~4.4 GB; the live server holds another):

```bash
cd ~/phase-a/src
rab_run() {  # $1 image  $2 label  $3 config  $4 subject(s)
  sudo podman run --rm -i --network docs-bridge-net \
    -v $3:/config/config.yaml:ro,z -v ~/phase-a/out:/out:z \
    -v ~/phase-a/src/notes/gate2:/q:ro,z \
    --entrypoint python $1 - run --questions /q/questions-teamcenter.yaml \
    --subject $4 --label $2 --out /out/06-retrieval-$2.json \
    < tools/eval/retrieval_ab.py 2>&1 | tee ~/phase-a/out/06-retrieval-$2.txt
}
rab_run $OLD_SERVER                old $CFG                     teamcenter
rab_run localhost/docs-bridge:$NEW new ~/phase-a/config-gate.yaml teamcenter-2512,teamcenter-whats-new
python3 tools/eval/retrieval_ab.py compare --old ~/phase-a/out/06-retrieval-old.json \
  --new ~/phase-a/out/06-retrieval-new.json --json ~/phase-a/out/06-compare.json \
  | tee ~/phase-a/out/06-compare.txt
{ date; free -m; } >> ~/phase-a/out/B4-ram-gate.txt  # RAM after a real query load on both
```

**Expect `VERDICT: PASS`**: no question answered by the old setup goes unanswered by
the new one, and every control that abstained still abstains. **Ignore the top-k
overlap** it reports: it compares chunk ids, and the re-parse changed the chunks, so
low overlap is expected and means nothing here. hit@6 is the new baseline.

**A MISS needs a look before it counts.** Open the question in `06-retrieval-new.json`:
if the labelled `doc` is in the top-k but the `contains` text no longer matches, the
label is stale (docling 2.130 parsed the passage differently) — fix the label, re-run
`compare`. If the doc is gone from the top-k, it is a real regression.

**If it FAILS on real regressions, stop.** Nothing has been switched; the only thing
to undo is the paused timer (B.2). Bring `B*`, `06-*` and `07-*` back.

### 6b — the two-version view (report only, 5 min)

Not a gate — a look at what callers will see when they ask about both versions:

```bash
rab_run localhost/docs-bridge:$NEW both $NEW_CFG teamcenter-2512,teamcenter-2606
```

Its hit@k is informational (the labels point at 2512 docs; a 2606 copy of the same
text counts through `also_in`). Then two or three manual searches through the REST
mirror on a topic you know changed in 2606, checking that results carry `subjects` with both versions where the text is
the same, and the 2606 passage where it differs.

## Bring the results back

```bash
rsync -a --exclude '*.tar' vhost2:phase-a/out/ notes/phase-a/     # from the repo root
```

---

## The switch-over (operator repo, after steps 5, 7 and 6 pass — step 6 is the gate)

Not done here. What `containers-at-home` (`applications/docs-bridge.yml`, the
`docs_bridge` contract and `_templates/docs-bridge-config.yaml.j2`) must change:

1. **The models image must exist on the host before any image build.** Load it from
   the step-2 tar on a fresh host; never rebuild it from deployed images.
2. **Tag fully qualified**: `localhost/docs-bridge:…` and `localhost/ingest-worker:…`
   — the server is currently under `docker.io/library/`.
3. **Build from the commit that passed** — tag it on `mac-mini` (`git tag
   macmini-rebuild <sha from step 0>`) and pin the deploy to it
   (`-e docs_bridge_ref=macmini-rebuild`; `prep.yml` defaults to the `mac-mini`
   branch tip). Simplest of all: re-use the step-3 images as they are.
4. **The config contract:** the three Teamcenter subjects replace `teamcenter`
   (names, collections, descriptions), the other subjects' collections get their
   `_docling130` names, and `manifest_path` becomes
   `/data/state/manifest-subject-keyed.sqlite` — all exactly as in `$NEW_CFG`; diff
   the rendered config against it. **The template must render per-subject
   `include` / `exclude`** — today it renders only name, dir, collection and
   description, and without them the 2512 / what's-new split silently collapses (2512
   would ingest the what's-new docs too).
5. **The restore path must know the new names.** Backups discover collections live;
   the deploy's "Recover collections" step restores by configured name — check it maps
   the three new collections and the new manifest file.
6. Point the ingest unit and the server at the new tags; restart the server (seconds,
   the only interruption). Then **re-start the timer** paused in B.2
   (`sudo systemctl start docs-bridge-ingest.timer`) and check `systemctl is-enabled`
   on it — in August it was active but not enabled, which a reboot silently undoes.
7. Next morning: the nightly sync ran (`journalctl -u docs-bridge-ingest`), reported
   0/0/0 for every subject, and `verify-corpus.py` still exits 0 for
   each (with `--manifest` and `--collection`, as in B.6).
8. Optional: one line in `server.instructions` on versions — e.g. "Teamcenter docs
   exist per version; use the version the user names, and when they name none, ask
   or search both and say which version each passage is from." The catalog of
   subjects (from the descriptions) is composed in automatically.

**Rollback:** the old config (subject `teamcenter`, old manifest path), the old
image tags, restart the server, re-start the timer on the old worker. The old images,
the old manifest and the old collections (`teamcenter`, `aig`, `pq`) were never
written by any step above — **as long as they exist.** Keep them for a rollback window
(a week or two of normal use), then take a final snapshot of each, drop them (`curl -X
DELETE localhost:6333/collections/<name>`) and archive the old manifest; that also
gives the RAM back (B.4).

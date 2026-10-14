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
> runtime, into a **new manifest** and **new collections**, with Teamcenter split
> by version, each subject in its own dir:
>
> | Subject | Collection | Dir (container) | Content |
> |---|---|---|---|
> | `teamcenter-2512` | `teamcenter_2512` | `/data/docs/teamcenter-2512` | `Siemens/Teamcenter 2512/*`, flattened, minus its 3 What's New PDFs |
> | `teamcenter-whats-new` | `teamcenter_whats_new` | `/data/docs/teamcenter-whats-new` | `Siemens/Teamcenter What's New/*` (14.3 … 2606) + the 2 Gateway What's New PDFs |
> | `teamcenter-2606` | `teamcenter_2606` | `/data/docs/teamcenter-2606` | **later** (B.5), when the 2606 docs arrive |
> | `sow` (+ `ecosys`, `polarion`) | `sow_docling130` (…) | unchanged | rebuilt as they are |
>
> The new dirs are **reflink copies** (btrfs, `cp --reflink=always`): no extra disk, and
> the old `/data/docs/teamcenter` stays exactly as the old stack knows it, so rollback
> needs no file moves. It is removed at the end of the rollback window.
> `Teamcenter 2512/Whats New in Teamcenter 2512.pdf` is byte-identical to
> `whats_new_tc_2512.pdf` and is not copied. The other subjects are rebuilt into
> `<collection>_docling130` because a fresh manifest sees all their files as new and
> the new point ids would not overwrite the old points — a sync into the old
> collection would leave every chunk there twice. `ecosys` and `polarion` have no files
> yet (nothing to build); `sow` is 33 files / 2.1k chunks.
>
> Steps 0–3 were run on 2026-09-29 for the previous plan; the models image (step 2)
> stands, the **images must be rebuilt** (step 3) because they predate the identity
> change.

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

## Step B — the rebuild (one night)

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

### B.1 — lay out the new dirs and write the new config (10 min)

Reflink copies (btrfs: instant, no extra disk; the old tree is only read). Run as
the owner of the docs (`ananchev`), no sudo:

```bash
D=$P/docs; SRC="$D/teamcenter/Siemens"
mkdir -p $D/teamcenter-2512 $D/teamcenter-whats-new
# 2512: everything in "Teamcenter 2512", flattened, minus its What's New PDFs
cp -a --reflink=always "$SRC/Teamcenter 2512/." $D/teamcenter-2512/
for f in "Whats New in Teamcenter 2512.pdf" \
         "Teamcenter Gateway for Enterprise Applications - Whats New.pdf" \
         "Teamcenter Gateway for SAP Business Suite - Whats New.pdf"; do
  rm "$D/teamcenter-2512/$f"
done
# what's new: the per-release guides + the two Gateway What's New (the 2512 duplicate
# "Whats New in Teamcenter 2512.pdf" == whats_new_tc_2512.pdf is not copied)
cp -a --reflink=always "$SRC/Teamcenter What's New/." $D/teamcenter-whats-new/
cp -a --reflink=always "$SRC/Teamcenter 2512/Teamcenter Gateway for Enterprise Applications - Whats New.pdf" \
                       "$SRC/Teamcenter 2512/Teamcenter Gateway for SAP Business Suite - Whats New.pdf" \
                       $D/teamcenter-whats-new/
# the new trees together = the old one, minus the one duplicate
old=$(find $D/teamcenter -type f | wc -l); a=$(find $D/teamcenter-2512 -type f | wc -l)
b=$(find $D/teamcenter-whats-new -type f | wc -l)
echo "old $old = 2512 $a + whats-new $b + 1 duplicate ? $((a + b + 1))" | tee ~/phase-a/out/B1-split.txt
ls $D/teamcenter-whats-new | tee -a ~/phase-a/out/B1-split.txt
```

**Expect** the two numbers to be equal (14075 today) and 9 PDFs in what's-new. The
doc_ids change with the flattening (`Siemens/Teamcenter 2512/X.pdf` → `X.pdf`); step 6
rewrites the question labels to match.

Then derive the new config from the live one — same Qdrant, chunking, batch and
server settings; new subjects; new manifest. The descriptions are what an LLM reads
to pick a subject: each names its version, and they point at each other.

```bash
sudo podman run --rm -i -v $CFG:/in.yaml:ro,z --entrypoint python localhost/ingest-worker:$NEW - \
  > $NEW_CFG <<'EOF'
import yaml
c = yaml.safe_load(open("/in.yaml"))
old = [s for s in c["subjects"] if s["name"] == "teamcenter"]
assert len(old) == 1, "expected exactly one teamcenter subject"
others = [s for s in c["subjects"] if s["name"] != "teamcenter"]
for s in others:                                      # rebuilt into new collections too
    s["collection"] += "_docling130"
c["subjects"] = [
    {"name": "teamcenter-2512", "dir": "/data/docs/teamcenter-2512",
     "collection": "teamcenter_2512",
     "description": (
         "Siemens Teamcenter PLM, release 2512 — installation, configuration, "
         "administration, Active Workspace, BMIDE data modelling, server/client APIs, "
         "SOA & ITK development and customization (incl. the SOA API reference), and "
         "the Active Integration Gateway (T4S/T4EA/T4O) for SAP and enterprise-"
         "application integration (AIG runs on and is configured through Teamcenter, "
         "so AIG install/config topics live here). Use for how Teamcenter 2512 works; "
         "for what changed between releases use teamcenter-whats-new.")},
    {"name": "teamcenter-whats-new", "dir": "/data/docs/teamcenter-whats-new",
     "collection": "teamcenter_whats_new",
     "description": (
         "Siemens Teamcenter release notes — the What's New guide of each release "
         "(Teamcenter 14.3, 2312, 2406, 2412, 2506, 2512, 2606) and the Teamcenter "
         "Gateway (T4S, T4EA) What's New for 2512. Use for which release introduced, "
         "changed or removed a feature, and for upgrade planning; for how a feature "
         "works in detail, search that release's own documentation pool "
         "(e.g. teamcenter-2512).")},
] + others
c["manifest_path"] = "/data/state/manifest-subject-keyed.sqlite"
print(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
EOF
grep -nE 'name:|dir:|collection:|manifest_path' $NEW_CFG
```

`others` are the remaining subjects (sow, ecosys, polarion): same names, dirs and
descriptions, new collection names. They are not gated (the question set is
teamcenter's); B.6 verifies them. Check the scan sees what B.1's copy laid out:

```bash
sudo podman run --rm -v $P/docs:/data/docs:ro,z -v $NEW_CFG:/config/config.yaml:ro,z \
  --entrypoint python localhost/ingest-worker:$NEW -c '
from docs_bridge import config
from docs_bridge.parse import scan
cfg = config.load("/config/config.yaml")
for s in cfg.subjects:
    print(s.name, len(scan(s, cfg)))
' 2>&1 | tee -a ~/phase-a/out/B1-split.txt
```

**Expect** the counts to match the file counts above (minus any unsupported
suffixes, which the old corpus skipped too).

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

A full teamcenter ingest was ~13 h on the old stack, so this is one night. `-d` keeps the run alive if SSH drops. Live queries get slower
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
for s in sow; do                                      # the small ones (ecosys, polarion: no files yet)
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

Qdrant now grows a second copy of teamcenter next to the old one (and 2606 later),
plus the worker and the live server, in 16 GB:

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
change the plan** (e.g. snapshot and drop the old collection right after the switch,
accepting a restore-based rollback, and before 2606 is added).

### B.5 — later: 2606, when its docs arrive

Not part of the first rebuild. When the 2606 documentation is on the host, lay it out
flat like 2512 (`$P/docs/teamcenter-2606/X.pdf`, an SOA Reference subdir if it ships
one), add the subject to `$NEW_CFG` — and, after the switch-over, to the operator
contract — then one night:

```yaml
  - name: teamcenter-2606
    dir: /data/docs/teamcenter-2606
    collection: teamcenter_2606
    description: "Siemens Teamcenter PLM, release 2606 — <same scope as teamcenter-2512>.
      Use for how Teamcenter 2606 works; for what changed between releases use
      teamcenter-whats-new."
```

```bash
rebuild teamcenter-2606
sudo podman logs -f rebuild-teamcenter-2606
```

(and B.4's RAM loop again), then B.6 and 6b. After the switch-over this is no longer
a runbook step: the nightly timer picks up a new subject by itself.

### B.6 — check the rebuild (10 min), after both nights

```bash
sudo podman rm rebuild-teamcenter-whats-new rebuild-sow rebuild-teamcenter-2512
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
# the flattened layout drops the "Siemens/Teamcenter 2512/" prefix from every doc_id
sed 's#doc: "Siemens/Teamcenter 2512/#doc: "#' ~/phase-a/src/notes/gate2/questions-teamcenter.yaml \
  > ~/phase-a/src/notes/gate2/questions-teamcenter-flat.yaml
grep -c 'doc: "Siemens' ~/phase-a/src/notes/gate2/questions-teamcenter-flat.yaml   # expect 0
```

Then, **one image at a time** (each holds ~4.4 GB; the live server holds another):

```bash
cd ~/phase-a/src
rab_run() {  # $1 image  $2 label  $3 config  $4 subject(s)  $5 questions file
  sudo podman run --rm -i --network docs-bridge-net \
    -v $3:/config/config.yaml:ro,z -v ~/phase-a/out:/out:z \
    -v ~/phase-a/src/notes/gate2:/q:ro,z \
    --entrypoint python $1 - run --questions /q/$5 \
    --subject $4 --label $2 --out /out/06-retrieval-$2.json \
    < tools/eval/retrieval_ab.py 2>&1 | tee ~/phase-a/out/06-retrieval-$2.txt
}
rab_run $OLD_SERVER                old $CFG                      teamcenter \
  questions-teamcenter.yaml
rab_run localhost/docs-bridge:$NEW new ~/phase-a/config-gate.yaml teamcenter-2512,teamcenter-whats-new \
  questions-teamcenter-flat.yaml
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

### 6b — the two-version view (after B.5; report only, 5 min)

Not a gate — a look at what callers will see when they ask about both versions:

```bash
rab_run localhost/docs-bridge:$NEW both $NEW_CFG teamcenter-2512,teamcenter-2606 \
  questions-teamcenter-flat.yaml
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
4. **The config contract:** `teamcenter-2512` and `teamcenter-whats-new` replace
   `teamcenter` (names, dirs, collections, descriptions), the other subjects'
   collections get their `_docling130` names, and `manifest_path` becomes
   `/data/state/manifest-subject-keyed.sqlite` — all exactly as in `$NEW_CFG`; diff the
   rendered config against it. The subjects' `dir`s must exist under the payload root
   (`prep.yml` creates missing ones, empty — harmless, B.1 already filled them).
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
gives the RAM back (B.4). Remove the old `$P/docs/teamcenter` tree at the same time
(the reflink copies own their data; nothing else references the old paths).

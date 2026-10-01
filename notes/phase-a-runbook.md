# Phase A runbook — the toolchain upgrade, built and gated on vhost2

> **Cold start — where this stands (2026-10-01).** Read this block, then jump to the
> first step not marked done.
>
> **The plan (revised 2026-10-01 with the user).** The Mac mini moves to the new
> toolchain by **re-embedding `teamcenter` into a side collection,
> `teamcenter_after_ort130`**, with the new worker, while the old server keeps serving
> the old collection. The new server is switched to the new collection only after the
> gate (step 6) passes — no downtime, and the old collection plus the old images stay
> as the rollback. In parallel, and independently, the EC2 image is built
> (`multi-user-rollout-plan.md` §8 — no EC2 collection before its open decisions are
> settled). EC2 never reuses anything from the Mac mini.
>
> **Order:** 0 ✔ → 0b ✔ → 1 ✔ → 2 ✔ → 3 ✔ → 4 ✔ (diagnostic) → **5 → R (re-embed,
> overnight; 8 runs alongside) → 6 (the gate) → 7 → switch-over.**
>
> **Why the re-embed (history).** Step 4 (2026-09-29): same model bytes, 445/445
> identical tokens, but onnxruntime 1.20.1 → 1.30.0 moves the dynamic-INT8 vectors to
> cosine mean 0.987 / min 0.979 (bar ≥ 0.999) — the INT8 kernels changed. The new
> runtime is exactly as far from the stored vectors as the old one (0.982060 vs
> 0.982061), so querying the old collection with the new runtime would probably have
> worked; but re-embedding makes the collection and its query runtime one consistent
> pair, replaces the June vectors made by builds that no longer exist (0b.1: every image
> still on the host, oldest 2026-07-02, carries bge-m3 `68086607db73f9c3`; the June
> builds are gone), and costs one night with no downtime. Re-embedding is the answer
> *this once*, not the upgrade policy: `stack-upgrade-strategy.md` §7.
>
> **What exists.** Branch `multi-user-rollout`: new pins + per-image locks, frozen
> models image (`localhost/docs-bridge-models:frozen-2026-09`, saved as a tar next to
> the Qdrant snapshots), FlagEmbedding removed, `parse.py` (speaker notes, alt-text,
> RapidOCR, Office format options), the tools (`embed_upgrade_parity.py`,
> `retrieval_ab.py`, `reembed_to_collection.py`) and the question set
> (`notes/gate2/questions-teamcenter.yaml`, 45 entries; the user is verifying the labels,
> step 8). The new images `localhost/{docs-bridge,ingest-worker}:2026.09-phase-a` are
> built on vhost2 (step 3). Results land in `~/phase-a/out/` on vhost2 and come back to
> `notes/phase-a/`.
>
> **Open things to watch.** RAM while the re-embed runs and after (step R.4 logs it —
> the host has 16 GB: live server ~4.4 GB, worker ~4.8 GB, and Qdrant now holds two
> copies of `teamcenter`); disk is not a concern. The Mac mini images must come from
> the commit noted in step 0, never from a later commit of the branch, which will carry
> the EC2 work (switch-over, item 3).

Executes `multi-user-rollout-plan.md` §4 under decision **A** (2026-09-28): upgrade
everything on vhost2 — one new ingest-worker for both corpora, one new server image
for the Mac mini. Since 2026-10-01 the Mac mini gets a re-embedded `teamcenter`
(step R) and the gate is retrieval, old image + old collection vs new image + new
collection (step 6); step 4 is the diagnostic (status above). Nothing here touches the
running stack: every image gets a new tag, `:latest` is never retagged, the live server
keeps serving the old collection, and the old collection is only ever read. The one
change to the running host is that the ingest timer is **paused** from step R until
the switch-over, so the old collection stops changing while it is copied. The switch-over is the operator repo's, after the gates
(last section).

What changed in the repo, and why it matters for these steps:

| Change | Consequence here |
|---|---|
| docling 2.15.1 → 2.130.0; onnxruntime 1.20.1 → 1.30.0; tokenizers 0.19.1 → 0.23.2; numpy 1.26 → 2.4.6; torch 2.5.1 → 2.14.0+cpu | the embedder **runtime** changes under stored vectors → step 4 measures it, step 6 decides |
| **Models are no longer exported at build.** Both images copy them from `localhost/docs-bridge-models` | step 2 must run **before** anything replaces the deployed images; it takes the exact bytes out of them |
| Full transitive **locks** per image (`images/lock.sh`) | builds never re-resolve; the embedder runtime is provably identical in both images |
| FlagEmbedding backend removed | `embedding_backend: flagembedding` in a config now fails fast |
| `parse.py`: speaker notes, alt-text, RapidOCR, Office format options | only newly ingested documents change; `teamcenter`'s unchanged files are never re-parsed (hash-delta) |

Root podman needs `sudo` on vhost2, so these are yours to run. Every step writes into
`~/phase-a/out/`; bring that directory back into the repo's `notes/phase-a/` when
done and I will write the results up.

Throughout:

```bash
NEW=2026.09-phase-a                                   # tag for the new images
OLD_WORKER=localhost/ingest-worker:latest             # deployed; embedded teamcenter
OLD_SERVER=docker.io/library/docs-bridge:latest       # deployed; note the namespace
MODELS=localhost/docs-bridge-models:frozen-2026-09
CFG=/data/docker/docs-bridge/config/config.yaml
TC_NEW=teamcenter_after_ort130                        # step R's target collection
```

These are plain shell variables: a new shell or a reconnect loses them, and an empty
`$MODELS` turns `-t $MODELS .` into "tag `.`: invalid reference format". Keep the
block in `~/phase-a/vars.sh` and `source` it after every login.

---

## Step 0 — stage the source, and look at what is deployed (3 min)

From the repo root on the **workstation**, at the Phase A commit:

```bash
git rev-parse --short HEAD                            # note it in the results
ssh vhost2 'mkdir -p ~/phase-a/out'
rsync -a --delete --exclude .git ./ vhost2:phase-a/src/
```

Then on **vhost2**:

```bash
sudo podman images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Created}}' \
  | grep -E 'ingest-worker|docs-bridge' | tee ~/phase-a/out/00-images.txt
for img in $OLD_WORKER $OLD_SERVER; do
  echo "== $img"
  sudo podman run --rm --entrypoint sh $img -c \
    'sha256sum /opt/bge-m3-onnx/* /opt/bge-reranker-onnx/* 2>/dev/null;
     python -c "import onnxruntime,tokenizers,numpy; print(onnxruntime.__version__, tokenizers.__version__, numpy.__version__)"'
done 2>&1 | tee ~/phase-a/out/00-models.txt
```

**Expect** the four `/opt/bge-m3-onnx/*` hashes to be **identical in both images**
(the worker embedded the docs, the server embeds the queries — they must be the same
model), and `1.20.1 0.19.1 1.26.x` in both. **If the bge-m3 hashes differ, stop**:
that is a pre-existing query/doc mismatch, and it decides which bytes to freeze.

## Step 0b — were all the stored vectors made by ONE model? (~12 min)

Step 0 compares today's two images. It cannot see the past: `teamcenter` was embedded
in several runs by worker images that were each **rebuilt**, and every rebuild
re-exported the model:

| Run | What it embedded | Chunks |
|---|---|---|
| Jun 24-25 | the first PDF guides | ~27.8k still stored |
| Jun 27-28 | 13,707 SOA reference pages (+ a resume run; a worker rebuild landed mid-run) | ~81.5k |
| Jun (AIG) | 211 AIG API pages | ~2.6k |
| Aug 31 | 37 PDFs re-ingested (`~/reingest-*.txt`) | ~26.7k |

If one of those rebuilds produced a slightly different model, part of the collection
was measured with a different ruler — and step 4 would not notice, because it
compares the old and new *runtime* against the same stored vectors. Two checks, the
cheap definitive one first.

**0b.1 — hash every worker image still on the host (2 min).** Rebuilds usually leave
the previous image behind, untagged. If the June and August builds are still here,
their model fingerprints answer the question directly.

```bash
for id in $(sudo podman images -a -q | sort -u); do
  mnt=$(sudo podman image mount $id 2>/dev/null) || continue
  f=$mnt/opt/bge-m3-onnx/model.int8.onnx
  if sudo test -f $f; then
    echo "$(sudo podman image inspect $id --format '{{.Created}}') $id" \
         "$(sudo sha256sum $f | cut -c1-16)" \
         "$(sudo podman image inspect $id --format '{{.RepoTags}}')"
  fi
  sudo podman image unmount $id >/dev/null 2>&1
done | sort | tee ~/phase-a/out/00b-image-history.txt
```

One line per image that carries an embedder, oldest first: creation time, image id,
the first 16 hex of the model's sha256, and its tags (`[]` = an untagged old build).
**Read it as:** every line with the same fingerprint → the rebuilds reproduced the
model byte for byte, and 0b.2 only confirms it. Different fingerprints → note which
dates they span; 0b.2 then shows whether the difference reaches the vectors. Only
some builds still here → 0b.2 covers the rest.

**0b.2 — measure it (~10 min, read-only).** Runs in the **current** worker (today's
ruler). It splits `teamcenter` into the runs above, re-embeds 3 windows of 64
neighbouring chunks per run — each chunk alone and inside its window, the way ingest
batched — and compares each run's stored vectors against that run's own normal
wobble (the INT8 batch effect: how far two measurements with the same model land).

```bash
cd ~/phase-a/src
sudo podman run --rm -i --network docs-bridge-net \
  -v $CFG:/config/config.yaml:ro,z -v ~/phase-a/out:/out:z \
  -v ~/reingest-batch3.txt:/r/batch3.txt:ro,z -v ~/reingest-rest34.txt:/r/rest34.txt:ro,z \
  --entrypoint python $OLD_WORKER - history --collection teamcenter \
  --recent-docs /r/batch3.txt /r/rest34.txt --windows 3 \
  --json /out/00b-history.json < tools/eval/embed_upgrade_parity.py \
  2>&1 | tee ~/phase-a/out/00b-history.txt
```

**Expect `VERDICT: CONSISTENT`** — every group's stored vectors within 2x its own
wobble, `outside` near 0% everywhere. The `reingested-aug31` row is the control:
almost certainly embedded by today's worker (step 0's image date confirms), so it
shows what "same ruler" looks like on real data; the June rows should look like it.

## Step 1 — decide before freezing

- **0 and 0b agree (same fingerprints and/or CONSISTENT)** → the collection was made
  by one model, and it is the one in today's worker. Freeze it: step 2.
- **0b.2 says SUSPECT for a group** → **stop.** That group was measured with a
  different ruler than today's worker, and freezing today's bytes would lock the
  mismatch in. Bring `00b-*` back first. The JSON names the affected documents; the
  likely fix is re-embedding just that group with the frozen model (a targeted
  re-ingest via `tools/ingest/reingest-docs.py`), not all of `teamcenter` — but which
  bytes to freeze is decided from the evidence, not before it.

## Step 2 — freeze the models (1 min) — before anything else is built

```bash
cd ~/phase-a/src
sudo podman build -f images/models/Dockerfile \
  --build-arg WORKER_IMAGE=$OLD_WORKER --build-arg SERVER_IMAGE=$OLD_SERVER \
  -t $MODELS .
# the frozen bytes must be the step-0 bytes (a scratch image has no shell: copy out)
sudo podman create --name frozen-check $MODELS /x >/dev/null
sudo podman cp frozen-check:/opt ~/phase-a/out/frozen-opt && sudo podman rm frozen-check
sha256sum ~/phase-a/out/frozen-opt/*/* | tee ~/phase-a/out/02-frozen.txt
sudo rm -rf ~/phase-a/out/frozen-opt
# keep it: the only copy of these bytes once the old images are gone
sudo podman save -o ~/phase-a/out/docs-bridge-models-frozen-2026-09.tar $MODELS
```

**Expect** the hashes in `02-frozen.txt` to equal the step-0 ones (bge-m3 from the
worker, reranker from the server). Move the `.tar` next to the Qdrant snapshots —
it is ~1.1 GB and it is the reference artifact from now on; do not bring it into git.

## Step 3 — build the new images (~5 min)

```bash
cd ~/phase-a/src
sudo podman build -f images/docs-bridge/Dockerfile   -t localhost/docs-bridge:$NEW   . 2>&1 | tail -3
sudo podman build -f images/ingest-worker/Dockerfile -t localhost/ingest-worker:$NEW . 2>&1 | tail -3
for img in localhost/docs-bridge:$NEW localhost/ingest-worker:$NEW; do
  echo "== $img"
  sudo podman run --rm --entrypoint sh $img -c \
    'sha256sum /opt/bge-m3-onnx/*;
     python -c "from importlib.metadata import version as v; print({p: v(p) for p in [\"onnxruntime\",\"tokenizers\",\"numpy\",\"qdrant-client\"]})";
     pip list 2>/dev/null | grep -ci nvidia'
done 2>&1 | tee ~/phase-a/out/03-new-images.txt
```

**Expect** the same bge-m3 hashes as step 0 in both, `onnxruntime 1.30.0, tokenizers
0.23.2, numpy 2.4.6` in both, and `0` nvidia packages. (Both builds were test-built
rootless on vhost2 with placeholder models on 2026-09-28: server ~0.5 GB, worker
~2.1 GB before the ~0.6 GB model.)

## Step 4 — embed parity against the stored vectors: the diagnostic (~10 min)

> **Ran 2026-09-29: `VERDICT: FAIL` on bar 3 only, and no longer blocking.** Bars 1, 2
> and 4 pass (same model, same tokens, no new drift vs the stored vectors); bar 3
> (old vs new ≥ 0.999) fails at 0.979 — the onnxruntime 1.30 INT8 kernels, not the
> tokenizer. Bar 3 is stricter than the noise the collection already holds (the old
> runtime itself reproduces stored vectors at 0.982), so it cannot be the gate; step 6
> is. Read this step's output as the *why* when step 6 disagrees. The original text
> follows unchanged.

Three dumps of the same 400 `teamcenter` chunks + the 45 questions: the **old worker**
(the runtime that made the stored vectors) and **both** new images (the new ingest
runtime and the new query runtime — the lock says they are identical; this proves it).

```bash
cd ~/phase-a/src
dump() {  # $1 image  $2 output name  [$3...] extra args
  local img=$1 name=$2; shift 2
  sudo podman run --rm -i --network docs-bridge-net \
    -v $CFG:/config/config.yaml:ro,z -v ~/phase-a/out:/out:z \
    -v ~/phase-a/src/notes/gate2:/q:ro,z \
    --entrypoint python $img - dump --collection teamcenter --n 400 \
    --queries /q/questions-teamcenter.yaml --out /out/$name "$@" \
    < tools/eval/embed_upgrade_parity.py
}
dump $OLD_WORKER                  embed-old-worker.json
dump localhost/ingest-worker:$NEW embed-new-worker.json --ids-from /out/embed-old-worker.json
dump localhost/docs-bridge:$NEW   embed-new-server.json --ids-from /out/embed-old-worker.json
for side in worker server; do
  sudo podman run --rm -i -v ~/phase-a/out:/out:z --entrypoint python localhost/docs-bridge:$NEW \
    - compare --old /out/embed-old-worker.json --new /out/embed-new-$side.json \
    --json /out/04-parity-$side.json < tools/eval/embed_upgrade_parity.py \
    2>&1 | tee ~/phase-a/out/04-parity-$side.txt
done
```

**Both must print `VERDICT: PASS`** *(original bar — see the note at the top of this step).* The bar is in the tool's docstring: same model
bytes; old-vs-new dense cosine ≥ 0.999 for every text and sparse Jaccard ≥ 0.95;
and the new runtime no further from the stored vectors than the old runtime itself
(re-embedding never reproduces a stored vector exactly — ingest ran in batches and
INT8 scales depend on the batch — so the old runtime's own distance is the baseline).

~~**If it fails, stop here.**~~ Superseded 2026-09-29: a bar-3 failure with bars 1, 2
and 4 passing goes on to step 6. Still stop here if bar 1 (model) or bar 4 (new drift
vs stored) fails, or if the tokens moved (hold `tokenizers` at 0.19.1). The step-6
fallbacks are listed there.

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
scores the old runtime produced) — if so the second is a formality. *After step 4 it
may not:* the reranker is dynamic INT8 on the same new kernels, so it can move the way
the embedder did. Then the second run decides, and it is the meaningful one anyway. The second
reports how far the INT8 reranker sits from fp32 on the new runtime; the bar (plan
§5.2) is **no worse than on the old one**: max |Δ| ≤ 1.03, top-6 ordered-identical
≥ 2/8. `--threads 6` because C = 6 on this host (Gate 1).

## Step R — re-embed `teamcenter` into `teamcenter_after_ort130` (overnight, ~13 h)

The new worker re-embeds every chunk of the live collection into a new one:
`tools/eval/reembed_to_collection.py` reads each point's text and payload, embeds
with the frozen INT8 model on the new runtime, and upserts with the **same point id and
payload**. No re-parse, no manifest, and the source collection is only read. The same
point ids are what keep step 6 comparable chunk by chunk and the question labels valid.
`--batch 64` matches the Mac mini's ingest batch, so the new vectors carry the same
kind of batch effect the old ones do.

**R.0 — baseline the source (2 min).** Whatever is wrong with the old collection is
copied, so look first:

```bash
cd ~/phase-a/src
sudo python3 tools/ingest/verify-corpus.py --subject teamcenter \
  2>&1 | tee ~/phase-a/out/R0-verify-old.txt          # expect exit 0, no drifted docs
curl -s localhost:6333/collections/teamcenter | python3 -c \
  'import json,sys; r=json.load(sys.stdin)["result"]; print(r["status"], r["points_count"])' \
  | tee ~/phase-a/out/R0-count-old.txt
```

If verify reports drift, repair it first (`reingest-docs.py`, with the old worker), or
the new collection inherits the hole.

**R.1 — probe (5 min).** Proves the tool on the new stack and measures its speed:

```bash
cd ~/phase-a/src
sudo podman run --rm --network docs-bridge-net \
  -v $CFG:/config/config.yaml:ro,z \
  -v ~/phase-a/src/tools/eval/reembed_to_collection.py:/reembed.py:ro,z \
  --entrypoint python localhost/ingest-worker:$NEW /reembed.py \
  --config /config/config.yaml --source teamcenter --target $TC_NEW \
  --model-dir /opt/bge-m3-onnx --int8 --batch 64 --fresh --limit 2000 \
  2>&1 | tee ~/phase-a/out/R1-probe.txt
```

**Expect** a clean `DONE ~2000 chunks` and a s/chunk near 0.34 (the old runtime's
figure). Multiply by the R.0 count for the full run's length; much slower → bring
`R1-probe.txt` back before committing a night to it.

**R.2 — pause the nightly ingest.** From here until the switch-over the old collection
must not change, or the copy misses it:

```bash
systemctl list-timers --all | grep docs-bridge | tee ~/phase-a/out/R2-timer.txt
sudo systemctl stop docs-bridge-ingest.timer
systemctl is-active docs-bridge-ingest.service        # must not say "active" (a run in flight)
```

To undo (step 6 fails, or the plan is abandoned): `sudo systemctl start
docs-bridge-ingest.timer` — the old worker resumes on the old collection as before.

**R.3 — the full run, detached (overnight).** `-d` keeps it running if SSH drops;
`--fresh` drops the probe's partial collection first. Live queries get slower while it
runs (it competes for the same cores), so start it in the evening:

```bash
cd ~/phase-a/src
sudo podman run -d --name reembed-tc --network docs-bridge-net \
  -v $CFG:/config/config.yaml:ro,z \
  -v ~/phase-a/src/tools/eval/reembed_to_collection.py:/reembed.py:ro,z \
  --entrypoint python localhost/ingest-worker:$NEW /reembed.py \
  --config /config/config.yaml --source teamcenter --target $TC_NEW \
  --model-dir /opt/bge-m3-onnx --int8 --batch 64 --fresh
sudo podman logs -f reembed-tc                        # Ctrl-C detaches; the run goes on
```

**R.4 — log RAM while it runs.** The host has 16 GB, and nobody has measured it with
two copies of `teamcenter` in Qdrant plus a re-embed plus the live server. Start this
right after R.3; it stops by itself when the re-embed container exits:

```bash
sudo -v
sudo nohup sh -c 'while podman ps -q --filter name=reembed-tc | grep -q .; do
  date; free -m | sed -n 2p
  podman stats --no-stream --format "{{.Name}} {{.MemUsage}}"
  sleep 600
done' > ~/phase-a/out/R4-ram.txt 2>&1 &
```

In the morning: `available` in `free -m` should never have approached 0, and no
container should have restarted (`sudo podman ps -a`). After the run, one more reading
— the steady state the Mac mini lives with until the old collection is dropped:

```bash
{ date; free -m; sudo podman stats --no-stream; } >> ~/phase-a/out/R4-ram.txt
```

**A RAM problem here is the one finding that would change the plan** (e.g. drop the
old collection sooner, keeping it as a snapshot instead of live).

**R.5 — check the copy (5 min), after `DONE`:**

```bash
sudo podman logs reembed-tc 2>&1 | tail -3 | tee ~/phase-a/out/R5-done.txt
sudo podman rm reembed-tc
curl -s localhost:6333/collections/$TC_NEW | python3 -c \
  'import json,sys; r=json.load(sys.stdin)["result"]; print(r["status"], r["points_count"])' \
  | tee ~/phase-a/out/R5-count-new.txt
cd ~/phase-a/src
sudo python3 tools/ingest/verify-corpus.py --subject teamcenter --collection $TC_NEW \
  2>&1 | tee ~/phase-a/out/R5-verify-new.txt
```

**Expect** `green` and the same `points_count` as R.0 (the tool skips points with
empty text, so any shortfall must equal the number of those — verify names any doc
that lost chunks), and verify exit 0. Wait for `green` before step 6: Qdrant indexes
after the upload, and a search against a `yellow` collection measures the indexer.

The timer stays paused until the switch-over.

## Step 6 — THE GATE: old image + old collection vs new image + new collection (~25 min)

**Run it after step R and step 8**, so it scores the collection you will switch to,
against verified labels, once.

The new image reads a copy of the live config with only `teamcenter`'s collection
changed. It goes outside `~/phase-a/out`, because that directory comes back into the
repo:

```bash
sudo podman run --rm -i -v $CFG:/in.yaml:ro,z --entrypoint python localhost/docs-bridge:$NEW - \
  > ~/phase-a/config-after.yaml <<'PY'
import yaml
c = yaml.safe_load(open("/in.yaml"))
hit = [s for s in c["subjects"] if s["name"] == "teamcenter"]
assert len(hit) == 1, "expected exactly one teamcenter subject"
hit[0]["collection"] = "teamcenter_after_ort130"
print(yaml.safe_dump(c, sort_keys=False))
PY
grep -n 'teamcenter' ~/phase-a/config-after.yaml      # the subject now names the new collection
```

Then, **one image at a time** (each holds ~4.4 GB; the live server holds another):

```bash
cd ~/phase-a/src
rab_run() {  # $1 image  $2 label  $3 config
  sudo podman run --rm -i --network docs-bridge-net \
    -v $3:/config/config.yaml:ro,z -v ~/phase-a/out:/out:z \
    -v ~/phase-a/src/notes/gate2:/q:ro,z \
    --entrypoint python $1 - run --questions /q/questions-teamcenter.yaml \
    --subject teamcenter --label $2 --out /out/06-retrieval-$2.json \
    < tools/eval/retrieval_ab.py 2>&1 | tee ~/phase-a/out/06-retrieval-$2.txt
}
rab_run $OLD_SERVER                old $CFG
rab_run localhost/docs-bridge:$NEW new ~/phase-a/config-after.yaml
python3 tools/eval/retrieval_ab.py compare --old ~/phase-a/out/06-retrieval-old.json \
  --new ~/phase-a/out/06-retrieval-new.json --json ~/phase-a/out/06-compare.json \
  | tee ~/phase-a/out/06-compare.txt
{ date; free -m; } >> ~/phase-a/out/R4-ram.txt        # RAM after a real query load on both
```

The comparison is chunk-ID based, which is valid because step R kept the point ids.

**Expect `VERDICT: PASS`**: no question answered by the old setup goes unanswered by
the new one, and every control that abstained still abstains. hit@6 itself is the
baseline for later, not a gate — and it is only as good as the labels (step 8). Also
look at the top-k overlap it reports: a PASS with a large reshuffle is worth bringing
back before switching.

**If it FAILS, stop.** Nothing has been switched, so the only thing to undo is the
paused timer (R.2). Bring `04-*`, `06-*` and `R*` back; the cause (labels, the
re-embed, or the new runtime itself) is decided from them before any next attempt.
Pinning onnxruntime 1.20.1 stays the last resort: it costs RapidOCR (docling's ONNX
OCR extras need ≥ 1.24.1) and would tie the work-archive ingest to the pin too.

## Step 7 — smoke-ingest real files into a SCRATCH collection (~10 min)

Proves the new worker end to end (parse → embed → upsert) without touching the live
manifest or collections. Put a handful of representative files in a scratch dir —
**one deck with speaker notes, one DOCX with figures, one scanned PDF, one normal
PDF** (from the work archive if it is on the host yet, otherwise anything similar):

```bash
mkdir -p ~/phase-a/smoke/docs ~/phase-a/smoke/state ~/phase-a/smoke/cache
cp <your files> ~/phase-a/smoke/docs/
# the live config with ONLY the subjects, manifest path, filters and OCR replaced
sudo podman run --rm -i -v $CFG:/in.yaml:ro,z --entrypoint python localhost/ingest-worker:$NEW - \
  > ~/phase-a/smoke/config.yaml <<'EOF'
import yaml
c = yaml.safe_load(open("/in.yaml"))
c["subjects"] = [{"name": "phase_a_smoke", "dir": "/data/docs/smoke", "collection": "phase_a_smoke"}]
c["manifest_path"] = "/data/state/manifest-smoke.sqlite"
c["include"], c["exclude"] = [], []
c.setdefault("parse", {})["ocr"] = True
print(yaml.safe_dump(c, sort_keys=False))
EOF
grep -nE 'manifest_path|phase_a_smoke|ocr' ~/phase-a/smoke/config.yaml
sudo podman run --rm --network docs-bridge-net \
  -v ~/phase-a/smoke/docs:/data/docs/smoke:ro,z \
  -v ~/phase-a/smoke/state:/data/state:z \
  -v ~/phase-a/smoke/cache:/data/cache:z \
  -v ~/phase-a/smoke/config.yaml:/config/config.yaml:ro,z \
  localhost/ingest-worker:$NEW sync --subject phase_a_smoke --verbose \
  2>&1 | tee ~/phase-a/out/07-smoke-ingest.txt
# what landed — speaker notes and figure captions should be visible in the text
curl -s -X POST localhost:6333/collections/phase_a_smoke/points/scroll \
  -H 'Content-Type: application/json' \
  -d '{"limit":200,"with_payload":["doc_id","section_path","text"]}' \
  > ~/phase-a/out/07-smoke-chunks.json
grep -o 'Speaker notes:[^"]\{0,80\}\|Figure:[^"]\{0,80\}' ~/phase-a/out/07-smoke-chunks.json | head
# re-run must be a no-op (hash-delta): expect 0/0/0
sudo podman run --rm --network docs-bridge-net \
  -v ~/phase-a/smoke/docs:/data/docs/smoke:ro,z -v ~/phase-a/smoke/state:/data/state:z \
  -v ~/phase-a/smoke/cache:/data/cache:z -v ~/phase-a/smoke/config.yaml:/config/config.yaml:ro,z \
  localhost/ingest-worker:$NEW sync --subject phase_a_smoke 2>&1 | tail -2
# clean up the scratch collection
curl -s -X DELETE localhost:6333/collections/phase_a_smoke
```

The config rewrite keeps the live settings (Qdrant host, chunking, ingest batch
size) and replaces only the subjects, the manifest path, the global filters and OCR —
check the `grep` shows the smoke manifest, the smoke subject and `ocr: true`. **Expect** a clean exit, `Speaker notes:`
and `Figure:` lines if your files carry notes and alt-text, text from the scanned
PDF, and `0/0/0` on the re-run. The first run downloads docling's new layout models
into the scratch cache (a few hundred MB); the real cache fills the same way on the
first nightly run after the switch.

## Step 8 — verify the question labels (you, ~45 min, alongside step R, BEFORE step 6)

`notes/gate2/questions-teamcenter.yaml`: 42 questions + 3 controls, each label
checked by script to match a real chunk (0 misses, 2026-09-28) — but drafted by me
from random chunks. For each: read the `evidence`, decide whether that chunk really
answers the question as a Teamcenter user would mean it, fix `expect` if not, and
set `status: verified` (or `drop`). Step 6's verdict is only as good as these.

## Bring the results back

```bash
rsync -a --exclude '*.tar' vhost2:phase-a/out/ notes/phase-a/     # from the repo root
```

---

## The switch-over (operator repo, after steps 5–7 pass — step 6 is the gate)

Not done here. What the operator repo (`applications/docs-bridge.yml`) must change,
because the Dockerfiles now expect things the old build did not, and because the
Mac mini now serves a different collection:

1. **The models image must exist on the host before any image build.** Load it from
   the step-2 tar (`podman load -i …`) on a fresh host; never rebuild it from images
   that are about to be replaced.
2. **Tag fully qualified**: `localhost/docs-bridge:…` and `localhost/ingest-worker:…`
   — the server is currently under `docker.io/library/`.
3. **Build from the commit that passed the gates** — the one noted in step 0 (tag it,
   e.g. `git tag macmini-phase-a <sha>`), **not** the tip of the branch: the branch
   carries on with the EC2 work, and anything built from a later commit is untested on
   the Mac mini. The build pulls the locked wheels (network at build, as before) but
   **no longer pulls HF weights**. Simplest of all: re-use the step-3 images as they are.
4. **The `teamcenter` subject's `collection` becomes `teamcenter_after_ort130`** in the
   rendered `config.yaml` — for the server and the ingest unit alike (the subject name,
   its `dir` and the manifest stay as they are; the manifest tracks documents, not
   collections, and the copy has the same documents).
5. **The restore path must know the new name.** Backups discover collections live, so
   the new one is snapshotted without changes; but the deploy's "Recover collections"
   step restores by the configured collection name — check it maps
   `teamcenter_after_ort130` to its own snapshot before relying on a restore.
6. Point the ingest unit and the server at the new tags; restart the server (seconds,
   the only interruption). Then **re-start the timer** paused in R.2
   (`sudo systemctl start docs-bridge-ingest.timer`), and check `systemctl is-enabled`
   on it — in August it was active but not enabled, which a reboot silently undoes.
7. First nightly ingest after the switch downloads the new docling layout models into
   `/data/docs-bridge-payload/cache`, and writes into the new collection — check the
   next morning that `sync` ran and `verify-corpus.py --collection
   teamcenter_after_ort130` still exits 0.

**Rollback:** the old config (old collection name), the old image tags, restart the
server, timer on the old worker. The old images are untouched and the old collection
was only ever read — **as long as it exists.** Keep it for a rollback window (a week or
two of normal use), then take a final snapshot of it and drop it
(`curl -X DELETE localhost:6333/collections/teamcenter`); that also gives the RAM back
(R.4). Until then Qdrant carries both.

# Gate 1 runbook — the exact commands, and the results

Executes `multi-user-rollout-plan.md` §2 (+ §3) on vhost2 against **today's
image and today's pins** — that is the point of Gate 1: a hardware decision must
never be entangled with dependency churn. No repo change below alters the running
stack; the three harnesses are new files under `tools/eval/`, and the one edit to an
existing file (`--model-id` on the reranker exporter) leaves the baked v2-m3 export
byte-identical.

**Host:** vhost2 (`192.168.2.199`) — M2 Pro, Asahi, aarch64, 10 cores, 16 GiB.
Containers run under **root podman**; the scripts are already staged in
`~/gate1/` and every command writes into `~/gate1/out/`.

Run them in one session: `ssh -t vhost2`, then `cd ~/gate1`.

**Two things to know before starting.** Each step runs a **fresh throwaway
container** rather than `exec`-ing into the live server, so the server keeps serving
and never shares a process with a measurement. And the box has ~5 GiB free of 16
(the server holds ~4.4), so **do not run searches against docs-bridge while step 2
is timing** — CPU contention is the one thing that would silently corrupt C.

---

## Step 0 — confirm the names this runbook assumes (seconds)

```bash
sudo podman ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
sudo podman network ls
```

Expected: a `docs-bridge` and a `qdrant` container, images `localhost/docs-bridge:latest`
and an `ingest-worker:latest` available, plus a `docs-bridge-net` network. If any
name differs, substitute it below.

## Step 1 — §2.2 the fixture (~4 min)

The one artifact every later gate consumes. 8 validated queries × (the production
40 RRF candidates + a deeper 80-candidate oracle pool), with the current CPU-INT8
scores as the baseline column.

```bash
sudo podman run --rm -i --network docs-bridge-net \
  -v /data/docker/docs-bridge/config/config.yaml:/config/config.yaml:ro,z \
  -v ~/gate1/out:/out:z \
  --entrypoint python localhost/docs-bridge:latest - \
  < dump_rerank_fixture.py > out/fixture.json
```

Progress goes to the terminal, JSON to the file. It also prints the pair-token
distribution, which is §3's `max_length` review for free.

## Step 2 — §2.1 the thread-scaling curve → **C** (~12 min)

The number the entire fleet shape depends on, and the one thing in the plan that
has never been measured. Scores the fixture's first 40-pair batch with a fresh
`OnnxReranker(threads=N)` per N, N = 1,2,3,4,6,8,10.

```bash
sudo podman run --rm -i \
  -v /data/docker/docs-bridge/config/config.yaml:/config/config.yaml:ro,z \
  -v ~/gate1/out:/out:z \
  -e TS_FIXTURE=/out/fixture.json \
  --entrypoint python localhost/docs-bridge:latest - \
  < thread_scaling.py 2>&1 | tee out/thread_scaling.txt
```

No Qdrant, no embedder, one model resident at a time — so it is also the leanest
step memory-wise. Reads out C two independent ways and says so if they disagree.

## Step 3 — §3 export the smaller reranker (~10 min, needs egress)

`bge-reranker-base` is ~278M against v2-m3's ~568M. This is the experiment that
may delete the problem: no new hardware, no architecture change. Runs in a
throwaway `ingest-worker` container because that image already carries
torch+transformers; `onnx` is the one build-only dep it lacks.

```bash
sudo podman run --rm \
  -v ~/gate1/export_bge_reranker_onnx.py:/work/export.py:ro,z \
  -v ~/gate1/models:/models:z \
  --entrypoint bash localhost/ingest-worker:latest -lc '
    set -e
    pip install --no-cache-dir "onnx>=1.17"
    python /work/export.py --model-id BAAI/bge-reranker-base --out /models/bge-reranker-base-onnx
    python /work/export.py --model-id BAAI/bge-reranker-base --out /models/bge-reranker-base-onnx --quantize-only
  ' 2>&1 | tee out/export-base.txt
```

Two processes on purpose (export, then quantize) so torch's model is never resident
alongside the fp32 ONNX being quantized — the peak that OOM-killed the Pi. If it
still OOMs on 16 GiB, stop the server for this step or run it on the M2.

## Step 4a — validate the A/B harness against the baseline (~5 min)

`rerank_ab.py` re-implements `OnnxReranker`'s ~15 lines so a probe never needs an
image rebuild (and so it runs on a bare GPU box). Prove that stand-in is faithful
**before** trusting any comparison it makes: re-score the fixture with the baked
baseline model and reproduce the baseline column.

```bash
sudo podman run --rm -i \
  -v ~/gate1/out:/out:z \
  --entrypoint python localhost/docs-bridge:latest - \
  --fixture /out/fixture.json --backend onnx --selftest --repeats 1 \
  --json /out/ab-selftest.json \
  < rerank_ab.py 2>&1 | tee out/ab-selftest.txt
```

Expect `SELFTEST: max |candidate - baseline| = 0.00e+00 => FAITHFUL` and
`VERDICT: PASS`. Anything else invalidates step 4b, so stop here if it does.

## Step 4b — §3 score `bge-reranker-base` on the fixture (~6 min)

```bash
sudo podman run --rm -i \
  -v ~/gate1/out:/out:z \
  -v ~/gate1/models/bge-reranker-base-onnx:/models/base:ro,z \
  --entrypoint python localhost/docs-bridge:latest - \
  --fixture /out/fixture.json --backend onnx --model-dir /models/base \
  --repeats 3 --json /out/ab-base.json \
  < rerank_ab.py 2>&1 | tee out/ab-base.txt
```

Same fixture, same §2.4 bar (top-3 ordered-identical, top-6 Jaccard ≥ 0.83), with
disagreements adjudicated by the deeper-pool oracle instead of being assumed to be
regressions.

## Step 5 — §2.3 the GPU probe (~30 min, T4)

Host spec and rationale: `gate1-gpu-probe-host.md`. The host is **transient** — it
can be recreated at any moment, so everything it produces comes back into
`notes/gate1/` and nothing is kept on it. It consumes the step 1 fixture and
`tools/eval/rerank_ab.py`, nothing else.

**Host as provisioned (2026-09-28):** `g4dn.xlarge` (Tesla T4 15 GB, 4 vCPU, 15 GiB),
eu-central-1, AMI *Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Ubuntu
24.04)*: driver 595.91, torch 2.12.1+cu130, Python 3.13 in the **`/opt/pytorch`
venv** (the system `python3` has no torch — always use `/opt/pytorch/bin/python`).
Reached as `ssh bnl-docsbridge-dev-host`, which proxies over SSM and needs the
**`remote-session`** AWS profile (the `default` IAM user lacks `ssm:StartSession`);
the ssh config entry sets `AWS_PROFILE=remote-session` in its `ProxyCommand`.

### 5.1 — stage (from the workstation, repo root)

The fixture is **not in git**; it lives on vhost2. `scp -3` relays it through the
workstation. Copy the exact file — a regenerated fixture is a new baseline.

```bash
ssh bnl-docsbridge-dev-host 'mkdir -p ~/gate1/out'
scp -3 vhost2:gate1/out/fixture.json bnl-docsbridge-dev-host:gate1/
scp tools/eval/rerank_ab.py bnl-docsbridge-dev-host:gate1/
```

### 5.2 — set up (on the host)

```bash
ssh bnl-docsbridge-dev-host
tmux new -s gate1          # an SSM drop must not kill a run; reattach: tmux a -t gate1
cd ~/gate1
PY=/opt/pytorch/bin/python
sha256sum fixture.json     # MUST be 2fca6773e343641483b8480e0a14784f773fe4b8e11f6428fa7ed014dc4e3b81
$PY -m pip install -q transformers
{ date -u; nvidia-smi; sha256sum fixture.json
  $PY -c "import sys,torch,transformers,tokenizers; print(sys.version); print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0)); print('transformers', transformers.__version__, 'tokenizers', tokenizers.__version__)"
} 2>&1 | tee out/gpu-env.txt
```

### 5.3 — the runs (on the host, in tmux)

Weights (~2.3 GB) download on first use; the harness loads the model *before* its
warm-up and timing, so the download never lands in a measurement.

```bash
# runs 1-2: fp32 (the precision reference) then fp16 (the candidate)
for p in fp32 fp16; do
  $PY rerank_ab.py --fixture fixture.json --backend torch \
      --model-id BAAI/bge-reranker-v2-m3 --device cuda --precision $p \
      --repeats 10 --json out/gpu-$p.json 2>&1 | tee out/gpu-$p.txt
done

# runs 3-4: fp16 + length-bucketed sub-batches (why: gate1-findings-explained.md, test 5)
for b in 4 8; do
  $PY rerank_ab.py --fixture fixture.json --backend torch \
      --model-id BAAI/bge-reranker-v2-m3 --device cuda --precision fp16 \
      --bucket $b --repeats 10 --json out/gpu-fp16-bucket$b.json 2>&1 \
      | tee out/gpu-fp16-bucket$b.txt
done
```

**No bf16 run on the T4** — Turing has no bf16; it would fall back and mislead. It
belongs to a second-point instance (L4/A10G), if one is taken.

A run exits non-zero on `VERDICT: FAIL`; the loops carry on regardless, which is what
we want — a failing verdict is a result, not an error.

### 5.4 — bring everything back (from the workstation, repo root)

```bash
scp 'bnl-docsbridge-dev-host:gate1/out/*' notes/gate1/
```

Expect nine files: `gpu-env.txt` plus a `.json` and `.txt` per run. Then the instance
holds nothing of value and can be terminated or recycled.

### 5.5 — the fp32 reference run (once, next time a GPU host exists)

Since the 2026-09-28 decision, fp32 is the A/B reference **and** the judge that
adjudicates disagreements, and the judge needs fp32's scores over the 80-chunk deep
pool. `gpu-fp32.json` predates `--score-pool`, so it can set the bar but cannot
adjudicate — which is why fp16 + bucket 4 currently shows an *unadjudicated*
difference (a byte-identical duplicate swap on the email query). One fp32 run with
`--score-pool` fixes that; `--selftest` against the old run proves it is the same
reference, not a new one:

```bash
$PY rerank_ab.py --fixture fixture.json --backend torch \
    --model-id BAAI/bge-reranker-v2-m3 --device cuda --precision fp32 \
    --repeats 3 --score-pool --reference gpu-fp32.json --selftest \
    --json out/ref-fp32.json 2>&1 | tee out/ref-fp32.txt
```

(stage `notes/gate1/gpu-fp32.json` next to `rerank_ab.py` first). Expect
`SELFTEST … FAITHFUL`; then `notes/gate1/ref-fp32.json` is the canonical reference.
Every later A/B — including re-judging any saved run offline — is then:

```bash
python tools/eval/rerank_ab.py --fixture fixture.json --backend replay \
    --replay notes/gate1/<run>.json --reference notes/gate1/ref-fp32.json
```

---

# Results

Measured 2026-09-26 on vhost2 (M2 Pro, Asahi, aarch64, 10 cores, 16 GiB), against
`docker.io/library/docs-bridge:latest` and the live `teamcenter` collection
(**138,550 points**). Raw logs and result JSON are in `gate1/`; `fixture.json`
(1.0 MB) stays on the host at `~/gate1/out/` and is regenerable from step 1.

## Corrected commands

Two image names in the original runbook were wrong and are fixed above: the server
image is tagged **`docker.io/library/docs-bridge:latest`**, not `localhost/...`. It
was built from a bare `docs-bridge:latest`, which podman normalises to the Docker Hub
namespace — unlike `ingest-worker`, which the ingest unit correctly qualifies as
`localhost/`. Worth reconciling: a locally-built image sitting under
`docker.io/library/` is one `podman pull` from being replaced by a registry image.

## §2.2 — the fixture (PASS)

8 queries, 320 production pairs, 80-candidate oracle pool per query.

| | Measured | Plan said |
|---|---|---|
| rerank, `top_n=40` | **14.17 s/query** | ~11 s |
| per pair | **354 ms** | ~270 ms |
| implied `multi_top_n=60` | **~21 s** | ~16 s |
| embed | 133 ms/query | — |

**The plan's headline latency was ~30% optimistic.** Use 14.2 s.

Baseline sanity, all good:
- **Control holds.** `how to bake sourdough bread`: max −5.60, **0/40 pairs positive**,
  against +2.28…+6.14 top-1 for real queries. The rerank sign genuinely separates
  answers from abstention, so the control is a real tripwire.
- **`top_n` cannot be lowered — confirmed independently.** The deepest RRF rank
  feeding a production top-6 is 37, 37, 35, 34, 32 on five of seven real queries.
  `top_n: 30` would drop co-top chunks on most queries.
- **Pair lengths: p50=100, p90=385, p95=420, max=512, only 1/320 at the cap.**
  So §3's `max_length` review is answered: trimming 512 buys ~nothing.

## §2.1 — C = 6 on this host

| threads | 1 | 2 | 3 | 4 | **6** | 8 | 10 |
|---|---|---|---|---|---|---|---|
| s / 40 pairs | 50.91 | 25.86 | 17.58 | 13.43 | **9.33** | 11.05 | 10.88 |
| speedup | 1.00 | 1.97 | 2.90 | 3.79 | **5.46** | 4.61 | 4.68 |
| efficiency | 1.00 | 0.98 | 0.97 | 0.95 | **0.91** | 0.58 | 0.47 |

Both knee readings agree: **C = 6**. Scaling is near-linear to 6, then **reverses** —
8 and 10 threads are *slower* than 6.

That is the M2 Pro's **6 performance + 4 efficiency cores**: ORT partitions a batch
across its intra-op pool and every thread barriers on the slowest, so E-cores make
P-cores wait. C = 6 equalling the P-core count is strong evidence the mechanism is
core heterogeneity, not memory bandwidth.

**Two consequences.**

1. **Today's config is ~14% slower than necessary.** `search.py` never passes
   `threads=`, so ORT takes all 10 cores: 10.88 s where 6 gives 9.33 s on the
   identical batch. Plumbing `rerank.threads` (already in plan §6.1) is a free win
   on vhost2, independent of every other decision.
2. **C = 6 must NOT be used to size EC2.** It is substantially an artifact of
   heterogeneous cores, which no EC2 x86 instance has; there the curve will flatten
   for a different reason (memory bandwidth) and likely at a different N. The harness
   is ready — **re-run `thread_scaling.py` on the chosen instance type and hand THAT
   number to the AWS repo.** This is now a §7 deliverable, not a settled number.

## §3 — the cheap alternatives: both REJECTED

Both were measured against the §2.4 bar (top-3 ordered-identical, top-6 Jaccard
≥ 0.83) with every disagreement adjudicated by the deep-pool oracle rather than
assumed to be a regression.

### Harness validation first (PASS)

`rerank_ab.py`'s stand-in reproduced the baseline column at
**max |delta| = 0.00e+00**, top-3 identical and Jaccard 1.00 on all 8 queries. So
everything below is a real comparison, not a re-implementation artifact.

### `bge-reranker-base` (278M vs 568M): 3.3x faster, FAIL

**108 ms/pair → 4.33 s/query** (6.49 s at `multi_top_n=60`). But top-3 was
ordered-identical on **0 of 7** real queries, Jaccard fell to **0.00** on
`TC_transfer_area site preference`, and the oracle confirmed **regressions on 4 of 7**
(3 ties/improvements).

The `TC_transfer_area` case is the clearest read: the two models' top-6 sets overlap
in *nothing*, each recalling its own deep gold at 1.00 and the other's at 0.00. When
two rerankers disagree that completely, the tie-break is which is the better judge —
and v2-m3 is 2x the parameters and the stronger published model. The control agrees:
base scores the known-negative **−3.86** against v2-m3's **−5.60**, i.e. it separates
answers from noise with a visibly smaller margin. Rejected.

### Length-bucketed sub-batches: 2.0–2.9x faster, FAIL

The §2.2 length distribution (median 100 tokens, p95 420) meets a tokenizer that
pads to batch-longest, so one long pair drags all 40 up to ~500 and attention is
quadratic in that width. Bucketing cuts padded token-positions to 38%:

| bucket | s/query | speedup | padding | max score delta | top-6 moved | oracle |
|---|---|---|---|---|---|---|
| 4 | 4.93 | 2.86x | 38% | 1.41 | 6/8 | **2 regressions** |
| 8 | 5.80 | 2.43x | 44% | 1.03 | 6/8 | **3 regressions** |
| 16 | 6.76 | 2.08x | 51% | 1.01 | 6/8 | not run |
| 40 (today) | 14.10 | 1.00x | 100% | **0.0000** | 0/8 | control |

Bucket 40 reproducing the baseline at delta exactly 0 validates the sort/un-permute
round-trip, and the 2.86x tracks the 2.6x padding cut slightly superlinearly, as
attention's quadratic term predicts. **The mechanism is confirmed; the quality is
not.**

**Why it fails is the useful part.** Padding is mathematically inert — masked
attention cannot see it. What moves is the **dynamic INT8 quantization**:
`quantize_dynamic` computes activation scales per input tensor at runtime, so
changing what shares a batch changes every score in it, here by up to **1.41
logits** on a scale whose real top-1 values are +2.28…+6.14.

(`rerank_ab` measured bucket 4 at 5.32 s vs `rerank_batching`'s 4.93 s because the
wrapper tokenizes twice — once for lengths, once to score. A production
implementation tokenizes once, so **4.93 s is the honest figure** for the lever.)

**This does not close the lever; it relocates it.** Static (calibrated) INT8
quantization fixes activation scales at export time, which would make batch
composition irrelevant and bucketing numerically inert — turning a rejected 2.86x
into a free one. That needs its own A/B against the baseline (static vs dynamic is
itself a numerics change) and the §2.2 fixture is already a ready-made calibration
set. **Recorded as a Gate 2 candidate, not done here.**

## §2.3 — GPU: T4 is 26x faster; the "FAIL" is the reference, not the GPU

Measured 2026-09-28 on `g4dn.xlarge` (Tesla T4, eu-central-1), torch 2.12.1+cu130,
transformers 5.17.0, fixture sha256 `2fca6773…3b81` (the recorded baseline, not a
regeneration). Logs + JSON: `gate1/gpu-*.{txt,json}`, environment `gate1/gpu-env.txt`.

### Speed

| run | s/query `top_n=40` | ms/pair | vs CPU-INT8 | `multi_top_n=60` |
|---|---|---|---|---|
| CPU INT8 (vhost2, baseline) | 14.17 | 354 | 1.0x | ~21 |
| T4 fp32 | 2.80 | 70.0 | 5.2x | 4.20 |
| **T4 fp16** | **0.54** | **13.5** | **26x** | **0.81** |
| T4 fp16 + bucket 4 | 0.33 | 8.2 | 44x | 0.49 |
| **T4 fp16 + bucket 8** | **0.29** | **7.3** | **47x** | **0.44** |

The plan projected ~0.5 s for a T4; measured 0.54 s. fp32 → fp16 is 5.2x, which is
the T4's tensor cores (fp16 only on Turing). Timing is end to end per query:
tokenize, host→device copy, forward, sync.

**Bucketing pays on the GPU too — my pre-run guess that it might not was wrong.**
40 pairs padded to ~500 tokens is ~20k token-positions, which *is* enough to fill a
T4, so cutting padding still converts into speed: 1.86x at bucket 8. Bucket 4 is
slower than 8 because 10 sequential launches start to cost more than the extra
padding saves. (Both include the harness's double tokenization; production would
tokenize once.)

### Quality: the harness says FAIL, and the reason is the reference

All four GPU runs print the **same** `VERDICT: FAIL` — top-3 differs on email and
TC_transfer_area, Jaccard 0.71/0.71/0.50 on three queries, "oracle-confirmed
regression" on three. **fp32 fails identically to fp16.** fp32 is the model *as
published*, with no quantization at all, so a bar it cannot pass is not measuring the
GPU. It is measuring how far production's INT8 sits from the real model.

Pair-by-pair score deltas (from the now-kept `scores`; baseline from the fixture):

| comparison | max \|Δ\| | median \|Δ\| | top-6 ordered-identical | top-6 same set |
|---|---|---|---|---|
| **INT8 (production) vs fp32** — what INT8 costs | **1.031** | 0.181 | **2/8** | 5/8 |
| fp32 vs fp16 — what fp16 costs | 0.0095 | 0.0017 | **8/8** | 8/8 |
| fp16 vs fp16 + bucket 8 | 0.0156 | 0.0000 | **8/8** | 8/8 |
| fp16 vs fp16 + bucket 4 | 0.0117 | 0.0010 | 7/8 * | 8/8 |

\* the one "miss" is chunks #2 and #5 on the email query swapping places at 0.002
apart — and they are **byte-identical texts** (same md5). The LLM receives the same
six texts either way.

So **fp16 is the published model to ~0.01**, and **production INT8 is ~100x further
from it** (up to 1.03 logits). The GPU does not perturb rankings; it *un-perturbs*
them.

**The oracle is circular when the judge itself changes.** It defines "gold" as each
scorer's own top-6 over the deep pool, so each scorer always recalls its own gold at
1.00, and the accept rule (candidate ≥ baseline on *both* golds) reduces to "top-6
set unchanged". Every set change is therefore labelled a regression — see the three
flagged queries: each is an exact mirror (0.83 vs 1.00 / 1.00 vs 0.83). The method
is sound where it came from (`validate_union.py`, fixed judge, changing retrieval);
it cannot adjudicate between two *judges*. **This also weakens the
"oracle-confirmed" wording in §3** — see below.

**Reading the disputed chunks directly** (the only independent check available
without human labels), INT8 vs fp32 is a wash, not a verdict either way:

- **TC_transfer_area** — the top 18 candidates score within 0.45 of each other
  (+3.37…+3.81), so the top-6 cut is close to arbitrary under either numerics. fp32
  keeps `'Site preference.'` (a 16-character heading fragment) at rank 6 and drops
  the actual answer (`TC_transfer_area=/tmp …`, "Valid Values") to rank 17, **0.04
  below the cut**. INT8 happens to keep it. Chance at a near-tie, in INT8's favour.
- **SAP basic unit of measure / SAP BOM origin** — the swaps are at ranks 5-7 with
  scores between −0.6 and +0.0, i.e. chunks the model already says are *not* answers
  (both sides are `ums_import_unit_definitions` examples; both BOM chunks are
  tangential). A tie.
- **BOM data in AIG, email, EPM** — reorders inside an unchanged top-6 set; email's
  are among byte-identical duplicates.
- **Control** — sourdough scores **−6.09** on the GPU vs −5.60 on INT8: the abstention
  margin gets *wider*, not narrower.

### Bucketing vs its pre-set bar (`gate1-findings-explained.md` test 5)

Bar, set before the run: top-6 ordered-identical to unbucketed fp16 on 8/8 **and**
max |Δ| ≤ 0.05.

- **bucket 8: PASS** — 8/8, max 0.0156.
- **bucket 4: formally 7/8, max 0.0117** — the single miss is the byte-identical swap
  above. Substantively inert; recorded as a miss on the letter of the bar.

Bucketing's worst shift is **~90x smaller** in fp16 than under INT8 (0.016 vs 1.41).
That confirms test 4's diagnosis — the damage was INT8's per-batch calibration, not
padding or masking — and so backs the Gate 2 static-quantization idea.

## Decision

**DECIDED 2026-09-28: the GPU path (T4-class, x86_64, fp16, length-bucketed at 8),
and the quality bar's reference is switched from CPU-INT8 to fp32.** The §2.4 bar
was set with the CPU-INT8 ranking as the reference; this run shows that reference is
itself ~1 logit off the unquantized model, so no *more* precise scorer can pass it.
The bar now means "faithful to the model as published", and fp16 on the T4 passes it:
faithful to 0.01 and top-6 ordered-identical 8/8. Changing a pre-set bar after the
results was done openly and on evidence that does not depend on judgement — fp32
itself fails the INT8 bar. Considered and not taken: a one-off waiver for the GPU
while keeping the INT8 bar (leaves the next A/B facing the same broken rule), and
blocking on human labels of the disputed chunks (worth doing later, not needed for
the hardware decision).

- **Speed:** 0.54 s/query plain, **0.29 s bucketed**, against 14.2 s — 26-47x. The
  host note's reading table puts this squarely in "≲ 1 s: the GPU path wins outright".
  Serially one T4 reranks ~6,700 searches/hour (fp16) or ~12,000 (bucket 8), against
  ~500/hour for a 16-vCPU CPU box (14.2 s/search, ~2 concurrent at C≈6 — Gate 1's priced
  CPU fallback). Rerank only — embed (CPU,
  ~0.13 s) and Qdrant are on top, and concurrency on one GPU was not measured.
- **Quality:** no evidence of harm, and a wider abstention margin. The INT8 vs fp32
  disagreements are boundary noise between near-equal chunks.
- **Second instance (L4/A10G): not needed to decide.** The T4 already clears
  sub-second with margin; an L4 would only add a price/performance slope, which
  matters for §7 sizing, not for the CPU-vs-GPU question.
- **Still to validate in §6.3 packaging:** this was torch; production will likely run
  ONNX Runtime's CUDA provider (or torch) in an image — re-run this same fixture on
  that build, judged against **fp32**, not INT8.

**What this changes elsewhere.**

- **§3's "oracle-confirmed regressions" were produced by the same circular oracle**
  and should be read as "the top-6 set changed", not "got worse". `bge-reranker-base`
  stays rejected on its other evidence (top-3 changed on 7/7, a disjoint top-6 on
  TC_transfer_area, control margin −3.86 vs −5.60) — but that is now a judgement, not
  a confirmed regression. INT8 bucketing is moot on the GPU path; on the CPU path it
  should be re-judged against fp32 alongside static quantization (Gate 2).
- **The A/B harness now takes an fp32 reference (done 2026-09-28).**
  `rerank_ab.py --reference <run.json>` judges against a saved run instead of the
  fixture's INT8 column, adjudicates with that reference as a *fixed* judge over the
  deep pool (the circular own-view oracle is gone), and `--backend replay` re-judges
  any saved run offline. Re-judged against `gpu-fp32.json`: fp16 **PASS**, fp16 +
  bucket 8 **PASS**, fp16 + bucket 4 one unadjudicated duplicate swap — pending the
  §5.5 reference run with pool scores.
- **Chunking (Gate 2 input):** a 16-character heading fragment (`'Site preference.'`)
  scores +3.55 and can take a top-6 slot. That is a chunker problem, not a reranker
  one.

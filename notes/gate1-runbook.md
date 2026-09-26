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

## Step 5 — §2.3 the GPU probe (deferred)

Blocked on a GPU host, not on this repo. The probe and the full host spec are in
`gate1-gpu-probe-host.md`; it consumes the step 1 fixture and nothing else.

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

## §2.3 — GPU: not yet run

No host yet; quota granted. Spec and probe in `gate1-gpu-probe-host.md`.

Gate 1's outcome makes this the **only remaining candidate for a large win**, and
notably the only one that does not trade quality away: both rejected levers failed by
perturbing INT8 numerics, whereas fp16 strictly *reduces* numerical error relative to
INT8. The ranking bar should therefore be easier for the GPU to clear, not harder —
and the fp32-first run exists to prove exactly that.

## Decision

**Deferred, pending §2.3 — and that is the correct outcome, not an incomplete one.**
Gate 1's job was to stop anyone sizing a fleet or rebuilding an image on guesses,
and it has:

- The real cost of a search is **14.2 s**, not 11 s.
- **A CPU pool is the fallback, and it is expensive.** At 14.2 s and C≈6 (host-
  specific), a 16-vCPU box sustains ~2 concurrent searches — ~500/hour.
- **Neither cheap lever survives contact with the quality bar.** A 3.3x smaller
  model and a 2.9x batching change both lose answers; §0.1's "fewer/shorter pairs or
  a smaller model" options are now measured and closed.
- **One free win is in hand regardless:** set `rerank.threads` (≈14% on vhost2).
- **One new lever was found and priced:** static quantization, which would unlock
  bucketing's 2.86x for free if it holds up.

So the hardware question now rests entirely on the GPU number, exactly as §2.3
framed it — but with the CPU fallback's price, and the cheap alternatives'
failure, both measured rather than assumed.

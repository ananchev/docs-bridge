# Gate 1 in plain English — what we tested, and what we learned

A companion to `gate1-runbook.md`, which has the commands, tables and raw numbers.
This file explains what those numbers *mean* to someone who has not been living
inside `rerank.py`. Measured 2026-09-26 on vhost2 against the live 138,550-point
`teamcenter` collection.

## The setup

A search has two stages.

**Stage 1 (~0.2 s).** Qdrant finds ~40 plausible chunks by vector similarity —
dense and sparse, fused with RRF.

**Stage 2 (~14 s).** A *cross-encoder* — a 568M-parameter model — reads your
question **together with** each of those 40 chunks and scores how well that chunk
answers it. The best 6 go to the LLM. Unlike stage 1, which compares pre-computed
vectors, this model actually reads both texts at once. That is why it is the quality
lever, and why it is slow.

Stage 2 is ~98% of the wait. One user shrugs at 14 seconds; fifty users through
LibreChat do not. Before buying hardware we needed three things nobody had measured:
how many cores one search really uses, whether something cheaper would do, and
whether a GPU is worth a rebuild.

## Test 1 — "how many cores does one search consume?"

Capacity is just *total cores ÷ cores per search*, and that divisor had never been
measured. So we ran the identical 40-chunk batch with the model restricted to 1, 2,
3, 4, 6, 8, then 10 threads.

It speeds up almost perfectly to 6 cores — 5.5x faster than one core — and then
**gets worse**: 8 and 10 threads are slower than 6.

That is the M2 Pro's **6 performance + 4 efficiency cores**. The model splits a batch
into equal pieces and cannot finish until the slowest piece does, so handing work to
efficiency cores makes the fast cores sit and wait.

Two things follow:

- **A free win.** Today the code passes no thread limit, so it grabs all 10 — which
  is **14% slower** than just using 6. Plumbing `rerank.threads` (already planned in
  §6.1) banks that.
- **A warning.** C = 6 describes *this chip*. No EC2 x86 instance has fast/slow
  cores, so this number cannot size the fleet. The harness must be re-run on the
  chosen instance type, and *that* number goes to the AWS repo.

## Test 2 — "what does a search really cost?"

**14.2 s**, where the plan had ~11 s on record. It was **30% optimistic**, and
`"all"`-subject searches are ~21 s.

Three side results worth having:

- **The candidate list cannot be shortened.** Answers routinely pull in chunk #32-37
  out of 40, so cutting to 30 would start losing real answers — independently
  confirming what the plan already believed on older evidence.
- **The abstention tripwire works.** The deliberate nonsense question ("how to bake
  sourdough bread") scores −5.60 with **zero** positive chunks, while real questions
  score +2.3 to +6.1. The model genuinely knows when it has nothing, which makes it a
  usable safety check for every later experiment.
- **The median question+chunk pair is only 100 tokens**, against a 512 limit almost
  nothing reaches. That looked like a dead end. It became test 4.

### Why the old 270 ms/pair number was not wrong either

The earlier measurement (~270 ms/pair, ~8.3 s at `top_n=30`) was taken on a **batch
of 30**, on a smaller corpus. We now measure ~354 ms/pair on a **batch of 40**. Test
4 explains why those are consistent: the model pads every pair in a batch up to the
**longest pair in that batch**, so a bigger batch is more likely to contain a long
pair, which raises the cost of *every* pair in it.

So "latency = `top_n` × a constant" is slightly wrong — the per-pair cost itself
creeps up with `top_n`, and the growth is a little faster than linear. Worth
re-testing directly (time the same fixture at `top_n` 30 vs 40) before anyone
extrapolates to `multi_top_n=60`.

## Test 3 — "just use a smaller model?"

`bge-reranker-base` is half the size (278M vs 568M). We exported it and scored the
same fixture: **3.3x faster, 4.3 s per search.**

Then quality collapsed. On **all 7** real questions the top-3 changed. On
`TC_transfer_area site preference` the two models' top-6 had **nothing whatsoever in
common**.

When two models disagree that completely, each naturally thinks its own picks are
best — so "they disagree" proves nothing by itself. The tie-break is which model is
the better *judge*, and there the evidence is one-sided: v2-m3 is twice the size and
the stronger published model, and on the nonsense control the small model's margin
shrinks from −5.60 to −3.86 — measurably worse at knowing when to abstain.
**Rejected.**

## Test 4 — "stop paying for padding?"

This came out of test 2's token measurement. The model processes a batch as a
**rectangle**: all 40 pairs get padded to the length of the longest, ~500 tokens,
even though the median is 100. Roughly 60% of the compute is padding — and attention
cost grows with the *square* of that width.

The fix is obvious: sort the 40 by length and process them in small same-length
groups. It worked exactly as predicted — padding fell to 38% of the token-positions,
and it ran **2.86x faster (4.9 s)**.

And it changed the answers, on 6 of 8 questions, with scores moving up to **1.41
points** on a scale where good answers score +2 to +6.

The reason matters, because it is **not the padding**. Padding is invisible to the
model by construction — masked attention cannot see it. The culprit is **INT8
compression**: to run fast on CPU the model stores numbers in 8 bits, and
`quantize_dynamic` calibrates that conversion *from whatever is in the current
batch*. Change who shares a batch, and every score in it shifts.

**Rejected as implemented — but this is a fixable cause, not a dead end.** If the
conversion is calibrated **once at build time** (static quantization) instead of per
batch, regrouping becomes mathematically invisible and the 2.86x becomes free. The
fixture we just built is exactly the calibration data that needs. This is now the
most promising cheap idea on the table, logged for Gate 2.

## Where this leaves the decision

Gate 1's job was to stop anyone sizing a fleet or rebuilding an image on guesses. It
did — but the honest outcome is **"decision deferred to the GPU probe"**, not
"decision made":

- **Both cheap escapes are closed by measurement, not assumption.** A smaller model
  and smarter batching each buy ~3x and each lose answers. The plan's "fewer/shorter
  pairs, or a smaller model" options are now spent.
- **The CPU fallback is priced, and it is expensive:** 14.2 s per search, ~2
  concurrent searches per 16-vCPU box, ~500 searches/hour.
- **One free win is banked regardless:** wire up `rerank.threads` (~14% here).
- **The GPU is the only remaining candidate for a large win** — and the only one that
  does not trade quality away. Both failures came from *perturbing* INT8 numerics,
  whereas fp16 on a GPU *reduces* numerical error versus INT8. The quality bar should
  therefore be **easier** for the GPU to clear, not harder. That is also why the host
  spec runs fp32 first: it separates what the GPU costs from what INT8 was already
  costing.

## Picking this up again

1. **GPU host** — spec in `gate1-gpu-probe-host.md` (`g4dn.xlarge`, PyTorch DLAMI,
   x86_64 not Graviton). `scp` in `fixture.json` + `tools/eval/rerank_ab.py`, run
   three probes, terminate the instance. Then Gate 1 closes.
2. **The fixture lives on vhost2** at `~/gate1/out/fixture.json` (1.0 MB, kept out of
   git). Regenerate with step 1 of `gate1-runbook.md` if it is lost — but note a
   regenerated fixture is a *new baseline*, so do not mix its scores with the results
   already recorded here.
3. **Everything staged on vhost2** is in `~/gate1/` (four harnesses + the exporter).
   The harnesses themselves live in `tools/eval/` and are committed.

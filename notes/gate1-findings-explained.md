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
most promising cheap idea on the table, logged for Gate 2. *(Later decided not
to pursue: the Mac mini keeps today's INT8, which works fine for one user, and the
team's GPU server does not need INT8 at all.)*

## Test 5 — "is a GPU fast enough?" (the T4 probe, and bucketing again)

*Planned and run 2026-09-28 on a `g4dn.xlarge` (T4); commands in `gate1-runbook.md`
step 5, full tables in its §2.3 results. The plan comes first, the results after.*

Same fixture, same bar, same oracle as tests 3 and 4, but the scorer is the
**original v2-m3 weights on an NVIDIA T4** (`g4dn.xlarge`) instead of the INT8 ONNX
export on vhost2's CPU. There are four runs, each asking one question:

| Run | What changes | The question it answers |
|---|---|---|
| **fp32** | GPU, full precision | What does the GPU cost in *answers*? Any ranking difference here comes from INT8 on the CPU side, not from the GPU. |
| **fp16** | GPU, half precision | The actual candidate: how fast is it, and does it pass the bar? |
| **fp16 + bucket 8** | fp16, scored in length-sorted groups of 8 | Does test 4's trick come back for free? |
| **fp16 + bucket 4** | same, groups of 4 | Same question, pushed further. |

### Why bucketing gets a second chance

Test 4 did not reject bucketing because of the padding. It rejected it because of
**INT8's per-batch calibration**: `quantize_dynamic` picks its 8-bit scale from
whatever is in the batch, so regrouping the batch moved every score, by up to 1.41.

fp16 on a GPU has **no such calibration step**. The weights are ordinary half-
precision numbers and nothing is re-scaled at runtime based on who else is in the
batch. Masked attention still cannot see padding. So, in principle, regrouping goes
back to being what it should always have been: **a pure speed change.**

"In principle" because of one small effect. A GPU picks a different matrix-multiply
routine for each matrix shape, and different routines add the same numbers up in a
different order. In fp16 (about 3 significant digits) that shows up as tiny score
jitter, likely thousandths of a point, not INT8's 1.41. It could flip only two chunks
that were already practically tied. The runs measure how big it really is, rather
than assuming it.

### Why it might not be faster on a GPU

On the CPU, compute is the bottleneck, so cutting ~60% of the padded tokens gave
most of that back as speed (2.86x). **A GPU is different.** 40 pairs is a *small*
batch for it, and it probably is not full even with the padding. Splitting the batch
into 10 groups of 4 means 10 back-to-back trips instead of one, and each trip pays a
fixed overhead (launching the work on the GPU, copying the inputs over, Python in
between). So on a GPU, bucket 4 could easily be **slower** than one batch, and
bucket 8 is the middle ground. The harness also tokenizes twice when bucketing (once
to measure lengths, once to score). That was negligible against 14 s, but it is a
visible share of a sub-second search. Either result is useful: a speedup is a free
optimisation for the GPU path, and a slowdown tells us the GPU prefers one big batch.

### How the bucket runs are judged

Two comparisons, not one:

1. **Against the CPU-INT8 baseline**, like every other candidate: the §2.4 bar plus
   the oracle, printed by the harness itself. This says whether bucketed fp16 is
   acceptable *as a production reranker*.
2. **Against unbucketed fp16, pair by pair.** This is the comparison that isolates
   bucketing, because it keeps model, precision and hardware fixed. Comparison 1
   alone would mix in every difference fp16 already has from INT8. This is why
   `rerank_ab.py --json` now keeps each pair's score. It used to keep only the top-6
   IDs, which is enough to judge against the baseline but not to diff two GPU runs.

**Bar for comparison 2, set before the results:** top-6 **ordered-identical** to
unbucketed fp16 on all 8 queries, **and** max |score delta| **≤ 0.05**, about 45x
below the smallest real top-1 score (+2.28) and about 30x below INT8's 1.41. If both hold,
bucketing is numerically inert on fp16 and its speed is the only thing left to read.
Two more outcomes are possible. Rankings could still hold while deltas land above
0.05; that is recorded and judged by the oracle, not waved through. Rankings could
move; that would be the surprise and would need explaining before anything ships.

**A side benefit either way:** if bucketing turns out inert in fp16, that is direct
evidence that test 4's failure really was INT8's calibration and not something about
padding or masking. That strengthens the case for the Gate 2 static-quantization
idea, which exists to remove exactly that calibration from the CPU path.

### What happened — speed

| | per search | vs today |
|---|---|---|
| today, CPU on vhost2 | 14.2 s | — |
| T4, fp32 | 2.8 s | 5x |
| **T4, fp16** | **0.54 s** | **26x** |
| **T4, fp16, groups of 8** | **0.29 s** | **47x** |

The plan's back-of-envelope guess for a T4 was ~0.5 s. It measured 0.54 s.

fp16 is 5x faster than fp32 on the same card because the T4 has dedicated
half-precision hardware ("tensor cores") and nothing comparable for fp32.

**Grouping by length helped, and my prediction above was wrong.** I guessed 40
pairs might be too small a job to fill a GPU. But 40 pairs × ~500 padded tokens is
~20,000 token-positions, which keeps a T4 busy, so removing the padding still paid
off: 1.9x with groups of 8. Groups of 4 were a little slower than groups of 8,
because at 10 back-to-back trips the per-trip overhead starts to show. That part
of the prediction held.

### What happened — quality, and why "FAIL" is the wrong reading

The harness printed **VERDICT: FAIL** for all four runs. It flagged the same four
questions every time, with the same numbers. Taken at face value, the GPU "loses
answers" the way the smaller model did. It does not, and fp32 is the reason we ran it
first.

Think of three versions of the same model:

- **fp32**: the model exactly as its authors published it, full precision.
- **fp16**: the same model with half-precision numbers (the GPU candidate).
- **INT8**: a compressed copy made so it runs fast on a CPU (today's production).

The harness grades every candidate against **INT8's answers**. **fp32 failed exactly
the same way fp16 did**, and fp32 *is* the original. So the test was not measuring
the GPU. It measured how far the compressed copy has drifted from the original, and
**no more-accurate model could ever pass it.** It is like calibrating a new ruler
against an old one that got bent: the straighter the new ruler, the worse it
"fails".

The pair-by-pair numbers make that concrete. Scores are on a scale where good
answers score +2 to +6:

- **fp16 vs the original: never more than 0.01 apart**, and the same top 6 in the
  same order on all 8 questions.
- **INT8 vs the original: up to 1.03 apart**, and the top 6 in the same order on only
  2 of 8.

So the GPU does not disturb today's rankings. **It removes a disturbance INT8 already
puts there.**

### Then which answers are better, INT8's or the original's?

The harness has a referee for this, the "deep-pool oracle", and here it breaks down.
It works by asking each model to rank a bigger pool of 80 chunks and treating that
ranking as the answer key. That is fine when the *judge* stays the same and only the
retrieval changes, which is what it was built for. Here the judge itself changes.
**Each model gets graded against its own answer key**, so each scores 100% on its
own and loses on the other's. Any change becomes a "regression". You can see it in
the log: every flagged question is an exact mirror, 0.83 vs 1.00 one way and 1.00 vs
0.83 the other.

So I read the disputed chunks myself:

- **TC_transfer_area**: the top 18 chunks all score within 0.45 of each other, so
  "which 6" is close to a coin toss under either model. The original model happens to
  keep a 16-character heading (`Site preference.`) and push the actual answer
  (`TC_transfer_area=/tmp`, its valid values) to 17th place, **0.04 below the
  cut**. INT8 happens to keep the answer. That is luck at a near-tie, this time in
  INT8's favour.
- **The two SAP questions**: the swaps are between chunks the model already scores
  near or below zero, i.e. "not really an answer" on both sides. A tie.
- **The other three**: the same six chunks, reordered. On the email question some of
  them are literally the same text stored twice.
- **The nonsense control** scores **−6.09** on the GPU against −5.60 today: it is
  *better* at saying "I have nothing".

**Net: no evidence the GPU loses answers, and a slightly clearer "no".**

### What happened — the grouping test (comparison 2)

Against the bar written down before the run (same top 6, same order, on all 8, and
no score moving more than 0.05):

- **Groups of 8: pass.** 8/8, largest shift 0.016.
- **Groups of 4: 7/8 by the letter of the bar, largest shift 0.012.** The one "miss"
  is two chunks with byte-identical text swapping places, 0.002 apart, so the LLM
  gets the same text either way. Inert in substance, but recorded as a miss.

Under INT8 the same regrouping moved scores by up to 1.41. In fp16 it moves them by
0.016, **about 90x less**. That confirms test 4's diagnosis: the damage came from
INT8's per-batch calibration, not from padding. It is also good news for the Gate 2
idea of calibrating INT8 once at build time.

## Where this leaves the decision

**Decided (2026-09-28): go GPU.** A T4-class NVIDIA card, x86_64, fp16, with length
grouping at 8. That cuts the rerank from 14.2 s to ~0.3 s, and it is the only lever
Gate 1 found that does not trade answers for speed.

**And the quality bar changed, deliberately, after the results.** It used to compare
everything to today's INT8 rankings. Test 5 showed those rankings are themselves ~1
point off the real model, so under that rule nothing more accurate than INT8 could
ever pass. It was like grading against an answer key with typos in it. **From now on,
candidates are judged against the real model (fp32).** fp16 passes that cleanly.
Changing a rule after seeing results is normally a red flag. It is fair here because
the reason does not depend on anyone's judgement: the original model itself fails
the old rule. The alternatives were a one-off exception for the GPU (the next
comparison would hit the same broken rule) or having an expert grade the disputed
chunks (worth doing some day, not needed to pick hardware).

What else Gate 1 leaves behind:

- **The smaller model (test 3) stays rejected, on softer ground.** Its "confirmed
  regressions" came from the same self-graded referee, so they only prove the top 6
  *changed*. It still changed the top 3 on every question, disagreed completely on
  one, and got worse at saying "I have nothing". That makes the rejection a judgement,
  not a proof. With the GPU at 0.3 s, it no longer matters.
- **The CPU fallback is priced:** 14.2 s per search, ~500 searches/hour per 16-vCPU
  box, against ~6,700-12,000/hour of rerank for one T4 (not counting the other stages
  or several users hitting one card at once).
- **Free win, still valid for the CPU box:** wire up `rerank.threads` (~14%).
- **A second GPU type (L4) is not needed to decide.** The T4 already clears the bar
  with room to spare. An L4 would only tell us whether paying more buys meaningfully
  more, which is a sizing question for later.
- **To fix before Gate 2 relies on it:** teach `rerank_ab.py` to use a saved fp32 run
  as its reference and as a fixed referee. Plan §5.2's A/B has the same problem.
- **Spotted in passing (chunking):** a 16-character heading, `Site preference.`,
  scores like a real answer and can steal a top-6 slot. That is the chunker's
  problem, not the reranker's.

## Picking this up again

1. **GPU probe: done.** Results are in `notes/gate1/gpu-*` and `gate1-runbook.md`
   §2.3. To re-run, recreate the host from `gate1-gpu-probe-host.md` and follow
   `gate1-runbook.md` step 5 exactly. Nothing needs to survive on the instance.
2. **The fixture lives on vhost2** at `~/gate1/out/fixture.json` (1.0 MB, kept out of
   git). Regenerate with step 1 of `gate1-runbook.md` if it is lost — but note a
   regenerated fixture is a *new baseline*, so do not mix its scores with the results
   already recorded here.
3. **Everything staged on vhost2** is in `~/gate1/` (four harnesses + the exporter).
   The harnesses themselves live in `tools/eval/` and are committed.

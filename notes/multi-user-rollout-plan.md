# Multi-user rollout: three gates, a toolchain upgrade, and a GPU reranker

**Goal.** Take docs-bridge from a single-user LAN appliance to a shared retrieval
service for a work team (<50 users, LibreChat front end, EC2 host, **new corpus and
new database**) — without guessing at any of the three decisions that drive the
design: how many cores a query needs, whether a GPU is worth it, and whether the
upgraded toolchain still retrieves as well.

**Scope of this document.** Product-side only: the `docs_bridge` package, the two
images, the config contract, and the measurement gates. Everything here is
reproducible on any host with a container runtime.

**Non-scope — owned by the AWS/operator repo.** Terraform, AMI selection, GPU service
quotas, security groups, load balancer and health-check configuration, ECR, instance
sizing, secrets delivery, backup targets, the ingest schedule. This plan **emits the
numbers and config keys** that repo needs (§7) and consumes nothing from it. The
deployment boundary stays exactly as `docs-bridge-deployment-boundary` defines it: a
container image plus a mounted `config.yaml`.

**One external schedule dependency worth starting immediately:** a fresh AWS account
has a **zero** vCPU quota for On-Demand G instances. Gate 1 cannot run until that
increase is granted, and it can take days. Request it before anything else in this
plan; the request itself lives in the other repo.

Measurements marked **(measured)** come from vhost2 (M2 Pro, Asahi, 10 cores) and are
reproduced in `docs-bridge-build-validation-state` / `docs-bridge-mcp-server`.
Everything marked **(projected)** must be replaced by a Gate 1 or Gate 2 number before
it is used for sizing.

---

## 0. Why this plan exists

Three findings, all verified in this repo's code or from the pinned dependency source,
define the work:

**0.1 Query latency is FLOP-bound, not tuning-bound.** `rerank.py:score()` batches all
candidates into one `session.run`, and latency is dead linear at **~354 ms per
(query, passage) pair (measured, Gate 1 §2.2 — supersedes the earlier ~270 ms)** —
the batch buys nothing because a single pair's matmuls already saturate every core.
At the deployed `rerank.top_n: 40` that is **~14.2 s per query (measured)**, ~21 s
for a multi-subject or `"all"` search at `multi_top_n: 60`. `top_n` is closed as a lever: `recall_topn.py` and
`validate_union.py` showed 30 drops co-top chunks and 16 changes the answer on 16/20
queries. So the only levers are **fewer/shorter pairs, a smaller model, or different
silicon.**

**0.2 The MCP path serialises the whole process.** `server.py:59-77` registers
`list_subjects` and `search` as **plain sync functions**. In the pinned
`fastmcp==2.14.7`, `FunctionTool.run` executes them inline on the asyncio event loop
(`tools/tool.py:377-383`: `result = type_adapter.validate_python(arguments)`, awaited
only if awaitable; `tools/tool_manager.py:160` is a bare `await tool.run(...)`). There
is no thread offload anywhere in that chain. `server.py:156` is a single
`uvicorn.run(app)`. So for the ~11 s a rerank runs, that process serves **nothing
else** — including `/healthz`. The REST mirror is unaffected (`rest.py` wraps every
call in `run_in_threadpool`), which is why a REST smoke test cannot detect this.

Consequence on any managed load balancer: health checks landing inside a search mark
the target unhealthy and deregister it, so under load the pool sheds exactly the
instances doing work. **This is the one item that is a hard prerequisite for a
multi-user deployment**, independent of hardware.

**0.3 Capacity is total cores ÷ cores-per-query.** Slicing one box into N containers
does not add throughput (the FLOPs are fixed); adding hardware does, close to
linearly. The missing quantity is **C = the number of cores a single search usefully
consumes.** `OnnxReranker.__init__` already accepts `threads=` (`rerank.py:33`) but
`search.py:59-66` never passes it, so every session takes every core. **Gate 1
measured C = 6 on vhost2 — and found 8 and 10 threads SLOWER than 6**, because the
M2 Pro has 6 performance + 4 efficiency cores and ORT barriers on its slowest
thread. So C = 6 is a property of this silicon, not of the model: it must be
re-measured on the x86 target before it sizes anything (§7). The free consequence
that does transfer: not passing `threads=` costs ~14% here.

---

## 1. The three gates

| Gate | Question it answers | Blocking on | Cost |
|---|---|---|---|
| **1. Hardware** | How many cores does a query need, and is a GPU worth a rebuild? | Nothing — runs against today's image | ~half a day |
| **2. Retrieval** | Does the upgraded toolchain + fp16 reranker retrieve as well? | Gate 1 decision + the upgrade landing | ~1 day |
| **3. Concurrency** | Does one instance stay alive and fair under N concurrent users? | The §5 code changes | ~half a day |

The **toolchain upgrade sits between Gate 1 and Gate 2** (§4). Gate 1 deliberately
runs on the *current* pins so a hardware decision is never entangled with dependency
churn; Gate 2 then validates the upgraded stack once, against the hardware Gate 1
chose.

**Hard sequencing lock: no bulk ingest before the upgrade lands.** A new docling
brings a new `HybridChunker`, so chunk boundaries move. With a new database that costs
nothing — but only if the corpus is parsed **once**, by the final toolchain. Any test
ingest done before §4 must be thrown away.

### 1.1 Two deployment targets, two different corpora (decided 2026-09-28)

Both are required in the end; neither is the other's fallback.

| | Mac mini (vhost2) | AWS GPU (EC2) |
|---|---|---|
| Corpus | existing `teamcenter` (138k chunks, **old toolchain**) — stays | new work-archive corpus (SKF/HxGN, **new toolchain**); first a Teamcenter 14.2 test collection (§8 3b) |
| Ingested on | vhost2 | **the EC2 instance itself** (decided 2026-10-27) |
| Embedder | CPU, dynamic INT8 | **GPU, fp16 or fp32, as its own service** (decided 2026-10-27) |
| Reranker | CPU, dynamic INT8, `threads` tuned — **unchanged** | CUDA, fp16, length-bucketed at 8 |
| Quality check | **old vs new image on the same `teamcenter`** (chunk-ID comparable), fp32 judge | absolute: labelled questions per corpus + fp16-vs-fp32 numerics; for the 14.2 test collection, the Mac mini's 14.2 as baseline |

*The rows above were the 2026-09-28 starting point; the Mac mini rebuilt from scratch in
October (§8 step 3) and the EC2 column is as decided on 2026-10-27. The text below is
kept as the reasoning of the time.*

One codebase: the targets differ only in reranker runtime and config
(`rerank.device` / `rerank.precision`, §6.3), never in the embedder. Consequences:

- **The sequencing lock above binds the EC2 corpus**, not `teamcenter`.
- **§4.1's "the pins dissolve" holds only for EC2.** The Mac mini's server queries
  137k vectors embedded by the *old* runtime, so upgrading its image makes
  `stack-upgrade-strategy.md`'s embed-parity gate against `teamcenter` blocking again.
- **§5.1's cross-architecture check is blocking for EC2**: its corpus is embedded on
  aarch64 (vhost2) and queried on x86.

**Update 2026-09-29 (user): the Mac mini's ingest and database are never reused on
EC2 — EC2 gets its own ingest and its own collections.** Nothing embedded for
`teamcenter` is ever served from EC2, and no EC2 decision is constrained by the Mac
mini's stored vectors. ~~Open: *where* EC2's corpus is embedded.~~ **Decided
2026-10-27: on the EC2 instance**, so the cross-architecture bullet above no longer
applies.

Also from Phase A (2026-09-29): an onnxruntime upgrade alone moved the INT8 embedder's
vectors to cosine ~0.98 with identical model bytes and tokens, so "never in the
embedder" above is not something a shared codebase guarantees — only a frozen runtime
does. Re-embedding on every upgrade does not scale; the long-run policy (freeze the
embedder as model + runtime behind its own service, fingerprint each collection, gate
at the retrieval level, stable precision for new corpora) is
`stack-upgrade-strategy.md` §7. Its consequences for EC2 are noted in §5.1 and §6.3.

---

## 2. Gate 1 — measure before building

Runs entirely against the **existing** `docs-bridge:latest` image and the existing
populated `teamcenter` collection on vhost2. No repo changes, no new images.

### 2.1 The thread-scaling curve (yields C)

A ~15-line script constructing `OnnxReranker(model_dir, threads=N)` directly for
N = 1,2,4,6,8,10 and scoring a fixed set of 40 real pairs. Run it the way every other
harness in `tools/eval/` runs:

```bash
sudo docker exec -i docs-bridge python - < tools/eval/thread_scaling.py
```

Output: seconds per query vs threads. **C is where the curve flattens.** The vhost2
note "more threads do nothing" tells us it is flat *at* 10; it does not tell us
whether it flattened at 4 or at 10, and those imply very different fleets:

| If C ≈ | A 16-vCPU box serves | Fleet shape |
|---|---|---|
| 4 | 4 concurrent searches at ~11 s | fewer, larger instances; 4 permits each |
| 8 | 2 concurrent searches at ~11 s | 8-vCPU instances, 1–2 permits each |
| 16 | 1 search at a time | scale by instance count |

### 2.2 The fixture dump (the only test data the GPU probe needs)

The reranker is a pure function of (query, passage) — it needs no Qdrant, no corpus,
no manifest, no server. So the fixture is one JSON file, dumped from the live stack:
for each of the 7 user-validated queries (the `recall_topn.py` set: T4X / BGS / SAP,
the two exact-identifier needles, plus the sourdough known-negative control), write
the query, its 40 RRF candidate passages, and **the current CPU-INT8 scores**. That
last column is the baseline every later gate compares against.

This fixture is the most reusable artifact in the plan: Gate 2's reranker A/B consumes
the same file.

### 2.3 The GPU probe — torch, not onnxruntime

On a GPU box (provisioned by the other repo; a Deep Learning Base AMI carries the
driver and container toolkit):

```python
m = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3") \
        .half().cuda().eval()
```

Score the §2.2 fixture, time it, compare ordering to the baseline column.

**Deliberately not onnxruntime-gpu at this stage.** This gate tests physics, not
packaging; ORT fp16 lands in the same ballpark, and paying the CUDA/ORT packaging cost
to learn whether a GPU is 20× faster is backwards. Zero repo changes; if the number
disappoints, the whole GPU track dies for the price of an afternoon and a few dollars
of instance time.

**Projected payoff:** ~10^13 FLOPs per query at fp16 puts a T4 near ~0.5 s and an L4
near ~0.3 s **(projected)** — i.e. sub-second queries instead of 11 s, and the §0.1
queueing arithmetic stops being a design input at all.

### 2.4 Exit criteria

**Status (2026-09-28): all of Gate 1 measured. Results, commands and raw logs in
`gate1-runbook.md` + `gate1/`.** C = 6 (host-specific), baseline 14.2 s/query, both
cheap alternatives in §3 rejected, and **§2.3: a T4 in fp16 reranks in 0.54 s (26x),
0.29 s length-bucketed at 8 (47x)**, faithful to the unquantized model to 0.01 and
8/8 top-6. **DECIDED 2026-09-28: GPU**, and the bar below **switched its reference
from CPU-INT8 to fp32**: the INT8 reference is itself up to 1.03 logits from fp32 (top-6 ordered-
identical on only 2/8), so fp32 fails it exactly as fp16 does; the reference must
become **fp32**, and the oracle a *fixed* fp32 judge — as specified it grades each
scorer against its own deep top-6, which labels every set change a regression.

- **C measured**, with the seconds-vs-threads curve recorded in this file.
- **GPU latency measured** on real pairs, and ranking compared to the **fp32
  reference** (originally the CPU-INT8 baseline — superseded, see status) using the
  established bar: top-3 ordered-identical, top-6 Jaccard ≥ 0.83.
  fp16 is strictly more precise than INT8, so a *difference* is acceptable where it is
  an improvement — adjudicate with the reranker-as-oracle method from
  `validate_union.py` (score a deeper pool to define the global-best top-6, then ask
  which ranking contains them) rather than treating the INT8 order as ground truth.
- **A decision, written down:** CPU pool, or GPU. Everything after this branches on it.

---

## 3. Gate 1 also prices the cheap alternative

Before committing to either hardware path, Gate 1's fixture makes one more experiment
nearly free, and it may delete the problem: **a smaller reranker.**
`bge-reranker-base` is ~278M against v2-m3's ~568M — roughly 2× faster on the same
CPU, no new hardware, no architecture change. Combined with a `max_length` review
(chunks target 400 tokens, so 512 is *mostly* justified — measure how many pairs
actually reach the cap before trimming), 11 s could become 4–5 s. At 4 s a single
8-vCPU box sustains ~900 searches/hour and the pool question evaporates.

Same fixture, same bar. Run it alongside §2.3 — the two are independent and the
cheaper one winning is the best outcome available.

**Outcome (measured, `gate1-runbook.md`): both cheap levers lost.**
`bge-reranker-base` is **3.3x faster (4.33 s/query) but fails** — top-3 identical on
0 of 7 queries, Jaccard 0.00 on one, "oracle-confirmed" regressions on 4 of 7 (the
oracle turned out circular when the judge changes — §2.4 status; read as "top-6 set
changed"), and a
control margin that shrinks from −5.60 to −3.86. The `max_length` review came back
**negative as a lever** (p50 = 100 tokens, only 1 pair in 320 reaches the 512 cap) —
but it exposed a different one: the tokenizer pads to batch-longest, so 40 mixed-
length pairs all compute at ~500 tokens. **Length-bucketing cuts padding to 38% and
is 2.86x faster, and it also fails the bar** — not because padding matters (masked
attention cannot see it) but because `quantize_dynamic` recomputes activation scales
per batch, moving scores up to 1.41 logits. That points at a real follow-up:
**static/calibrated INT8 would make bucketing numerically inert and its 2.86x free**
(the §2.2 fixture is a ready calibration set). **Not pursued (decided 2026-09-28):** the Mac mini stays on today's dynamic INT8 without bucketing — it works fine for its single user, and the GPU target does not need INT8 at all.

---

## 4. The toolchain upgrade (between Gate 1 and Gate 2)

`notes/stack-upgrade-strategy.md` holds the full analysis; it is **not** duplicated
here. This section records only what changes now that (a) the database is new and
(b) an x86/GPU target may be in play.

### 4.1 Why the upgrade is much cheaper than that document assumes

That document's blocking gate is **vector parity against 137,013 stored `teamcenter`
vectors** — the reason `tokenizers==0.19.1` and `onnxruntime==1.20.1` are pinned at
all (`pyproject.toml` says so explicitly: *"MUST match what ingest tokenized with, or
query vectors drift from doc vectors"*). **With a new database there are no stored
vectors to preserve, so that constraint dissolves.** The `onnxruntime>=1.24.1` demanded
by `docling-slim`'s ONNX extras stops being a conflict and becomes just an upgrade.

This is the single strongest argument for doing the upgrade **now**, before the corpus
exists, rather than ever again.

### 4.2 The invariant that replaces the old pins

Delete the *values* of those pins, keep the *property* they protected:

> **The embedder runtime and tokenizer must be identical in the ingest-worker and the
> docs-bridge server images, and the corpus must be ingested only after the final
> version bump.**

Both images already install the same `pyproject.toml` dependency set, so this is
structurally satisfied — it just has to be stated, and enforced by ordering (upgrade,
then ingest, never the reverse). Record the resolved versions in this file when the
branch lands, so a future bump has a known-good baseline to diff against.

**Resolved 2026-09-28 (Phase A, `notes/phase-a-runbook.md`) — the known-good baseline:**

| | before (2025-01 set) | after | where |
|---|---|---|---|
| docling / -core / -parse / -ibm-models | 2.15.1 / 2.14.0 / 3.0.0 / 3.1.2 | **2.130.0 / 2.99.0 / 7.22.1 / 4.0.3** | worker |
| onnxruntime | 1.20.1 | **1.30.0** | both — embedder runtime |
| tokenizers | 0.19.1 | **0.23.2** | both — embedder runtime |
| numpy | <2 (1.26) | **2.4.6** (2.5 needs py3.12) | both — embedder runtime |
| transformers | 4.44.2 | **5.17.0** | worker (the chunker's token counter) |
| torch | 2.5.1 | **2.14.0+cpu** (PyTorch CPU index — the PyPI aarch64 wheel pulls ~15 CUDA packages) | worker |
| FlagEmbedding | 1.3.3 | **removed** | — |
| fastmcp / uvicorn / qdrant-client / PyYAML | 2.14.7 / 0.49.0 / 1.12.1 / 6.0.2 | unchanged (fastmcp: §9) | — |
| python | 3.11 | 3.11 | both |

The invariant is now enforced three ways rather than stated: full transitive **locks
per image** (`images/*/requirements*.lock`, regenerated by `images/lock.sh`, which
fails if the embedder runtime differs between the worker and either server lock); the
**models are a frozen artifact** (`images/models/`), no longer re-exported by every
build from whatever torch/onnxruntime it installed — which would have silently
changed the model under stored vectors on exactly this bump; and the runtime change
itself is **gated** against the stored `teamcenter` vectors
(`tools/eval/embed_upgrade_parity.py`, runbook step 4).

### 4.3 What moves

- `docling`, `docling-core[chunking]`, `docling-parse`, `docling-ibm-models` →
  current (that document records 2.130.0, Aug 2026; re-check at branch time).
- `onnxruntime` → current, or `onnxruntime-gpu` for the GPU path (§6.3).
- `tokenizers` / `transformers` → whatever the new docling set resolves to.
- `numpy<2` → likely liftable; the `<2` cap exists for the old docling/FlagEmbedding
  set. Resolve deliberately and comment it, in the style the file already uses.
- `torch` → still arrives via docling. **`FlagEmbedding` fallback backend: DROPPED
  (decided 2026-09-28)** — ONNX is the default and has been since the INT8 adoption.
  `docling-slim[standard]` still pulls torch, so the worker image does not shrink from
  that alone.

### 4.4 What to turn on while the file is open

Tier 1 and Tier 2 from `stack-upgrade-strategy.md` §4 — all of which matter far more
for an Office-heavy team corpus than they did for generated Doxygen/Javadoc Markdown:

- **PPTX speaker notes** (not extracted today; for decks the narrative lives there).
- **Register format options for DOCX/PPTX.** `parse.py:90` passes `format_options` for
  `InputFormat.PDF` only, so every enrichment flag silently no-ops on Office formats.
- **Authored alt-text.** DOCX/PPTX images carry a human-written `descr`; zero inference
  cost, better than any generated caption, ignored by docling in every version.
  Today those images are parsed and then dropped at `parse.py:108`.
- **RapidOCR (ONNX)** in place of torch EasyOCR, and **fix the wrong comment at
  `parse.py:83`** — OCR does *not* run on every page by default; with
  `force_full_page_ocr=False` it runs only on detected bitmap regions.

Tier 3 (picture description) stays deferred until a pilot library produces real figure
counts, per that document.

### 4.5 What the upgrade invalidates

- **The chunk-yield and parse cost model.** `work-archive-ingest-plan.md` §4 projects
  ~58,000 chunks at 1.23 s/page and 0.342 s/chunk **(measured on the old toolchain)**.
  A new chunker and new enrichment change both. Re-measure on the pilot library
  (`SKF/Portfolio - PLM Portfolio Documents`, 215 files) and replace those numbers
  before any bulk run.
- **Any pre-upgrade test ingest.** See the sequencing lock in §1.

---

## 5. Gate 2 — retrieval parity and quality

Runs after §4 lands, on the hardware Gate 1 chose.

### 5.1 Embed determinism (blocking)

`tools/eval/validate_embed_parity.py`, sampling real chunk texts. The question is no
longer "old runtime vs new" — it is **"does the worker's embedder produce the same
vectors as the server's"** under the upgraded stack, and **"is the embedder
deterministic across processes and hosts"** (relevant if the worker runs on Graviton
and the server on x86). Existing bar: **dense cosine ≥ 0.999, sparse Jaccard ≥ 0.95.**
Cheap, and a failure here is the only thing in this plan that can invalidate the
corpus after it is built — so it runs before the bulk ingest, not after.

**The embedder stays CPU/INT8 even on the GPU path.** It is ~20 ms of an 11 s query,
so a GPU buys it nothing, and it is the one component whose numerics are baked into
stored data. Keeping it on CPU means the GPU change has **zero** vector-parity
surface.

> **Reopened 2026-09-29.** The argument above is about query latency; Phase A showed
> the bigger cost is *upgrades*: dynamic INT8 moves every vector when onnxruntime
> changes (cos ~0.98), and its batch effect leaves ~0.98 of noise in the stored vectors,
> so no vector-level bar can be tight. EC2's corpus does not exist yet and is served
> from a GPU, so an fp16 (or fp32) embedder is cheap there — at ingest, where a GPU
> turns the CPU's ~0.34 s/chunk into a fraction of it, and for any later re-embed.
> Expected to be far less runtime-sensitive than INT8; to be measured, not assumed.
> `stack-upgrade-strategy.md` §7 (items 1, 2, 4). And the 0.999 bar here should be
> read as the diagnostic, with retrieval on labelled questions as the gate (§7 item 3).
> **Decided 2026-10-27: EC2's embedder runs fp16 or fp32 on the GPU, as its own
> service** (§8 3b); which precision is measured on the 14.2 test collection.

### 5.2 Reranker A/B (blocking for the GPU path)

~~Modify `tools/eval/recall_topn.py`~~ — **dropped 2026-09-28**: reranker numerics are
covered by `rerank_ab.py` (fixture + fp32 judge, offline replay). End-to-end old vs
new on the Mac mini goes through a new `tools/eval/retrieval_ab.py` instead: the same
labelled question set run through the real `Searcher` in the old and the new image
against the same `teamcenter`, compared by chunk ID and judged against fp32 (§1.1).
Compare, on the §2.2 fixture:

1. CPU INT8, old toolchain (the recorded baseline)
2. CPU INT8, new toolchain
3. GPU fp16, new toolchain (GPU path only)

Bar: top-3 ordered-identical and top-6 Jaccard ≥ 0.83 against the **fp32 reference**
(`notes/gate1/ref-fp32.json` once `gate1-runbook.md` step 5.5 has run; until then
`gpu-fp32.json`, which sets the bar but cannot adjudicate — decided 2026-09-28, §2.4
status; `rerank_ab.py --reference`), with disagreements
adjudicated by a **fixed fp32 judge** over the deeper pool, not by each scorer's own
deep top-6 (circular when the judge changes). Rows 1-2 (CPU INT8) are therefore
measured *against* fp32 too; today INT8 sits up to 1.03 logits from it, which is the
number any new INT8 or static-quantized export has to not make worse.

### 5.3 First-stage sanity (advisory)

The RRF/dense/sparse balance was settled by measurement (`diag_channels.py`,
`validate_union.py`: keep RRF, keep sparse, no union, no DBSF, no Elasticsearch) —
**on the old chunker.** New chunk boundaries could move that balance. Re-run
`diag_channels.py` once on the new corpus and confirm `top_n: 40` still covers the
gold set. Advisory: act only if worstHit exceeds 40.

### 5.4 Cost-model refresh (output, not a gate)

Record, from the pilot library: parse s/page, chunks/page, embed s/chunk, and the
resulting bulk-ingest projection. These numbers go straight into the other repo's
capacity planning.

---

## 6. Gate 3 — concurrency and liveness

### 6.1 The code changes (prerequisites, not optional)

| Change | File | Effort |
|---|---|---|
| Make both MCP tools `async` and offload via `anyio.to_thread.run_sync` — mirroring what `rest.py` already does | `server.py:59-77` | ~5 lines |
| `stateless_http=True` so any pool member can serve any request | `server.py:123` | 1 line |
| A semaphore (permits = vCPU ÷ C from Gate 1) around the rerank, so the threadpool cannot oversubscribe the cores | `search.py` | ~10 lines |
| Plumb `threads=` into `OnnxReranker` / `OnnxEmbedder` from config, so cores can be budgeted per session | `search.py:59-66`, `config.py` | ~10 lines |
| Startup warm-up: one throwaway search, so the first user does not pay the ~9.4 s model load **(measured)** and the unguarded lazy-load properties never race | `server.py` | ~5 lines |
| Optional: bound tool duration with `anyio.fail_after`, so a queued search fails with a clean MCP error instead of hanging past LibreChat's own timeout | `server.py` | ~5 lines |

On `stateless_http`: verified present in `fastmcp==2.14.7`
(`server/server.py:2586-2632`). docs-bridge's tools are pure functions with no
server-side state between calls, so statelessness is safe here — and without it, MCP
sessions pin to one process and a round-robin pool breaks.

### 6.2 The test

1. **Liveness under load.** Poll `/healthz` every 500 ms while driving N concurrent
   MCP `tools/call` requests for `search`. **Bar: `/healthz` never exceeds ~1 s and
   never fails, at any N.** Before the §6.1 change this test fails at N=1 — which is
   also the cleanest way to prove the fix, so run it once *before* the change to
   record the failure.
2. **Fairness.** `list_subjects` must return promptly while searches are queued.
3. **Degradation shape.** Search latency should grow roughly linearly with queue depth,
   not pathologically. If it grows faster, the semaphore permits are too high for C.
4. **Pool statelessness.** Two containers behind a round-robin proxy, identical
   config: every MCP request must succeed regardless of which member serves it, and
   restarting one mid-conversation must not wedge the LibreChat agent.

### 6.3 GPU packaging (GPU path only, after Gate 1 says yes)

| Item | Effort | Note |
|---|---|---|
| fp16 export mode | ~15 lines | Mirror `--quantize-only` in `export_bge_reranker_onnx.py` using `onnxconverter_common.float16`. 568M × 2B ≈ 1.1 GB — under the 2 GB protobuf limit, so it stays a single self-contained file and avoids the external-data blob that once bloated the image to 4.9 GB. |
| Provider + config keys | trivial | `rerank.device` / `rerank.precision`, both **required** (no default, §8 3b). `cuda` requests `CUDAExecutionProvider` only, and startup fails unless the session really got it — no CPU fallback list, which is exactly what hides a CUDA/cuDNN mismatch. `cpu` keeps a CPU-only run possible, chosen explicitly. |
| CUDA base image | **~1 day — the real work** | Prefer keeping `python:3.11-slim-bookworm` and pulling CUDA in as pip wheels (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`) with `LD_LIBRARY_PATH` set, rather than switching to `nvidia/cuda:*-ubuntu22.04` (which ships Python 3.10, not 3.11). The wheel route keeps every pin and the `deps`→`exporter`→`runtime` cache structure intact. ORT's CUDA major/cuDNN major must match or you get a silent CPU fallback rather than an error. |
| x86_64 switch | ~half a day | Mechanical: the aarch64-only comments in both Dockerfiles, pins re-resolving on x86 (wheels exist). Two side benefits: the from-source `images/qdrant` build becomes unnecessary (4 K pages), and the M2 stops being the only builder — build in CI/on the instance and push to a registry, finally closing the "build once, distribute" item open since the Pi. |

**Embedder as its own service (proposed 2026-09-29, `stack-upgrade-strategy.md` §7;
ADOPTED for EC2 2026-10-27, so it can later move to another instance).**
EC2 gets a third image: the frozen embedding model plus its own pinned
onnxruntime (CUDA provider if the embedder goes fp16 on GPU), serving both the ingest
worker and the query server. The server and worker images then carry no embedder
runtime at all and upgrade independently of the vectors; the collection records the
embedder's fingerprint (model sha256 + runtime + precision), checked at startup.
Cheapest to decide before EC2's first collection exists.

**Architecture note (verified):** `onnxruntime-gpu` has **no aarch64 wheels below
1.29.0** — the 1.20.x line is x86_64/Windows only. So a Graviton+GPU (g5g) target
would force ORT nine minor versions forward on its own. **Go x86_64 for the GPU path.**

### 6.4 Test data for end-to-end

~~Do **not** re-ingest to test. Restore a `teamcenter` snapshot onto the test box.~~
**Superseded (2026-09-29 / 10-27):** EC2 never serves the Mac mini's vectors — and
they were made by a different embedder (CPU INT8) than EC2's. EC2 ingests its own
Teamcenter 14.2 test collection on the instance (§8 3b).

---

## 7. What crosses the boundary

What this repo hands the AWS/operator repo when the gates close:

**Numbers for sizing:** C (cores per query); measured service time per search
(single-subject and `all`); permits per instance; peak RSS for the server (~4.4 GB
**measured** on the old stack, re-measure) and for the worker (~4.8 GB **measured**);
bulk-ingest wall clock from §5.4.

**New config keys to render** (all in the mounted `config.yaml`, none of them secret;
**all required** — a missing key refuses to start, §8 3b; names final when implemented):

| Key | Purpose |
|---|---|
| `rerank.device` | `cpu` \| `cuda` |
| `rerank.precision` | `int8` \| `fp16` |
| `embed.url` (or similar) | address of the embedder service — server and worker |
| `embed.device`, `embed.precision` | the embedder service's own runtime: `cuda`, `fp16` \| `fp32` |
| `rerank.threads`, `embed.threads` | per-session core budget |
| `server.max_concurrent_searches` | semaphore permits |
| `server.warm_on_start` | startup warm-up toggle |

**Unchanged contract:** `DOCS_BRIDGE_TOKENS` (plural, one per client, independently
revocable) stays the auth surface; `/healthz` stays the only open route and is the
health-check target; `/mcp` stays streamable HTTP with a bearer header.

**Constraints the other repo must respect, and cannot see from the image:**

1. **Pool members are stateless and need no corpus.** The server reads Qdrant only —
   no manifest, no `/data/docs`. They need an identical `config.yaml` (differing
   `subjects`/`top_n` between members means answers vary by which one you hit) and the
   baked models. Small root volume, no data EBS.
2. **Ingest is single-writer.** `manifest.py:52` is a plain `sqlite3.connect(path)` —
   no WAL, no busy timeout. Exactly one host runs `sync`, and never two subjects
   concurrently against one manifest.
3. **Qdrant has no auth or TLS in this codebase.** `qdrant_io.py:48` is
   `QdrantClient(host=..., port=...)` — no api_key, no https parameter. Cross-instance
   traffic is unauthenticated plaintext; network isolation is the only control until
   that is changed here.
4. **The ingest host holds state; it needs a data EBS volume.** Qdrant storage, the
   manifest and `/data/docs` are not reproducible from the AMI + Ansible. They sit on
   the root disk only while the test collection is cheap to rebuild; a separate EBS
   data volume (with snapshots) comes before anything else lands. Never the
   `/opt/dlami/nvme` instance store — wiped on every stop/start.
5. **The embedder is a separate unit** (§6.3) — the server and the worker reach it by
   address, so the deploy must not assume it shares a host with either.

---

## 8. Sequence

1. **Request the GPU quota** (other repo) — do this first; it gates §2.3 by days.
2. **Gate 1** on today's image: thread-scaling (§2.1), fixture dump (§2.2), GPU probe
   (§2.3) and the smaller-reranker probe (§3) in parallel. **Decide the hardware.**
   — *§2.1, §2.2, §3 DONE 2026-09-26; §2.3 DONE 2026-09-28 on a T4: 0.29-0.54 s vs
   14.2 s. DECIDED: GPU; quality bar re-referenced to fp32 (§2.4).*
3. **The upgrade** (§4) on this branch: pins, Tier 1+2 enrichment, the `parse.py:83`
   comment fix, the §4.2 invariant recorded.
   — *Code DONE 2026-09-28 (decision A: upgrade everything on vhost2); images
   test-built rootless on vhost2. Awaiting `notes/phase-a-runbook.md` steps 0-7 on the
   host — step 4 (embed parity vs stored `teamcenter`) is blocking — and the
   operator-repo switch-over.*
   — *2026-10-14 → 10-17: the Mac mini rebuilt from scratch instead of re-embedding
   (subject-keyed manifest, per-collection embedder stamp, worker fonts, embed batch
   32 + token budget); retrieval gate adjudicated PASS, switched over 2026-10-17.
   Record: branch `mac-mini`, `notes/phase-a-runbook.md`, `notes/phase-a/`.*

   **From here the two targets are two branches** (2026-10-27): `mac-mini` (the Mac
   mini line) and `ec2-gpu` (this file's line, formerly `multi-user-rollout`). They
   share history up to the merge of `mac-mini` into `ec2-gpu` on 2026-10-27 and **may
   diverge for good**; later fixes are picked across deliberately, not merged by habit.

   **3b. EC2.** Decided 2026-10-27:
   - **Ingest runs on the EC2 instance** — the corpus is embedded where it is served,
     so §1.1's cross-architecture check does not apply. Needs an amd64 worker lock
     (`images/lock.sh` locks the worker for aarch64 only).
   - **Embedder fp16 or fp32 on the GPU, not INT8** (§5.1 reopened → decided; which of
     the two is measured on the test collection).
   - **The embedder is its own service** — a container with the frozen model and its
     own pinned onnxruntime, called over the network by the worker and the server, so
     it can move to another instance (`stack-upgrade-strategy.md` §7 item 1, adopted).
     The server and worker images then carry no embedder runtime.
   - **No silent defaults.** Every new key (`rerank.device`, `rerank.precision`, the
     embedder's address and precision, …) is **required**; a config without it refuses
     to start, naming the key. The Mac mini gets the keys explicitly if `mac-mini` ever
     takes this code — a missing key never means "today's behaviour".
   - **Hardware stays `g4dn.xlarge` (T4, 4 vCPU, 15 GiB, no swap) for now**, slow
     ingest accepted: something working end to end first; the instance is bumped for
     the full rebuild tests.
   - **Test collection: Teamcenter 14.2** (78 PDFs, 179 MB) — not the SKF pilot. The
     Mac mini's `teamcenter_14_2` (39,649 pts, CPU INT8) is the baseline; the gate is
     a short labelled 14.2 question set run against both.
   - **Users: LibreChat as a container on the host, reached through an SSM port
     forward**, with an external LLM API (egress via Zscaler). ALB, real ingress and
     Gate 3 (§6) come later.

   Still in this order before the first EC2 collection — image work is cheap to redo,
   a collection on the wrong embedder is a re-embed:
   1. *fp16 reranker on onnxruntime + CUDA* — the fp16 export (§6.3) is unwritten and
      §5.2's comparison against fp32 has not run on it.
   2. *No silent CPU fallback* — a CUDA/cuDNN mismatch makes onnxruntime run on CPU
      without an error: the server and the embedder check at startup that they got the
      CUDA provider and refuse to start otherwise. Also: exactly one of `onnxruntime` /
      `onnxruntime-gpu` may be installed (docling's OCR extras pull the CPU one).
   3. *The embedder service* — fp16/fp32 exports from a pinned HF revision as a new
      `docs-bridge-models` tag, the service image, and the fingerprint extended with
      precision and execution provider.
      **Exports DONE 2026-10-27:** `localhost/docs-bridge-models:gpu-2026-10` on the
      host (bge-m3 fp32 + fp16, reranker fp16; bge-m3 `5617a9f`, reranker `953dc6f`),
      built by `tools/onnx-export/build-gpu-models.sh`. ONNX fp32 = torch fp32 (embed
      cosine 0.99999994, rerank |Δ| 5e-6); fp16 vs fp32 on ORT CUDA: embed cosine
      ≥ 0.99985, rerank max |Δ| 0.017; deterministic, so rebuildable. Open from it
      (`tools/onnx-export/FINDINGS.md`): ORT fp16 reranks at 0.85 s/query against
      torch's 0.54 s — no fused attention yet (item 1, finding 6); fp16 moves sparse
      weights up to 0.07 of 0.38 — input to the embedder's precision (finding 7). The
      fingerprint must hash `model.onnx.data` too when the embedder runs fp32.

   **Host facts (inventory 2026-10-27):** Docker 29.8 + NVIDIA Container Toolkit 1.20
   already registered as a runtime, driver 595.91 / CUDA 13.2 (runs CUDA-12 wheels in
   containers), 96 GB root disk, and a 115 GB `/opt/dlami/nvme` **instance store that
   is wiped on every stop/start** — scratch only, never Qdrant or the manifest.

   **Not to forget — a separate EBS data volume.** For the test collection, Qdrant, the
   manifest and the documents live on the root disk. Before anything that is not
   cheaply reproducible lands there, the AWS repo adds a data EBS volume (and its
   snapshot/backup), because its README assumes the root disk holds nothing that the
   AMI + Ansible cannot rebuild.
4. **Gate 2** (§5): embed determinism, reranker A/B, first-stage sanity, cost refresh
   off the pilot library.
5. **Gate 3** (§6): the concurrency changes, the liveness test, pool statelessness;
   GPU packaging (§6.3) only if Gate 1 said yes.
6. **Bulk ingest** — one top-level library per run, `verify-corpus.py` after every
   batch, treating a clean exit as the unit of work (`work-archive-ingest-plan.md` §7).
7. **Hand §7 to the AWS repo.**

Steps 2 and 3 can overlap only in the sense that §4's dependency resolution can be
drafted while Gate 1 runs — Gate 1 must *measure* on the old pins.

---

## 9. Deliberately not doing

- **Bumping `fastmcp` to 3.x to fix the blocking.** Later fastmcp does offload sync
  tools by default (3.4.2: `run_in_thread: bool = True`, `function_tool.py:89,332`),
  and 3.x adds a per-tool `timeout`. But that is a major-version move in the
  dependency the entire HTTP surface rests on; `server.py:14` already flags
  `mcp.http_app(path=...)` and `.lifespan` as integration points to re-verify on a
  bump, and three LibreChat↔MCP gotchas have already been paid for on this surface.
  The 5-line async change achieves the same thing with no dependency movement. Revisit
  the bump later, deliberately, on its own branch.
- **Graviton + GPU (g5g).** §6.3 — no aarch64 `onnxruntime-gpu` below 1.29.0.
- **Moving the embedder to GPU.** ~20 ms of an 11 s query, and it is the one component
  with vector-parity exposure. Reranker only.
- **Re-litigating the first stage.** Keep RRF, keep sparse, no union, no DBSF, no
  Elasticsearch — all measured and settled. §5.3 only re-confirms it on new chunks.
- **A whole-page VLM pipeline, or switching parser.** Per
  `stack-upgrade-strategy.md` §4 and its "deliberately not doing".
- **Per-user auth or rate limiting inside docs-bridge.** Tokens stay per-*client*;
  user identity lives in LibreChat. Worth revisiting only if a per-user quota is
  actually needed.
- **Trusting manifest totals.** `verify-corpus.py` gates every ingest batch. The
  manifest records intent; Qdrant records fact.

---

## 10. Verified facts, and how to re-check them

Recorded so a future reader does not have to re-derive them:

| Fact | Evidence | Re-check |
|---|---|---|
| `fastmcp==2.14.7` runs sync tools **inline on the event loop** | `tools/tool.py:377-383`, `tools/tool_manager.py:160` — no `to_thread` in the chain | `docker exec docs-bridge python -c "import inspect,fastmcp.tools.tool as t;print(inspect.getsource(t.FunctionTool.run))"` |
| fastmcp 3.x offloads sync tools by default | 3.4.2 `tools/function_tool.py:89` (`run_in_thread: bool = True`), `:332` | same, against a 3.x install |
| `stateless_http` exists in 2.14.7 | `server/server.py:2586-2632` | as above, on `FastMCP.http_app` |
| `onnxruntime-gpu` has no aarch64 wheels before 1.29.0 | PyPI release files: 1.20.x = `manylinux*_x86_64` + `win_amd64` only; first `manylinux_2_34_aarch64` at 1.29.0 (cp311 present) | `curl -s https://pypi.org/pypi/onnxruntime-gpu/json` and filter filenames |
| Rerank is **~354 ms/pair**, latency ≈ `top_n × 354 ms` = **14.2 s** at `top_n=40` | `dump_rerank_fixture.py` on vhost2, 320 real pairs **(measured 2026-09-26; supersedes bench_search.py's ~270 ms)** | step 1 of `gate1-runbook.md` |
| Pair lengths are p50=100 / p95=420, and only 0.3% reach `max_length=512`; the tokenizer pads to batch-longest, so all 40 compute at ~500 | `dump_rerank_fixture.py` + `rerank_batching.py` **(measured)** | steps 1 and 2b of `gate1-runbook.md` |
| INT8 activation scales are batch-dependent, so re-batching moves scores up to 1.41 logits | `rerank_batching.py`, `rerank_ab.py --bucket` **(measured)** | step 4c of `gate1-runbook.md` |
| `top_n` cannot be lowered | `recall_topn.py`, `validate_union.py` **(measured)** | re-run after any chunker change (§5.3) |
| C (cores per query) | **6 on vhost2** — near-linear to 6 (efficiency 0.91), then *reverses*: 8 and 10 threads are slower, because 6 = the M2 Pro's P-core count **(measured)** | `thread_scaling.py`; **re-run on the x86 target — C=6 is silicon-specific and must not size EC2** |

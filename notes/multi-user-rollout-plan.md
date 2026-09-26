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
candidates into one `session.run`, and latency is dead linear at **~270 ms per
(query, passage) pair (measured)** — the batch buys nothing because a single pair's
matmuls already saturate every core. At the deployed `rerank.top_n: 40` that is
**~11 s per query (measured)**, ~16 s for a multi-subject or `"all"` search at
`multi_top_n: 60`. `top_n` is closed as a lever: `recall_topn.py` and
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
`search.py:59-66` never passes it, so every session takes every core and C has never
been measured. C sets permits-per-instance and therefore the entire fleet shape.

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

- **C measured**, with the seconds-vs-threads curve recorded in this file.
- **GPU latency measured** on real pairs, and ranking compared to the CPU-INT8
  baseline using the established bar: top-3 ordered-identical, top-6 Jaccard ≥ 0.83.
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

### 4.3 What moves

- `docling`, `docling-core[chunking]`, `docling-parse`, `docling-ibm-models` →
  current (that document records 2.130.0, Aug 2026; re-check at branch time).
- `onnxruntime` → current, or `onnxruntime-gpu` for the GPU path (§6.3).
- `tokenizers` / `transformers` → whatever the new docling set resolves to.
- `numpy<2` → likely liftable; the `<2` cap exists for the old docling/FlagEmbedding
  set. Resolve deliberately and comment it, in the style the file already uses.
- `torch` → still arrives via docling. Dropping the `FlagEmbedding` fallback backend
  is worth considering while the file is open (ONNX is the default and has been since
  the INT8 adoption), but `docling-slim[standard]` still pulls torch, so the worker
  image does not shrink from that alone.

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

### 5.2 Reranker A/B (blocking for the GPU path)

Modify `tools/eval/recall_topn.py` to parameterise the **reranker** instead of
`top_n`. It already drives the real `Searcher` inside the container against the real
config, so the change is small and the comparison stays honest. Compare, on the §2.2
fixture plus the live collection:

1. CPU INT8, old toolchain (the recorded baseline)
2. CPU INT8, new toolchain
3. GPU fp16, new toolchain (GPU path only)

Bar: top-3 ordered-identical and top-6 Jaccard ≥ 0.83 against the baseline, with
disagreements adjudicated by the deeper-pool oracle (§2.4) rather than assumed to be
regressions.

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
| Provider + config keys | trivial | `providers=["CUDAExecutionProvider","CPUExecutionProvider"]` plus `rerank.device` / `rerank.precision`, so **one image still runs CPU-only**. Keeps vhost2 working and preserves a CPU fallback. |
| CUDA base image | **~1 day — the real work** | Prefer keeping `python:3.11-slim-bookworm` and pulling CUDA in as pip wheels (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`) with `LD_LIBRARY_PATH` set, rather than switching to `nvidia/cuda:*-ubuntu22.04` (which ships Python 3.10, not 3.11). The wheel route keeps every pin and the `deps`→`exporter`→`runtime` cache structure intact. ORT's CUDA major/cuDNN major must match or you get a silent CPU fallback rather than an error. |
| x86_64 switch | ~half a day | Mechanical: the aarch64-only comments in both Dockerfiles, pins re-resolving on x86 (wheels exist). Two side benefits: the from-source `images/qdrant` build becomes unnecessary (4 K pages), and the M2 stops being the only builder — build in CI/on the instance and push to a registry, finally closing the "build once, distribute" item open since the Pi. |

**Architecture note (verified):** `onnxruntime-gpu` has **no aarch64 wheels below
1.29.0** — the 1.20.x line is x86_64/Windows only. So a Graviton+GPU (g5g) target
would force ORT nine minor versions forward on its own. **Go x86_64 for the GPU path.**

### 6.4 Test data for end-to-end

Do **not** re-ingest to test. Restore a `teamcenter` snapshot onto the test box — that
path is built, idempotent, and validated (fires only on collection-404 plus
manifest-absent).

---

## 7. What crosses the boundary

What this repo hands the AWS/operator repo when the gates close:

**Numbers for sizing:** C (cores per query); measured service time per search
(single-subject and `all`); permits per instance; peak RSS for the server (~4.4 GB
**measured** on the old stack, re-measure) and for the worker (~4.8 GB **measured**);
bulk-ingest wall clock from §5.4.

**New config keys to render** (all in the mounted `config.yaml`, none of them secret):

| Key | Purpose |
|---|---|
| `rerank.device` | `cpu` \| `cuda` |
| `rerank.precision` | `int8` \| `fp16` |
| `rerank.threads`, `embed.threads` | per-session core budget |
| `server.max_concurrent_searches` | semaphore permits |
| `server.warm_on_start` | startup warm-up toggle |

**Unchanged contract:** `DOCS_BRIDGE_TOKENS` (plural, one per client, independently
revocable) stays the auth surface; `/healthz` stays the only open route and is the
health-check target; `/mcp` stays streamable HTTP with a bearer header.

**Three constraints the other repo must respect, and cannot see from the image:**

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

---

## 8. Sequence

1. **Request the GPU quota** (other repo) — do this first; it gates §2.3 by days.
2. **Gate 1** on today's image: thread-scaling (§2.1), fixture dump (§2.2), GPU probe
   (§2.3) and the smaller-reranker probe (§3) in parallel. **Decide the hardware.**
3. **The upgrade** (§4) on this branch: pins, Tier 1+2 enrichment, the `parse.py:83`
   comment fix, the §4.2 invariant recorded.
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
| Rerank is ~270 ms/pair, latency ≈ `top_n × 270 ms` | `bench_search.py` on vhost2 **(measured)** | `sudo docker exec -i docs-bridge python - < tools/eval/bench_search.py` |
| `top_n` cannot be lowered | `recall_topn.py`, `validate_union.py` **(measured)** | re-run after any chunker change (§5.3) |
| C (cores per query) | **not yet measured** | §2.1 — the first thing this plan does |

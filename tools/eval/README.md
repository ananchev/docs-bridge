# tools/eval

Retrieval / embedding **quality + performance** tooling — the gates and diagnostics
used to validate the ONNX/INT8 stack and every change under it, and to tune
retrieval. (The ONNX-vs-FlagEmbedding gates of 2026-06, `validate_embed_parity.py`
and `retrieval_parity.py`, were retired with the FlagEmbedding backend on
2026-09-28; they are in git history.) Most of these drive the same code the server uses, so they run
**inside the running docs-bridge container** (it has the baked models, mounted
config, and the `qdrant` network alias). See each file's docstring header for the
full flag/env list.

## Scripts

| Script | Purpose |
|---|---|
| `embed_upgrade_parity.py` | **Phase A gate (blocking for the Mac mini)** — does a new embedder *runtime* (onnxruntime/tokenizers/numpy) put text where the old one did, and no further from the stored vectors than the old runtime itself? `dump` inside each image, `compare` anywhere. Checks model bytes, token ids, old-vs-new vectors, and drift vs stored. |
| `retrieval_ab.py` | **Gate 2, Mac mini old vs new** — the labelled question set (`notes/gate2/questions-*.yaml`) through the real `Searcher` in each image against the same collection: hit@k, MRR, control abstention; `compare` fails on any question the old image answered and the new one misses. |
| `reembed_to_collection.py` | Re-embed an existing collection's chunks with the ONNX backend into a NEW collection — the substrate for Gate 2 and the embed-speed A/B. |
| `recall_topn.py` | Decide whether lowering rerank `top_n` loses retrieval quality (one deep retrieve contains every smaller `top_n` as a prefix). |
| `diag_channels.py` | Decompose hybrid retrieval to find WHICH first-stage channel fails to surface the rerank-confirmed gold chunk. |
| `validate_union.py` | Score first-stage strategies against the globally-best (rerank-confirmed) answers, avoiding tie-jitter false losses. |
| `bench_search.py` | Measure `/v1/search` latency broken down by stage (retrieve vs rerank), to see what the rerank costs before tuning `top_n`. |
| `dump_rerank_fixture.py` | **Gate 1 §2.2** — dump the rerank fixture: the validated queries, their real RRF candidates, and the current CPU-INT8 scores. The one artifact every hardware/model comparison consumes (it needs no Qdrant, so it travels to a bare GPU box). |
| `thread_scaling.py` | **Gate 1 §2.1** — seconds-per-query vs `OnnxReranker(threads=N)`, yielding **C**, the cores one search usefully consumes. C sets semaphore permits and therefore the whole fleet shape. |
| `rerank_ab.py` | **Gate 1 §2.3 / §3, Gate 2 §5.2** — A/B any reranker (ONNX int8/fp32/fp16 — CUDA provider only with `--device cuda` — torch fp16/bf16 on CUDA, or `--backend docs-bridge`: the production `OnnxReranker`, run inside the server image) on the top-3 + Jaccard bar against a **reference**: a saved fp32 run via `--reference` (plan §2.4, since 2026-09-28), else the fixture's legacy CPU-INT8 column. Disagreements are adjudicated by the reference as a *fixed* judge over the deeper pool (needs a `--score-pool` reference run). `--backend replay --replay <run.json>` re-judges any saved run offline. Standalone: no Qdrant, no config, no `docs_bridge` install (except `--backend docs-bridge`). The torch backend needs `--revision`. Canonical reference: `notes/gate1/ref-fp32.json`. |

## Sample invocation

These stream a script into the container's Python (no install needed):

```bash
# inside the running stack, e.g. latency breakdown:
sudo docker exec -i docs-bridge python - < tools/eval/bench_search.py

# with overrides (each script reads its own env — see its docstring):
sudo docker exec -i \
  -e BENCH_QUERY='...' -e BENCH_SUBJECT=teamcenter -e BENCH_K=5 \
  docs-bridge python - < tools/eval/bench_search.py
```

The parity/re-embed gates take CLI args (e.g. `--ref`, source/target collections);
run the file with `--help` or read its header for specifics.

## Gate 1 (multi-user rollout)

`notes/gate1-runbook.md` has the exact, ordered commands for vhost2 and records the
results; `notes/gate1-gpu-probe-host.md` specifies the GPU host for §2.3. The three
scripts above are ordered: the fixture first, then the thread-scaling curve and the
reranker A/Bs, which both consume it.

Unlike the older harnesses, these run as a **fresh throwaway container** rather than
`exec` into the live server, so a measurement never shares a process with something
serving traffic:

```bash
sudo podman run --rm -i --network docs-bridge-net \
  -v /data/docker/docs-bridge/config/config.yaml:/config/config.yaml:ro,z \
  -v ~/gate1/out:/out:z \
  --entrypoint python localhost/docs-bridge:latest - \
  < dump_rerank_fixture.py > out/fixture.json
```

`rerank_ab.py` takes CLI flags (it also runs as a plain file on a GPU host); the
other two read env vars, like the older scripts here.

What was learned using these harnesses: [FINDINGS.md](FINDINGS.md).

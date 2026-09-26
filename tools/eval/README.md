# tools/eval

Retrieval / embedding **quality + performance** tooling — the gates and diagnostics
used to validate the ONNX/INT8 stack against the FlagEmbedding ground truth and to
tune retrieval. Most of these drive the same code the server uses, so they run
**inside the running docs-bridge container** (it has the baked models, mounted
config, and the `qdrant` network alias). See each file's docstring header for the
full flag/env list.

## Scripts

| Script | Purpose |
|---|---|
| `validate_embed_parity.py` | **Gate 1** — vector parity: does the ONNX (and INT8) backend reproduce FlagEmbedding on real chunk texts, against hard thresholds? |
| `retrieval_parity.py` | **Gate 2** — retrieval parity: does the ONNX stack return the same search results as the FlagEmbedding stack? The user-facing quality gate. |
| `reembed_to_collection.py` | Re-embed an existing collection's chunks with the ONNX backend into a NEW collection — the substrate for Gate 2 and the embed-speed A/B. |
| `recall_topn.py` | Decide whether lowering rerank `top_n` loses retrieval quality (one deep retrieve contains every smaller `top_n` as a prefix). |
| `diag_channels.py` | Decompose hybrid retrieval to find WHICH first-stage channel fails to surface the rerank-confirmed gold chunk. |
| `validate_union.py` | Score first-stage strategies against the globally-best (rerank-confirmed) answers, avoiding tie-jitter false losses. |
| `bench_search.py` | Measure `/v1/search` latency broken down by stage (retrieve vs rerank), to see what the rerank costs before tuning `top_n`. |
| `dump_rerank_fixture.py` | **Gate 1 §2.2** — dump the rerank fixture: the validated queries, their real RRF candidates, and the current CPU-INT8 scores. The one artifact every hardware/model comparison consumes (it needs no Qdrant, so it travels to a bare GPU box). |
| `thread_scaling.py` | **Gate 1 §2.1** — seconds-per-query vs `OnnxReranker(threads=N)`, yielding **C**, the cores one search usefully consumes. C sets semaphore permits and therefore the whole fleet shape. |
| `rerank_ab.py` | **Gate 1 §2.3 / §3, Gate 2 §5.2** — A/B any reranker (ONNX int8/fp32, or torch fp16/bf16 on CUDA) against the fixture's baseline, on the top-3 + Jaccard bar, with disagreements adjudicated by a deeper-pool oracle. Standalone: no Qdrant, no config, no `docs_bridge` install. |

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

# images/docs-bridge

The docs-bridge server image: hybrid search → rerank → MCP/REST. Built from the repo
root as context, with the models copied from a `docs-bridge-models` image
(`images/models/`) — never exported here. What was found building it:
[FINDINGS.md](FINDINGS.md).

## Targets

There is no default target and no default `MODELS_IMAGE`; a build without either fails.

| Target | Arch | Lock | Models | Rerank config |
|---|---|---|---|---|
| `runtime-cpu` | arm64 (Mac mini) or amd64 | `requirements-<arch>.lock` (`onnxruntime`) | `frozen-2026-09`: INT8 bge-m3 + reranker | `device: cpu, precision: int8, bucket: 0` |
| `runtime-gpu` | amd64 only | `requirements-amd64-gpu.lock` (`onnxruntime-gpu[cuda,cudnn]`) | a GPU tag, e.g. `gpu-2026-10`: the fp16 reranker only | `device: cuda, precision: fp16, bucket: 8` |

The embed config goes with the target: `runtime-cpu` embeds queries in-process
(`embed: {mode: local, device: cpu, precision: int8, model_dir: /opt/bge-m3-onnx}`);
`runtime-gpu` holds no embedder and embeds them through the embedder service
(`embed: {mode: service, url: ...}`, `images/embedder/`).

```bash
docker build -f images/docs-bridge/Dockerfile --target runtime-gpu \
    --build-arg MODELS_IMAGE=localhost/docs-bridge-models:gpu-2026-10 \
    -t localhost/docs-bridge:gpu-<commit> .
docker run --gpus all -v <dir with config.yaml>:/config:ro -e DOCS_BRIDGE_TOKEN=... \
    -p 127.0.0.1:8080:8080 localhost/docs-bridge:gpu-<commit>
```

Locks come from `images/lock.sh`, which also refuses a GPU lock holding the CPU
`onnxruntime` too, or a different onnxruntime version than the CPU locks.

## Startup checks

The server refuses to start, naming the cause, when:

- the `embed` section is missing or incomplete for its mode, or one of the keys it
  replaced (`embedding_int8`, `embedding_backend`, `onnx_model_dir`) is present
  (`config._embed`);
- `embed.mode: service` and the service does not answer, or answers with a
  fingerprint that does not match its id;
- `rerank.device`, `rerank.precision` or `rerank.bucket` is missing, or the old
  `rerank.int8` is present (`config._rerank`);
- the combination is impossible (INT8 on CUDA, fp16 on CPU) or the image has no file
  for that precision;
- more or fewer than one onnxruntime build is installed, or `cuda` is asked of the CPU
  build (`ort_session.open_session`);
- the session did not get the CUDA provider — e.g. the container was started without
  `--gpus` (FINDINGS 1).

The embedder is reached and the reranker loaded, and each runs once, before the
server listens, so these fail the start, not the first search. The server does not
compare the embedder's fingerprint with the collections' stamps — those live in the
ingest manifest, which a server does not read (plan §7, constraint 1).

## Verify

```bash
# the production reranker in this image vs the fp32 reference (Gate 1 fixture)
docker run --rm --gpus all --entrypoint python -v <repo>/tools/eval:/t:ro -v ~/gate1:/g \
    localhost/docs-bridge:gpu-<commit> /t/rerank_ab.py --fixture /g/fixture.json \
    --backend docs-bridge --model-dir /opt/bge-reranker-onnx --device cuda \
    --precision fp16 --bucket 8 --reference /g/ab/ref-fp32.json
```

Expect `VERDICT: PASS` at ~0.4 s/search on a T4.

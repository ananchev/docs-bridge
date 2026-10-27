# images/embedder

The embedder service image: bge-m3 on the GPU behind HTTP (`docs-bridge-embedder`,
`src/docs_bridge/embed_service.py`). EC2's ingest worker embeds chunks through it
and the server embeds queries through it (`embed.mode: service`), so documents and
queries come from one model and one runtime. It holds the frozen model, its own
pinned onnxruntime-gpu and an HTTP surface — no Qdrant client, no Docling, no torch —
so no other dependency bump can reach the vectors. What was found building it:
[FINDINGS.md](FINDINGS.md).

## Build

From the repo root as context, linux/amd64 only, with a GPU models tag
(`images/models/gpu.Dockerfile`). `MODELS_IMAGE` has no default.

```bash
docker build -f images/embedder/Dockerfile \
    --build-arg MODELS_IMAGE=localhost/docs-bridge-models:gpu-2026-10 \
    -t localhost/docs-bridge-embedder:gpu-<commit> .
```

The lock is `requirements.lock` (`images/lock.sh`: extras `embedder` + `gpu`); the
script refuses it if the CPU `onnxruntime`, `qdrant-client`, `torch`, `docling` or
`fastmcp` appears in it, or if its onnxruntime, numpy or tokenizers differ from the
CPU server lock's. The image ships bge-m3 at **fp32 and fp16**; the service runs the
one `--precision` names.

## Run

Every flag is required — the deploy states which model runs where, nothing is picked
by omission:

```bash
docker run -d --name embedder --gpus all --network <net> \
    localhost/docs-bridge-embedder:gpu-<commit> \
    --model-dir /opt/bge-m3-onnx --device cuda --precision fp16 \
    --host 0.0.0.0 --port 8090
```

| Flag | Values |
|---|---|
| `--model-dir` | `/opt/bge-m3-onnx` in this image |
| `--device` | `cuda` (`cpu` exists for completeness; this image has no INT8 file) |
| `--precision` | `fp16` \| `fp32` on `cuda` |
| `--host`, `--port` | where it listens; the worker and server use it as `embed.url` |

It runs as `nobody` and needs no volume. Ready after ~15 s on the T4 (session,
hashing the model files, one embed). There is **no auth**: like Qdrant it belongs on
the private network only (plan §7, constraint 3).

## Surface

| Route | Answer |
|---|---|
| `GET /healthz` | `{"status": "ok"}` |
| `GET /v1/fingerprint` | `{"id", "fingerprint"}` — model id and revision, precision, sha256 of every file that precision runs (+ sparse head, tokenizer), execution provider, onnxruntime build, tokenizers version |
| `POST /v1/embed` | `{"texts": [...]}` → `{"fingerprint_id", "dense": [[...]], "sparse": [{"indices", "values"}]}` |

Encodes run one at a time (the token budget bounds one run's GPU memory); a query
arriving during an ingest batch waits for that batch.

## Startup checks

The service exits, naming the cause, before it listens when:

- a flag is missing, or the device/precision pair cannot run (fp16 on CPU, INT8 on
  CUDA);
- the image has no file for that precision;
- more or fewer than one onnxruntime build is installed, or the session did not get
  the CUDA provider — e.g. started without `--gpus` (`ort_session.open_session`);
- a file's sha256 on disk differs from the one `meta.json` records.

## Verify

```bash
docker logs embedder | grep fingerprint      # the id the collection will be stamped with
curl -s <host>:8090/v1/fingerprint | python3 -m json.tool
```

Then a worker sync stamps the collection with that fingerprint, and the server logs
`embedder service <url>: BAAI/bge-m3 <precision> on CUDAExecutionProvider (<id>)` at
start. Service vs in-process `OnnxEmbedder` on the same GPU must be bit-identical
(FINDINGS 2).

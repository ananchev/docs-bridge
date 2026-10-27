# images/models

`docs-bridge-models` — the model artifacts, as a `FROM scratch` image that the runtime
images copy from (`COPY --from=models`). A model has its own identity (the sha256 of
its files); the runtime images never export one. One tag = one set of model bytes.

| Dockerfile | Tag | What it holds | Built by |
|---|---|---|---|
| `Dockerfile` | `frozen-2026-09` | the Mac mini's **INT8** bge-m3 + reranker, taken out of the images that embedded its vectors — not a re-export | `podman build` on vhost2 (header) |
| `gpu.Dockerfile` | `gpu-<yyyy-mm>` | the EC2 target's bge-m3 **fp32 + fp16** and reranker **fp16**, from pinned HF commits, checked on the CUDA provider | `tools/onnx-export/build-gpu-models.sh` only |

Paths are the same in both (`/opt/bge-m3-onnx/`, `/opt/bge-reranker-onnx/`); the file
inside picks the precision (`model.int8.onnx`, `model.onnx` + `model.onnx.data`,
`model.fp16.onnx`). The GPU image adds `/opt/docs-bridge-models.json`: the docs-bridge
commit, the HF revisions, every shipped file's sha256 and the checks' results.

Tags are fully qualified (`localhost/docs-bridge-models:<tag>`): a bare name resolves
to `docker.io/library/` and is one pull away from being replaced. A tag is never
rebuilt in place — the build script refuses an existing tag.

The GPU tag lives in the EC2 host's Docker store only (no registry yet). Losing it costs
a ~10-minute rebuild that reproduces the same bytes (`tools/onnx-export/FINDINGS.md`
finding 5), checked against the manifest committed for that tag in
[`notes/models/`](../../notes/models/).

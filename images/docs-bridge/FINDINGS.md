# images/docs-bridge — findings register

What was found building and running the server image, and how each item was
resolved. Numbering is stable. The operating manual is [README.md](README.md).

## Findings & peculiarities (and how we resolved them)

### 1. Without `--gpus`, onnxruntime-gpu silently runs on CPU
Started without `--gpus all` on 2026-10-27, the `runtime-gpu` image's session asked
for `CUDAExecutionProvider` alone and got `['CPUExecutionProvider']` — onnxruntime
substitutes CPU with a warning, not an error, and would have served every search at
CPU speed with a different (fp16-on-CPU) computation.
**Resolution:** `ort_session.open_session` compares the session's providers with the
one requested and raises; the server runs the reranker before it listens, so the
container exits at start with the reason. Verified: exit 1, "refusing to run on a
fallback".

### 2. Two onnxruntime builds in one environment
`onnxruntime` and `onnxruntime-gpu` install into the same `onnxruntime/` package dir;
with both present the last one installed wins and the other's metadata lies. Until
2026-10-27 `onnxruntime` was a base dependency of the package, so any GPU lock built
from it would have held both.
**Resolution:** onnxruntime moved to the extras `cpu` and `gpu`; each lock takes one
(`images/lock.sh` refuses a GPU lock with the CPU build in it), and
`open_session` refuses an environment with both or neither. The existing CPU locks
came out byte-identical apart from their header.

### 3. A Dockerfile default picked the models by omission
`MODELS_IMAGE` defaulted to `frozen-2026-09` and the last stage was the CPU runtime, so
a GPU build that forgot either flag would have produced an INT8 CPU image without
complaint.
**Resolution:** no default `MODELS_IMAGE`, and the last stage is `choose-a-target`,
which fails with the instruction. The `runtime-gpu` stage also refuses a non-amd64
`TARGETARCH`.

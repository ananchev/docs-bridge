# images/models-export

`docs-bridge-models-export` — the pinned toolchain that exports, converts and checks
model dirs (`tools/onnx-export/`). A **tool image, never deployed**; it holds no
scripts and no models. `tools/onnx-export/build-gpu-models.sh` builds it, tagged with
the docs-bridge commit, and runs it with the checkout mounted at `/src` (read-only),
the work dir at `/run-dir`, the HF cache at `/work/hf-cache`, `--gpus all`, and the
invoking user's uid.

| File | |
|---|---|
| `Dockerfile` | `python:3.11-slim-bookworm` + the lock; fails the build if onnxruntime has no CUDA provider compiled in |
| `requirements.in` | the toolchain: torch (+cpu), transformers, tokenizers, numpy (the worker's pins), onnx, onnxconverter-common, onnxruntime-gpu `[cuda,cudnn]` (the server's onnxruntime version) |
| `requirements.lock` | hashed, linux/x86_64, from `images/lock.sh` |

x86_64 only (the EC2 GPU host). torch is the CPU wheel — an export needs no GPU; the
CUDA libraries come as `nvidia-*` wheels with onnxruntime-gpu, CUDA 13 / cuDNN 9
(`tools/onnx-export/FINDINGS.md` finding 1). `images/lock.sh` refuses a lock where
the CPU `onnxruntime` sneaks in beside `-gpu`, or where `-gpu`'s version differs from
the server's onnxruntime.

Changing anything in the lock can change the exported bytes, i.e. the model identity
(finding 5): rebuild under a new models tag and compare the sha256s.

# tools/onnx-export

Offline **build steps** that turn the official BAAI models into the ONNX model dirs
the runtime loads, and check them. Never part of a runtime image: the runtime images
copy their models from a `docs-bridge-models` image (`images/models/`), built once
per model identity. What was learned building them: [FINDINGS.md](FINDINGS.md).

## Scripts

| Script | Purpose |
|---|---|
| `export_bge_m3_onnx.py` | `BAAI/bge-m3` (encoder + the real sparse head) → model dir for `docs_bridge.embed_onnx.OnnxEmbedder`. |
| `export_bge_reranker_onnx.py` | `BAAI/bge-reranker-v2-m3` (single-logit cross-encoder) → model dir for `docs_bridge.rerank.OnnxReranker`. `--model-id` exports another XLM-R/BERT cross-encoder into the same layout. |
| `_common.py` | What both share: the fp32 save layout, fp16 and INT8 conversion, `meta.json`. |
| `check_precision.py` | A model dir's acceptance check on the **CUDA provider only**: fidelity of `model.onnx` to the HF model (torch fp32), fp16 vs fp32, padding invariance. |
| `build-gpu-models.sh` | The GPU target's models image, end to end, on the EC2 host: exporter image → exports → checks → `localhost/docs-bridge-models:<tag>`. |

Both export scripts take a **required `--revision`**, a 40-hex HF commit (a branch
name is refused), and write:

| File | Variant |
|---|---|
| `model.onnx` + `model.onnx.data` | fp32, weights in one external-data file (2.3 GB > protobuf's 2 GB) |
| `model.fp16.onnx` | fp16 weights, fp32 inputs/outputs — `--fp16`, or `--fp16-only` on an existing dir |
| `model.int8.onnx` | dynamic INT8 (the CPU line) — `--quantize`, or `--quantize-only` |
| `tokenizer.json`, (`sparse_linear.npz`) | tokenizer; bge-m3's sparse head in numpy |
| `meta.json` | model id, revision, opset, sha256 + size of every file, the toolchain |

The `-only` modes are torch-free, so a conversion never shares RAM with torch's copy
of the model.

## Building the GPU models image

On the EC2 host, from a clean checkout of the branch (the image records the commit).
The host's checkout is Ansible's (`/opt/docs-bridge/src`, root-owned, synced by the
`docs_bridge_src` role in bnl-docs-bridge), hence `sudo`:

```bash
cd /opt/docs-bridge/src
sudo tools/onnx-export/build-gpu-models.sh --tag gpu-2026-10 \
    --work /opt/dlami/nvme/models-export --fixture /home/ubuntu/gate1/fixture.json
```

All three arguments are required. `--work` takes the HF cache, the export dirs, the
check results and `build.log` under `<work>/<tag>/`; it can be the instance store, since
the image is the artifact and the export is reproducible (FINDINGS 5). `--fixture` is
the Gate 1 rerank fixture (`tools/eval/dump_rerank_fixture.py`), whose real queries
and chunks the checks run on. About 10 minutes on a `g4dn.xlarge`, ~15 GB in `--work`.

The image ships bge-m3 **fp32 and fp16** (the embedder's precision is measured on the
14.2 test collection) and the reranker **fp16 only** (decided in Gate 1);
`/opt/docs-bridge-models.json` inside it names the commit, the revisions, every
shipped file's sha256 and both checks' results. Layout: `images/models/gpu.Dockerfile`.

## The checks

`check_precision.py --kind embed|rerank --model-dir … --fixture … --json …` exits
non-zero on a FAIL; its bars are constants in the script, documented in its header.
It answers "is this file the model, at this precision, on this provider" — not "does
retrieval still work": that is `tools/eval/rerank_ab.py` against the torch fp32
reference for the reranker, and the labelled question set for the embedder.

## The INT8 line

The Mac mini's models are the frozen INT8 image (`images/models/Dockerfile`), taken
out of the images that produced its vectors, not re-exported. A new INT8 export
(`--quantize`) is a new model identity and means re-embedding.

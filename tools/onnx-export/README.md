# tools/onnx-export

One-time, **offline build steps** that convert the official BAAI models to the ONNX
model dirs the runtime loads. Not part of the runtime image — you run these on a
build host (needs network to pull from Hugging Face), then ship/bake the output
dir. Reproducible and self-sourced (pulls official weights, exports via Optimum,
optionally makes an INT8 copy).

## Scripts

| Script | Purpose | Loaded by |
|---|---|---|
| `export_bge_m3_onnx.py` | Convert `BAAI/bge-m3` (dense + the real sparse head) to an ONNX model dir. | `docs_bridge.embed_onnx.OnnxEmbedder` |
| `export_bge_reranker_onnx.py` | Convert `BAAI/bge-reranker-v2-m3` (cross-encoder, single relevance logit) to an ONNX model dir. | `docs_bridge.rerank.OnnxReranker` |

## Sample invocation

```bash
# embedding model -> ONNX dir (+ optional INT8)
python tools/onnx-export/export_bge_m3_onnx.py --out ./models/bge-m3-onnx --quantize

# reranker -> ONNX dir (mirror of the above)
python tools/onnx-export/export_bge_reranker_onnx.py --out ./models/bge-reranker-onnx --quantize
```

See each file's docstring header for the exact flags (output path, `--quantize` /
`--quantize-only`). Note: quantizing needs ample RAM — see the build-host notes in
the project memory if a low-RAM host OOMs at the quantize step.

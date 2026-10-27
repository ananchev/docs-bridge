# tools

Operator / developer helpers, grouped by subject. Each subdir has its own README
with per-script descriptions and sample invocations.

| Subdir | What's in it |
|---|---|
| [`ingest/`](ingest/) | Get source docs from the Nextcloud archive into the corpus (Doxygen / Javadoc / SOA libs + WSDL → Markdown conversion, tag-driven copy, prune/reconcile, run the worker). |
| [`eval/`](eval/) | Retrieval & embedding quality gates and diagnostics (embed parity across a runtime upgrade, reranker A/B, end-to-end retrieval A/B on labelled questions, `top_n` recall, channel diagnosis, search latency). |
| [`onnx-export/`](onnx-export/) | Offline build steps: export the BAAI models to ONNX (fp32/fp16/INT8), check them on the CUDA provider, build the GPU models image. |

None of these are part of the `docs_bridge` runtime package — they're invoked
standalone or streamed into the running container.

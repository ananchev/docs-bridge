# tools

Operator / developer helpers, grouped by subject. Each subdir has its own README
with per-script descriptions and sample invocations.

| Subdir | What's in it |
|---|---|
| [`ingest/`](ingest/) | Get source docs from the Nextcloud archive into the corpus (Doxygen→Markdown conversion, tag-driven copy, prune/reconcile, run the worker). |
| [`eval/`](eval/) | Retrieval & embedding quality gates and diagnostics (ONNX/INT8 vs FlagEmbedding parity, `top_n` recall, channel diagnosis, search latency). |
| [`onnx-export/`](onnx-export/) | One-time offline build steps converting the BAAI models to the ONNX dirs the runtime loads. |

None of these are part of the `docs_bridge` runtime package — they're invoked
standalone or streamed into the running container.

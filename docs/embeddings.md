# Embeddings and reranking

docs-bridge uses two models from the BGE family, both run on CPU:

- **BGE-M3** — the embedder. One forward pass yields a **dense** semantic vector
  *and* a **sparse** lexical vector. Used for both ingest and query embedding.
- **BGE-reranker-v2-m3** — the cross-encoder reranker. Scores `(query, passage)`
  pairs jointly; the server's second-stage quality lever.

## Why BGE-M3 (dense + sparse from one model)

A hybrid retriever needs both a semantic signal (vector similarity) and a lexical
signal (exact-term overlap). The usual approach bolts a separate BM25 index onto a
dense vector store. BGE-M3 instead produces a learned **sparse** vector — token-id →
weight — from the *same* model as the dense vector, in a single pass. Both are
stored in Qdrant as named vectors on each chunk, and the server fuses them with RRF
([retrieval.md](retrieval.md)). One model, two complementary signals, no second
index to maintain.

The dense vector is 1024-dimensional (`embedding_dim`), cosine distance.

## The backend

`embed.get_embedder(cfg)` returns the embedder behind an `encode(texts) -> (dense,
sparse)` contract. There is one backend, `onnx` (`embed_onnx.OnnxEmbedder`:
onnxruntime + tokenizers, INT8 by default, ~4–5× the CPU throughput of the torch
path it was validated against in 2026-06 at on-par retrieval quality). The
FlagEmbedding/torch fallback was removed on 2026-09-28; its sparse vectors were in
the **same BGE-M3 token-id space**, so corpora it embedded stay queryable.

**Runtime parity.** The worker (documents) and the server (queries) must run the
embedder on the *same* onnxruntime, tokenizers and numpy, and the same model bytes —
or query vectors drift from stored ones. Both images install from locks that pin
those identically (`images/lock.sh` checks it), and both copy the model from one
frozen artifact. A runtime bump is gated by `tools/eval/embed_upgrade_parity.py`
against the stored vectors (`notes/phase-a-runbook.md` step 4).

### How the ONNX backend stays faithful

The strategy is *"encoder in ONNX, heads in numpy."* The big XLM-RoBERTa encoder is
exported to ONNX (and optionally INT8-quantized) — that is where the compute lives.
The two small BGE-M3 output heads are then applied in numpy at **full precision**,
so quantization never touches them:

- **Dense** = L2-normalized `last_hidden_state[:, 0]` (the `[CLS]` token).
- **Sparse** = `relu(last_hidden_state @ sparse_linear)` per token position, then
  **max-pooled per token id** with special tokens dropped. This is a faithful
  reimplementation of FlagEmbedding's token-weight processing, which is what
  guaranteed token-id parity with the (since removed) torch backend.

The ONNX model directory the loader reads contains: `model.onnx` (+
`model.int8.onnx`), `tokenizer.json`, `sparse_linear.npz` (the sparse head
weights), and `meta.json`. `embedding_int8: true` selects the INT8 file.

> **Tokenizer pinning matters.** The query must be tokenized exactly as the corpus
> was, or query and document vectors drift apart. The dependency pins
> (`tokenizers`, `transformers`) are deliberate for this reason — keep ingest and
> serve on the same tokenizer.

## The reranker

`rerank.OnnxReranker` loads BGE-reranker-v2-m3 from its own baked model dir
(`model.int8.onnx` + `tokenizer.json` + `meta.json`). It is an XLM-R
sequence-classification head emitting **one relevance logit per pair**; higher means
more relevant. The tokenizer carries the pair post-processor
(`<s> query </s></s> passage </s>`), so the server just hands it `(query, passage)`
tuples and reads back a logit per candidate.

A cross-encoder is far stronger than the bi-encoder embedder because it attends over
both texts at once — but for the same reason it cannot be run over a whole
collection. It runs only over the pooled retrieval candidates
([retrieval.md](retrieval.md#3-cross-encoder-rerank)).

## Producing the ONNX model dirs

The INT8 model dirs are produced **offline** by the export scripts in
[`tools/onnx-export/`](../tools/onnx-export/) — they pull the official BAAI weights
from Hugging Face, export via ONNX, and (with `--quantize`) write the INT8 copy:

```bash
python tools/onnx-export/export_bge_m3_onnx.py       --out ./models/bge-m3-onnx       --quantize
python tools/onnx-export/export_bge_reranker_onnx.py --out ./models/bge-reranker-onnx --quantize
```

The models are **baked in** — the serving container needs no network and no torch —
but since 2026-09 they are **not exported during an image build**. They live in a
separate models image (`images/models/Dockerfile`, tag
`localhost/docs-bridge-models:<version>`) that both app images copy from. Until then
each build re-exported them from whatever torch/transformers/onnx/onnxruntime it had
installed, which made the model a by-product of the dependency set: a dependency bump
would have silently produced a different `model.int8.onnx` under vectors the old one
had made. The current tag was taken byte-for-byte out of the images deployed before
the 2026-09 upgrade. A new model (e.g. the GPU fp16 reranker) is exported with the
scripts above and published as a new models tag. The models are host-invariant, so
baking them does not break the config-only portability model.

> The INT8 quantization step is memory-hungry (it loads the full-precision ONNX and
> works over copies of it). Run the export and the quantize as separate processes
> (`--quantize-only`) so their peaks don't stack.

## Tuning knobs

| Config | Effect |
|---|---|
| `embedding_backend` | `onnx` (the only backend). |
| `embedding_int8` | ONNX only: load `model.int8.onnx` vs `model.onnx`. |
| `onnx_model_dir` | Where the baked embedder model dir lives. |
| `ingest.batch_size` | Chunks per embed batch — the main ingest memory/throughput dial. |
| `rerank.enabled` | Turn the cross-encoder on/off (off → order by RRF score). |
| `rerank.int8` / `rerank.model_dir` / `rerank.max_length` | Reranker backend, model dir, and pair truncation length. |

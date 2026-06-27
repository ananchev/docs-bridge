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

## The two backends

`embed.get_embedder(cfg)` selects a backend from config and returns it behind a
common `encode(texts) -> (dense, sparse)` contract:

| `embedding_backend` | Module | Deps | When |
|---|---|---|---|
| `onnx` *(default)* | `embed_onnx.OnnxEmbedder` | onnxruntime, tokenizers | The validated default — runs the same model with no torch, optionally INT8-quantized for ~4–5× the CPU throughput of the torch path at on-par retrieval quality. |
| `flagembedding` | `embed.Embedder` | FlagEmbedding, torch | Reference/fallback. Useful for parity checks against the ONNX path. |

Imports are lazy, so only the chosen backend's heavy dependencies load. Both
backends emit sparse vectors in the **same BGE-M3 token-id space**, so a corpus
embedded by one backend stays queryable by the other.

### How the ONNX backend stays faithful

The strategy is *"encoder in ONNX, heads in numpy."* The big XLM-RoBERTa encoder is
exported to ONNX (and optionally INT8-quantized) — that is where the compute lives.
The two small BGE-M3 output heads are then applied in numpy at **full precision**,
so quantization never touches them:

- **Dense** = L2-normalized `last_hidden_state[:, 0]` (the `[CLS]` token).
- **Sparse** = `relu(last_hidden_state @ sparse_linear)` per token position, then
  **max-pooled per token id** with special tokens dropped. This is a faithful
  reimplementation of FlagEmbedding's token-weight processing, which is what
  guarantees token-id parity with the torch backend.

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

The application Dockerfiles run these in a throwaway `exporter` build stage and copy
only the INT8 files into the runtime image, so the models are **baked in** — the
serving container needs no network and no torch. The models are host-invariant, so
baking them does not break the config-only portability model; only host-*variable*
values stay in the mounted config.

> The INT8 quantization step is memory-hungry (it loads the full-precision ONNX and
> works over copies of it). On a low-RAM build host, export on a roomier machine and
> ship the resulting image, or split the export and quantize into separate processes
> so their peaks don't stack (which is what the Dockerfiles do).

## Tuning knobs

| Config | Effect |
|---|---|
| `embedding_backend` | `onnx` (default) or `flagembedding`. |
| `embedding_int8` | ONNX only: load `model.int8.onnx` vs `model.onnx`. |
| `onnx_model_dir` | Where the baked embedder model dir lives. |
| `ingest.batch_size` | Chunks per embed batch — the main ingest memory/throughput dial. |
| `rerank.enabled` | Turn the cross-encoder on/off (off → order by RRF score). |
| `rerank.int8` / `rerank.model_dir` / `rerank.max_length` | Reranker backend, model dir, and pair truncation length. |

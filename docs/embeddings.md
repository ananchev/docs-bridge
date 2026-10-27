# Embeddings and reranking

docs-bridge uses two models from the BGE family, each on a configured device and
precision — INT8 on CPU (the Mac mini), fp16/fp32 on CUDA (EC2):

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

## Where the embedder runs

`embed.get_embedder(cfg)` returns the embedder behind an `encode(texts) -> (dense,
sparse)` contract. The model always runs in `embed_onnx.OnnxEmbedder` (onnxruntime +
tokenizers, sessions opened by `ort_session.open_session`); `embed.mode` (required)
says in which process:

- **`service`** (EC2) — the **embedder service** (`embed_service.py`, image
  `images/embedder/`) runs it, and the worker and the server call it over HTTP
  (`embed.ServiceEmbedder`). Model, onnxruntime and precision exist in that one
  container, so documents and queries are embedded by one runtime by construction,
  and the worker and server images carry no embedder and upgrade without touching
  the vectors. The service can move to another instance: callers know only
  `embed.url`. Over HTTP the vectors are bit-identical to an in-process run.
- **`local`** (the CPU INT8 line) — `OnnxEmbedder` in the worker and the server
  process. Both must then run the *same* onnxruntime, tokenizers and numpy and the
  same model bytes, or query vectors drift from stored ones: both images install
  from locks that pin those identically (`images/lock.sh` checks it) and copy the
  model from one frozen artifact. A runtime bump is gated by
  `tools/eval/embed_upgrade_parity.py` against the stored vectors
  (`notes/phase-a-runbook.md` step 4).

The FlagEmbedding/torch backend was removed on 2026-09-28; its sparse vectors were
in the **same BGE-M3 token-id space**, so corpora it embedded stay queryable.

### The fingerprint

Every collection is stamped, in the manifest, with the **fingerprint** of the
embedder that filled it (`OnnxEmbedder.fingerprint`): the model id and revision,
the precision, the sha256 of every file that precision runs (for fp32 that includes
the 2.3 GB `model.onnx.data`) plus the sparse head and tokenizer, the execution
provider, and the onnxruntime build and tokenizers version. The worker refuses to
add vectors to a collection stamped with a different fingerprint — before it deletes
anything — so two embedders never mix in one collection; a new embedder means a new
collection. Where `meta.json` records a file's sha256, the bytes on disk must match.
In service mode every response carries the fingerprint's id, and a client whose
service changed under it refuses to continue.

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

The ONNX model directory the loader reads holds one file set per precision —
`model.onnx` + `model.onnx.data` (fp32), `model.fp16.onnx`, `model.int8.onnx` —
plus `tokenizer.json`, `sparse_linear.npz` (the sparse head weights) and
`meta.json`; the precision picks the file set.

> **Tokenizer pinning matters.** The query must be tokenized exactly as the corpus
> was, or query and document vectors drift apart. The dependency pins
> (`tokenizers`, `transformers`) are deliberate for this reason — keep ingest and
> serve on the same tokenizer.

## The reranker

`rerank.OnnxReranker` loads BGE-reranker-v2-m3 from its own baked model dir (the
file for `rerank.precision` + `tokenizer.json` + `meta.json`), in the server
process. It is an XLM-R
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
| `embed.mode` / `embed.url` | The embedder service, or in-process (`device`, `precision`, `model_dir`). |
| `ingest.batch_size` | Chunks per embed batch — the main ingest memory/throughput dial. |
| `rerank.enabled` | Turn the cross-encoder on/off (off → order by RRF score). |
| `rerank.device` / `rerank.precision` / `rerank.bucket` | Where and how the reranker runs (required). |
| `rerank.model_dir` / `rerank.max_length` | Reranker model dir and pair truncation length. |

# Architecture

docs-bridge is the **retrieval layer** of a RAG system. It turns a set of document
folders into a hybrid vector index and serves grounded, cited retrieval to an LLM.
Generation happens elsewhere — docs-bridge never calls an LLM.

## The three components

```
            ┌─────────────────────────────────────────────┐
            │                config.yaml                  │  (mounted, shared)
            └───────────────┬───────────────┬─────────────┘
                            │               │
   source docs              ▼               ▼
   on disk        ┌──────────────┐   ┌──────────────┐
   /data/docs/* ─▶│ ingest-worker│   │ docs-bridge  │◀─ MCP /mcp
                  │  (one-shot   │   │   server     │◀─ REST /v1/*
                  │   CLI)       │   │ (ASGI, long- │
                  └──────┬───────┘   │  running)    │
            parse+embed  │           └──────┬───────┘
                         ▼  upsert          │  hybrid search + rerank
                  ┌─────────────────────────▼──────┐
                  │             Qdrant             │
                  │   one collection per subject   │
                  └────────────────────────────────┘
```

| Component | Image | Lifetime | Responsibility |
|---|---|---|---|
| **ingest-worker** | `ingest-worker` | one-shot (`ingest sync`, then exits) | Detect changed files, parse them, embed the chunks, upsert to Qdrant, update the manifest. |
| **docs-bridge server** | `docs-bridge` | long-running ASGI service | Embed the query, hybrid-retrieve from Qdrant, rerank, return cited chunks over MCP and REST. |
| **Qdrant** | `qdrant` (built from source) | long-running | Vector store: named dense + sparse vectors per chunk, one collection per subject. |

Both application images are built from the **same repository** and install the
**same `docs_bridge` Python package**. They differ only in their extra
dependencies (the worker pulls Docling + torch; the server pulls FastMCP +
uvicorn) and their entrypoint.

## Why one shared core

The single most important property of a RAG retriever is that **query vectors and
document vectors live in the same space**. docs-bridge guarantees this structurally:
the worker and the server import the same `get_embedder()` and load the same
BGE-M3 ONNX model. There is no second embedding implementation to drift out of sync.

The retrieve→rerank logic is likewise a single code path (`search.Searcher`) shared
by the MCP tool and the REST endpoint, so those two surfaces cannot diverge either.

## The `docs_bridge` package

| Module | Role |
|---|---|
| `config.py` | Loads and validates `config.yaml` into typed dataclasses. The whole host-portability story is config-only — nothing host-specific is baked into an image. |
| `models.py` | Plain dataclasses shared across passes: `Subject`, `Chunk`, `DocState`, `SyncStats`. |
| `parse.py` | **Ingest pass 1.** Scans the subject dirs (with include/exclude filters), parses each file with Docling, and emits structure-aware `Chunk`s. |
| `embed.py` / `embed_onnx.py` | **Ingest pass 2 / query embedding.** BGE-M3 → dense + sparse vectors. `embed_onnx` is the default ONNX/INT8 backend; `embed` is the FlagEmbedding/torch fallback. `get_embedder()` picks one from config. |
| `manifest.py` | The SQLite manifest (persistent `docs` table for hash-delta detection) and the per-run `staged_chunks` scratch table. |
| `qdrant_io.py` | Collection creation, idempotent upserts, doc-scoped deletes. Owns the Qdrant payload/vector layout. |
| `sync.py` | The ingest orchestrator: scan → classify → pass 1 → pass 2. |
| `search.py` | The server's retrieve→rerank core (`Searcher`). |
| `rerank.py` | The ONNX/INT8 cross-encoder reranker. |
| `server.py` | Assembles the ASGI app: FastMCP tools + REST routes behind a bearer gate. |
| `rest.py` | The REST mirror of the MCP surface. |
| `cli.py` | The `ingest` console-script entrypoint. |

## Data flow end to end

1. **Ingest** ([ingestion.md](ingestion.md)) — `ingest sync` walks the subject
   directories, compares file hashes against the manifest, parses the new/changed
   files into chunks, embeds them, and upserts dense+sparse vectors into the
   subject's Qdrant collection. Deleted files have their points removed.
2. **Serve** ([retrieval.md](retrieval.md)) — a `search(subject, query, k)` call
   embeds the query with the same model, runs a dense and a sparse prefetch against
   each target collection, fuses them with RRF, reranks the pooled candidates with
   the cross-encoder, and returns the top `k` chunks with citations.
3. **Consume** ([server.md](server.md)) — an MCP client (or anything that can POST
   JSON) gets cited chunks back and hands them to an LLM for synthesis.

## Design properties worth knowing

- **Two-pass ingest, never co-resident.** Parsing (Docling) and embedding (BGE-M3)
  are heavy, separate model stacks. The worker fully releases the parser and stages
  chunks to SQLite before loading the embedder, so peak memory is bounded by the
  larger of the two, not their sum. See [ingestion.md](ingestion.md).
- **Idempotent, incremental sync.** A re-run with no file changes reports
  `0 new / 0 changed / 0 deleted` and touches nothing. This is the operational
  contract a scheduler relies on.
- **CPU-only serving.** Both serve-time models (embedder, reranker) run as
  ONNX/INT8 through onnxruntime. The server image carries no torch and no Docling.
- **Hybrid by construction.** BGE-M3 emits dense and sparse vectors from one
  forward pass, so the lexical signal is a real learned sparse vector — not a
  bolted-on BM25 over a separate index. See [embeddings.md](embeddings.md).
- **Config-only portability.** Host-varying knobs (batch size, on-disk vectors,
  quantization) live in the mounted `config.yaml`; the images are host-agnostic.

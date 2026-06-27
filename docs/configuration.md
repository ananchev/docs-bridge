# Configuration reference

A single `config.yaml` is **shared by both images** and mounted at
`/config/config.yaml` (override the path with the `DOCS_BRIDGE_CONFIG` env var).
The ingest-worker ignores the `rerank`/`server` blocks; the server ignores the
`parse`/`ingest` blocks. Start from [`config.example.yaml`](../config.example.yaml);
the loader and defaults live in `config.py`.

Host-varying values (batch size, on-disk vectors, quantization) belong here, not in
the image — that is the whole portability model: change the config, not the code.

## Top-level

| Key | Default | Notes |
|---|---|---|
| `embedding_model` | `BAAI/bge-m3` | The embedding model identity. The stack is built around BGE-M3; changing it implies re-exporting the ONNX dirs. |
| `embedding_dim` | `1024` | Dense vector size. Must match the model and the Qdrant collection. |
| `embedding_backend` | `onnx` | `onnx` (default) or `flagembedding`. See [embeddings.md](embeddings.md). |
| `embedding_int8` | `true` | ONNX only: load `model.int8.onnx` vs `model.onnx`. |
| `onnx_model_dir` | `/opt/bge-m3-onnx` | Where the baked embedder model dir lives (set by the image). |
| `manifest_path` | `/data/state/manifest.sqlite` | The SQLite manifest + staging DB. |
| `suffixes` | `.pdf .html .htm .md .docx .pptx` | File suffixes considered for ingest. |
| `include` / `exclude` | `[]` | Global glob filters, unioned with each subject's own. See [filters](#file-filters). |

## `chunk` (ingest)

| Key | Default | Notes |
|---|---|---|
| `target_tokens` | `400` | Token budget the HybridChunker aims for. |
| `overlap` | `60` | Chunk overlap. |
| `strategy` | `structure_aware` | Heading-aware chunking. |

## `parse` (ingest)

| Key | Default | Notes |
|---|---|---|
| `ocr` | `false` | EasyOCR is slow on CPU and needless for digital PDFs. Enable only for scanned corpora. |
| `table_structure` | `true` | Keep table detection — tables carry real content. |

## `qdrant`

| Key | Default | Notes |
|---|---|---|
| `host` | `qdrant` | Service name on the container network. |
| `port` | `6333` | |
| `on_disk_vectors` | `false` | `false` for a small corpus; `true` to keep vectors on disk as the corpus grows. |
| `quantization` | `none` | `none` or `scalar` (INT8 scalar quantization, kept in RAM). |

## `ingest`

| Key | Default | Notes |
|---|---|---|
| `batch_size` | `8` | Chunks per embed batch — the main memory/throughput dial. Raise it on a roomier host. |
| `two_pass` | `true` | Stage chunks to SQLite and release Docling before loading the embedder. |

## `rerank` (server only)

| Key | Default | Notes |
|---|---|---|
| `enabled` | `true` | Off → results ordered by RRF fusion score. |
| `model_dir` | `/opt/bge-reranker-onnx` | Baked INT8 cross-encoder dir. |
| `int8` | `true` | Load the INT8 reranker model. |
| `max_length` | `512` | Truncation length for each `(query, passage)` pair. |
| `top_n` | `30` | Candidate budget for a **single-subject** search. |
| `multi_top_n` | `60` | Candidate budget when a search **spans multiple** subjects. See [retrieval.md](retrieval.md#multi-subject-search). |

## `server` (server only)

| Key | Default | Notes |
|---|---|---|
| `host` | `0.0.0.0` | Bind address. |
| `port` | `8080` | |
| `default_k` | `6` | Chunks returned when `search` is called without `k`. |
| `prefetch_limit` | `50` | Dense / sparse candidates fetched **each** before RRF fusion. |
| `instructions` | `""` | MCP `instructions` policy string (grounding + citation + language). See [server.md](server.md#steering-the-consuming-llm). |

The static bearer token is **not** in the config — it is the `DOCS_BRIDGE_TOKEN`
environment variable.

## `subjects` (required)

A list of independent corpora. Each is a source directory mapped to a Qdrant
collection. At least one is required.

```yaml
subjects:
  - name: teamcenter            # the subject id used in search(subject=...)
    dir: /data/docs/teamcenter  # source directory (under the mounted /data volume)
    collection: teamcenter      # Qdrant collection name
    description: >              # optional: what this corpus is (surfaced to the LLM)
      Product installation, configuration, and administration guides.
    include: []                 # optional per-subject whitelist globs
    exclude:                    # optional per-subject blacklist globs
      - "api/index.html"
```

| Field | Required | Notes |
|---|---|---|
| `name` | yes | The id passed to `search`/`list_subjects`. |
| `dir` | yes | Source directory; scanned recursively. |
| `collection` | yes | Qdrant collection name. |
| `description` | no | Surfaced via `list_subjects` and composed into the MCP catalog so the model can pick the right subject. |
| `include` / `exclude` | no | Per-subject glob filters, unioned with the global ones. |

### File filters

`include`/`exclude` exist at two levels — global (top-level) and per-subject — and
are **unioned**:

- **`include`** is a whitelist: if any include pattern exists at either level, a
  file must match one.
- **`exclude`** is a blacklist and **wins** over include.

Patterns match the `doc_id` (the file's posix path relative to its subject `dir`)
with `fnmatch`, so `*` spans `/`. Omit both to ingest every suffix-matched file.
See the Doxygen example in [`config.example.yaml`](../config.example.yaml).

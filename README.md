# docs-bridge

A self-updating, multi-subject **documentation RAG stack** — the *retrieval* half
of a RAG system, packaged as container images plus a documented config contract.
It ingests document corpora, keeps a hybrid vector index fresh, and serves
grounded, **cited** retrieval to any LLM over **MCP** or **REST**.

No LLM runs here. docs-bridge does retrieval; your model does generation. That
keeps the stack small enough to run on a single CPU host with no GPU.

## What it does

- **Ingests** PDF / HTML / DOCX / PPTX / Markdown into structure-aware chunks
  that remember their heading path (via [Docling](https://github.com/DS4SD/docling)).
- **Embeds** each chunk with **BGE-M3**, which yields a **dense + sparse** vector
  in a single pass — one model gives you both semantic and lexical signal.
- **Stores** them in **[Qdrant](https://qdrant.tech)**, one collection per subject.
- **Serves hybrid search**: a dense and a sparse retrieval are fused with
  Reciprocal Rank Fusion, then a **cross-encoder reranks** the top candidates.
  Results come back as chunks with citations (source path, section, timestamp).
- **Stays fresh**: an incremental **hash-delta** sync classifies every file as
  new / changed / deleted and only re-parses what moved. Re-running is idempotent.
- **Multi-subject**: N independent corpora, each its own directory and collection.
  A query can target one subject, a list of them, or `all`.
- **CPU-only at serve time**: the embedder and reranker run as **ONNX/INT8** — no
  torch, no GPU, no network calls.

## How it works

```
        ┌───────────────┐   parse + embed   ┌──────────┐
 docs ─▶│ ingest-worker │ ────────────────▶ │  Qdrant  │
        │  (one-shot)   │   (hash-delta)    │ (vectors)│
        └───────────────┘                   └────┬─────┘
                                                 │ hybrid search
        ┌───────────────┐   search / rerank      │
 MCP ──▶│ docs-bridge   │ ◀──────────────────────┘
 REST ─▶│   server      │ ──▶ cited chunks ──▶ your LLM
        └───────────────┘
```

Two deployable images plus Qdrant, all sharing one `docs_bridge` Python core:

| Component | What it is |
|---|---|
| **ingest-worker** | A one-shot CLI (`ingest sync`). Parses changed docs, embeds them, upserts to Qdrant, updates a manifest. Run it on a schedule for incremental updates. |
| **docs-bridge server** | A long-running ASGI service exposing the MCP tools (`search`, `list_subjects`) and a REST mirror. Hybrid retrieve → rerank → cite. |
| **Qdrant** | The vector store. Built from source so it runs on 16 KB-page kernels (see [`images/qdrant`](images/qdrant/)). |

The two images share the `docs_bridge` package so the **query embedder is byte-for-byte
the same model as the ingest embedder** — query and document vectors live in the
same space. See [`docs/architecture.md`](docs/architecture.md) for the full picture.

## Quick start

Prerequisites: a container runtime (Docker or Podman with its Docker-compatible
socket) on a Linux host. The images are built for `aarch64`/`arm64`.

```bash
# 1. Configure: copy the example and edit the subjects + paths.
cp config.example.yaml config.yaml

# 2. Build the three images (build context is the repo root for the two app images).
docker build -f images/ingest-worker/Dockerfile -t ingest-worker:latest .
docker build -f images/docs-bridge/Dockerfile  -t docs-bridge:latest  .
docker build -t qdrant:local images/qdrant

# 3. Put the vector store and the apps on one network.
docker network create docs-bridge
docker run -d --name qdrant --network docs-bridge -v qdrant-data:/qdrant/storage qdrant:local

# 4. Drop source docs under the subject dirs you configured (e.g. ./data/docs/<subject>/),
#    then run an incremental sync. The worker exits when done.
docker run --rm --network docs-bridge \
  -v "$PWD/config.yaml:/config/config.yaml:ro" -v "$PWD/data:/data" \
  ingest-worker:latest sync --subject all

# 5. Serve. DOCS_BRIDGE_TOKEN gates every route except /healthz.
docker run -d --name docs-bridge --network docs-bridge -p 8080:8080 \
  -e DOCS_BRIDGE_TOKEN=change-me \
  -v "$PWD/config.yaml:/config/config.yaml:ro" -v "$PWD/data:/data" \
  docs-bridge:latest

# 6. Query (REST mirror of the MCP search tool).
curl -s localhost:8080/v1/search -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{"subject": "all", "query": "how do I configure X?", "k": 6}'
```

Point any MCP client at `http://<host>:8080/mcp` (streamable HTTP, same bearer
token) to expose `search` and `list_subjects` as tools to an LLM.

> **The `ingest-worker` is a one-shot CLI by design** — schedule it (systemd timer,
> cron, k8s CronJob, …) for nightly incremental syncs. It runs the parse and embed
> passes one after another so the two heavy model stacks are never resident at once.

## Configuration

A single `config.yaml` is **shared by both images** (mounted at
`/config/config.yaml`). The ingest-worker ignores the `rerank`/`server` blocks;
the server ignores the `parse`/`ingest` blocks. Start from
[`config.example.yaml`](config.example.yaml); every key is documented in
[`docs/configuration.md`](docs/configuration.md).

## Documentation

| Doc | Covers |
|---|---|
| [architecture.md](docs/architecture.md) | The components, the shared core, and how data flows through the stack. |
| [ingestion.md](docs/ingestion.md) | The two-pass hash-delta sync: scan → classify → parse → embed → upsert. |
| [retrieval.md](docs/retrieval.md) | The search path: query embed → hybrid RRF → cross-encoder rerank → citations; multi-subject search. |
| [embeddings.md](docs/embeddings.md) | BGE-M3 dense+sparse, the ONNX/INT8 backends, the reranker, and the offline export tools. |
| [data-model.md](docs/data-model.md) | The Qdrant collection layout and payload, deterministic point ids, and the SQLite manifest schema. |
| [configuration.md](docs/configuration.md) | Reference for every `config.yaml` key. |
| [server.md](docs/server.md) | The MCP + REST surface, auth, and how the server steers a consuming LLM. |

Helper scripts (corpus prep, ONNX export, evaluation harnesses) live under
[`tools/`](tools/) — they are not part of the runtime package.

## Deployment

docs-bridge is **not self-deploying**. This repo is the *product*: container
images, the config contract, and the application source. Deployment concerns —
secrets, runtime install, the sync schedule, and backup/restore — are owned by a
separate **operator** (automation) repo that treats docs-bridge as just-another-app.

**The boundary is a container image plus a mounted `config.yaml`.** Nothing else
crosses it: the operator never needs the app internals, and this repo never needs
the inventory/secrets/backup topology. That keeps docs-bridge portable across
hosts with config changes only — no code or image rebuild.

## Repository layout

```
src/docs_bridge/   the shared RAG core (config, parse, embed, qdrant_io, search, server)
images/            Dockerfiles: ingest-worker, docs-bridge server, qdrant (16K-page build)
tools/             offline helpers: onnx-export, corpus ingest, eval harnesses
docs/              architecture & app-logic documentation
config.example.yaml the config contract, annotated
```

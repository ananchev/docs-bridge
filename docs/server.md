# The server: MCP + REST surface

The docs-bridge server (`docs-bridge-server`) is one ASGI app exposing the same
retrieval core through two surfaces, both behind a static bearer token. Source:
`server.py`, `rest.py`, `search.py`.

```
                    ┌─────────────── BearerAuth (ASGI) ───────────────┐
  MCP client  ──▶   │  /mcp        FastMCP (streamable HTTP)          │
  HTTP client ──▶   │  /v1/search  REST  ─┐                           │
                    │  /v1/subjects REST ─┼─▶ one Searcher instance   │
                    │  /healthz    open ──┘   (models loaded once)    │
                    └─────────────────────────────────────────────────┘
```

A single `Searcher` backs both surfaces, so the MCP tools and the REST endpoints
run the **identical** retrieve→rerank path and cannot diverge.

## MCP surface (`/mcp`)

Streamable-HTTP MCP via FastMCP. Two tools:

### `list_subjects() -> list[dict]`

Lists the available corpora and whether each has a populated vector collection
(live point count from Qdrant). Callers — and the LLM — use this to discover what
exists and pick the right `subject` before searching.

### `search(subject, query, k=default_k) -> list[dict]`

Hybrid-retrieve then rerank the most relevant chunks for `query`. `subject` is a
single name, a **list** of names (to span related corpora), or `"all"`. Results are
reranked globally across the chosen pools, each tagged with its `subject`, and
cited (`source_path`, `section`, `last_updated`). Retrieval only — **no LLM
synthesis**. See [retrieval.md](retrieval.md).

## REST mirror

A thin mirror of the same surface, useful for health checks and `curl`-based
validation (a green REST smoke test means the MCP tool is green too, since they
share the `Searcher`):

| Route | Method | Maps to |
|---|---|---|
| `/healthz` | GET | liveness — **the only open route** |
| `/v1/subjects` | GET | `list_subjects()` |
| `/v1/search` | POST | `search()` — JSON body `{subject, query, k?}` |

`/v1/search` validates that `subject` and `query` are present (400 otherwise),
returns 404 for an unknown subject, and runs the (synchronous) Searcher in a
threadpool so it never blocks the event loop or the MCP stream.

```bash
curl -s localhost:8080/v1/search -H 'Authorization: Bearer <token>' \
  -H 'Content-Type: application/json' \
  -d '{"subject": ["teamcenter","power-query"], "query": "...", "k": 6}'
```

## Authentication

A pure-ASGI bearer gate (`BearerAuth`) wraps the whole app:

- The token is read from the **`DOCS_BRIDGE_TOKEN`** environment variable.
- Every route **except `/healthz`** requires `Authorization: Bearer <token>`.
- It **fails closed**: a missing or non-matching token → `401`. If
  `DOCS_BRIDGE_TOKEN` is unset, every non-health request 401s (and the server logs a
  warning at startup).
- It is implemented as pure ASGI (not `BaseHTTPMiddleware`) specifically so it never
  buffers the MCP SSE stream.

The gate is a clean seam: swapping the static token for an OAuth flow would change
only this layer, nothing in the retrieval core.

## Steering the consuming LLM

Retrieval quality is only half the job — the model also has to *use* the results
well (ground its answer, cite, answer in the user's language). docs-bridge exposes
two config-driven channels for that:

- **`server.instructions`** — a generic policy string returned in the MCP
  `initialize` handshake.
- **Per-subject `description`s** — the single source of truth for what each corpus
  contains. The server auto-composes them into an "Available corpora" catalog
  appended to the instructions, and surfaces them through `list_subjects` and the
  `search` tool docstring. So a new subject becomes self-describing to the model
  with no policy-text change.

Both are set in `config.yaml`, so guidance is tunable by re-rendering the config and
restarting — no image rebuild.

> **Note on MCP clients:** not every MCP client injects a server's `instructions`
> into the model's context. The most portable place for model-facing guidance is the
> **tool channel** — tool docstrings and tool *output* (like `list_subjects`) — which
> clients reliably surface. The `description`-driven catalog is designed to ride that
> channel, not rely on `instructions` alone.

## Running it

```bash
docker run -d --name docs-bridge --network docs-bridge -p 8080:8080 \
  -e DOCS_BRIDGE_TOKEN=change-me \
  -v ./config.yaml:/config/config.yaml:ro -v ./data:/data \
  docs-bridge:latest
```

The server binds `server.host:server.port` (default `0.0.0.0:8080`) and loads its
config from `/config/config.yaml` (override via `DOCS_BRIDGE_CONFIG`). It carries no
torch and no Docling — query embedding and reranking go through the baked ONNX/INT8
models. See [embeddings.md](embeddings.md).

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

- Valid tokens come from **`DOCS_BRIDGE_TOKENS`** (plural — comma- or
  whitespace-separated, **one secret per client**) unioned with **`DOCS_BRIDGE_TOKEN`**
  (singular, kept for back-compat with single-token deployments).
- Every route **except `/healthz`** requires `Authorization: Bearer <token>` matching
  one of the configured tokens.
- The match is **constant-time** (`hmac.compare_digest`) so response timing can't leak
  token bytes.
- It **fails closed**: a missing or non-matching token → `401`. If no tokens are set,
  every non-health request 401s (and the server logs a warning at startup).
- It is implemented as pure ASGI (not `BaseHTTPMiddleware`) specifically so it never
  buffers the MCP SSE stream.

**Per-client tokens** let each consumer carry its own secret — e.g. LibreChat, a local
Claude Code / opencode, a teammate — so any one can be **revoked independently** by
dropping it from `DOCS_BRIDGE_TOKENS` and restarting, without disturbing the others.
Mint them as random secrets (`openssl rand -hex 32`) and keep them in the operator's
vault, not in `config.yaml` (secrets never go in the rendered config).

The gate is a clean seam: swapping the static tokens for an OAuth flow would change
only this layer, nothing in the retrieval core.

## Connecting a local coding agent (Claude Code / opencode)

Both Claude Code and opencode can consume docs-bridge as a **remote MCP server over
streamable HTTP with a bearer header** — no server-side change beyond giving the client
its own token. The `search` / `list_subjects` tools then appear in the agent, so it can
retrieve from the Teamcenter corpus while you code (the agent's own model backend — e.g.
OpenRouter — is unaffected; docs-bridge only provides the *tools*, never the LLM).

> **Reachability:** the server is **LAN-only** and reached over the home **VPN**; point
> clients at the vhost2 LAN address (e.g. `http://192.168.2.199:8080/mcp`). This requires
> the container to publish `8080` on a VPN-reachable interface (an operator/deploy
> concern — the default `127.0.0.1:8080` publish is loopback-only and not reachable from
> the laptop). There is **no public edge**; off-VPN there is no access.
>
> **Likely future change:** if VPN-only proves inconvenient, this can be exposed publicly
> through the existing NPM + Cloudflare (mTLS / CF Access) pattern, keeping the bearer
> behind that edge. Not built yet — noted here as the anticipated next step.

**Claude Code** (user scope shown; drop `--scope user` for the current project):

```bash
claude mcp add --transport http docs-bridge \
  http://192.168.2.199:8080/mcp \
  --header "Authorization: Bearer <your-laptop-token>" \
  --scope user
```

or equivalently in `.mcp.json` / `~/.claude.json`:

```json
{
  "mcpServers": {
    "docs-bridge": {
      "type": "http",
      "url": "http://192.168.2.199:8080/mcp",
      "headers": { "Authorization": "Bearer <your-laptop-token>" }
    }
  }
}
```

**opencode** — in `opencode.json` (project) or `~/.config/opencode/opencode.json`
(global):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "docs-bridge": {
      "type": "remote",
      "url": "http://192.168.2.199:8080/mcp",
      "enabled": true,
      "headers": { "Authorization": "Bearer <your-laptop-token>" }
    }
  }
}
```

Sanity-check the path end-to-end before wiring the client (run this on the VPN):

```bash
curl -s http://192.168.2.199:8080/v1/subjects \
  -H 'Authorization: Bearer <your-laptop-token>' | jq
# 200 + subject list = reachable & authorized;  401 = token/exposure problem
```

Then, in the agent, `list_subjects` should show the populated `teamcenter` corpus and
`search` should return cited chunks.

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
  -e DOCS_BRIDGE_TOKENS="librechat-token,laptop-token" \
  -v ./config.yaml:/config/config.yaml:ro -v ./data:/data \
  docs-bridge:latest
```

The server binds `server.host:server.port` (default `0.0.0.0:8080`) and loads its
config from `/config/config.yaml` (override via `DOCS_BRIDGE_CONFIG`). It carries no
torch and no Docling — query embedding and reranking go through the baked ONNX/INT8
models. See [embeddings.md](embeddings.md).

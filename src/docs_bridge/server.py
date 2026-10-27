"""docs-bridge server: FastMCP tools + REST mirror on one ASGI app (design §9/§15).

Surface (all behind a static bearer token, except /healthz):
  - MCP (streamable-http) at /mcp   ->  tools: list_subjects(), search()
  - REST mirror                     ->  GET /healthz, GET /v1/subjects, POST /v1/search

Auth is static bearer tokens (DOCS_BRIDGE_TOKENS / DOCS_BRIDGE_TOKEN) — the design's
chosen model: the data-compliant clients send a static header (§14). A SET of tokens
is accepted, one per client (LibreChat, a local Claude Code / opencode, ...), each
independently revocable. It is kept as a clean ASGI middleware seam so that IF a
consumer MCP client (claude.ai) is ever required, only this layer swaps to the OAuth
`mcp-auth` AS — nothing else changes.

NOTE (verify on the Pi): the FastMCP integration points used here are `mcp.http_app(
path=...)` returning a Starlette app and `.lifespan`. Pin fastmcp exactly once these
are confirmed against the installed version.
"""

from __future__ import annotations

import hmac
import logging
import os

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount

from . import config
from .config import Config
from .rest import build_rest_routes
from .search import Searcher

log = logging.getLogger(__name__)


def build_mcp(searcher: Searcher, cfg: Config):
    """The MCP tool surface (design §9). Thin wrappers over the shared Searcher so
    the MCP tools and the REST mirror are the identical retrieve->rerank path."""
    from fastmcp import FastMCP

    # The MCP `instructions` steer the consuming LLM. Composed from the operator's
    # generic policy (cfg.server.instructions: grounding + citation + language) PLUS an
    # auto-generated catalog of corpora built from the subject descriptions — so the
    # policy text stays corpus-agnostic and a new subject appears here automatically
    # (single source of truth = Subject.description). See ServerCfg.instructions.
    catalog = "\n".join(
        f"- {s.name}: {s.description}" for s in cfg.subjects if s.description
    )
    parts = [cfg.server.instructions.strip()]
    if catalog:
        parts.append(
            "Choose the `subject` whose description best matches the question. "
            "Available corpora:\n" + catalog
        )
    instructions = "\n\n".join(p for p in parts if p) or None
    mcp = FastMCP(name="docs-bridge", instructions=instructions)

    @mcp.tool
    def list_subjects() -> list[dict]:
        """List the available documentation corpora (subjects) and whether each has
        a populated vector collection."""
        return searcher.list_subjects()

    @mcp.tool
    def search(
        subject: str | list[str], query: str, k: int = cfg.server.default_k
    ) -> list[dict]:
        """Hybrid-retrieve (dense+sparse) then rerank the most relevant chunks for
        `query`. `subject` selects the corpus/corpora: a single subject name, a LIST of
        names to span related pools, or "all" to search every pool. Call list_subjects
        first to see the available subjects and their descriptions (incl. which relate);
        for a cross-cutting question, pass the several relevant subjects. Results are
        reranked GLOBALLY across the chosen pools and each chunk is tagged with its
        `subject`; cited (source_path, section, last_updated) — retrieval only, no
        LLM synthesis. Across several pools, a passage whose text is identical in more
        than one is returned ONCE: `subjects` lists every pool it was found in and
        `also_in` their citations — e.g. text unchanged between two product versions.
        A pool missing from `subjects` does NOT mean the text differs there."""
        return searcher.search(subject, query, k)

    return mcp


def _load_tokens() -> set[str]:
    """Valid bearer tokens, unioned from DOCS_BRIDGE_TOKENS (plural: comma- or
    whitespace-separated, one secret per client) and DOCS_BRIDGE_TOKEN (singular,
    kept for back-compat with the original single-token deployment). Per-client
    tokens let each consumer (LibreChat, a local Claude Code / opencode, a teammate)
    be revoked independently instead of sharing one secret."""
    raw = " ".join(
        os.environ.get(k, "") for k in ("DOCS_BRIDGE_TOKENS", "DOCS_BRIDGE_TOKEN")
    )
    return {t for t in raw.replace(",", " ").split() if t}


class BearerAuth:
    """Pure-ASGI bearer-token gate. Pure ASGI (not BaseHTTPMiddleware) so it never
    buffers the MCP SSE stream. Fails CLOSED: a protected path whose Authorization
    header matches none of the configured tokens -> 401; /healthz (and any
    open_paths) pass through. Accepts a SET of tokens (one per client); the match is
    constant-time (hmac.compare_digest) so response timing can't leak token bytes."""

    def __init__(self, app, tokens: set[str], open_paths: set[str]) -> None:
        self.app = app
        self.expected = [f"Bearer {t}" for t in tokens]
        self.open_paths = open_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "") not in self.open_paths:
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"").decode()
            if not any(hmac.compare_digest(auth, e) for e in self.expected):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


def build_app(cfg: Config):
    """Assemble the combined ASGI app: REST routes + the MCP app mounted at /mcp,
    wrapped in the bearer gate. One Searcher instance -> models loaded once."""
    searcher = Searcher(cfg)
    mcp = build_mcp(searcher, cfg)
    mcp_app = mcp.http_app(path="/mcp")  # Starlette app exposing MCP at /mcp

    # REST routes are matched first; the Mount is the catch-all that carries /mcp.
    # The MCP app owns the lifespan (its streamable-http session manager), so the
    # parent must run it.
    app = Starlette(
        routes=[*build_rest_routes(searcher), Mount("/", app=mcp_app)],
        lifespan=mcp_app.lifespan,
    )

    tokens = _load_tokens()
    if not tokens:
        log.warning(
            "no bearer tokens set (DOCS_BRIDGE_TOKENS / DOCS_BRIDGE_TOKEN) — "
            "all non-health requests will 401"
        )
    else:
        log.info("bearer auth: %d token(s) loaded", len(tokens))
    return BearerAuth(app, tokens=tokens, open_paths={"/healthz"})


def serve() -> None:
    """Console-script entrypoint (`docs-bridge-server`)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = config.load()
    app = build_app(cfg)

    import uvicorn

    log.info("docs-bridge server on %s:%s", cfg.server.host, cfg.server.port)
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    serve()

"""docs-bridge server: FastMCP tools + REST mirror on one ASGI app (design §9/§15).

Surface (all behind a static bearer token, except /healthz):
  - MCP (streamable-http) at /mcp   ->  tools: list_subjects(), search()
  - REST mirror                     ->  GET /healthz, GET /v1/subjects, POST /v1/search

Auth is a static bearer token (DOCS_BRIDGE_TOKEN) — the design's chosen model: the
data-compliant client is LibreChat, which sends a static header (§14). It is kept as
a clean ASGI middleware seam so that IF a consumer MCP client (claude.ai) is ever
required, only this layer swaps to the OAuth `mcp-auth` AS — nothing else changes.

NOTE (verify on the Pi): the FastMCP integration points used here are `mcp.http_app(
path=...)` returning a Starlette app and `.lifespan`. Pin fastmcp exactly once these
are confirmed against the installed version.
"""

from __future__ import annotations

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

    mcp = FastMCP(name="docs-bridge")

    @mcp.tool
    def list_subjects() -> list[dict]:
        """List the available documentation corpora (subjects) and whether each has
        a populated vector collection."""
        return searcher.list_subjects()

    @mcp.tool
    def search(subject: str, query: str, k: int = cfg.server.default_k) -> list[dict]:
        """Hybrid-retrieve (dense+sparse) then rerank the most relevant chunks for
        `query` within `subject`. Returns cited chunks (source_path, section,
        last_updated) — retrieval only, no LLM synthesis."""
        return searcher.search(subject, query, k)

    return mcp


class BearerAuth:
    """Pure-ASGI bearer-token gate. Pure ASGI (not BaseHTTPMiddleware) so it never
    buffers the MCP SSE stream. Fails CLOSED: a protected path with no/!matching
    token -> 401; /healthz (and any open_paths) pass through."""

    def __init__(self, app, token: str, open_paths: set[str]) -> None:
        self.app = app
        self.token = token
        self.open_paths = open_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "") not in self.open_paths:
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"").decode()
            if not self.token or auth != f"Bearer {self.token}":
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

    token = os.environ.get("DOCS_BRIDGE_TOKEN", "")
    if not token:
        log.warning("DOCS_BRIDGE_TOKEN is not set — all non-health requests will 401")
    return BearerAuth(app, token=token, open_paths={"/healthz"})


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

"""Thin, bearer-authed REST mirror of the MCP surface (design §15).

The Pi has no chat UI, so validation is curl against these routes — they wrap the
exact same Searcher path as the MCP `search()` tool, so a green REST smoke test
means the MCP tool is green too. Auth is enforced by the ASGI middleware in
server.py (these handlers assume the request already passed it), except /healthz
which the middleware leaves open.

Searcher calls are synchronous (qdrant-client + onnxruntime), so they run in a
threadpool to avoid blocking the event loop / the MCP SSE stream.
"""

from __future__ import annotations

import logging

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .search import Searcher

log = logging.getLogger(__name__)


def build_rest_routes(searcher: Searcher) -> list[Route]:
    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def subjects(_: Request) -> JSONResponse:
        data = await run_in_threadpool(searcher.list_subjects)
        return JSONResponse({"subjects": data})

    async def search(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        subject, query = body.get("subject"), body.get("query")
        if not subject or not query:
            return JSONResponse(
                {"error": "both 'subject' and 'query' are required"}, status_code=400
            )
        try:
            results = await run_in_threadpool(
                searcher.search, subject, query, body.get("k")
            )
        except KeyError as e:  # unknown subject
            return JSONResponse({"error": str(e)}, status_code=404)
        except Exception:
            log.exception("search failed")
            return JSONResponse({"error": "internal error"}, status_code=500)
        return JSONResponse({"results": results})

    return [
        Route("/healthz", healthz, methods=["GET"]),
        Route("/v1/subjects", subjects, methods=["GET"]),
        Route("/v1/search", search, methods=["POST"]),
    ]

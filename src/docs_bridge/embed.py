"""Pass 2 - embed.

BGE-M3 produces a dense vector AND a sparse (lexical) vector in a single forward
pass, which is exactly the hybrid signal we store in Qdrant (design §12.4: dense
+ sparse in one model, not a separate BM25). The model runs in
`embed_onnx.OnnxEmbedder`; this module keeps the pieces both callers share -- the
`SparseVec` container, the `get_embedder` factory the ingest worker and the server
both call, and `ServiceEmbedder`, the client of the embedder service.

Where the model runs is configured (`embed.mode`, config.EmbedCfg), never defaulted:
  service  the embedder service (embed_service.py) -- EC2. Model, runtime and
           precision live in that one container, so the worker and the server cannot
           disagree about them, and they upgrade without touching the vectors
           (stack-upgrade-strategy.md §7 item 1).
  local    OnnxEmbedder in this process -- the CPU INT8 line.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)


class EmbedderProto(Protocol):
    """The encode contract every embedder exposes."""

    def encode(self, texts: list[str]) -> tuple[list[list[float]], list["SparseVec"]]:
        ...

    @property
    def fingerprint(self) -> dict:
        """Identity of everything that decides the vectors (stamped per collection)."""
        ...


def fingerprint_id(fingerprint: dict) -> str:
    """A short, stable name for a fingerprint: what the service tags every response
    with, so a client notices when the embedder behind the address changed."""
    blob = json.dumps(fingerprint, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def get_embedder(cfg: "Config") -> EmbedderProto:
    """The configured embedder. The import is lazy so the parse pass never loads
    onnxruntime alongside Docling."""
    e = cfg.embed
    if e.mode == "service":
        return ServiceEmbedder(e.url)
    if e.mode == "local":
        from .embed_onnx import OnnxEmbedder

        return OnnxEmbedder(e.model_dir, device=e.device, precision=e.precision)
    raise ValueError(f"embed.mode {e.mode!r}: service or local")


class ServiceEmbedder:
    """The embedder service (embed_service.py) behind the local encode() contract.

    On construction it fetches the service's fingerprint -- an unreachable service
    fails here, at startup, not at the first search. Every /v1/embed response carries
    the id of the fingerprint that produced it; a different id than the one fetched
    (the service restarted on another model or precision mid-run) raises instead of
    mixing two embedders' vectors under one collection stamp."""

    # One request = one ingest batch (ingest.batch_size chunks) or one query. A fp32
    # batch of 32 x 512 tokens takes ~1 s on the T4; this only bounds a hung service.
    TIMEOUT_S = 300.0

    def __init__(self, url: str) -> None:
        import httpx

        self.url = url.rstrip("/")
        self._http = httpx.Client(base_url=self.url, timeout=self.TIMEOUT_S)
        got = self._call("GET", "/v1/fingerprint")
        self._fingerprint, self._id = got["fingerprint"], got["id"]
        if fingerprint_id(self._fingerprint) != self._id:
            raise RuntimeError(f"embedder service {self.url}: fingerprint does not "
                               f"match its id {self._id}")
        log.info("embedder service %s: %s %s on %s (%s)", self.url,
                 self._fingerprint.get("model_id"), self._fingerprint.get("precision"),
                 self._fingerprint.get("provider"), self._id)

    @property
    def fingerprint(self) -> dict:
        return self._fingerprint

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        import httpx

        try:
            r = self._http.request(method, path, json=body)
        except httpx.HTTPError as e:
            raise RuntimeError(f"embedder service {self.url}{path}: {e!r}") from e
        if r.status_code != 200:
            raise RuntimeError(f"embedder service {self.url}{path}: HTTP "
                               f"{r.status_code}: {r.text[:500]}")
        return r.json()

    def encode(self, texts: list[str]) -> tuple[list[list[float]], list["SparseVec"]]:
        got = self._call("POST", "/v1/embed", {"texts": texts})
        if got["fingerprint_id"] != self._id:
            raise RuntimeError(
                f"embedder service {self.url} changed under this process: it was "
                f"{self._id} at start, it now answers as {got['fingerprint_id']} -- "
                f"restart against the embedder the collection was stamped with"
            )
        if len(got["dense"]) != len(texts) or len(got["sparse"]) != len(texts):
            raise RuntimeError(f"embedder service {self.url}: {len(texts)} texts sent, "
                               f"{len(got['dense'])} / {len(got['sparse'])} vectors back")
        sparse = [SparseVec(s["indices"], s["values"]) for s in got["sparse"]]
        return got["dense"], sparse


class SparseVec:
    """Qdrant-ready sparse vector: parallel indices/values lists."""

    __slots__ = ("indices", "values")

    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values

"""Retrieval core for the docs-bridge server (design §9).

ONE retrieve->rerank path, shared by the MCP `search()` tool and the REST mirror
(`POST /v1/search`) so they cannot diverge. Steps:

  1. embed the query with the SAME ONNX/INT8 BGE-M3 as ingest (get_embedder) — this
     is what guarantees query vectors live in the same space as the stored doc
     vectors. No re-implementation; the embedder is the shared source of truth.
  2. hybrid retrieve from Qdrant: a dense-vector prefetch + a sparse-vector prefetch,
     fused server-side with Reciprocal Rank Fusion (design §12.4: dense+sparse in one
     model, not a separate BM25).
  3. rerank the fused top_n with the cross-encoder (rerank.OnnxReranker), keep top-k.
  4. return chunks with citations (source_path, section, last_updated, scores).

Models are loaded LAZILY and once (design §9: "loaded once, used across all subject
collections"): /healthz and /v1/subjects stay light, and the heavy embedder+reranker
only load on the first real search — friendlier to the 8GB Pi.
"""

from __future__ import annotations

import logging

from qdrant_client import models as qm

from . import qdrant_io
from .config import Config
from .embed import get_embedder
from .qdrant_io import DENSE, SPARSE

log = logging.getLogger(__name__)

_SNIPPET_CHARS = 320


class Searcher:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._client = None
        self._embedder = None
        self._reranker = None

    # --- lazily-loaded resources ---------------------------------------------

    @property
    def client(self):
        if self._client is None:
            self._client = qdrant_io.connect(self.cfg)
        return self._client

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(self.cfg)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None and self.cfg.rerank.enabled:
            from .rerank import OnnxReranker

            self._reranker = OnnxReranker(
                self.cfg.rerank.model_dir,
                use_int8=self.cfg.rerank.int8,
                max_length=self.cfg.rerank.max_length,
            )
        return self._reranker

    # --- tools ----------------------------------------------------------------

    def list_subjects(self) -> list[dict]:
        """Configured subjects + whether each has a populated Qdrant collection.
        Live-checked against Qdrant (matches the backup's live-discovery view)."""
        out = []
        for s in self.cfg.subjects:
            exists = self.client.collection_exists(s.collection)
            points = (
                self.client.count(s.collection, exact=True).count if exists else 0
            )
            out.append(
                {
                    "name": s.name,
                    "description": s.description,
                    "collection": s.collection,
                    "points": points,
                    "populated": points > 0,
                }
            )
        return out

    def search(self, subject: str, query: str, k: int | None = None) -> list[dict]:
        k = k or self.cfg.server.default_k
        collection = self.cfg.subject(subject).collection  # raises on unknown subject
        if not self.client.collection_exists(collection):
            log.warning("collection %s does not exist yet (subject %s)", collection, subject)
            return []

        dense_list, sparse_list = self.embedder.encode([query])
        dense, sparse = dense_list[0], sparse_list[0]

        res = self.client.query_points(
            collection_name=collection,
            prefetch=[
                qm.Prefetch(query=dense, using=DENSE, limit=self.cfg.server.prefetch_limit),
                qm.Prefetch(
                    query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=SPARSE,
                    limit=self.cfg.server.prefetch_limit,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=self.cfg.rerank.top_n,
            with_payload=True,
        )
        points = res.points
        if not points:
            return []

        # Rerank the fused candidates; fall back to the fusion score if disabled.
        reranker = self.reranker
        if reranker is not None:
            texts = [(p.payload or {}).get("text", "") for p in points]
            scores = reranker.score(query, texts)
            order = sorted(range(len(points)), key=lambda i: scores[i], reverse=True)
            ranked = [(points[i], float(scores[i])) for i in order[:k]]
        else:
            ranked = [(p, float(p.score)) for p in points[:k]]

        return [self._to_result(p, score) for p, score in ranked]

    @staticmethod
    def _to_result(point, score: float) -> dict:
        pl = point.payload or {}
        text = pl.get("text", "")
        return {
            "score": score,
            "source_path": pl.get("source_path"),
            "section": pl.get("section_path"),
            "last_updated": pl.get("last_updated"),
            "doc_id": pl.get("doc_id"),
            "chunk_id": pl.get("chunk_id"),
            "snippet": text[:_SNIPPET_CHARS],
            "text": text,
        }

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

    def search(
        self, subject: "str | list[str]", query: str, k: int | None = None
    ) -> list[dict]:
        k = k or self.cfg.server.default_k
        subjects = self._resolve_subjects(subject)   # raises on an unknown name
        multi = len(subjects) > 1
        # Total candidates handed to the (slow) reranker. Single pool -> top_n
        # (unchanged); spanning pools -> the larger multi_top_n so cross-pool hits
        # aren't squeezed out. Bounded regardless of pool count (rerank is the cost).
        budget = self.cfg.rerank.multi_top_n if multi else self.cfg.rerank.top_n

        dense_list, sparse_list = self.embedder.encode([query])
        dense, sparse = dense_list[0], sparse_list[0]

        # One RRF retrieval per existing collection; keep each candidate paired with
        # its subject so results stay attributable.
        per_subject: list[list[tuple[object, str]]] = []
        for s in subjects:
            if not self.client.collection_exists(s.collection):
                log.warning("collection %s missing (subject %s); skipped", s.collection, s.name)
                continue
            res = self.client.query_points(
                collection_name=s.collection,
                prefetch=[
                    qm.Prefetch(query=dense, using=DENSE, limit=self.cfg.server.prefetch_limit),
                    qm.Prefetch(
                        query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                        using=SPARSE,
                        limit=self.cfg.server.prefetch_limit,
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=budget,
                with_payload=True,
            )
            if res.points:
                per_subject.append([(p, s.name) for p in res.points])

        # Round-robin the per-pool RRF lists then cap at the budget: each pool's TOP
        # candidates go in first, fairly, total <= budget. For a single pool this is
        # just its top-`budget` in order (identical to the old single-collection path).
        candidates = self._interleave(per_subject)[:budget]
        if not candidates:
            return []

        # Rerank the gathered candidates GLOBALLY (cross-encoder scores are comparable
        # across pools); fall back to the fusion score if the reranker is disabled.
        reranker = self.reranker
        if reranker is not None:
            texts = [(p.payload or {}).get("text", "") for p, _ in candidates]
            scores = reranker.score(query, texts)
            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            ranked = [(candidates[i][0], candidates[i][1], float(scores[i])) for i in order[:k]]
        else:
            order = sorted(
                range(len(candidates)),
                key=lambda i: candidates[i][0].score or 0.0, reverse=True,
            )
            ranked = [
                (candidates[i][0], candidates[i][1], float(candidates[i][0].score or 0.0))
                for i in order[:k]
            ]

        return [self._to_result(p, subj, score) for p, subj, score in ranked]

    def _resolve_subjects(self, subject: "str | list[str]") -> list:
        """Normalize the `subject` arg into a list of Subjects. Accepts a single name,
        a list of names, or the literal "all" (every configured subject)."""
        if isinstance(subject, str):
            if subject.strip().lower() == "all":
                return list(self.cfg.subjects)
            return [self.cfg.subject(subject)]
        return [self.cfg.subject(name) for name in subject]

    @staticmethod
    def _interleave(lists: list) -> list:
        """Round-robin merge: lists[0][0], lists[1][0], ..., lists[0][1], lists[1][1], ..."""
        out = []
        for i in range(max((len(x) for x in lists), default=0)):
            for lst in lists:
                if i < len(lst):
                    out.append(lst[i])
        return out

    @staticmethod
    def _to_result(point, subject: str, score: float) -> dict:
        pl = point.payload or {}
        text = pl.get("text", "")
        return {
            "score": score,
            "subject": subject,
            "source_path": pl.get("source_path"),
            "section": pl.get("section_path"),
            "last_updated": pl.get("last_updated"),
            "doc_id": pl.get("doc_id"),
            "chunk_id": pl.get("chunk_id"),
            "snippet": text[:_SNIPPET_CHARS],
            "text": text,
        }

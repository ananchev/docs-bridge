"""Retrieval core for the docs-bridge server (design §9).

ONE retrieve->rerank path, shared by the MCP `search()` tool and the REST mirror
(`POST /v1/search`) so they cannot diverge. Steps:

  1. embed the query with the SAME BGE-M3 as ingest (get_embedder) — on EC2 the very
     same embedder service the worker used, on the CPU line the same in-process
     model — so query vectors live in the space of the stored doc vectors.
  2. hybrid retrieve from Qdrant: a dense-vector prefetch + a sparse-vector prefetch,
     fused server-side with Reciprocal Rank Fusion (design §12.4: dense+sparse in one
     model, not a separate BM25).
  3. rerank the fused top_n with the cross-encoder (rerank.OnnxReranker), keep top-k;
     across several pools, identical passages fold into one result (_merge_identical).
  4. return chunks with citations (source_path, section, last_updated, scores).

Models are loaded once (design §9: "loaded once, used across all subject
collections"), on first use; server.build_app makes that first use its startup
check, so a server that cannot embed or rerank does not start.
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
                device=self.cfg.rerank.device,
                precision=self.cfg.rerank.precision,
                bucket=self.cfg.rerank.bucket,
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
            scores = [float(x) for x in reranker.score(query, texts)]
        else:
            scores = [float(p.score or 0.0) for p, _ in candidates]
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        ranked = [(candidates[i][0], candidates[i][1], scores[i]) for i in order]

        if not multi:
            return [self._to_result(p, subj, score) for p, subj, score in ranked[:k]]
        return self._merge_identical(ranked, k)

    @classmethod
    def _merge_identical(cls, ranked: list, k: int) -> list[dict]:
        """Top-k DISTINCT passages across pools. Versioned corpora (Teamcenter 2512 /
        2606) share most of their text word for word, so without this a both-versions
        search spends half of k on identical pairs. An identical passage (whitespace-
        normalised) folds into the best-scored result: `subjects` lists every pool it
        came from and `also_in` carries the other citations. Only candidates that
        reached the reranker are seen, so a subject MISSING from `subjects` is not
        evidence the text differs there — presence means same, absence means unknown."""
        results: list[dict] = []
        by_text: dict[str, dict] = {}
        for p, subj, score in ranked:
            pl = p.payload or {}
            key = " ".join(pl.get("text", "").split())
            hit = by_text.get(key)
            if hit is not None:
                if subj not in hit["subjects"]:
                    hit["subjects"].append(subj)
                hit["also_in"].append({
                    "subject": subj,
                    "source_path": pl.get("source_path"),
                    "section": pl.get("section_path"),
                    "doc_id": pl.get("doc_id"),
                    "chunk_id": pl.get("chunk_id"),
                })
            elif len(results) < k:
                hit = by_text[key] = cls._to_result(p, subj, score)
                results.append(hit)
            # else: a new passage past k — skipped, but keep scanning so the chosen
            # results still collect their identical copies from the other pools.
        return results

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
            "subjects": [subject],   # >1 when identical text came from several pools
            "also_in": [],           # the other pools' citations for that same text
            "source_path": pl.get("source_path"),
            "section": pl.get("section_path"),
            "last_updated": pl.get("last_updated"),
            "doc_id": pl.get("doc_id"),
            "chunk_id": pl.get("chunk_id"),
            "snippet": text[:_SNIPPET_CHARS],
            "text": text,
        }

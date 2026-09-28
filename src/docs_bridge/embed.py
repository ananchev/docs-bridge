"""Pass 2 - embed.

BGE-M3 produces a dense vector AND a sparse (lexical) vector in a single forward
pass, which is exactly the hybrid signal we store in Qdrant (design §12.4: dense
+ sparse in one model, not a separate BM25). The backend is ONNX/INT8
(`embed_onnx.OnnxEmbedder`), validated as the default on 2026-06-21. The
FlagEmbedding/torch fallback was removed on 2026-09-28 (multi-user rollout §4.3):
ONNX had been the only backend in use since the INT8 adoption, and dropping it
removes FlagEmbedding's transitive pins from the upgrade. This module keeps the
backend-neutral pieces: the `SparseVec` container and the `get_embedder` factory
that both the ingest worker and the server call, so they always load the same model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)


class EmbedderProto(Protocol):
    """The encode contract every embedder backend exposes."""

    def encode(self, texts: list[str]) -> tuple[list[list[float]], list["SparseVec"]]:
        ...


def get_embedder(cfg: "Config") -> EmbedderProto:
    """The configured embedder. ONNX is the only backend; the import is lazy so the
    parse pass never loads onnxruntime alongside Docling."""
    backend = getattr(cfg, "embedding_backend", "onnx")
    if backend == "onnx":
        from .embed_onnx import OnnxEmbedder

        return OnnxEmbedder(cfg.onnx_model_dir, use_int8=cfg.embedding_int8)
    raise ValueError(
        f"unknown embedding_backend {backend!r}: only 'onnx' is supported "
        f"('flagembedding' was removed on 2026-09-28)"
    )


class SparseVec:
    """Qdrant-ready sparse vector: parallel indices/values lists."""

    __slots__ = ("indices", "values")

    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values

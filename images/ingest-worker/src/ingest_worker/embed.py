"""Pass 2 - embed.

BGE-M3 produces a dense vector AND a sparse (lexical) vector in a single forward
pass, which is exactly the hybrid signal we store in Qdrant (design §12.4: dense
+ sparse in one model, not a separate BM25). This module owns the FlagEmbedding
import so the model loads only after the parse pass has released Docling.

API note: pinned to FlagEmbedding==1.3.3. `BGEM3FlagModel.encode(...,
return_dense=True, return_sparse=True)` returns a dict with `dense_vecs`
(ndarray) and `lexical_weights` (list of {token_id_str: weight}).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class SparseVec:
    """Qdrant-ready sparse vector: parallel indices/values lists."""

    __slots__ = ("indices", "values")

    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values


class Embedder:
    def __init__(self, model_name: str, use_fp16: bool = False) -> None:
        from FlagEmbedding import BGEM3FlagModel

        # use_fp16=False: CPU on the Pi has no fp16 speedup and we want bit-for-bit
        # parity with the M2. Flip per-host later via config if the M2 benefits.
        log.info("loading embedding model %s", model_name)
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def encode(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec]]:
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = [vec.tolist() for vec in out["dense_vecs"]]
        sparse = [self._to_sparse(lw) for lw in out["lexical_weights"]]
        return dense, sparse

    @staticmethod
    def _to_sparse(lexical_weights: dict) -> SparseVec:
        # lexical_weights maps stringified token ids -> weights; drop zeros.
        indices: list[int] = []
        values: list[float] = []
        for tok, weight in lexical_weights.items():
            w = float(weight)
            if w <= 0.0:
                continue
            indices.append(int(tok))
            values.append(w)
        return SparseVec(indices=indices, values=values)

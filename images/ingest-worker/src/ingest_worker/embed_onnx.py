"""Pass 2 - embed, ONNX backend (alternative to the FlagEmbedding/torch path).

Runs the SAME BAAI/bge-m3 model as `embed.py`, but through ONNX Runtime instead
of torch — the goal is CPU throughput (and an optional INT8 weight quantization).
It is a drop-in for `embed.Embedder`: same `encode(texts) -> (dense, sparse)`
contract, and sparse vectors land in the *same BGE-M3 token-id index space* so
they stay compatible with documents embedded by the torch backend.

Design (strategy "encoder in ONNX, heads in numpy"):
  * The big XLM-RoBERTa encoder is exported to ONNX (and optionally INT8-quantized)
    by tools/export_bge_m3_onnx.py — that is where the speed lives.
  * The two tiny BGE-M3 heads are applied here in numpy, full-precision, so INT8
    never touches them:
      - DENSE  = L2-normalize(last_hidden_state[:, 0])           (the [CLS] token)
      - SPARSE = relu(last_hidden_state @ sparse_linear) -> per-token weights,
                 then max-pooled per input token id (specials dropped). This is a
                 faithful reimplementation of FlagEmbedding's _process_token_weights,
                 which is what guarantees token-id parity with embed.py.

The export writes a small model dir: model.onnx (+ model.int8.onnx), tokenizer.json,
sparse_linear.npz, and meta.json. This module loads from that dir; it does NOT
import torch or FlagEmbedding.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from .embed import SparseVec  # reuse the exact same Qdrant-ready sparse container

log = logging.getLogger(__name__)


class OnnxEmbedder:
    def __init__(
        self,
        model_dir: str,
        use_int8: bool = True,
        max_length: int = 8192,
        threads: int | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        meta_path = os.path.join(model_dir, "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        log.info(
            "loading ONNX embedder from %s (int8=%s, base=%s)",
            model_dir,
            use_int8,
            meta.get("model_id", "BAAI/bge-m3"),
        )

        onnx_file = "model.int8.onnx" if use_int8 else "model.onnx"
        onnx_path = os.path.join(model_dir, onnx_file)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"{onnx_path} not found; run tools/export_bge_m3_onnx.py first "
                f"(and pass --quantize for the int8 file)"
            )

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

        # Sparse head: Linear(1024 -> 1). npz holds weight (1, 1024) and bias (1,).
        head = np.load(os.path.join(model_dir, "sparse_linear.npz"))
        self._sparse_w = head["weight"].astype(np.float32).reshape(-1)  # (1024,)
        self._sparse_b = float(head["bias"].reshape(-1)[0])

        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        # Pad to the longest sequence in each batch (dynamic) so short batches stay cheap.
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(pad_id=pad_id or 1, pad_token="<pad>")

        # Special token ids to exclude from the sparse vector (same set FlagEmbedding
        # drops: cls/bos, sep/eos, pad, unk). Missing ones resolve to None and are
        # harmless.
        self._special = {
            self.tokenizer.token_to_id(t)
            for t in ("<s>", "</s>", "<pad>", "<unk>")
        }
        self._special.discard(None)

    def encode(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec]]:
        encs = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attn = np.array([e.attention_mask for e in encs], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attn}
        if "token_type_ids" in self._input_names:  # some exports keep it; XLM-R = zeros
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}

        # last_hidden_state: (batch, seq, 1024)
        hidden = self.session.run(None, feeds)[0]

        dense = self._dense(hidden)
        sparse = [
            self._sparse(hidden[i], input_ids[i], attn[i]) for i in range(len(texts))
        ]
        return dense, sparse

    # --- heads (numpy, full precision) ----------------------------------------

    @staticmethod
    def _dense(hidden: np.ndarray) -> list[list[float]]:
        cls = hidden[:, 0, :]  # [CLS] pooling, as in BGE-M3
        norm = np.linalg.norm(cls, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return (cls / norm).astype(np.float32).tolist()

    def _sparse(
        self, hidden: np.ndarray, ids: np.ndarray, attn: np.ndarray
    ) -> SparseVec:
        # token_weights = relu(hidden . w + b), one weight per token position.
        w = np.maximum(0.0, hidden @ self._sparse_w + self._sparse_b)  # (seq,)
        # Max-pool per token id over real (non-pad) positions, dropping specials.
        # Mirrors FlagEmbedding._process_token_weights -> identical token-id space.
        best: dict[int, float] = {}
        for pos in range(len(ids)):
            if attn[pos] == 0:
                continue
            tok = int(ids[pos])
            if tok in self._special:
                continue
            weight = float(w[pos])
            if weight <= 0.0:
                continue
            if weight > best.get(tok, 0.0):
                best[tok] = weight
        return SparseVec(indices=list(best.keys()), values=list(best.values()))

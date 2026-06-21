"""Cross-encoder reranker over (query, passage) pairs — ONNX/INT8, no torch.

The docs-bridge server's second stage (design §9): after the hybrid dense+sparse
retrieve fuses a candidate set, BGE-reranker-v2-m3 re-scores each (query, passage)
pair jointly and we keep the top-k. Unlike the bi-encoder embedder, a cross-encoder
sees both texts at once, so it is the quality lever — and the slow one, which is
why it runs only over the ~top_n fused candidates, not the whole collection.

Mirror of embed_onnx.OnnxEmbedder: loads a baked model dir (model.int8.onnx +
tokenizer.json + meta.json) produced by tools/export_bge_reranker_onnx.py on the
M2. BAAI/bge-reranker-v2-m3 is an XLM-R sequence-classification head emitting ONE
logit per pair; higher = more relevant. The tokenizer.json carries the pair
post-processor (`<s> query </s></s> passage </s>`), so we just hand it pairs.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

log = logging.getLogger(__name__)


class OnnxReranker:
    def __init__(
        self,
        model_dir: str,
        use_int8: bool = True,
        max_length: int = 512,
        threads: int | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        meta_path = os.path.join(model_dir, "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        log.info(
            "loading ONNX reranker from %s (int8=%s, base=%s)",
            model_dir,
            use_int8,
            meta.get("model_id", "BAAI/bge-reranker-v2-m3"),
        )

        onnx_file = "model.int8.onnx" if use_int8 else "model.onnx"
        onnx_path = os.path.join(model_dir, onnx_file)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"{onnx_path} not found; run tools/export_bge_reranker_onnx.py first "
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

        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(pad_id=pad_id or 1, pad_token="<pad>")

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance logit for each (query, passage). Order matches `passages`."""
        if not passages:
            return []
        # Pair encoding: the loaded tokenizer.json post-processor inserts the
        # XLM-R separators, so encode_batch over (query, passage) tuples is correct.
        encs = self.tokenizer.encode_batch([(query, p) for p in passages])
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attn = np.array([e.attention_mask for e in encs], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attn}
        if "token_type_ids" in self._input_names:  # XLM-R = all zeros
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}

        # logits: (batch, 1) for the single-label relevance head -> flatten.
        logits = self.session.run(None, feeds)[0]
        return np.asarray(logits, dtype=np.float32).reshape(-1).tolist()

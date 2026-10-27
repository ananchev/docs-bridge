"""Cross-encoder reranker over (query, passage) pairs — ONNX, no torch.

The docs-bridge server's second stage (design §9): after the hybrid dense+sparse
retrieve fuses a candidate set, BGE-reranker-v2-m3 re-scores each (query, passage)
pair jointly and we keep the top-k. Unlike the bi-encoder embedder, a cross-encoder
sees both texts at once, so it is the quality lever — and the slow one, which is
why it runs only over the ~top_n fused candidates, not the whole collection.

Loads a model dir from the models image (tools/onnx-export/ layout): the file is
picked by precision — model.int8.onnx (CPU, the Mac mini), model.fp16.onnx (CUDA,
EC2), model.onnx (fp32) — plus tokenizer.json and meta.json. Device and precision
are configured, never defaulted (config.RerankCfg). BAAI/bge-reranker-v2-m3 is an
XLM-R sequence-classification head emitting ONE logit per pair; higher = more
relevant. The tokenizer.json carries the pair post-processor
(`<s> query </s></s> passage </s>`), so we just hand it pairs.

`bucket` > 0 scores the candidates length-sorted in sub-batches of that size: each
sub-batch pads only to its own longest pair, which cut padding — 60% of the FLOPs on
the fixture — and halved fp16 latency on the T4 (Gate 1; 0.89 -> 0.40 s/search on ORT
CUDA, notes/gate1/ort-fp16*.txt). Under dynamic INT8 it moves scores (activation
scales are per batch), so the CPU INT8 line runs with bucket 0.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from .ort_session import open_session

log = logging.getLogger(__name__)

FILES = {"int8": "model.int8.onnx", "fp32": "model.onnx", "fp16": "model.fp16.onnx"}


class OnnxReranker:
    def __init__(
        self,
        model_dir: str,
        device: str,
        precision: str,
        bucket: int,
        max_length: int,
        threads: int | None = None,
    ) -> None:
        from tokenizers import Tokenizer

        if precision not in FILES:
            raise ValueError(f"precision {precision!r}: expected one of {', '.join(FILES)}")
        onnx_path = os.path.join(model_dir, FILES[precision])
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"{onnx_path} not found -- the models image this server was built from "
                f"has no {precision} reranker (images/models/README.md)"
            )
        meta = json.load(open(os.path.join(model_dir, "meta.json")))
        log.info(
            "loading ONNX reranker %s (%s on %s, bucket=%s, %s@%s)",
            onnx_path, precision, device, bucket,
            meta.get("model_id"), str(meta.get("revision"))[:12],
        )
        self.session = open_session(onnx_path, device, threads)
        self._input_names = {i.name for i in self.session.get_inputs()}
        self.bucket = bucket

        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(pad_id=pad_id or 1, pad_token="<pad>")

    def _run(self, input_ids: np.ndarray, attn: np.ndarray) -> np.ndarray:
        feeds = {"input_ids": input_ids, "attention_mask": attn}
        if "token_type_ids" in self._input_names:  # XLM-R = all zeros
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        # logits: (batch, 1) for the single-label relevance head -> flatten.
        return np.asarray(self.session.run(None, feeds)[0], dtype=np.float32).reshape(-1)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance logit for each (query, passage). Order matches `passages`."""
        if not passages:
            return []
        # Pair encoding: the loaded tokenizer.json post-processor inserts the
        # XLM-R separators, so encode_batch over (query, passage) tuples is correct.
        encs = self.tokenizer.encode_batch([(query, p) for p in passages])
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attn = np.array([e.attention_mask for e in encs], dtype=np.int64)
        if not self.bucket:
            return self._run(input_ids, attn).tolist()

        # Padding is on the right, so a sub-batch is the same rows cut to its own
        # longest pair -- identical tokens to encoding the sub-batch on its own.
        lens = attn.sum(axis=1)
        order = np.argsort(lens, kind="stable")
        out = np.empty(len(passages), dtype=np.float32)
        for i in range(0, len(order), self.bucket):
            idx = order[i:i + self.bucket]
            width = int(lens[idx].max())
            out[idx] = self._run(input_ids[idx, :width], attn[idx, :width])
        return out.tolist()

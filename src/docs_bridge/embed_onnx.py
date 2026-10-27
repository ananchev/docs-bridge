"""Pass 2 - embed, ONNX backend (the only backend; the FlagEmbedding/torch path it
replaced was removed 2026-09-28).

Runs BAAI/bge-m3 through ONNX Runtime at a configured device and precision, never
defaulted: INT8 on CPU (the Mac mini line), fp16 or fp32 on CUDA (EC2, inside the
embedder service, embed_service.py). It implements the `encode(texts) -> (dense,
sparse)` contract of the old torch backend, and sparse vectors land in the *same
BGE-M3 token-id index space*.

Design (strategy "encoder in ONNX, heads in numpy"):
  * The big XLM-RoBERTa encoder is exported to ONNX (fp32, fp16, INT8) by
    tools/onnx-export/export_bge_m3_onnx.py — that is where the speed lives.
  * The two tiny BGE-M3 heads are applied here in numpy, full-precision, so INT8
    and fp16 never touch them (the fp16 graph returns fp32 hidden states):
      - DENSE  = L2-normalize(last_hidden_state[:, 0])           (the [CLS] token)
      - SPARSE = relu(last_hidden_state @ sparse_linear) -> per-token weights,
                 then max-pooled per input token id (specials dropped). This is a
                 faithful reimplementation of FlagEmbedding's _process_token_weights,
                 which is what guarantees token-id parity with embed.py.

The export writes a model dir (tools/onnx-export/README.md): one file set per
precision (FILES), tokenizer.json, sparse_linear.npz and meta.json. This module
loads from that dir; it does NOT import torch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import cached_property

import numpy as np

from .embed import SparseVec  # reuse the exact same Qdrant-ready sparse container

log = logging.getLogger(__name__)

# Upper bound on batch_size x padded_length^2 for one ONNX run (see encode()): a batch
# of 32 chunks of up to 512 tokens — the shape the 3.7 GB peak was measured on.
TOKEN_BUDGET = 32 * 512 * 512


# The files each precision runs, as written by tools/onnx-export/ (_common.py): fp32
# keeps its weights in an external-data file, which decides the vectors as much as
# the graph does -- hashing model.onnx alone would fingerprint 1 MB of graph and miss
# 2.3 GB of weights.
FILES = {
    "int8": ("model.int8.onnx",),
    "fp32": ("model.onnx", "model.onnx.data"),
    "fp16": ("model.fp16.onnx",),
}
HEAD_FILES = ("sparse_linear.npz", "tokenizer.json")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class OnnxEmbedder:
    def __init__(
        self,
        model_dir: str,
        device: str,
        precision: str,
        max_length: int = 8192,
        threads: int | None = None,
    ) -> None:
        from tokenizers import Tokenizer

        from .ort_session import open_session

        if precision not in FILES:
            raise ValueError(f"precision {precision!r}: expected one of {', '.join(FILES)}")
        for f in FILES[precision] + HEAD_FILES + ("meta.json",):
            if not os.path.exists(os.path.join(model_dir, f)):
                raise FileNotFoundError(
                    f"{os.path.join(model_dir, f)} not found -- the models image this "
                    f"was built from has no {precision} embedder (images/models/README.md)"
                )
        self._meta = json.load(open(os.path.join(model_dir, "meta.json")))
        log.info(
            "loading ONNX embedder %s (%s on %s, %s@%s)",
            model_dir, precision, device,
            self._meta.get("model_id"), str(self._meta.get("revision"))[:12],
        )
        self._model_dir = model_dir
        self.device = device
        self.precision = precision
        self.session = open_session(
            os.path.join(model_dir, FILES[precision][0]), device, threads
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
        # Same tokenizer without padding, only to measure each text's length for the
        # batch split in encode().
        self._len_tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self._len_tokenizer.enable_truncation(max_length=max_length)

        # Special token ids to exclude from the sparse vector (same set FlagEmbedding
        # drops: cls/bos, sep/eos, pad, unk). Missing ones resolve to None and are
        # harmless.
        self._special = {
            self.tokenizer.token_to_id(t)
            for t in ("<s>", "</s>", "<pad>", "<unk>")
        }
        self._special.discard(None)

    @cached_property
    def fingerprint(self) -> dict:
        """What decides the vectors: the model bytes, the precision, and the runtime
        and execution provider that run them. Phase A step 4 showed identical model
        bytes and tokens still move INT8 dense vectors to cosine ~0.987 across
        onnxruntime 1.20 -> 1.30, so the runtime is part of the identity; fp16 and
        fp32 of one model are two identities, and so are CPU and CUDA kernels.
        Stamped per collection by the manifest (manifest.stamp_collection).

        Hashes every file the precision runs plus the heads (2.3 GB for fp32), once,
        on first use -- at the embedder service's start, or by the ingest worker in
        local mode; never on a local-mode server's start. Where meta.json records a
        file's sha256 (every export since tools/onnx-export/_common.py), the bytes on
        disk must match it."""
        import tokenizers

        from .ort_session import installed_build

        recorded = self._meta.get("files") or {}
        files = {}
        for f in FILES[self.precision] + HEAD_FILES:
            files[f] = _sha256(os.path.join(self._model_dir, f))
            want = (recorded.get(f) or {}).get("sha256")
            if recorded and want != files[f]:
                raise RuntimeError(
                    f"{self._model_dir}/{f}: sha256 {files[f][:16]} on disk, meta.json "
                    f"records {str(want)[:16]} -- not the exported model"
                )
        return {
            "model_id": self._meta.get("model_id"),
            "revision": self._meta.get("revision"),
            "precision": self.precision,
            "files": files,
            "provider": self.session.get_providers()[0],
            "onnxruntime": installed_build(),
            "tokenizers": tokenizers.__version__,
        }

    def encode(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec]]:
        """Embed `texts`, splitting the batch whenever padding would make it too big.

        Each run pads to the batch's longest text, and attention memory grows with
        batch x length^2 — measured 2026-10-15/16 on vhost2: one 587-token chunk took a
        64-batch to 7.6 GB, and table-of-contents dot leaders (785 tokens) killed a
        32-batch at a 10 GB cap. Consecutive texts are grouped so that
        len(group) x longest^2 stays within TOKEN_BUDGET (32 x 512^2). Batches that
        already fit — every batch of typical chunks — run unchanged as one group, so
        their vectors are exactly what an unsplit run gives."""
        lengths = [len(e.ids) for e in self._len_tokenizer.encode_batch(texts)]
        dense: list[list[float]] = []
        sparse: list[SparseVec] = []
        start, longest = 0, 0
        for i, n in enumerate(lengths):
            grown = max(longest, n)
            if i > start and (i - start + 1) * grown * grown > TOKEN_BUDGET:
                d, s_ = self._encode_group(texts[start:i])
                dense += d; sparse += s_
                start, grown = i, n
            longest = grown
        d, s_ = self._encode_group(texts[start:])
        return dense + d, sparse + s_

    def _encode_group(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec]]:
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

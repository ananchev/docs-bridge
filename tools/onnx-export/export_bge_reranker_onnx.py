"""Convert BAAI/bge-reranker-v2-m3 to an ONNX model dir for rerank.OnnxReranker.

One-time, offline BUILD step (mirror of export_bge_m3_onnx.py). Pulls the official
cross-encoder weights from HF, exports the XLM-R sequence-classification model to
ONNX (it emits a single relevance logit per (query, passage) pair), and optionally
makes an INT8 copy. The output dir is what docs_bridge.rerank.OnnxReranker loads.

Run (on a >=16GB host — the M2; the 8GB Pi OOMs on quantize):
    pip install '.[ingest,export]'        # torch+transformers come from .[ingest]
    python tools/export_bge_reranker_onnx.py --out /opt/bge-reranker-onnx
    python tools/export_bge_reranker_onnx.py --out /opt/bge-reranker-onnx --quantize-only

`--model-id` exports a DIFFERENT cross-encoder into the same dir layout, which is
what prices the cheap alternative to new hardware (multi-user-rollout-plan §3):
bge-reranker-base is ~278M against v2-m3's ~568M, so roughly 2x faster on the same
CPU with no architecture change. Both are XLM-R, so the tokenizer pair
post-processor and the single-logit head are unchanged:

    python tools/onnx-export/export_bge_reranker_onnx.py \
        --model-id BAAI/bge-reranker-base --out /opt/bge-reranker-base-onnx --quantize

Outputs in --out:
    model.onnx          full-precision cross-encoder (logits)
    model.int8.onnx     dynamic-INT8 copy               (with --quantize/-only)
    tokenizer.json      XLM-R tokenizer (carries the pair post-processor)
    meta.json           provenance (model id + revision + opset)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
OPSET = 17


def _quantize(out_dir: str) -> None:
    """Dynamic INT8 of the cross-encoder. Torch-free, so it runs as its own process
    where torch's model is not resident alongside the fp32 ONNX being loaded for
    quantization -- that combined peak OOM-kills the 8GB Pi (see the Dockerfile)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print("[quantize] cross-encoder -> INT8 ...")
    quantize_dynamic(
        os.path.join(out_dir, "model.onnx"),
        os.path.join(out_dir, "model.int8.onnx"),
        weight_type=QuantType.QInt8,
    )


def _write_meta(out_dir: str, model_id: str, revision: str, quantized: bool) -> None:
    print("[meta] writing meta.json ...")
    json.dump(
        {
            "model_id": model_id,
            "revision": revision,
            "opset": OPSET,
            "head": "sequence-classification, 1 logit per (query, passage) pair",
            "quantized": quantized,
        },
        open(os.path.join(out_dir, "meta.json"), "w"),
        indent=2,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output model dir")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help="HF cross-encoder to export (default: %(default)s). Any "
                         "single-logit sequence-classification reranker works; the "
                         "output dir layout is what OnnxReranker loads.")
    ap.add_argument("--revision", default="main", help="HF revision/commit to pin")
    ap.add_argument("--quantize", action="store_true",
                    help="also write model.int8.onnx in THIS process (fine on >=16GB hosts)")
    ap.add_argument("--quantize-only", action="store_true",
                    help="ONLY quantize an existing model.onnx, no torch export. Run as a "
                         "SEPARATE process after a plain export so quantize never shares "
                         "RAM with torch -- required to fit the 8GB Pi (see Dockerfile).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Torch-free quantize path: assumes a prior `--out <dir>` export wrote model.onnx.
    if args.quantize_only:
        _quantize(args.out)
        _write_meta(args.out, args.model_id, args.revision, quantized=True)
        print(f"done (quantize-only) -> {args.out}")
        return 0

    # 1. Export the XLM-R sequence-classification model to ONNX. We wrap it to emit
    #    ONLY logits (the relevance score); the pair tokenization is applied at
    #    runtime by OnnxReranker via the saved tokenizer.json post-processor.
    print(f"[1/2] exporting {args.model_id}@{args.revision} cross-encoder -> ONNX ...")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, revision=args.revision
    ).eval()
    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    tok.save_pretrained(args.out)  # writes tokenizer.json (fast tokenizer)

    # Segment ids: XLM-R (v2-m3 AND bge-reranker-base) has type_vocab_size == 1, so
    # they carry no information and the graph stays a two-input one -- byte-identical
    # to every export before this flag existed. A BERT-family cross-encoder encodes
    # the passage as segment B, and omitting that input would silently degrade every
    # pair score instead of failing, so export it when the model actually uses it.
    use_types = getattr(model.config, "type_vocab_size", 1) > 1
    print(f"[1/2] token_type_ids: {'exported' if use_types else 'omitted'} "
          f"(type_vocab_size={getattr(model.config, 'type_vocab_size', 1)})")

    class _Scorer(torch.nn.Module):
        def __init__(self, m: torch.nn.Module, types: bool) -> None:
            super().__init__()
            self.m = m
            self.types = types

        def forward(self, input_ids, attention_mask, token_type_ids=None):  # noqa: ANN001
            if self.types:
                return self.m(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).logits
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    enc = tok(["query probe"], ["passage probe"], return_tensors="pt", padding=True)
    names = ["input_ids", "attention_mask"]
    inputs = [enc["input_ids"], enc["attention_mask"]]
    if use_types:
        names.append("token_type_ids")
        inputs.append(enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])))
    onnx_path = os.path.join(args.out, "model.onnx")
    with torch.no_grad():
        torch.onnx.export(
            _Scorer(model, use_types),
            tuple(inputs),
            onnx_path,
            input_names=names,
            output_names=["logits"],
            dynamic_axes={
                **{n: {0: "batch", 1: "seq"} for n in names},
                "logits": {0: "batch"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
        )

    # Make sure tokenizer.json is present (some configs only write the slow files).
    tok_json = os.path.join(args.out, "tokenizer.json")
    if not os.path.exists(tok_json):
        from huggingface_hub import hf_hub_download

        shutil.copy(
            hf_hub_download(args.model_id, "tokenizer.json", revision=args.revision),
            tok_json,
        )

    # Release the torch model before any in-process quantize (prefer two processes
    # on small hosts: a plain export then a separate `--quantize-only`).
    del model, tok, enc
    import gc

    gc.collect()

    if args.quantize:
        _quantize(args.out)
    else:
        print("[quantize] skipping INT8 (no --quantize)")

    _write_meta(args.out, args.model_id, args.revision, quantized=args.quantize)
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

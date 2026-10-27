"""Convert BAAI/bge-reranker-v2-m3 to an ONNX model dir for rerank.OnnxReranker.

One-time, offline BUILD step (mirror of export_bge_m3_onnx.py). Pulls the official
cross-encoder weights from HF at a PINNED commit, exports the XLM-R
sequence-classification model to ONNX (one relevance logit per (query, passage)
pair), and optionally writes an fp16 and/or INT8 copy. The output dir is what
docs_bridge.rerank.OnnxReranker loads.

Run inside the exporter image (images/models-export/), which pins the toolchain:
    python tools/onnx-export/export_bge_reranker_onnx.py \
        --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --out /work/bge-reranker-onnx --fp16

`--model-id` exports a DIFFERENT cross-encoder into the same dir layout, which is
what priced the cheap alternative to new hardware (multi-user-rollout-plan §3):
bge-reranker-base is ~278M against v2-m3's ~568M. Both are XLM-R, so the tokenizer
pair post-processor and the single-logit head are unchanged.

Outputs in --out (layout: _common.py):
    model.onnx + model.onnx.data   fp32 cross-encoder (logits)
    model.fp16.onnx                fp16, fp32 in/out                  (with --fp16)
    model.int8.onnx                dynamic-INT8 copy                  (with --quantize)
    tokenizer.json                 XLM-R tokenizer (carries the pair post-processor)
    meta.json                      provenance + sha256 of every file
"""

from __future__ import annotations

import argparse
import gc
import os
import re

import _common

DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
OPSET = 17


def _meta(args) -> dict:
    return {"model_id": args.model_id, "revision": args.revision, "opset": OPSET,
            "head": "sequence-classification, 1 logit per (query, passage) pair"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output model dir")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help="HF cross-encoder to export (default: %(default)s). Any "
                         "single-logit sequence-classification reranker works; the "
                         "output dir layout is what OnnxReranker loads.")
    ap.add_argument("--revision", required=True,
                    help="HF commit (40 hex) to export -- a branch name like 'main' "
                         "is refused: it moves, and the scores would move with it")
    ap.add_argument("--fp16", action="store_true", help="also write model.fp16.onnx")
    ap.add_argument("--quantize", action="store_true", help="also write model.int8.onnx")
    ap.add_argument("--fp16-only", action="store_true",
                    help="ONLY convert an existing model.onnx to fp16, no torch")
    ap.add_argument("--quantize-only", action="store_true",
                    help="ONLY quantize an existing model.onnx, no torch. Run as a "
                         "SEPARATE process after a plain export on small hosts -- "
                         "torch + the fp32 ONNX together OOM-killed the 8 GB Pi.")
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        ap.error(f"--revision {args.revision!r} is not a 40-hex commit")

    os.makedirs(args.out, exist_ok=True)

    if args.fp16_only or args.quantize_only:
        if args.fp16_only:
            _common.to_fp16(args.out)
        if args.quantize_only:
            _common.quantize_int8(args.out)
        _common.write_meta(args.out, _meta(args))
        print(f"done (conversion only) -> {args.out}")
        return 0

    # 1. Export the sequence-classification model, wrapped to emit ONLY logits; the
    #    pair tokenization is applied at runtime via tokenizer.json's post-processor.
    #    dynamo=False: the TorchScript exporter, as for every model dir before this one.
    print(f"[1/2] exporting {args.model_id}@{args.revision} cross-encoder -> ONNX fp32 ...")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, revision=args.revision, dtype=torch.float32
    ).eval()
    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    tok.save_pretrained(args.out)  # writes tokenizer.json (fast tokenizer)

    # Segment ids: XLM-R (v2-m3 AND bge-reranker-base) has type_vocab_size == 1, so
    # they carry no information and the graph stays a two-input one. A BERT-family
    # cross-encoder encodes the passage as segment B, and omitting that input would
    # silently degrade every pair score instead of failing, so export it when the
    # model actually uses it.
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

    def _export(path: str) -> None:
        with torch.no_grad():
            torch.onnx.export(
                _Scorer(model, use_types),
                tuple(inputs),
                path,
                input_names=names,
                output_names=["logits"],
                dynamic_axes={
                    **{n: {0: "batch", 1: "seq"} for n in names},
                    "logits": {0: "batch"},
                },
                opset_version=OPSET,
                do_constant_folding=True,
                dynamo=False,
            )

    _common.export_fp32(_export, args.out)

    if not os.path.exists(os.path.join(args.out, "tokenizer.json")):
        _common.copy_hf_file(args.model_id, "tokenizer.json", args.revision,
                             os.path.join(args.out, "tokenizer.json"))

    # Release torch's copy before a conversion loads the ONNX one.
    del model, tok, enc, inputs
    gc.collect()

    print("[2/2] conversions ...")
    if args.fp16:
        _common.to_fp16(args.out)
    if args.quantize:
        _common.quantize_int8(args.out)

    _common.write_meta(args.out, _meta(args))
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

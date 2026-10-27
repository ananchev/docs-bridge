"""Convert the official BAAI/bge-m3 weights to an ONNX model dir for OnnxEmbedder.

One-time, offline BUILD step (not part of any runtime image). Reproducible and
self-sourced: it pulls the official weights from HF at a PINNED commit, exports the
XLM-R encoder to ONNX, extracts the real sparse head, and optionally writes an fp16
and/or INT8 copy. The output dir is what docs_bridge.embed_onnx.OnnxEmbedder loads.

Run inside the exporter image (images/models-export/), which pins the toolchain:
    python tools/onnx-export/export_bge_m3_onnx.py \
        --revision 5617a9f61b028005a4858fdac845db406aefb181 --out /work/bge-m3-onnx --fp16

Outputs in --out (layout: _common.py):
    model.onnx + model.onnx.data   fp32 encoder (last_hidden_state)
    model.fp16.onnx                fp16 encoder, fp32 in/out          (with --fp16)
    model.int8.onnx                dynamic-INT8 encoder               (with --quantize)
    sparse_linear.npz              the Linear(1024->1) sparse head, full precision
    tokenizer.json                 XLM-R tokenizer
    meta.json                      provenance + sha256 of every file

Heads (dense pooling + sparse linear) are intentionally NOT exported into the
graph; OnnxEmbedder applies them in numpy at full precision so a reduced precision
only ever touches the big encoder. tools/onnx-export/check_precision.py compares a
reduced variant against fp32 before it is adopted.
"""

from __future__ import annotations

import argparse
import gc
import os
import re

import numpy as np

import _common

MODEL_ID = "BAAI/bge-m3"
OPSET = 17


def _meta(args) -> dict:
    return {"model_id": MODEL_ID, "revision": args.revision, "opset": OPSET,
            "dense": "cls+l2norm",
            "sparse": "relu(linear) max-pooled per token id, specials dropped"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output model dir")
    ap.add_argument("--revision", required=True,
                    help="HF commit (40 hex) to export -- a branch name like 'main' "
                         "is refused: it moves, and the vectors would move with it")
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

    # 1. Export the XLM-R encoder with torch.onnx directly (no optimum: it keeps
    #    relocating its exporter between releases), wrapped to emit ONLY
    #    last_hidden_state -- the heads are applied later in numpy by OnnxEmbedder.
    #    dynamo=False: the TorchScript exporter, as for every model dir before this one.
    print(f"[1/3] exporting {MODEL_ID}@{args.revision} encoder -> ONNX fp32 ...")
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(MODEL_ID, revision=args.revision,
                                      dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=args.revision)
    tok.save_pretrained(args.out)  # writes tokenizer.json (fast tokenizer)

    class _Encoder(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):  # noqa: ANN001
            return self.m(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

    enc = tok(["parity probe"], return_tensors="pt", padding=True)

    def _export(path: str) -> None:
        with torch.no_grad():
            torch.onnx.export(
                _Encoder(model),
                (enc["input_ids"], enc["attention_mask"]),
                path,
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "last_hidden_state": {0: "batch", 1: "seq"},
                },
                opset_version=OPSET,
                do_constant_folding=True,
                dynamo=False,
            )

    _common.export_fp32(_export, args.out)

    # 2. Pull the real sparse head (Linear 1024->1) and store it as numpy.
    print("[2/3] extracting sparse_linear head ...")
    from huggingface_hub import hf_hub_download

    sp = hf_hub_download(MODEL_ID, "sparse_linear.pt", revision=args.revision)
    sd = torch.load(sp, map_location="cpu")
    np.savez(
        os.path.join(args.out, "sparse_linear.npz"),
        weight=sd["weight"].detach().cpu().numpy().astype(np.float32),  # (1, 1024)
        bias=sd["bias"].detach().cpu().numpy().astype(np.float32),      # (1,)
    )

    if not os.path.exists(os.path.join(args.out, "tokenizer.json")):
        _common.copy_hf_file(MODEL_ID, "tokenizer.json", args.revision,
                             os.path.join(args.out, "tokenizer.json"))

    # Release torch's ~2.3 GB copy before a conversion loads the ONNX one.
    del model, tok, enc, sd
    gc.collect()

    # 3. Optional reduced-precision copies of the encoder only.
    print("[3/3] conversions ...")
    if args.fp16:
        _common.to_fp16(args.out)
    if args.quantize:
        _common.quantize_int8(args.out)

    _common.write_meta(args.out, _meta(args))
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Convert the official BAAI/bge-m3 weights to an ONNX model dir for OnnxEmbedder.

One-time, offline BUILD step (not part of the runtime image). Reproducible and
self-sourced: it pulls the official weights from HF, exports the XLM-R encoder to
ONNX via Optimum, extracts the real sparse head, and (optionally) makes an INT8
copy. The output dir is what ingest_worker.embed_onnx.OnnxEmbedder loads.

Run (on a host with torch+transformers available, e.g. after the benchmark frees RAM):
    pip install onnx onnxruntime          # torch+transformers come from the base image
    python tools/export_bge_m3_onnx.py --out /data/cache/bge-m3-onnx --quantize

Outputs in --out:
    model.onnx          full-precision encoder (last_hidden_state)
    model.int8.onnx     dynamic-INT8 encoder            (with --quantize)
    sparse_linear.npz   the Linear(1024->1) sparse head, full precision
    tokenizer.json      XLM-R tokenizer
    meta.json           provenance (model id + revision + opset)

Heads (dense pooling + sparse linear) are intentionally NOT exported into the
graph; OnnxEmbedder applies them in numpy at full precision so INT8 only ever
touches the big encoder. Parity with FlagEmbedding is then proven by
tools/validate_embed_parity.py before adoption.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

MODEL_ID = "BAAI/bge-m3"
OPSET = 17


def _quantize(out_dir: str) -> None:
    """Dynamic INT8 of the encoder only. No torch import, so this can run as its own
    process where torch's ~2.3GB bge-m3 model is not resident alongside the fp32 ONNX
    being loaded for quantization -- that combined peak OOM-killed the 8GB Pi."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print("[quantize] encoder -> INT8 ...")
    quantize_dynamic(
        os.path.join(out_dir, "model.onnx"),
        os.path.join(out_dir, "model.int8.onnx"),
        weight_type=QuantType.QInt8,
    )


def _write_meta(out_dir: str, revision: str, quantized: bool) -> None:
    print("[meta] writing meta.json ...")
    json.dump(
        {
            "model_id": MODEL_ID,
            "revision": revision,
            "opset": OPSET,
            "dense": "cls+l2norm",
            "sparse": "relu(linear) max-pooled per token id, specials dropped",
            "quantized": quantized,
        },
        open(os.path.join(out_dir, "meta.json"), "w"),
        indent=2,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output model dir")
    ap.add_argument("--revision", default="main", help="HF revision/commit to pin")
    ap.add_argument("--quantize", action="store_true",
                    help="also write model.int8.onnx in THIS process (fine on >=16GB hosts)")
    ap.add_argument("--quantize-only", action="store_true",
                    help="ONLY quantize an existing model.onnx, no torch export. Run as a "
                         "SEPARATE process after a plain export so the quantize never shares "
                         "RAM with torch -- required to fit the 8GB Pi (see Dockerfile).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Torch-free quantize path: assumes a prior `--out <dir>` export wrote model.onnx.
    if args.quantize_only:
        _quantize(args.out)
        _write_meta(args.out, args.revision, quantized=True)
        print(f"done (quantize-only) -> {args.out}")
        return 0

    # 1. Export the XLM-R encoder to ONNX with torch.onnx directly (no optimum).
    #    optimum keeps relocating its onnx exporter between releases/extras, so we use
    #    torch + transformers (both already in the base image) and wrap the model to
    #    emit ONLY last_hidden_state -- that single tensor is the encoder; the dense and
    #    sparse heads are applied later in numpy by OnnxEmbedder.
    print(f"[1/4] exporting {MODEL_ID}@{args.revision} encoder -> ONNX ...")
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(MODEL_ID, revision=args.revision).eval()
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
    onnx_path = os.path.join(args.out, "model.onnx")
    with torch.no_grad():
        torch.onnx.export(
            _Encoder(model),
            (enc["input_ids"], enc["attention_mask"]),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
        )

    # 2. Pull the real sparse head (Linear 1024->1) and store it as numpy.
    print("[2/4] extracting sparse_linear head ...")
    import torch
    from huggingface_hub import hf_hub_download

    sp = hf_hub_download(MODEL_ID, "sparse_linear.pt", revision=args.revision)
    sd = torch.load(sp, map_location="cpu")
    np.savez(
        os.path.join(args.out, "sparse_linear.npz"),
        weight=sd["weight"].detach().cpu().numpy().astype(np.float32),  # (1, 1024)
        bias=sd["bias"].detach().cpu().numpy().astype(np.float32),      # (1,)
    )

    # 3. Make sure tokenizer.json is present (Optimum usually copies it).
    tok = os.path.join(args.out, "tokenizer.json")
    if not os.path.exists(tok):
        from huggingface_hub import hf_hub_download as _dl

        shutil.copy(_dl(MODEL_ID, "tokenizer.json", revision=args.revision), tok)

    # Release the torch model before any in-process quantize so its ~2.3GB does not
    # stack on top of the fp32 ONNX load. (On small hosts, prefer two processes:
    # a plain export then a separate `--quantize-only` -- see the Dockerfile.)
    del model, tok, enc, sd
    import gc

    gc.collect()

    # 4. Optional dynamic INT8 quantization of the encoder only.
    if args.quantize:
        _quantize(args.out)
    else:
        print("[quantize] skipping INT8 (no --quantize)")

    _write_meta(args.out, args.revision, quantized=args.quantize)
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

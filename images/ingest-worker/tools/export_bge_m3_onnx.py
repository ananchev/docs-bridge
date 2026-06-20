"""Convert the official BAAI/bge-m3 weights to an ONNX model dir for OnnxEmbedder.

One-time, offline BUILD step (not part of the runtime image). Reproducible and
self-sourced: it pulls the official weights from HF, exports the XLM-R encoder to
ONNX via Optimum, extracts the real sparse head, and (optionally) makes an INT8
copy. The output dir is what ingest_worker.embed_onnx.OnnxEmbedder loads.

Run (on a host with torch available, e.g. after the benchmark frees RAM):
    pip install "optimum[exporters]" onnx onnxruntime
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output model dir")
    ap.add_argument("--revision", default="main", help="HF revision/commit to pin")
    ap.add_argument("--quantize", action="store_true", help="also write model.int8.onnx")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Export the XLM-R encoder to ONNX (Optimum handles dynamic axes + opset).
    print(f"[1/4] exporting {MODEL_ID}@{args.revision} encoder -> ONNX ...")
    from optimum.onnxruntime import ORTModelForFeatureExtraction

    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_ID, revision=args.revision, export=True
    )
    model.save_pretrained(args.out)  # writes model.onnx + tokenizer.json + config.json

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

    # 4. Optional dynamic INT8 quantization of the encoder only.
    if args.quantize:
        print("[3/4] quantizing encoder -> INT8 ...")
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            os.path.join(args.out, "model.onnx"),
            os.path.join(args.out, "model.int8.onnx"),
            weight_type=QuantType.QInt8,
        )
    else:
        print("[3/4] skipping INT8 (no --quantize)")

    print("[4/4] writing meta.json ...")
    json.dump(
        {
            "model_id": MODEL_ID,
            "revision": args.revision,
            "dense": "cls+l2norm",
            "sparse": "relu(linear) max-pooled per token id, specials dropped",
            "quantized": bool(args.quantize),
        },
        open(os.path.join(args.out, "meta.json"), "w"),
        indent=2,
    )
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

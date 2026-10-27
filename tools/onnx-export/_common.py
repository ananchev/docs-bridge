"""Steps both export scripts share: fp32 save layout, fp16/INT8 conversion, meta.json.

The model dir a script writes, by variant:
    model.onnx + model.onnx.data   fp32 graph + its weights in ONE external-data file
                                   (568M params x 4 B = 2.3 GB, over protobuf's 2 GB
                                   cap, so it cannot be a single file)
    model.fp16.onnx                fp16 weights, fp32 inputs/outputs (keep_io_types),
                                   1.1 GB -- under the cap, one self-contained file
    model.int8.onnx                dynamic INT8 (the CPU line; unchanged)
    meta.json                      model id + revision + opset + every file's sha256 +
                                   the toolchain that wrote them

Every conversion is torch-free and reads model.onnx from disk, so it can run as its own
process (`--fp16-only`, `--quantize-only`) where torch's copy of the model is not
resident alongside the ONNX one.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile

FP32_FILE = "model.onnx"
FP32_DATA = "model.onnx.data"
FP16_FILE = "model.fp16.onnx"
INT8_FILE = "model.int8.onnx"


def export_fp32(torch_export, out_dir: str) -> None:
    """Run `torch_export(path)` into a temp dir, then re-save as model.onnx + ONE data file.

    torch writes a >2 GB model as one external file PER TENSOR (hundreds of files named
    after the weights); re-saving collapses them into model.onnx.data, which is what the
    models image copies and what the fingerprint hashes."""
    import onnx

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        raw = os.path.join(tmp, FP32_FILE)
        torch_export(raw)
        model = onnx.load(raw, load_external_data=True)
        for f in (FP32_FILE, FP32_DATA):
            if os.path.exists(os.path.join(out_dir, f)):
                os.remove(os.path.join(out_dir, f))
        onnx.save_model(model, os.path.join(out_dir, FP32_FILE),
                        save_as_external_data=True, all_tensors_to_one_file=True,
                        location=FP32_DATA, size_threshold=1024)
    for f in (FP32_FILE, FP32_DATA):  # onnx writes the data file 0600 (FINDINGS 3)
        os.chmod(os.path.join(out_dir, f), 0o644)
    onnx.checker.check_model(os.path.join(out_dir, FP32_FILE))  # path form: >2 GB ok


def to_fp16(out_dir: str) -> None:
    """fp32 -> fp16 weights, inputs/outputs kept fp32 so callers feed and read the same
    dtypes as the fp32 model. Shape inference is skipped: it serialises the model to one
    protobuf, which the 2.3 GB fp32 graph cannot be."""
    import onnx
    from onnxconverter_common import float16

    print(f"[fp16] {FP32_FILE} -> {FP16_FILE} ...")
    model = onnx.load(os.path.join(out_dir, FP32_FILE), load_external_data=True)
    half = float16.convert_float_to_float16(model, keep_io_types=True,
                                            disable_shape_infer=True)
    del model
    onnx.save_model(half, os.path.join(out_dir, FP16_FILE))
    onnx.checker.check_model(os.path.join(out_dir, FP16_FILE))


def quantize_int8(out_dir: str) -> None:
    """Dynamic INT8 of the encoder/cross-encoder -- the CPU line's model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"[quantize] {FP32_FILE} -> {INT8_FILE} ...")
    quantize_dynamic(os.path.join(out_dir, FP32_FILE), os.path.join(out_dir, INT8_FILE),
                     weight_type=QuantType.QInt8)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _versions() -> dict:
    import importlib.metadata as md

    out = {}
    for pkg in ("torch", "transformers", "onnx", "onnxconverter-common",
                "onnxruntime", "onnxruntime-gpu", "tokenizers", "numpy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def write_meta(out_dir: str, fields: dict) -> None:
    """meta.json from what is ON DISK, so a later `--fp16-only` / `--quantize-only` run
    records its file without forgetting the earlier ones. `toolchain` is this process's;
    `exported_with` (the torch export's toolchain) is kept from the first write."""
    path = os.path.join(out_dir, "meta.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    for k in ("model_id", "revision"):
        if old.get(k) not in (None, fields[k]):
            raise SystemExit(f"{out_dir} holds {k}={old[k]!r}, this run is "
                             f"{fields[k]!r} -- use a fresh --out")
    files = {}
    for f in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, f)
        if f != "meta.json" and os.path.isfile(p):
            files[f] = {"sha256": _sha256(p), "bytes": os.path.getsize(p)}
    meta = {**fields,
            "variants": [v for v, f in (("fp32", FP32_FILE), ("fp16", FP16_FILE),
                                        ("int8", INT8_FILE)) if f in files],
            "files": files,
            "exported_with": old.get("exported_with") or _versions(),
            "toolchain": _versions()}
    print(f"[meta] {path}: {', '.join(meta['variants'])}")
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)


def copy_hf_file(model_id: str, name: str, revision: str, dest: str) -> None:
    from huggingface_hub import hf_hub_download

    shutil.copy(hf_hub_download(model_id, name, revision=revision), dest)

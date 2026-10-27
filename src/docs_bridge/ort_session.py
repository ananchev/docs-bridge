"""One way to open an onnxruntime session, so every model in the process gets the same
guarantees (plan §8 3b items 1-2):

  * exactly ONE onnxruntime build installed. `onnxruntime` and `onnxruntime-gpu` both
    install the same `onnxruntime` package dir; with both present, whichever was
    installed last wins and the other's metadata lies about what runs.
  * `cuda` requests CUDAExecutionProvider ONLY -- no CPU fallback list -- and the
    session must report it. A CUDA/cuDNN mismatch otherwise makes onnxruntime run the
    whole graph on CPU with nothing but a warning: slower by an order of magnitude,
    and fp16 on CPU is a different computation.
  * `cpu` is CPU only, chosen explicitly.

Nothing is defaulted: the caller passes the device it was configured with.
"""

from __future__ import annotations

import importlib.metadata as md
import logging

log = logging.getLogger(__name__)

PROVIDERS = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}
# Which precisions each device runs, for the embedder and the reranker alike: INT8
# is dynamic-quantized for CPU, fp16 needs the GPU's tensor cores.
PRECISIONS = {"cpu": ("int8", "fp32"), "cuda": ("fp16", "fp32")}


def check_precision(section: str, device: str, precision: str, where: str) -> None:
    """Refuse a device / precision pair that cannot run, naming `section` (a config
    section or a flag set) and `where` (the file or command it came from)."""
    if device not in PRECISIONS:
        raise ValueError(f"{section}.device {device!r} in {where}: cpu or cuda")
    if precision not in PRECISIONS[device]:
        raise ValueError(f"{section}.precision {precision!r} on {device} in {where}: "
                         f"{' or '.join(PRECISIONS[device])} -- INT8 is dynamic-"
                         f"quantized for CPU, fp16 needs the GPU's tensor cores")


def _builds() -> list[str]:
    out = []
    for dist in ("onnxruntime", "onnxruntime-gpu"):
        try:
            out.append(f"{dist}=={md.version(dist)}")
        except md.PackageNotFoundError:
            pass
    return out


def installed_build() -> str:
    """The one onnxruntime build installed, e.g. 'onnxruntime-gpu==1.30.0', or raise."""
    builds = _builds()
    if len(builds) != 1:
        raise RuntimeError(f"exactly one onnxruntime build must be installed, found "
                           f"{builds or 'none'}")
    return builds[0]


def open_session(path: str, device: str, threads: int | None = None):
    """An InferenceSession on `device` ('cpu' | 'cuda'), or an exception that says why not."""
    import onnxruntime as ort

    if device not in PROVIDERS:
        raise ValueError(f"device {device!r}: expected one of {', '.join(PROVIDERS)}")
    build = installed_build()
    if device == "cuda":
        if not build.startswith("onnxruntime-gpu=="):
            raise RuntimeError(f"device 'cuda' needs onnxruntime-gpu; installed: {build}")
        ort.preload_dlls()  # CUDA/cuDNN from the nvidia-* wheels, not the system

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    want = PROVIDERS[device]
    session = ort.InferenceSession(path, sess_options=so, providers=[want])
    got = session.get_providers()
    if not got or got[0] != want:
        raise RuntimeError(
            f"{path}: asked for {want}, the session got {got} -- refusing to run on a "
            f"fallback (driver, --gpus, or a CUDA/cuDNN major onnxruntime does not expect)"
        )
    log.info("onnxruntime session %s on %s (%s)", path, want, build)
    return session

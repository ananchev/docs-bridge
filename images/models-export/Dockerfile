# docs-bridge-models-export: the pinned toolchain that makes and checks model dirs.
#
# A TOOL image, never deployed. It holds no scripts and no models: the docs-bridge
# checkout is mounted at /src and a work dir at /work, so what ran is the commit that
# was checked out (tools/onnx-export/build-gpu-models.sh records it). Run with
# `--gpus all`: check_precision.py runs the fp16 graph on the CUDA provider.
#
# x86_64 only -- it runs on the EC2 GPU host. torch is the +cpu wheel (an export needs
# no GPU, and the CUDA torch would add ~4 GB); the CUDA libraries come with
# onnxruntime-gpu's [cuda,cudnn] extras as nvidia-* wheels, CUDA 13 / cuDNN 9, which
# the host's driver (595, CUDA 13.2) runs. No nvidia/cuda base image needed.
#
#   docker build -f images/models-export/Dockerfile -t docs-bridge-models-export:<sha> .
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/work/hf-cache HF_HUB_DISABLE_TELEMETRY=1

# torch is the +cpu wheel from the PyTorch index (see images/lock.sh for why).
COPY images/models-export/requirements.lock /tmp/requirements.lock
RUN pip install --require-hashes --no-deps -r /tmp/requirements.lock \
        --extra-index-url https://download.pytorch.org/whl/cpu \
 && rm /tmp/requirements.lock \
 && python -c "import onnxruntime as o; assert 'CUDAExecutionProvider' in o.get_available_providers(), o.get_available_providers()"

WORKDIR /src

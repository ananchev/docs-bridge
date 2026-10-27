#!/usr/bin/env bash
# Regenerate the per-image dependency locks from pyproject.toml.
#
# Each image installs its lock (exact versions + hashes for every transitive dep),
# then the docs_bridge package with --no-deps, so a build never re-resolves and two
# builds of the same commit get the same wheels. Run from the repo root after any
# change to pyproject.toml, commit the locks with it, and re-run the embed parity
# gate (notes/phase-a-runbook.md) if onnxruntime, tokenizers or numpy moved.
#
# Needs uv (https://docs.astral.sh/uv/); runs on any host — the target platform is
# a flag, not the machine it runs on.
#
#   images/ingest-worker/requirements.lock      linux/arm64 only: the worker runs on
#                                               vhost2 (both corpora are ingested there)
#   images/docs-bridge/requirements-arm64.lock  the Mac mini server (vhost2)
#   images/docs-bridge/requirements-amd64.lock  a CPU server on x86_64
#   images/docs-bridge/requirements-amd64-gpu.lock  the EC2 GPU server: onnxruntime-gpu
#                                               INSTEAD of onnxruntime (extra `gpu`)
#   images/models-export/requirements.lock      the exporter (x86_64, the EC2 GPU host):
#                                               from its own requirements.in, not from
#                                               pyproject -- it installs no docs_bridge
#
# torch comes from the PyTorch CPU index. On PyPI the linux/aarch64 torch 2.14 wheel
# is CUDA-enabled and drags in ~15 nvidia-* packages (several GB) that an M2 can
# never use; the +cpu wheel is the same code without them.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=3.11
CPU_TORCH=(--index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match)

uv pip compile pyproject.toml --extra ingest --extra cpu --python-version "$PY" \
    --python-platform aarch64-manylinux_2_28 "${CPU_TORCH[@]}" \
    --generate-hashes -q -o images/ingest-worker/requirements.lock

for arch in arm64:aarch64 amd64:x86_64; do
    uv pip compile pyproject.toml --extra server --extra cpu --python-version "$PY" \
        --python-platform "${arch#*:}-manylinux_2_28" \
        --generate-hashes -q -o "images/docs-bridge/requirements-${arch%%:*}.lock"
done
uv pip compile pyproject.toml --extra server --extra gpu --python-version "$PY" \
    --python-platform x86_64-manylinux_2_28 \
    --generate-hashes -q -o images/docs-bridge/requirements-amd64-gpu.lock

uv pip compile images/models-export/requirements.in --python-version "$PY" \
    --python-platform x86_64-manylinux_2_28 "${CPU_TORCH[@]}" \
    --generate-hashes -q -o images/models-export/requirements.lock

# The §4.2 invariant, checked mechanically: the embedder runtime must be identical
# in the worker and in both server locks.
fail=0
for pkg in numpy onnxruntime tokenizers; do
    v=$(grep -h "^${pkg}==" images/ingest-worker/requirements.lock \
        images/docs-bridge/requirements-{arm64,amd64}.lock | sed 's/ .*//' | sort -u)
    if [ "$(echo "$v" | wc -l)" -ne 1 ]; then
        echo "INVARIANT BROKEN: $pkg differs across images:"; echo "$v"; fail=1
    else
        echo "ok  $v (all images)"
    fi
done
if grep -q "^nvidia-" images/ingest-worker/requirements.lock; then
    echo "CUDA packages leaked into the worker lock (torch not from the CPU index)"; fail=1
fi
# The GPU locks (exporter, GPU server): onnxruntime-gpu ONLY -- the CPU build must
# not sneak in beside it (plan §8 3b item 2) -- at the CPU locks' version.
srv=$(grep "^onnxruntime==" images/docs-bridge/requirements-amd64.lock | sed 's/ .*//')
for lock in images/models-export/requirements.lock images/docs-bridge/requirements-amd64-gpu.lock; do
    if grep -q "^onnxruntime==" "$lock"; then
        echo "the CPU onnxruntime leaked into $lock beside onnxruntime-gpu"; fail=1
    fi
    gpu=$(grep "^onnxruntime-gpu==" "$lock" | sed 's/ .*//; s/-gpu//')
    [ "$gpu" = "$srv" ] && echo "ok  $gpu (-gpu) in $lock" \
        || { echo "$lock: onnxruntime-gpu ($gpu) differs from the CPU locks' ($srv)"; fail=1; }
done
for pkg in numpy tokenizers; do
    a=$(grep -h "^${pkg}==" images/docs-bridge/requirements-amd64.lock | sed 's/ .*//')
    b=$(grep -h "^${pkg}==" images/docs-bridge/requirements-amd64-gpu.lock | sed 's/ .*//')
    [ "$a" = "$b" ] || { echo "GPU server lock has $b, CPU server $a"; fail=1; }
done
exit $fail

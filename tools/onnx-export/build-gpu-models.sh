#!/usr/bin/env bash
# Build the GPU target's models image: export -> fp16 -> check on CUDA -> image.
#
# Runs ON the EC2 GPU host (x86_64, Docker + the nvidia runtime), from a clean
# docs-bridge checkout; HF and PyPI are reached through the host's egress.
#
#   tools/onnx-export/build-gpu-models.sh --tag gpu-2026-10 \
#       --work /opt/dlami/nvme/models-export --fixture ~/gate1/fixture.json
#
# Produces localhost/docs-bridge-models:<tag> (images/models/gpu.Dockerfile). Every
# step fails the build; nothing is defaulted:
#   1. the exporter image (images/models-export/), tagged with the checkout's commit
#   2. bge-m3 and bge-reranker-v2-m3 at the commits pinned below -> fp32 + fp16
#   3. check_precision.py on each, CUDA provider only -- a FAIL stops before the image
#   4. the image from exactly the shipped files, plus /opt/docs-bridge-models.json
#
# --work holds the HF cache, the export dirs and the check results; it may be scratch
# (the instance store): the image is the artifact. A rebuild re-downloads ~4.5 GB.
#
# Shipped: bge-m3 fp32 AND fp16 (which one the embedder service runs is measured on the
# 14.2 test collection, plan §8 3b), the reranker fp16 only (decided, Gate 1). The
# reranker's fp32 export stays in --work: check_precision.py's reference, not served.
set -euo pipefail

# The pinned HF commits. Both models' last change is mid-2024; Gate 1's GPU runs and
# the Mac mini's frozen INT8 models used these same commits (via `main`).
BGE_M3_REV=5617a9f61b028005a4858fdac845db406aefb181
RERANKER_REV=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e

die() { echo "build-gpu-models.sh: $*" >&2; exit 1; }

TAG='' WORK='' FIXTURE=''
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG="${2:?}"; shift 2 ;;
        --work) WORK="${2:?}"; shift 2 ;;
        --fixture) FIXTURE="${2:?}"; shift 2 ;;
        *) die "unknown argument '$1' (--tag, --work, --fixture; all required)" ;;
    esac
done
[ -n "$TAG" ] || die "--tag is required (e.g. gpu-2026-10)"
[ -n "$WORK" ] || die "--work is required (a dir with ~15 GB free)"
[ -n "$FIXTURE" ] || die "--fixture is required (the Gate 1 rerank fixture.json)"
[[ "$TAG" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || die "--tag '$TAG' is not a valid image tag"
[ -f "$FIXTURE" ] || die "no fixture at $FIXTURE"

SRC="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SRC"
[ -z "$(git status --porcelain)" ] || die "$SRC has uncommitted changes -- the image records the commit, so build from a clean checkout"
COMMIT=$(git rev-parse HEAD)
IMAGE="localhost/docs-bridge-models:$TAG"
EXPORTER="localhost/docs-bridge-models-export:${COMMIT:0:12}"
docker image inspect "$IMAGE" >/dev/null 2>&1 && die "$IMAGE already exists -- a tag names one build; pick a new one or remove it deliberately"
docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"' || die "docker has no nvidia runtime"

mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
RUN="$WORK/$TAG"
[ -e "$RUN" ] && die "$RUN exists -- remove it, or use a new --tag"
mkdir -p "$RUN" "$WORK/hf-cache"
cp "$FIXTURE" "$RUN/fixture.json"
exec > >(tee "$RUN/build.log") 2>&1
echo "== $IMAGE from $COMMIT ($(date -u +%FT%TZ))"

echo "== 1/4 exporter image $EXPORTER"
docker build -q -f images/models-export/Dockerfile -t "$EXPORTER" .

# As the invoking user, so the work dir stays theirs; HF_HOME is in the image env.
exporter() {
    docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "$SRC:/src:ro" -v "$WORK/hf-cache:/work/hf-cache" -v "$RUN:/run-dir" \
        "$EXPORTER" "$@"
}

echo "== 2/4 exports"
exporter python tools/onnx-export/export_bge_m3_onnx.py \
    --revision "$BGE_M3_REV" --out /run-dir/bge-m3-onnx --fp16
exporter python tools/onnx-export/export_bge_reranker_onnx.py \
    --revision "$RERANKER_REV" --out /run-dir/bge-reranker-onnx --fp16

echo "== 3/4 fp16 vs fp32 on CUDA"
exporter python tools/onnx-export/check_precision.py --kind embed \
    --model-dir /run-dir/bge-m3-onnx --fixture /run-dir/fixture.json --json /run-dir/check-embed.json
exporter python tools/onnx-export/check_precision.py --kind rerank \
    --model-dir /run-dir/bge-reranker-onnx --fixture /run-dir/fixture.json --json /run-dir/check-rerank.json

echo "== 4/4 image $IMAGE"
CTX="$RUN/context"
mkdir -p "$CTX/bge-m3-onnx" "$CTX/bge-reranker-onnx"
for f in model.onnx model.onnx.data model.fp16.onnx sparse_linear.npz tokenizer.json meta.json; do
    ln "$RUN/bge-m3-onnx/$f" "$CTX/bge-m3-onnx/$f"
done
for f in model.fp16.onnx tokenizer.json meta.json; do
    ln "$RUN/bge-reranker-onnx/$f" "$CTX/bge-reranker-onnx/$f"
done
python3 - "$CTX" "$RUN" "$TAG" "$COMMIT" "$EXPORTER" <<'PY'
import json, os, sys
ctx, run, tag, commit, exporter = sys.argv[1:]
out = {"tag": tag, "docs_bridge_commit": commit, "exporter_image": exporter, "models": {}}
for d in ("bge-m3-onnx", "bge-reranker-onnx"):
    meta = json.load(open(os.path.join(run, d, "meta.json")))
    shipped = sorted(f for f in os.listdir(os.path.join(ctx, d)) if f != "meta.json")
    out["models"][d] = {"model_id": meta["model_id"], "revision": meta["revision"],
                        "files": {f: meta["files"][f] for f in shipped},
                        "toolchain": meta["exported_with"]}
for kind in ("embed", "rerank"):
    c = json.load(open(os.path.join(run, f"check-{kind}.json")))
    out[f"check_{kind}"] = {k: v for k, v in c.items() if k != "queries"}
json.dump(out, open(os.path.join(ctx, "docs-bridge-models.json"), "w"), indent=2)
PY
docker build -q -f images/models/gpu.Dockerfile -t "$IMAGE" "$CTX"
docker image ls "$IMAGE"
echo "== done: $IMAGE   (record: $RUN)"

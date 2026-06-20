#!/usr/bin/env bash
# eval-onnx.sh — run the whole ONNX/INT8 embed evaluation on vhost2, side-by-side
# with the existing FlagEmbedding stack. Nothing here touches ingest-worker:latest
# or the `aig` collection; the ONNX vectors land in a separate collection (aig_onnx).
#
# RUN AFTER the FlagEmbedding benchmark finishes (conversion needs ~6-8 GB RAM).
# Run from the ingest-worker dir of the `embed-onnx-int8` branch checked out on vhost2.
#
# Stages (pass one, or 'all'):
#   build     build ingest-worker:onnx (FROM :latest + onnx/export extras + tools/)
#   convert   export official BAAI/bge-m3 -> ONNX + INT8 into the model cache
#   gate1     vector parity vs FlagEmbedding (HALTS the script if fp32 conversion wrong)
#   populate  re-embed aig -> aig_onnx with the INT8 model (also = embed-speed A/B)
#   gate2     retrieval parity: does aig_onnx return the same chunks as aig?
#   all       build -> convert -> gate1 -> populate -> gate2
#
# Usage:  sudo ./eval-onnx.sh all      (or: build | convert | gate1 | populate | gate2)
set -euo pipefail

STAGE="${1:-all}"

IMAGE=ingest-worker:onnx
BASE_IMAGE="${BASE_IMAGE:-docker.io/library/ingest-worker:latest}"
BUILD_CTX="${BUILD_CTX:-$PWD}"                       # the ingest-worker dir on vhost2

PAYLOAD=/data/docs-bridge-payload
CONFIG=/data/docker/docs-bridge/config/config.yaml
NET=docs-bridge-net

MODEL=/data/cache/bge-m3-onnx                        # container path (cache is mounted)
SUBJECT=aig
CAND=aig_onnx
N_PARITY=200
N_RETRIEVAL=100
K=10

# docker run with the qdrant network + cache + config mounted (gate1/populate/gate2).
dr() {
  sudo docker run --rm --network "$NET" \
    -v "$PAYLOAD/cache:/data/cache:z" \
    -v "$CONFIG:/config/config.yaml:ro,z" \
    --entrypoint python "$IMAGE" "$@"
}

banner() { echo; echo "==================== $* ===================="; }

do_build() {
  banner "BUILD $IMAGE (from $BASE_IMAGE)"
  sudo docker build -f Dockerfile.onnx --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t "$IMAGE" "$BUILD_CTX"
}

do_convert() {
  banner "CONVERT BAAI/bge-m3 -> $MODEL (fp32 + INT8)"
  # No qdrant network needed; cache holds the HF weights from the benchmark.
  time sudo docker run --rm \
    -v "$PAYLOAD/cache:/data/cache:z" \
    --entrypoint python "$IMAGE" \
    tools/export_bge_m3_onnx.py --out "$MODEL" --quantize
}

do_gate1() {
  banner "GATE 1 — vector parity (fp32 must pass; INT8 reported)"
  dr tools/validate_embed_parity.py \
    --config /config/config.yaml --subject "$SUBJECT" \
    --model-dir "$MODEL" --n "$N_PARITY"
}

do_populate() {
  banner "POPULATE $CAND with INT8 (= embed-speed A/B vs 1.64 s/chunk)"
  time dr tools/reembed_to_collection.py \
    --config /config/config.yaml --source "$SUBJECT" --target "$CAND" \
    --model-dir "$MODEL" --int8 --fresh
}

do_gate2() {
  banner "GATE 2 — retrieval parity ($CAND vs $SUBJECT)"
  dr tools/retrieval_parity.py \
    --config /config/config.yaml --ref "$SUBJECT" --cand "$CAND" \
    --model-dir "$MODEL" --int8 --n "$N_RETRIEVAL" --k "$K"
}

case "$STAGE" in
  build)    do_build ;;
  convert)  do_convert ;;
  gate1)    do_gate1 ;;
  populate) do_populate ;;
  gate2)    do_gate2 ;;
  all)
    do_build
    do_convert
    do_gate1      # set -e: a failed fp32 gate stops here, before populate/gate2
    do_populate
    do_gate2
    banner "EVAL COMPLETE — read Gate 1 + Gate 2 + the populate s/chunk together"
    ;;
  *) echo "unknown stage '$STAGE' (build|convert|gate1|populate|gate2|all)"; exit 2 ;;
esac

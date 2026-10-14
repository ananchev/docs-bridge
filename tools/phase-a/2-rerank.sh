#!/usr/bin/env bash
# Step 2 (runbook step 5): the same INT8 reranker bytes on the new onnxruntime — do the
# scores still match what the old runtime produced on the Gate 1 fixture? ~8 min.
set -uo pipefail
. "$(dirname "$0")/vars.sh"
cd "$SRC"

rab() {  # $1 output name  [$2...] extra args
  local name=$1; shift
  sudo podman run --rm -i -v $HOME/gate1/out:/fx:ro,z -v $OUT:/out:z \
    -v $SRC/notes/gate1:/ref:ro,z \
    --entrypoint python localhost/docs-bridge:$NEW - \
    --fixture /fx/fixture.json --backend onnx --model-dir /opt/bge-reranker-onnx \
    --threads 6 --repeats 1 --json /out/$name.json "$@" \
    < tools/eval/rerank_ab.py > $OUT/$name.txt 2>&1
}

say "new runtime vs the scores the old runtime produced (~4 min)"
rab 2-rerank-vs-old-int8 --selftest
tail -5 $OUT/2-rerank-vs-old-int8.txt
grep -q FAITHFUL $OUT/2-rerank-vs-old-int8.txt && ok "FAITHFUL" || bad "not FAITHFUL"

say "distance to the fp32 reference model (~4 min, reported, judged by me)"
rab 2-rerank-vs-fp32 --reference /ref/gpu-fp32.json
tail -8 $OUT/2-rerank-vs-fp32.txt
finish "Next: ./3-smoke.sh"

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

say "new runtime vs the scores the old runtime produced (~4 min; reported, not gated)"
rab 2-rerank-vs-old-int8 --selftest
grep -E 'SELFTEST' $OUT/2-rerank-vs-old-int8.txt
echo "(two INT8 runtimes may differ from each other by more than either does from fp32)"

say "both runtimes vs the fp32 reference (~4 min) — the gate"
rab 2-rerank-vs-fp32 --reference /ref/gpu-fp32.json
grep -E '^speed' $OUT/2-rerank-vs-fp32.txt
python3 $SRC/tools/phase-a/rerank_vs_fp32.py $HOME/gate1/out/fixture.json \
  $SRC/notes/gate1/gpu-fp32.json $OUT/2-rerank-vs-fp32.json | tee $OUT/2-rerank-analysis.txt
grep -q 'VERDICT: PASS' $OUT/2-rerank-analysis.txt \
  && ok "new runtime ranks as close to fp32 as the old one" || bad "new runtime ranks further from fp32"
finish "Next: ./3-smoke.sh"

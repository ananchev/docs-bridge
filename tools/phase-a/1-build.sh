#!/usr/bin/env bash
# Step 1 (runbook steps 2+3): check the frozen models image, build the two new images
# from this source, and check they carry the frozen bge-m3 bytes and this commit's code.
# ~10 min. Asks for your sudo password.
set -uo pipefail
. "$(dirname "$0")/vars.sh"
cd "$SRC"

say "frozen models image"
if sudo podman image exists $MODELS; then ok "$MODELS present"
else
  say "not present — loading it from the tar"
  sudo podman load -i $OUT/docs-bridge-models-frozen-2026-09.tar && ok "loaded" || bad "could not load $MODELS"
fi
[ "$FAILED" = 0 ] || finish

say "building localhost/docs-bridge:$NEW (server)"
sudo podman build -f images/docs-bridge/Dockerfile -t localhost/docs-bridge:$NEW . \
  > $OUT/1-build-server.log 2>&1 && ok "built" || bad "server build — see $OUT/1-build-server.log"
say "building localhost/ingest-worker:$NEW (worker)"
sudo podman build -f images/ingest-worker/Dockerfile -t localhost/ingest-worker:$NEW . \
  > $OUT/1-build-worker.log 2>&1 && ok "built" || bad "worker build — see $OUT/1-build-worker.log"
[ "$FAILED" = 0 ] || finish

say "checking what is inside both images"
want=$(grep bge-m3-onnx $OUT/02-frozen.txt | awk '{print $1}' | sort)
for img in localhost/docs-bridge:$NEW localhost/ingest-worker:$NEW; do
  echo "-- $img"
  got=$(sudo podman run --rm --entrypoint sh $img -c 'sha256sum /opt/bge-m3-onnx/*' | awk '{print $1}' | sort)
  [ "$got" = "$want" ] && ok "bge-m3 bytes = the frozen ones" || bad "bge-m3 bytes differ from 02-frozen.txt"
  info=$(sudo podman run --rm --entrypoint python $img -c '
from importlib.metadata import version as v, distributions
import docs_bridge.manifest as m
print(v("onnxruntime"), v("tokenizers"), v("numpy"), hasattr(m, "EmbedderMismatchError"),
      sum(1 for d in distributions() if "nvidia" in (d.metadata["Name"] or "").lower()))')
  echo "   onnxruntime tokenizers numpy new-code nvidia-pkgs: $info"
  [ "$info" = "1.30.0 0.23.2 2.4.6 True 0" ] && ok "versions, code and no CUDA as expected" \
    || bad "expected '1.30.0 0.23.2 2.4.6 True 0'"
done 2>&1 | tee $OUT/1-images.txt
say "worker image: a real substitute for non-embedded Courier (docling renders pages)"
mono=$(sudo podman run --rm --entrypoint fc-match localhost/ingest-worker:$NEW Courier 2>&1)
echo "   fc-match Courier -> $mono" | tee -a $OUT/1-images.txt
grep -qi "nimbus" <<< "$mono" && ok "Courier -> Nimbus Mono (URW)" | tee -a $OUT/1-images.txt \
  || bad "no URW Courier substitute in the worker image" | tee -a $OUT/1-images.txt
grep -q FAIL $OUT/1-images.txt && FAILED=1
if [ -s $OUT/2-rerank-analysis.txt ]; then
  finish "Next: ./8-reparse-pdfs.sh (2-rerank already passed; it tests the server image, unchanged by a worker fix)"
else
  finish "Next: ./2-rerank.sh"
fi

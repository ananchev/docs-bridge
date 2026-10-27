# docs-bridge-models:<gpu tag> -- the GPU target's models, fp16/fp32, from a pinned
# HF commit. Built ONLY by tools/onnx-export/build-gpu-models.sh, whose build context
# is the assembled export (never the repo): exactly the files below, already checked
# fp16-vs-fp32 on the CUDA provider. The INT8 line is the other Dockerfile here.
#
#   /opt/bge-m3-onnx/          model.onnx + model.onnx.data (fp32), model.fp16.onnx,
#                              sparse_linear.npz, tokenizer.json, meta.json
#   /opt/bge-reranker-onnx/    model.fp16.onnx, tokenizer.json, meta.json
#   /opt/docs-bridge-models.json   what is in this image: tag, commit, revisions,
#                              sha256 per shipped file, the precision checks' results
FROM scratch
COPY bge-m3-onnx/ /opt/bge-m3-onnx/
COPY bge-reranker-onnx/ /opt/bge-reranker-onnx/
COPY docs-bridge-models.json /opt/docs-bridge-models.json

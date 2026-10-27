# tools/onnx-export — findings register

What was found building the GPU target's model exports, and how each item was
resolved. Numbering is stable: code comments cite "finding N". The operating manual
is [README.md](README.md).

## Findings & peculiarities (and how we resolved them)

### 1. onnxruntime-gpu 1.30 is CUDA 13, not CUDA 12
Plan §6.3 assumed CUDA 12 wheels (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`). The
1.30.0 wheel's `[cuda,cudnn]` extras are `nvidia-cuda-runtime~=13.0` and
`nvidia-cudnn-cu13~=9.0`.
**Resolution:** the exporter installs `onnxruntime-gpu[cuda,cudnn]==1.30.0` on
`python:3.11-slim-bookworm` and calls `onnxruntime.preload_dlls()` before a session;
the host's driver (595.91, CUDA 13.2) runs it — CUDA provider confirmed on the T4
2026-10-27. No `nvidia/cuda` base image. The server and embedder images take the
same route.

### 2. transformers 5 traces the attention mask with tracer warnings
The TorchScript export of both models warns that `masking_utils.py` and
`sdpa_attention.py` turn tensor comparisons into constants. Traced on an unpadded
probe, that could have baked "no mask" into the graph: every padded batch would then
score wrong, and fp16 and fp32 would still agree with each other.
**Resolution:** checked — the reranker scores a padded batch of 8 identically to
scoring each pair alone (max |delta| 6e-6, fp32), and bge-m3 likewise (cosine
0.9999999). `check_precision.py` now checks padding invariance on every run, and
fidelity to the torch model (embed cosine 0.99999994, rerank |delta| 5e-6).

### 3. onnx writes the external-data file mode 0600
`onnx.save_model(..., save_as_external_data=True)` created `model.onnx.data` as
0600. Copied into a `FROM scratch` image, a non-root container user could not read
it — a failure that would show only at the embedder's first start.
**Resolution:** `_common.export_fp32` sets 0644 on both fp32 files.

### 4. Fixture candidates tie, so a 0.002 shift can reorder rank 6
fp16 vs fp32 on the reranker: max |logit delta| 0.017, but top-6 order differed on
`TC_transfer_area site preference`. There three candidates (one a 16-character
heading fragment) tie at 3.5554 in fp32, and a fourth sits ~0.005 below; fp16 moves
it 0.002 above them.
**Resolution:** `check_precision.py` judges numerics (|delta| ≤ 0.05) and only
reports top-6 agreement (7/8). Ordering is judged by `tools/eval/rerank_ab.py`
against the torch fp32 reference with the fixed judge over the deep pool
(gate1-runbook §5.5).

### 5. The export is deterministic
Two exports of the reranker in the exporter image produced byte-identical
`model.onnx` and `model.onnx.data`.
**Resolution:** a models image can be rebuilt from the exporter lock and the two
pinned revisions, so its export dir may live on the instance store. The sha256s in
`meta.json` are how a rebuild proves it made the same model — a different hash is a
different model identity, which for bge-m3 means re-embedding.

### 6. ORT fp16 runs the reranker at 0.85 s/query; torch fp16 did 0.54 s — open
On the T4, the fixture's 40-candidate queries take 3.00 s (fp32) and 0.85 s (fp16)
median on ORT CUDA, against Gate 1's 2.80 s / 0.54 s for torch. The plain fp16 graph
has no fused attention or LayerNorm kernels.
**Open:** a server-side matter (plan §8 3b item 1): `onnxruntime.transformers`'
BERT optimizer (fused attention) on top of the fp16 graph is the candidate, measured
with `rerank_ab.py`, and length-bucketing at 8 (Gate 1) still applies. If it changes
the graph, the result is a new file and a new tag.

### 7. fp16 moves bge-m3's sparse weights more than its dense vector
Dense: fp16 vs fp32 cosine ≥ 0.99985 on 256 real texts. Sparse: the per-token weight
`relu(linear(h))` moves by up to 0.070, where the largest weight is 0.378.
**Open:** an input to the embedder's precision choice (fp16 vs fp32, plan §8 3b),
decided on the labelled 14.2 question set — the sparse channel is where an fp16
embedder would show it. fp16 embeds 3.6x faster (2.4 s vs 8.7 s for 256 texts).

### 8. torch 2.14 deprecates the TorchScript exporter
`torch.onnx.export` has defaulted to the torch.export ("dynamo") exporter since 2.9;
the scripts pass `dynamo=False` and get a DeprecationWarning.
**Resolution:** kept for now — it is the exporter every model dir so far came from.
When it is removed, switching exporter produces a different graph, i.e. a new model
identity: re-run both checks and treat bge-m3 as a re-embed.

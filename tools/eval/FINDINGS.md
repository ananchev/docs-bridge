# tools/eval — findings register

What was found using the evaluation harnesses, and how each item was resolved.
Numbering is stable. The operating manual is [README.md](README.md).

## Findings & peculiarities (and how we resolved them)

### 1. On a near-tie at rank k, a single query's verdict is rounding noise
`TC_transfer_area site preference` has four byte-identical `Site preference.`
fragments tied at rank 6 in fp32 (3.5554), with another chunk ~0.005 below. On
2026-10-27 five fp16 variants of the same model split PASS/FAIL on that query with no
pattern — unbucketed fails and bucket 8 passes, with LayerNorm fused the reverse —
because each moves the scores by a few thousandths (`notes/gate1-runbook.md`,
"2026-10-27").
**Resolution:** the bar is applied as set; a FAIL caused by such a flip is reported as
a FAIL, with the tie shown, and the configuration decision rests on the
configuration actually deployed. The cause — the chunker emitting heading fragments
that score like content — is a Gate 2 (chunking) item, not a reranker one. Before
reading any rank-k verdict, check the reference's gap at rank k.

### 2. A reranker comparison must reach the production class in its image
`rerank_ab.py`'s ONNX backend re-implements `OnnxReranker` so it can run anywhere,
which also means it cannot show what the server image's onnxruntime, CUDA libraries
and model file do.
**Resolution:** `--backend docs-bridge` imports `docs_bridge.rerank.OnnxReranker` and
passes `--bucket` to it; run inside the server image it scored identically (max |Δ|
0.0) to the re-implemented backend on the same model and settings.

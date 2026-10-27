# images/embedder — findings register

What was found building the embedder service and wiring the worker and server to it,
and how each item was resolved. Numbering is stable. The operating manual is
[README.md](README.md).

## Findings & peculiarities (and how we resolved them)

### 1. The fingerprint hashed the fp32 graph, not its weights
`OnnxEmbedder.fingerprint` hashed `model.int8.onnx` or `model.onnx`. The INT8 file is
self-contained, but an fp32 export keeps its weights in `model.onnx.data` (2.3 GB;
protobuf's 2 GB limit): hashing `model.onnx` alone fingerprints the ~1 MB graph, so
two fp32 models with different weights would have stamped identically. It also
recorded `int8: true|false` — fp16 and fp32 would both read `false` — and not the
execution provider.
**Resolution:** the fingerprint hashes every file the precision runs
(`embed_onnx.FILES`: fp32 = `model.onnx` + `model.onnx.data`), full sha256, plus the
sparse head and tokenizer, and records `precision`, `provider` and the onnxruntime
*build* (`onnxruntime-gpu==1.30.0`, not just the version). Where `meta.json` records
a sha256, the bytes on disk must match it. Hashing takes 6.1 s (fp32) and 3.1 s
(fp16) at the service's start (T4 host, image layers on the root EBS).

### 2. Over HTTP the vectors are bit-identical to an in-process run
Measured 2026-10-27 on the T4, 256 Gate 1 fixture texts in batches of 32, the
service against `OnnxEmbedder` in the same image: dense max |Δ| 0.0, 0 sparse vectors
differing, at fp16 and fp32. JSON carries each float32 as the double it widens to,
and `repr` round-trips doubles exactly. Cost: ~3% on a batch (fp16 5.99 s vs 5.73 s
for the 256 texts), and a single query embeds in 7.8 ms (fp16) / 11.0 ms (fp32)
including the HTTP round trip.
**Resolution:** JSON, no binary encoding. The service changes where the model runs,
not what it computes.

### 3. A refused embedder could leave changed docs without points
`sync` deleted the old points of changed docs and only then, in `_drain_staged`,
checked the collection's stamp. A worker on the wrong embedder therefore deleted the
changed docs' points and stopped — found by pointing a worker at the fp32 service
after the collection was filled at fp16. The next run with the right embedder resumed
the staged chunks, so nothing was lost for good, but in between those docs were not
searchable.
**Resolution:** `sync._stamped_embedder` checks the stamp before PASS 2 deletes
anything, and the resume path deletes the staged docs' points before draining them —
a refused run leaves the manifest holding the new hash while Qdrant still holds the
old chunks. Verified: fp16 collection, `server.md` shrunk from 12 to 4 chunks, sync
against fp32 refused (63 points, server.md 12, unchanged), sync against fp16 resumed
(55 points, server.md 4, nothing stale), next sync a no-op.

### 4. onnxruntime 1.30 tries to persist a telemetry device id
As `nobody` (no writable home), the service logs "Failed to persist telemetry device
ID; using an in-memory identifier" at start.
**Resolution:** none needed — a warning, the session runs. Not a reason to give the
container a writable home.

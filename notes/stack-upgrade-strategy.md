# docs-bridge stack upgrade strategy

**Goal.** Move the ingest stack from docling 2.15.1 (2025-01-10) to current
(2.130.0, Aug 2026) **without disturbing the existing `teamcenter` corpus**, and pick
up the rich-content handling that arrived in the intervening ~20 months.

**Parallelism.** This work is independent of the work-archive build
(`notes/work-archive-ingest-plan.md`) until that plan's step 4. Steps 1–2 there
(keep-list, LibreOffice conversion) touch no code and can run concurrently. The only
ordering constraint: **the upgrade must land before the archive bulk ingest**, so the
new corpus is parsed once, by the final toolchain.

The Pi is out of scope — vhost2 (M2 Pro, Asahi Fedora, aarch64) is the only runtime
host.

---

## 1. Why upgrade

Current pin: `docling==2.15.1`, `docling-core[chunking]==2.14.0`,
`docling-ibm-models==3.1.2`, `docling-parse==3.0.0` (pyproject `[ingest]` extra).
Everything below landed after that pin.

| Capability | Landed | Relevance |
|---|---|---|
| PPTX presenter/speaker notes | PR #474, merged 2025-03-19 | **Not extracted today.** For decks the narrative lives in the notes; slide bodies are bullet fragments. 748 PPTX in the archive. |
| Enrichment for formats without page images (DOCX/HTML) | **2.52.0** (2025-09-11), PR #2251 | Prerequisite for any image handling in Office files |
| Enrichment extended to PPTX/XLSX | **2.55.x**, PR #2401 | ditto |
| `do_picture_description` + picture serializers | — | The only route from an embedded image to indexed text |
| RapidOCR (ONNX backend) | — | Replaces torch-based EasyOCR; fits the all-ONNX ethos of this stack |
| `docling-slim[...]` extras split | — | Fine-grained installs; a path to dropping torch from the worker image |
| DOCX/PPTX backend fixes | ~115 releases | List groups across tables, OOXML heading-level clamping, slide positions, grouped shapes |

### Images in Office files are parsed and then discarded today

Verified against the v2.15.1 source: `msword_backend.py:513` and
`mspowerpoint_backend.py:273` both call `doc.add_picture(… ImageRef.from_pil(…))`, so
the bytes reach the `DoclingDocument`. But with no enrichment a `PictureItem`
serializes to nothing, and `src/docs_bridge/parse.py:107` drops empty chunks:

```python
text = (ch.text or "").strip()
if not text:
    continue
```

Every diagram, screenshot and chart in a DOCX or PPTX is extracted and thrown away.

### A correction to the code comment

`src/docs_bridge/parse.py:80` says *"OCR (EasyOCR) runs on every page by default"*.
That is wrong, and wrong in our favour. In v2.15.1's `base_ocr_model.py`, with
`force_full_page_ocr=False`:

- bitmap coverage > `BITMAP_COVERAGE_TRESHOLD` (0.75) → OCR the full page
- coverage between `bitmap_area_threshold` and 0.75 → OCR **only the detected bitmap
  rects**
- below → no OCR at all

For digital PDFs with a few figures, OCR runs on the figures only. OCR was disabled
based on a cost higher than the real one. Fix the comment as part of this work.

---

## 2. What is at risk — and what is not

### Not at risk: the existing chunks

Hash-delta classifies unchanged files as unchanged, so `teamcenter` never re-parses and
keeps its current chunks and point ids. A new `HybridChunker` affects only newly
ingested documents. The archive corpus is a separate collection. **A docling upgrade
alone cannot disturb `teamcenter`.**

### At risk: the embedder, via transitive pins

The danger is not docling — it is what docling drags under the embedder:

| Current pin | Modern requirement | Consequence if it moves |
|---|---|---|
| `onnxruntime==1.20.1` (base dep) | `docling-slim[models-onnxruntime]`, `[feat-ocr-rapidocr-onnx]` want `>=1.24.1` | Runtime under the validated BGE-M3 INT8 embed **and** the INT8 reranker changes |
| `tokenizers==0.19.1` (base dep, deliberate) | `docling-core>=2.96` pulls newer `transformers`/`tokenizers` | Query vectors could drift from the 137,013 stored `teamcenter` doc vectors |

pyproject already spells out why that tokenizers pin exists:

> MUST match what ingest tokenized with, or query vectors drift from doc vectors.

Note the plain `docling-slim[standard]` extra does **not** pull onnxruntime — the
conflict only bites when taking the ONNX OCR/VLM extras, which is exactly what we want.
So the conflict is real, not hypothetical.

`docling-slim[standard]` still pulls `torch>=2.2.2`, so the worker image does not shrink
from the extras split by itself.

---

## 3. The parity gate

The gate already exists — this is the same methodology that cleared INT8 adoption
(`docs-bridge-fastembed-evaluation`, Gates 1 and 2).

**Gate A — vector parity (blocking).** `tools/eval/validate_embed_parity.py`, sampling
real chunk texts from the populated `teamcenter` collection, old runtime as ground truth
vs new. **Pass condition: dense cosine ≥ 0.999 and sparse Jaccard ≥ 0.95.** The baked
`model.int8.onnx` is byte-identical across the upgrade, so this is purely a runtime /
tokenizer question; it should pass trivially, and if it does not, we learn before
spending ~27 h of archive ingest.

**Gate B — retrieval parity (advisory).** `tools/eval/retrieval_parity.py`,
chunks-as-queries overlap@10 against `teamcenter`. Only worth running if Gate A comes
back below 1.0.

If Gate A fails, the fallback is to hold `tokenizers`/`onnxruntime` and take
`docling-slim[standard]` without the ONNX extras — losing RapidOCR but keeping the
upgrade. Decide that only if it happens.

---

## 4. Rich content — what to turn on

Cost ordering matters. On Asahi there is no Metal/MPS, so MLX is unavailable and every
model runs on CPU. Docling's own published benchmarks (M3 Max, one page) put the
whole-page VLM pipeline at **102 s/page via Transformers** vs 6.2 s/page via MLX — the
CPU path is the one we would be on. A whole-page VLM pipeline is not viable here and is
not proposed.

Picture *description* is different: it runs per extracted figure, not per page.

**Tier 1 — free, do it.**

- Upgrade → PPTX speaker notes, DOCX/PPTX backend fixes.
- Register format options for the Office formats. `parse.py:85` currently passes
  `format_options` for `InputFormat.PDF` **only**, which silently no-ops every
  enrichment flag on DOCX/PPTX — the exact bug tracked as docling-jobkit #145. The fix
  shape:
  ```python
  format_options[InputFormat.DOCX] = WordFormatOption(pipeline_options=opts)
  format_options[InputFormat.PPTX] = PowerpointFormatOption(pipeline_options=opts)
  ```
- Extract authored **alt-text**. DOCX and PPTX images carry a `descr` attribute a human
  wrote. Zero inference cost, better than any generated caption, and docling ignores it
  in every version. Needs a small preprocessor or a custom picture serializer.

**Tier 2 — cheap, targeted.**

- `parse.ocr: true` with RapidOCR (ONNX). Selective bitmap OCR, and only ~7% of the
  archive's PDFs are image-only. Per docling's paper, disabling OCR saves ~60% of
  runtime on x86 — so the delta here is bounded and small.

**Tier 3 — the picture-description chain.**

The full path exists upstream and is documented: `do_picture_description` populates
`item.meta.description.text`; a ~20-line `AnnotationPictureSerializer` subclassing
`MarkdownPictureSerializer` emits it into chunk text; passed to the chunker via
`HybridChunker(serializer_provider=…)`. Without that serializer the descriptions are
generated and then dropped — same failure as today.

Two backend choices for the description model:

| Option | Fit | Cost |
|---|---|---|
| `granite-docling-258M` ONNX/INT8, self-exported and baked | Maps exactly onto the existing two-stage Dockerfile + `tools/onnx-export/` pattern used for bge-m3 and bge-reranker | CPU inference per figure; needs its own measurement |
| Remote VLM via `PictureDescriptionApiOptions` + `enable_remote_services=True`, pointed at OpenRouter | The key is already in the stack (`librechat_secrets.openrouter_api_key`), and retrieved corpus text already goes to OpenRouter at query time | One-shot per figure, hash-keyed; ingest is batch so latency is irrelevant |

The remote option is materially cheaper and the exposure is a strict subset of what
already leaves the LAN per query — but it is a deliberate decision, especially given the
client-confidential material flagged in the archive plan §6. Decide before the bulk run,
because re-running enrichment means re-parsing.

Defer Tier 3 until Tier 1–2 are in and the pilot has measured real figure counts.

---

## 5. Sequence

1. **Branch.** `docling-upgrade` off `main`, same shape as `embed-onnx-int8`.
2. **Bump the pins** in `pyproject.toml` `[ingest]`: `docling`, `docling-core`,
   `docling-parse`, `docling-ibm-models`. Resolve the `onnxruntime` / `tokenizers`
   conflicts deliberately, and record the decision in the pyproject comments the way the
   existing pins are documented.
3. **Re-verify the two API notes** in `parse.py`'s module docstring —
   `docling.chunking.HybridChunker` and `chunk.meta.headings` — which the docstring
   already flags as things to re-check on a bump.
4. **Gate A.** Build the image, run `validate_embed_parity.py` against `teamcenter`.
   **Stop here if it fails.**
5. **Tier 1 + 2** — format options for DOCX/PPTX, alt-text, RapidOCR, fix the OCR
   comment at `parse.py:80`.
6. **Smoke-test** on a handful of representative archive files (one deck with notes, one
   DOCX with figures, one scanned PDF) before any bulk run.
7. **Hand off** to the archive plan's step 4 (pilot library).
8. **Tier 3** — only after the pilot produces real figure counts and parse timings.

### Deliberately not doing

- **Switching parser.** MinerU scores higher on READoc (~73 vs ~59) and Marker is the
  safer single-tool default, but this stack's citations, `section_path` and manifest are
  all built on `DoclingDocument`. Moving sideways means re-validating retrieval end to
  end for a benchmark delta that does not describe our corpus. Forward on docling.
- **Re-ingesting `teamcenter`.** It works, it stays. Any future re-ingest is its own
  decision with its own ~13 h price tag (137,013 chunks × 0.342 s).

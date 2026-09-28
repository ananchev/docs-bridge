# Work archive → docs-bridge: ingest plan

**Scope.** Build a second docs-bridge corpus from the frozen SKF and Hexagon (HxGN)
work archives living on the Nextcloud host (`vhost`, 192.168.2.3) under
`/nvme/containers/nextcloud/data/ananchev/files/{SKF,HxGN}`.

**Non-scope.** The existing `teamcenter` corpus is untouched by everything in this
document. Nothing here re-parses, re-embeds or re-chunks it. See
`notes/stack-upgrade-strategy.md` for the one thing that *could* affect it (transitive
pin drift under the embedder) and the gate that catches it.

Measurements in this document were taken 2026-09-26 by scanning the live trees.
Projections are marked as such — replace them with pilot measurements before
committing to the bulk run (step 4).

---

## 1. What is actually there

**18,867 files, 46.32 GB.** Almost none of it is text.

| Class | Files | Size |
|---|---:|---:|
| PDF / DOCX / PPTX / HTML / TXT | 9,536 | 13.09 GB |
| XLSX | 932 | 1.41 GB |
| Legacy Office (doc, ppt, xls, lwp, prz) | 907 | 1.44 GB |
| Video / audio (mp4, trec, wrf) | 124 | 9.66 GB |
| Archives (zip) | 144 | 5.46 GB |
| Loose images (png, jpg, gif, tif, ai, svg) | 3,158 | 0.70 GB |
| CAD (rvm, nwd, dgn, sldprt, par, asm, dwg, dft) | 943 | 1.06 GB |
| Email + OneNote (eml, msg, onepkg, `.1`) | 528 | 0.60 GB |
| Other (2 Windows/VS **ISOs** = 5.73 GB, ppsx, accdb, exe, xaml, js) | 2,283 | 12.63 GB |

### Three structural findings

**The corpus is frozen.** File mtimes run 2014–2019 (2018 alone: 6,875 files /
15.2 GB); after 2019 there are 9 files total. This is an archive, not a living
document set.

> Consequence: no nightly timer, no hash-delta churn, no incremental design. It is a
> one-shot build paid for once — which also makes expensive one-time enrichment
> (OCR, picture description) affordable in a way it never is for a corpus that
> re-syncs.

**The taxonomy already exists.** The top level is ~40 SharePoint/Teams document
libraries named by engagement — `SKF/Portfolio - PLM Portfolio Documents`,
`SKF/PhoenixPilot - WP1 DTw Prototype`, `SKF/PLM - CAx Documents`,
`HxGN/Total Quantum - Production Pilot Work Documents`,
`HxGN/EQUINOR - MEDB RFP - Partner Collaboration`, `HxGN/Bayer EcoSys - Documents`.
Nothing needs inventing; the job is to *preserve* what is there.

**OneDrive is the mess.** `SKF/OneDrive` = 8,548 files / 17.07 GB and
`HxGN/OneDrive` = 911 / 5.19 GB — half the byte total. It holds the two Windows/Visual
Studio ISOs (5.73 GB), the conference videos, `Expenses/`, `GoodNotes/`,
`timesheet.xlsx`, and a scraped WordPress site (2,039 `.htm` from `gpm.skf.com`) —
mixed in with genuinely good material: `Projects data` (1,507 files), `Work Documents`,
`Roadmap and Transformation Planning`, `Windchill`, and HxGN `Products Documentation`.

---

## 2. The filtering funnel

Mechanical filters only — no content judgement. This is what
`tools/ingest/archive-keeplist.py` produces against the live manifest:

```
input    18,867 files   46.32 GB
KEEP      5,978 files   11.84 GB
DROP     12,889 files   34.48 GB

dropped by reason:
  ext               12,239 files   30.48 GB   not a document/spreadsheet format
  noise                216 files    0.12 GB   Expenses, GoodNotes, ISOs, scraped site
  dup                  174 files    0.25 GB   same basename + same size
  junk-path            134 files    2.31 GB   under obsolete/old/archive/...
  sheet-too-big        110 files    1.32 GB   spreadsheet over 2 MB
  empty                 16 files    0.00 GB
```

**46.32 GB → 11.84 GB / 5,978 files**, a 74% cut by volume.

`dup` looks small at 174 files because the reasons are evaluated in order and most
duplicates — the repeated videos and zips — are already gone under `ext`. Counted
across the whole archive before any other filter, duplicate basename+size pairs account
for **2,911 files / 4.64 GB**: OneDrive and the SharePoint libraries hold the same
material, and items such as `HEXAGON_PPM Equinor Day 1 Presentation V1.0.pptx` exist in
two directories plus an `Obsolete/` sibling.

Final set by extension:

| Ext | Files | Size |
|---|---:|---:|
| pdf | 2,217 | 2.14 GB |
| docx | 1,426 | 0.70 GB |
| pptx | 747 | 7.48 GB |
| xlsx | 739 | 0.11 GB |
| doc | 487 | 0.56 GB |
| txt | 124 | 0.04 GB |
| xls | 99 | 0.03 GB |
| xlsm | 76 | 0.02 GB |
| ppt | 60 | 0.77 GB |
| rtf | 3 | 0.00 GB |

And by top-level library — the shape of the eventual per-library ingest batches:

```
2,207 files   2.65 GB  SKF/OneDrive
  438 files   1.32 GB  HxGN/OneDrive
  599 files   1.12 GB  HxGN/Total Quantum - Documents
  301 files   0.92 GB  HxGN/Total Quantum - Production Pilot Work Documents
  219 files   0.52 GB  SKF/Portfolio - PLM Portfolio Documents
  167 files   0.86 GB  HxGN/EQUINOR - MEDB RFP - General
  131 files   0.45 GB  HxGN/BASF RFI - Documents
  125 files   0.11 GB  HxGN/Total Quantum - Qualification Pilot Deliverables
  124 files   0.36 GB  SKF/PLM - PLM Team Documents
  105 files   0.11 GB  HxGN/Aker BP EcoSys - Aker BP Collaboration
   84 files   0.21 GB  SKF/PhoenixPilot - WP1 DTw Prototype
   66 files   1.35 GB  HxGN/EQUINOR - MEDB RFP - Partner Collaboration
   65 files   0.52 GB  SKF/PLM - Operations and Support Documents
   15 files   0.30 GB  SKF/PLM - Project Assurance Documents
```

Note `SKF/OneDrive` still carries 2,207 files after the mechanical pass — it is the one
library where a human eye is genuinely needed (§8).

7.48 GB of the remainder is PPTX with embedded video — 747 decks whose *text* totals
about 6 million characters. The information content of the whole archive is roughly
**75 million characters ≈ 40 MB of text**.

### Moving it

Copy the filtered 11.8 GB to the runtime host (`vhost2`) once. Do not build a mount,
do not build a sync: the corpus is frozen, so a single `rsync --files-from=<keeplist>`
is the entire transfer story.

**The filtering cannot live in `config.yaml`.** The xlsx cut is size-based and the
config's `include`/`exclude` are fnmatch globs matched against the `doc_id`
(`src/docs_bridge/parse.py:_included`). Filtering therefore belongs in the copy stage —
a new sibling of `tools/ingest/ingest-copy.sh`. Unlike the existing flows it should
**not** be Nextcloud-tag-driven: tagging 5,856 files by hand is not viable. Drive it
from the keep-list file instead (§8).

---

## 3. The spreadsheet trap

Read this before starting. Text volume per file type, sampled on `vhost`
(OOXML = `zipfile` + tag-strip, stdlib; PDF = `pypdf` in a throwaway venv):

| Type | Files | Median chars | Mean chars | Est. total | **Est. chunks** |
|---|---:|---:|---:|---:|---:|
| PDF | 2,510 | — | 20,374 | 51.1 M | **~32,000** |
| DOCX | 1,520 | 3,646 | 11,542 | 17.5 M | ~11,000 |
| PPTX | 784 | 3,019 | 7,805 | 6.1 M | ~3,800 |
| **XLSX** | 932 | 17,192 | **1,326,681** | **1,236 M** | **~773,000** |

The xlsx mean is **77× the median**. Distribution: p10 = 1.0k, p50 = 17.2k,
p75 = 121k, p90 = 4.0 M, p99 = 42.0 M, max = **46.0 M** characters.

The fat tail is one thing — dated weekly data-migration exports:

```
45.96 Mchars  HxGN/…/7- Data migration/02 - TEPUK - Elgin Franklin/20190612_ADL Export_Elgin Wellhead Platform A.xlsx
43.89 Mchars  HxGN/…/20190605_ADL Export_Elgin Wellhead Platform A.xlsx
42.04 Mchars  HxGN/…/20190522_ADL Export_Elgin Wellhead Platform A.xlsx
35.08 Mchars  HxGN/…/MERIDIAN_EDMS Database/DocList_Full_AO-STA.xlsx
```

Ingested naively these produce **~773,000 chunks — 5.6× the entire existing
`teamcenter` corpus (137,013)** — of near-identical asset registries with no prose
value, costing ~73 h of embed on their own.

**79% of spreadsheets are ≤ 200k chars (mean 28k ≈ 17 chunks).** A single size
threshold collapses the problem and keeps everything worth having. The funnel uses
2 MB on file size as the proxy; tighten to a char threshold if you want precision.

---

## 4. Cost model

Filtered set: **~58,000 chunks** over **~58,000 PDF pages**.

Using measured numbers from this stack (`docs-bridge-build-validation-state`): docling
parse = **1.23 s/page** on vhost2 (21.5 min for a 1,050-page PDF, OCR off, docling
2.15.1); BGE-M3 INT8 embed = **0.342 s/chunk** on vhost2.

| Pass | Estimate |
|---|---|
| Docling parse — PDF pages | **~20 h** |
| Docling parse — Office | ~1–2 h |
| BGE-M3 INT8 embed | **~5.5 h** |
| **Total wall clock** | **~27 h**, one-shot |

### The bottleneck inverts

On `teamcenter`, embed dominated and parse was noise — that corpus is mostly generated
Markdown (Doxygen/Javadoc). Here it is 58,000 real PDF pages, and **parse costs 3–4×
the embed**. Plan capacity, batching and babysitting around pass 1, not pass 2.

### Pass 1 has no resume path

This is the failure shape that produced the 37-silently-missing-PDFs incident
(`manifest-vs-qdrant-silent-drift`) and the resume gap (`docs-bridge-ingest-resume-gap`).
A 20 h pass 1 is exactly the run that gets killed two thirds of the way through.

**Discipline:** ingest one top-level library at a time, and run
`tools/ingest/verify-corpus.py` after every batch. Treat a clean exit as the unit of
work; never trust a healthy-looking manifest total.

---

## 5. Formats needing pre-conversion, and what gets dropped

Docling has no binary-OLE support. These need a one-shot LibreOffice pass
(`soffice --headless --convert-to docx|pptx|xlsx`) before ingest:

| Ext | Files | Size | Note |
|---|---:|---:|---|
| doc | 487 | 0.557 GB | includes `SKF/PLM - Operations and Support Documents` |
| ppt | 61 | 0.774 GB | |
| xls | 99 | 0.027 GB | |
| **total** | **647** | **1.36 GB** | real content — worth the conversion |

LibreOffice is **not installed on either host**. It is only needed for this one pass;
run it on vhost2 and discard.

Dropped for v1 (no docling support, low value, or both):

| Ext | Files | Verdict |
|---|---:|---|
| eml, msg | 359 | drop — email threads, poor chunk shape |
| vsdx, vsd | 257 | drop — diagrams, no text layer worth having |
| `.1`, onepkg | 167 | **reconsider** — OneNote; `Portfolio Notebook`, `PLM Management Notebook`, `PLM \| Unite`. Likely your own meeting notes = highest signal-per-byte in the archive. No docling path; needs a OneNote export step. |
| mpp | 55 | drop — MS Project schedules |
| lwp, prz | 170 | drop — Lotus WordPro/Freelance; LibreOffice reads `.lwp` only |

### Scanned PDFs

**7% of PDFs are image-only (~169 files corpus-wide)** — lower than expected. They are
mostly signed deliverables (`01.Business Requirements for EcoSys Implementation
030818_signed.pdf`), scanned handwritten notes, and P&ID drawings. Only 2% are "mixed"
(>30% low-text pages without being fully scanned).

This is a small, bounded problem. Docling's selective bitmap OCR handles it — OCR runs
only on detected bitmap regions unless `force_full_page_ocr` is set. It is **not** a
reason to reach for a VLM pipeline. See `notes/stack-upgrade-strategy.md` §4.

---

## 6. Corpus organization

**One subject, not forty.** The `doc_id` is the path relative to the subject dir
(`parse.py:doc_id_for`), so `HxGN/Total Quantum - Production Pilot Work Documents/7-
Data migration/…` already appears in every citation. The taxonomy travels in the
`doc_id`; splitting into per-project collections buys only the ability to scope a query
to an engagement you would rarely know in advance. Hybrid retrieval + the INT8
cross-encoder handle the discrimination.

Two cases where a split does earn its keep:

**Client-confidential material.** `EQUINOR - MEDB RFP`, `BASF RFI`, `Total Quantum`,
`Bayer EcoSys`, `Aker BP EcoSys` are commercial RFP and deliverable content from named
customers. The query path ships retrieved chunks to OpenRouter for synthesis via
LibreChat (`librechat_secrets.openrouter_api_key`). Splitting `work` / `work-client`
lets the second be queried only against a local model. A decision to make once, up
front — not a reason to skip ingesting.

**Reference-shaped bodies that would dominate retrieval.** Two candidates:

- **IBM Global Services Method** — 1,978 HTML files under
  `SKF/OneDrive/Work Documents/IBM Global Services Method/` (dated 2004). A genuine
  methodology reference, but generic and 22 years old; left in the main subject it will
  swamp any query about "method" or "delivery approach".
- **HxGN Products Documentation** — 226 files, 0.63 GB, vendor product docs rather than
  your own work.

Both are the same shape as the existing Doxygen/Javadoc corpora: reference, not
experience. Suggested layout:

| Subject | Collection | Content |
|---|---|---|
| `work` | `work` | SKF + HxGN internal/own-work libraries |
| `work-client` | `work_client` | named-customer RFP and deliverable material |
| `work-method` | `work_method` | IBM GSM + vendor product documentation |

Start with `work` alone; add the others only once the first is proving useful.

---

## 7. Execution sequence

1. **Build and review the keep-list** (§8). One file, ~5,900 lines. This is the only
   step where judgement is irreplaceable — everything downstream is mechanical. Do it
   before touching any code.
2. **LibreOffice pass** on the 647 legacy OLE files, on vhost2.
3. **Upgrade the stack** on a branch and run the embed-parity gate — see
   `notes/stack-upgrade-strategy.md`. This can proceed in parallel with steps 1–2; it
   only has to land before step 4.
4. **Pilot one library end to end.** `SKF/Portfolio - PLM Portfolio Documents`
   (215 files, 0.52 GB) is the right shape. Measure real parse s/page and chunk yield,
   and replace the projections in §4 before committing to the bulk run.
5. **Bulk ingest, one top-level library per run**, with `verify-corpus.py` after each.
6. **Smoke-test retrieval** in the LibreChat agent against a handful of questions you
   already know the answers to, before declaring the corpus done.

---

## 8. The keep-list

`tools/ingest/archive-keeplist.py` applies the §2 funnel to a `find` manifest and emits
a reviewable decision file. Standalone, stdlib only, never imports `docs_bridge` —
same convention as `doxy2md.py` / `javadoc2md.py`.

```bash
# 1. on the Nextcloud host - build the manifest
cd /nvme/containers/nextcloud/data/ananchev/files
find SKF HxGN -type f -printf '%s\t%TY-%Tm\t%p\n' > /tmp/corpus-manifest.tsv

# 2. generate decisions + the funnel summary
python3 archive-keeplist.py /tmp/corpus-manifest.tsv -o /tmp/keeplist.tsv --stats

# 3. review - flip column 1 by hand where you disagree

# 4. emit the rsync input and copy
python3 archive-keeplist.py --emit-paths /tmp/keeplist.tsv > /tmp/keep-paths.txt
rsync -a --files-from=/tmp/keep-paths.txt ./ vhost2:/data/docs/work/
```

**Format** — one row per file, tab-separated, with a header line:

```
decision  reason          bytes     month    path
KEEP      -               17605     2019-02  HxGN/Aker BP EcoSys - .../Actuals reporting Wellhead Platform Alliance.xlsx
DROP      sheet-too-big   4340478   2019-05  HxGN/Aker BP EcoSys - .../Combined esi.manage template v3.xlsm
KEEP      -               539816    2018-11  HxGN/Aker BP EcoSys - .../Cost management procedure.pdf
```

`decision` is the only column read back, so hand-editing is safe — sed, grep, a
spreadsheet, whatever. Use `--explain SUBSTRING` to see why any particular set of paths
was decided as it was.

**Re-running is safe.** A plain re-run regenerates decisions from scratch;
`--merge <old-keeplist>` diffs the old file against a fresh generation and carries
forward only the rows where you disagreed with the tool, marking them `manual`. So the
policy can be edited and re-run without losing review work.

### The policy block

The top of the script holds the whole policy as editable constants — it encodes this
archive's shape, not a law:

| Constant | Current value | What to reconsider |
|---|---|---|
| `DOC_EXTS` | pdf docx pptx doc ppt rtf odt odp md txt | add `htm/html` if you want the IBM GSM set (§6) |
| `SHEET_EXTS` | xlsx xls xlsm ods | drop entirely if spreadsheets prove worthless in the pilot |
| `SHEET_MAX_BYTES` | 2 MB | the §3 threshold; tighten to a char count for precision |
| `JUNK_SEGMENTS` | obsolete, old, archive, backup, superseded, deprecated, trash, temp, tmp | `old` is aggressive — check `--explain /old/` before trusting it |
| `NOISE_SUBSTRINGS` | Expenses, GoodNotes, SKF Visual Studio, GPM2 Website, timesheet | grows as you review |

### Where to spend review effort

The mechanical pass is good enough everywhere except one place. Ranked:

1. **`SKF/OneDrive` — 2,207 files / 2.65 GB kept.** Over a third of the surviving
   file count, and the least structured part of the archive. This is where to spend
   your attention. Good subtrees: `Projects data`, `Work Documents`,
   `Roadmap and Transformation Planning`, `Windchill`, the PI Munich/Berlin and PTC
   Forum presentation sets. Junk already caught: ISOs, `Expenses`, `GoodNotes`,
   the scraped `GPM2 Website`.
2. **`HxGN/OneDrive` — 438 files / 1.32 GB.** `_Desktop/` is a scratch dump;
   `Products Documentation` and `Templates` are real.
3. **The named-client libraries.** Not a quality question but a confidentiality one —
   decide here whether they go to `work-client` (§6) or are excluded.
4. **Everything else.** The SharePoint libraries are already curated; take the
   mechanical result.

A reasonable first pass is to drop nothing further, ingest the pilot library, and only
come back to trim once you have seen what retrieval actually surfaces.

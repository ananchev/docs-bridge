# Step 7 gate — adjudication (2026-10-16)

Run: old image + old `teamcenter` vs new image + `teamcenter-2512` (single pool, top_n 40
both sides), 45 draft-labelled questions, after the font fix and the PDF re-parse.

| | old | new |
|---|---|---|
| hit@6 | 41/42 | 39/42 |
| MRR | 0.76 | 0.74 |
| controls abstaining | 2/3 | 2/3 (same one, tc-c02 nginx, answers in both) |
| median latency | 14.7 s | **7.9 s** |

Missed by both: tc-025. The tool's verdict FAIL names two questions; both were traced
through the real retrieve → fuse → rerank path (tools: `~/phase-a/diag/search_probe.py`,
`chunk_text.py` on vhost2), against both collections.

## tc-002 "How to read BOM data from Teamcenter in AIG" — not a regression

The labelled chunk (`IndexedFieldMapping`, T4S/T4EA Basic Configuration Guide) is in the
new index with the same body text. Old: dense 51 / sparse 15 → fused #22 → rerank #2. New:
dense 272 / sparse 148 → never a candidate. The difference is the old chunk's **page
header/footer pollution**: "…Unpublished work. © 2025 Siemens 4-43 Software Version 2512
**Read more than one level of a Teamcenter BOM**" sat inside it — the only place it said
"BOM" and most of where it said "Teamcenter". docling 2.130 strips page furniture; the clean
chunk never mentions BOM, so this phrasing no longer finds it. The new top-6 are on topic
(`readBillOfMaterial` for Equipment/Order BOM, "Declare BOM Target Types").

## tc-007 "EPM_attach_related_objects" — not a regression (k=6 boundary)

The handler's reference sections rank old #6 and #8, new #8 and #11, in a tight rerank band
(top 10 all 4.2–4.7). Every new top-6 result is about this handler or its siblings (usage
example "Start action", its LOV example, EPM-attach-item-revision-target). The old #2 was a
junk footer chunk ("Software Version 2512 / EPM-attach-related-objects"), now gone.

## Verdict: PASS on adjudication

No substantive retrieval regression; both misses come from the parser being *cleaner*
(page furniture no longer leaks into chunks). Earlier failures of this gate were real and
are fixed: the two-pool budget split (gate config) and the font-less worker image
(non-embedded Courier → code blocks dropped; tc-012 `taxonomy -f=` now answered).

Label follow-ups for Gate 2 proper (not blocking): tc-002 should also accept the T4EA
"Read more than one level of a Teamcenter BOM" section; tc-007 is an identifier query
whose answer set is broad — expect ties at k.

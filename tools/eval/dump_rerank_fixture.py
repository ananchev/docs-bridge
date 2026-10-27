"""Dump the rerank fixture: the ONE test file every later gate compares against.

Gate 1 §2.2 of `notes/multi-user-rollout-plan.md`. The cross-encoder is a pure
function of (query, passage) — it needs no Qdrant, no corpus, no manifest, no
server. So the entire hardware/model comparison surface collapses to one JSON
file: the real queries, their real RRF candidate passages, and the CURRENT
CPU-INT8 scores. That last column is the baseline the GPU probe (§2.3), the
smaller-reranker probe (§3) and Gate 2's reranker A/B (§5.2) all measure against
— on hosts that will never have this corpus.

Runs INSIDE the running docs-bridge container, like every other harness here:

    sudo podman exec -i docs-bridge python - < tools/eval/dump_rerank_fixture.py \
        > notes/gate1/fixture.json

JSON goes to STDOUT; progress to STDERR, so that redirect is all it takes.

TWO retrieves per query, deliberately (they answer different questions):

  * the PRODUCTION set — prefetch=`server.prefetch_limit`, limit=`rerank.top_n`
    — is exactly what the server hands the reranker today, scored as ONE batch so
    its wall time is the real per-query rerank cost. Raising prefetch to reach a
    deeper pool would silently change this set (RRF fuses the two prefetch lists),
    which is why it is not reused for the oracle.
  * the ORACLE pool — prefetch=FIX_ORACLE_PREFETCH, limit=FIX_ORACLE_POOL — is the
    deeper pool §2.4 needs to define the global-best top-6 when a candidate
    reranker DISAGREES with this baseline. fp16 is strictly more precise than
    INT8, so a difference may be an improvement; the deep pool is how we tell,
    instead of treating the INT8 order as ground truth.

Scores are deduped by point id across the two batches (a pair scored in the
production batch keeps that score — same padding shape as production).

Env:
    FIX_SUBJECT           subject name to query          (default: teamcenter)
    FIX_TOP_N             production candidates          (default: rerank.top_n)
    FIX_ORACLE_POOL       deep pool depth, 0 disables    (default: 80)
    FIX_ORACLE_PREFETCH   per-channel prefetch for it    (default: 128)
    FIX_QUERYFILE         subject<TAB>query per line, overrides the built-in set
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time

from qdrant_client import models as qm

from docs_bridge import config
from docs_bridge.qdrant_io import DENSE, SPARSE
from docs_bridge.search import Searcher

# The user-validated set, carried over verbatim from diag_channels.py /
# validate_union.py so the fixture is comparable to every earlier measurement:
# five real answerable questions (T4X / BGS / SAP), two exact-identifier needles
# (where sparse is supposed to beat dense), and one known-negative control whose
# scores MUST stay negative — that is what proves the rerank sign still separates
# real answers from abstention noise on any new hardware or model.
DEFAULT_QUERIES = [
    ("How do we read properties of a Teamcenter relation?", False),
    ("How to read BOM data from Teamcenter in AIG", False),
    ("How to send an email from T4X?", False),
    ("How does SAP distinguish if a BOM line was created in SAP or via an external system?", False),
    ("How do we update basic unit of measure in SAP?", False),
    ("TC_transfer_area site preference", False),
    ("EPM_attach_related_objects", False),
    ("how to bake sourdough bread", True),   # known-negative control
]

SUBJECT = os.environ.get("FIX_SUBJECT", "teamcenter")
ORACLE_POOL = int(os.environ.get("FIX_ORACLE_POOL", "80"))
ORACLE_PREFETCH = int(os.environ.get("FIX_ORACLE_PREFETCH", "128"))


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_queries() -> list[tuple[str, str, bool]]:
    path = os.environ.get("FIX_QUERYFILE")
    if not path:
        return [(SUBJECT, q, ctl) for q, ctl in DEFAULT_QUERIES]
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            subj, _, q = line.partition("\t")
            out.append((subj.strip(), q.strip(), False))
    return out


cfg = config.load()
s = Searcher(cfg)
TOP_N = int(os.environ.get("FIX_TOP_N", str(cfg.rerank.top_n)))

if not cfg.rerank.enabled or s.reranker is None:
    raise SystemExit("rerank is disabled — there is no baseline column to dump.")

subject = cfg.subject(SUBJECT)          # raises on an unknown name
if not s.client.collection_exists(subject.collection):
    raise SystemExit(f"collection {subject.collection!r} does not exist")
points_total = s.client.count(subject.collection, exact=True).count


def retrieve(query_vecs, limit: int, prefetch: int):
    """The server's hybrid retrieve (search.py) with both knobs exposed."""
    dense, sparse = query_vecs
    res = s.client.query_points(
        collection_name=subject.collection,
        prefetch=[
            qm.Prefetch(query=dense, using=DENSE, limit=prefetch),
            qm.Prefetch(
                query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                using=SPARSE,
                limit=prefetch,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return res.points


def entry(point, rank: int) -> dict:
    pl = point.payload or {}
    return {
        "rrf_rank": rank,
        "id": point.id,
        "rrf_score": float(point.score) if point.score is not None else None,
        "doc_id": pl.get("doc_id"),
        "chunk_id": pl.get("chunk_id"),
        "source_path": pl.get("source_path"),
        "section": pl.get("section_path"),
        "last_updated": pl.get("last_updated"),
        "text": pl.get("text", ""),
    }


# A second tokenizer on the SAME tokenizer.json but WITHOUT truncation, so we can
# see the pair lengths the reranker actually truncates — §3's max_length review
# ("measure how many pairs reach the cap before trimming"). Cheap: no model.
from tokenizers import Tokenizer  # noqa: E402  (after cfg, so model_dir is known)

_len_tok = Tokenizer.from_file(os.path.join(cfg.rerank.model_dir, "tokenizer.json"))

rr_meta_path = os.path.join(cfg.rerank.model_dir, "meta.json")
rr_meta = json.load(open(rr_meta_path)) if os.path.exists(rr_meta_path) else {}

log(f"fixture | subject={SUBJECT} collection={subject.collection} "
    f"points={points_total} top_n={TOP_N} oracle_pool={ORACLE_POOL} "
    f"queries={len(load_queries())}")

out_queries = []
pair_lengths: list[int] = []
embed_ms: list[float] = []

for qi, (subj_name, query, is_control) in enumerate(load_queries(), 1):
    log(f"  [{qi}] {query[:60]}")

    t0 = time.perf_counter()
    d, sp = s.embedder.encode([query])
    embed_ms.append((time.perf_counter() - t0) * 1000.0)
    vecs = (d[0], sp[0])

    # --- production set: exactly what the server reranks today ----------------
    prod = retrieve(vecs, limit=TOP_N, prefetch=cfg.server.prefetch_limit)
    prod_entries = [entry(p, i + 1) for i, p in enumerate(prod)]

    t0 = time.perf_counter()
    prod_scores = s.reranker.score(query, [e["text"] for e in prod_entries])
    prod_rerank_s = time.perf_counter() - t0

    for e, sc in zip(prod_entries, prod_scores):
        e["baseline_score"] = float(sc)
    by_id = {e["id"]: float(sc) for e, sc in zip(prod_entries, prod_scores)}
    log(f"      production: {len(prod_entries)} pairs in {prod_rerank_s:.2f}s "
        f"({prod_rerank_s / max(1, len(prod_entries)) * 1000:.0f} ms/pair)")

    # --- oracle pool: deeper, for adjudicating later disagreements ------------
    oracle_entries: list[dict] = []
    oracle_rerank_s = 0.0
    if ORACLE_POOL > TOP_N:
        deep = retrieve(vecs, limit=ORACLE_POOL, prefetch=ORACLE_PREFETCH)
        oracle_entries = [entry(p, i + 1) for i, p in enumerate(deep)]
        todo = [e for e in oracle_entries if e["id"] not in by_id]
        if todo:
            t0 = time.perf_counter()
            deep_scores = s.reranker.score(query, [e["text"] for e in todo])
            oracle_rerank_s = time.perf_counter() - t0
            for e, sc in zip(todo, deep_scores):
                by_id[e["id"]] = float(sc)
        for e in oracle_entries:
            e["baseline_score"] = by_id[e["id"]]
        log(f"      oracle: {len(oracle_entries)} pooled, {len(todo)} newly scored "
            f"in {oracle_rerank_s:.2f}s")

    # pair token lengths, pre-truncation (production set only — that is the
    # distribution max_length actually acts on)
    for e in prod_entries:
        pair_lengths.append(len(_len_tok.encode(query, e["text"]).ids))

    out_queries.append({
        "query": query,
        "subject": subj_name,
        "control": is_control,
        "embed_ms": embed_ms[-1],
        "rerank_s": prod_rerank_s,
        "oracle_rerank_s": oracle_rerank_s,
        "candidates": prod_entries,
        "oracle_pool": oracle_entries,
    })

n_pairs = sum(len(q["candidates"]) for q in out_queries)
total_rerank_s = sum(q["rerank_s"] for q in out_queries)
cap = cfg.rerank.max_length
srt = sorted(pair_lengths)


def pct(xs: list[int], p: float) -> int:
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else 0


fixture = {
    "meta": {
        "purpose": "Gate 1 §2.2 rerank fixture — baseline = CPU INT8, this host",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "subject": SUBJECT,
        "collection": subject.collection,
        "collection_points": points_total,
        "top_n": TOP_N,
        "prefetch_limit": cfg.server.prefetch_limit,
        "oracle_pool": ORACLE_POOL,
        "oracle_prefetch": ORACLE_PREFETCH,
        "default_k": cfg.server.default_k,
        "baseline": {
            "kind": f"onnx-{cfg.rerank.device}-{cfg.rerank.precision}",
            "model_dir": cfg.rerank.model_dir,
            "device": cfg.rerank.device,
            "precision": cfg.rerank.precision,
            "bucket": cfg.rerank.bucket,
            "max_length": cap,
            "model_meta": rr_meta,
        },
        "embedder": {
            "model": cfg.embedding_model,
            "backend": getattr(cfg, "embedding_backend", None),
            "int8": getattr(cfg, "embedding_int8", None),
            "mean_ms": sum(embed_ms) / len(embed_ms) if embed_ms else None,
        },
        "timing": {
            "queries": len(out_queries),
            "production_pairs": n_pairs,
            "rerank_s_total": total_rerank_s,
            "rerank_s_mean_per_query": total_rerank_s / max(1, len(out_queries)),
            "rerank_ms_per_pair": total_rerank_s / max(1, n_pairs) * 1000.0,
        },
        "pair_tokens": {
            "n": len(srt),
            "cap": cap,
            "at_cap": sum(1 for x in srt if x >= cap),
            "at_cap_pct": (sum(1 for x in srt if x >= cap) / len(srt) * 100) if srt else 0.0,
            "p50": pct(srt, 50), "p90": pct(srt, 90), "p95": pct(srt, 95),
            "max": srt[-1] if srt else 0,
        },
    },
    "queries": out_queries,
}

json.dump(fixture, sys.stdout)
sys.stdout.write("\n")

m = fixture["meta"]
log("")
log(f"wrote fixture: {len(out_queries)} queries, {n_pairs} production pairs")
log(f"  rerank: {m['timing']['rerank_s_mean_per_query']:.2f} s/query at top_n={TOP_N} "
    f"({m['timing']['rerank_ms_per_pair']:.0f} ms/pair)")
log(f"  embed:  {m['embedder']['mean_ms']:.0f} ms/query")
log(f"  pair tokens: p50={m['pair_tokens']['p50']} p95={m['pair_tokens']['p95']} "
    f"max={m['pair_tokens']['max']} | at cap({cap}): "
    f"{m['pair_tokens']['at_cap']}/{m['pair_tokens']['n']} "
    f"({m['pair_tokens']['at_cap_pct']:.1f}%)")

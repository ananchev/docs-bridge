"""Validate first-stage strategies against the GLOBALLY-best answers (corrected method).

Earlier mistake: comparing a strategy's reranked top-k to ANOTHER arbitrary pool's
reranked top-k conflates 'different' with 'worse' and counts reranker tie-jitter as
loss. Fix: the reranker scores each (query, chunk) independently, so the true answer
is the highest-scoring chunks in the COLLECTION. Establish that gold once (rerank a
deep pool), then ask of each first-stage strategy only: does its candidate pool CONTAIN
the gold chunks? If yes, the reranker WILL surface them -> coverage = correctness, no
tie-ambiguity. And it compares RRF vs UNION fairly at equal pool size (= equal latency,
~270ms/candidate).

  GOLD (per query): rerank dense_top(DEEP) UNION sparse_top(DEEP), take top-K positive.
  strategies: RRF top-N  vs  union(dense top-Nd, sparse top-Ns).
  metric: how many of the K gold chunks the strategy's pool contains, and pool size.

    sudo docker exec -i docs-bridge python - < tools/validate_union.py
  (~5-6 min: one deep rerank per query for gold; strategy pools are retrieval-only.)

Env: VU_DEEP (gold pool depth per channel, default 100), VU_K (gold size, default 6).
"""

from __future__ import annotations

import os
import sys

from qdrant_client import models as qm

from docs_bridge import config
from docs_bridge.qdrant_io import DENSE, SPARSE
from docs_bridge.search import Searcher

# 50/channel keeps the gold rerank pool ~80-100 pairs (the global-best chunks all sit
# shallow; only tertiary ones live past 50). Bigger pools OOM/time-out the reranker.
DEEP = int(os.environ.get("VU_DEEP", "50"))    # per-channel depth for the gold pool
K = int(os.environ.get("VU_K", "6"))           # gold = top-K positive reranked

# strategies to compare: (label, kind, params). Pool sizes chosen to span ~equal
# latency bands so RRF vs union is a fair fight.
STRATS = [
    ("rrf-24", "rrf", 24),
    ("rrf-30", "rrf", 30),
    ("rrf-40", "rrf", 40),
    ("uni-20+8", "union", (20, 8)),
    ("uni-24+8", "union", (24, 8)),
    ("uni-24+12", "union", (24, 12)),
    ("uni-30+12", "union", (30, 12)),
]

QUERIES = [
    ("aig", "How do we read properties of a Teamcenter relation?"),
    ("aig", "How to read BOM data from Teamcenter in AIG"),
    ("aig", "How to send an email from T4X?"),
    ("aig", "How does SAP distinguish if a BOM line was created in SAP or via an external system?"),
    ("aig", "How do we update basic unit of measure in SAP?"),
    ("aig", "TC_transfer_area site preference"),
    ("aig", "EPM_attach_related_objects"),
    ("aig", "how to bake sourdough bread"),   # known-negative control (skipped)
]

cfg = config.load()
s = Searcher(cfg)


def encode(q):
    d, sp = s.embedder.encode([q])
    return d[0], sp[0]


def dense_pts(dense, collection, limit, payload=False):
    return s.client.query_points(collection_name=collection, query=dense, using=DENSE,
                                 limit=limit, with_payload=payload).points


def sparse_pts(sparse, collection, limit, payload=False):
    return s.client.query_points(
        collection_name=collection,
        query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
        using=SPARSE, limit=limit, with_payload=payload).points


def rrf_ids(dense, sparse, collection, limit):
    res = s.client.query_points(
        collection_name=collection,
        prefetch=[
            qm.Prefetch(query=dense, using=DENSE, limit=max(limit, 60)),
            qm.Prefetch(
                query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                using=SPARSE, limit=max(limit, 60)),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF), limit=limit, with_payload=False)
    return {p.id for p in res.points}, limit


def strat_pool(kind, params, dense, sparse, collection):
    if kind == "rrf":
        return rrf_ids(dense, sparse, collection, params)
    nd, ns = params
    d = {p.id for p in dense_pts(dense, collection, nd)}
    sp = {p.id for p in sparse_pts(sparse, collection, ns)}
    u = d | sp
    return u, len(u)


print(f"\nfirst-stage coverage of the GLOBAL-best answers  |  gold=top-{K} of "
      f"dense{DEEP}+sparse{DEEP} reranked\n")
hdr = f"{'query':<38}{'gold':>5}"
for label, _, _ in STRATS:
    hdr += f"{label:>11}"
print(hdr)
print("-" * len(hdr))

# agg[label] = list of (covered, gold_n, pool_size)
agg = {label: [] for label, _, _ in STRATS}

for qi, (subject, query) in enumerate(QUERIES, 1):
    print(f"  [{qi}/{len(QUERIES)}] {query[:50]}", file=sys.stderr, flush=True)
    try:
        collection = cfg.subject(subject).collection
        if not s.client.collection_exists(collection):
            continue
        dense, sparse = encode(query)

        # GOLD: rerank the deep union pool, keep top-K positive
        d = dense_pts(dense, collection, DEEP, payload=True)
        sp = sparse_pts(sparse, collection, DEEP, payload=True)
        merged = {}
        for p in list(d) + list(sp):
            merged[p.id] = p
        pool = list(merged.values())
        texts = [(p.payload or {}).get("text", "") for p in pool]
        scores = s.reranker.score(query, texts)
        order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
        gold = [pool[i].id for i in order[:K] if scores[i] > 0]

        if not gold:
            print(f"{query[:36]:<38}  (abstention, skipped)", flush=True)
            continue

        row = f"{query[:36]:<38}{len(gold):>5}"
        for label, kind, params in STRATS:
            pool_ids, size = strat_pool(kind, params, dense, sparse, collection)
            covered = sum(1 for g in gold if g in pool_ids)
            agg[label].append((covered, len(gold), size))
            mark = "" if covered == len(gold) else "*"   # * = missed a gold chunk
            row += f"{covered:>4}/{len(gold)}{mark:<1}{size:>4}"
        print(row, flush=True)
    except Exception as e:   # one bad query must not kill the whole run
        print(f"{query[:36]:<38}  ERROR: {type(e).__name__}: {e}", flush=True)

print("\n=== aggregate (answered queries) — coverage of the global-best answers ===")
print(f"{'strategy':<12}{'fullCover':>11}{'goldMissed':>12}{'poolMax':>9}{'~latency':>10}")
for label, _, _ in STRATS:
    rows = agg[label]
    if not rows:
        continue
    full = sum(1 for c, g, _ in rows if c == g)
    missed = sum(g - c for c, g, _ in rows)
    pmax = max(sz for _, _, sz in rows)
    print(f"{label:<12}{full:>7}/{len(rows):<3}{missed:>12}{pmax:>9}{pmax*0.27:>8.1f}s")

print("\nwinner = fewest goldMissed (ideally 0) at the smallest poolMax. Compare RRF vs "
      "union\nin the SAME latency band: if union covers more gold per candidate, it "
      "wins and sets\nretrieval.dense_k / sparse_k. '*' in the table marks a missed "
      "gold chunk.")
print()

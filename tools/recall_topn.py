"""Decide whether lowering rerank `top_n` loses retrieval quality (design §9).

`top_n` only truncates the RRF candidate list BEFORE the cross-encoder rerank, so a
single retrieve at the highest top_n contains every smaller top_n's answer as a
prefix: `points` come back in RRF order (index 0 = RRF rank 1), the reranker scores
them all, and `search()` keeps the top-k by rerank score. That means we can compare
top_n=16 against the top_n=30 baseline OFFLINE — one retrieve + one rerank per query,
no config change, no container restart, no second deployment.

Runs INSIDE the running docs-bridge container, exactly like bench_search.py:

    sudo docker exec -i docs-bridge python - < tools/recall_topn.py

It drives the SAME Searcher the server uses (baked INT8 models, mounted config,
`qdrant` alias), and replicates search.py's selection verbatim — including its tie
behavior: `sorted(range(n), key=score, reverse=True)[:k]` is a stable sort, so equal
rerank scores keep RRF order. Truncating to a smaller top_n uses the identical call
over range(top_n), so the comparison is apples-to-apples.

For each query it reports, per candidate top_n vs the BASE baseline:
  - top-3 ordered identical?         (the chunks an LLM leans on most)
  - top-k set Jaccard                 (how much the answer set drifts)
  - max RRF rank feeding the baseline top-k  (<= top_n  =>  that top_n is LOSSLESS)
Then an aggregate PASS/FAIL for top_n=16 against an overlap bar.

Override the query set with a newline-separated file (subject<TAB>query per line):
    sudo docker exec -i -e RECALL_QUERYFILE=/config/queries.tsv \
        docs-bridge python - < tools/recall_topn.py
"""

from __future__ import annotations

import os

from qdrant_client import models as qm

from docs_bridge import config
from docs_bridge.qdrant_io import DENSE, SPARSE
from docs_bridge.search import Searcher

# Baseline (current config default) and the candidates to test against it.
BASE = int(os.environ.get("RECALL_BASE", "30"))
COMPARE = [int(x) for x in os.environ.get("RECALL_COMPARE", "8,12,16,24").split(",")]

# Overlap bar for the top_n=16 verdict (user choice: tolerate a small tail swap):
#   - every query's top-3 must be ordered-identical to the baseline, AND
#   - every query's top-k set Jaccard >= JACCARD_MIN (>=0.83 = at most 1 of 6 swapped)
VERDICT_TOPN = int(os.environ.get("RECALL_VERDICT_TOPN", "16"))
JACCARD_MIN = float(os.environ.get("RECALL_JACCARD_MIN", "0.83"))

# Real corpus queries the user CONFIRMED are answerable in the aig docs (T4X / BGS /
# SAP integration), plus one deliberately-unanswerable control so we can verify the
# rerank +/- sign actually separates real answers from abstention noise.
DEFAULT_QUERIES = [
    ("aig", "How do we read properties of a Teamcenter relation?"),
    ("aig", "How to read BOM data from Teamcenter in AIG"),
    ("aig", "How to send an email from T4X?"),
    ("aig", "How does SAP distinguish if a BOM line was created in SAP or via an external system?"),
    ("aig", "How do we update basic unit of measure in SAP?"),
    # known-negative control (not in the corpus) — its scores should go NEGATIVE
    ("aig", "how to bake sourdough bread"),
]


def load_queries():
    path = os.environ.get("RECALL_QUERYFILE")
    if not path:
        return DEFAULT_QUERIES
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            subj, _, q = line.partition("\t")
            out.append((subj.strip(), q.strip()))
    return out


cfg = config.load()
s = Searcher(cfg)
K = cfg.server.default_k

# Override the per-vector prefetch depth to test whether top_n=30 is even ENOUGH
# (the reranker reaching to the pool edge means the baseline itself may be too
# shallow). RECALL_PREFETCH must be >= the largest pool you compare.
PREFETCH = int(os.environ.get("RECALL_PREFETCH", str(cfg.server.prefetch_limit)))

if not cfg.rerank.enabled or s.reranker is None:
    raise SystemExit("rerank is disabled — top_n only matters when reranking is on.")

POOL = max([BASE, *COMPARE])
queries = load_queries()


def retrieve(subject, query, limit):
    collection = cfg.subject(subject).collection
    if not s.client.collection_exists(collection):
        return None
    d, sp = s.embedder.encode([query])
    dense, sparse = d[0], sp[0]
    res = s.client.query_points(
        collection_name=collection,
        prefetch=[
            qm.Prefetch(query=dense, using=DENSE, limit=PREFETCH),
            qm.Prefetch(
                query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                using=SPARSE,
                limit=PREFETCH,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return res.points


def topk_ids(points, scores, top_n, k):
    """Replicates search.py: stable sort by rerank score over the first top_n
    RRF candidates, keep k. Returns the point ids in final ranked order."""
    n = min(top_n, len(points))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    return [points[i].id for i in order]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


print(f"\nrecall vs top_n  |  base={BASE}  compare={COMPARE}  k={K}  "
      f"prefetch_limit={PREFETCH}  queries={len(queries)}\n")

# agg[top_n] = list of (query, top3_match, jaccard, lossless, lost_pos, answered)
agg = {n: [] for n in COMPARE}
rank_headroom = []  # max RRF rank feeding each baseline top-k (lossless if <= top_n)

hdr = f"{'subject':<8}{'query':<40}{'ans':>2}{'mxRRF':>6}{'topS':>8}{'deepS':>8}"
for n in COMPARE:
    hdr += f"  t{n}:3 jac L"
print(hdr)
print("-" * len(hdr))

for subject, query in queries:
    points = retrieve(subject, query, POOL)
    if not points:
        print(f"{subject:<8}{query[:44]:<46}  (no/empty collection)")
        continue
    texts = [(p.payload or {}).get("text", "") for p in points]
    scores = s.reranker.score(query, texts)

    base_idx = sorted(range(min(BASE, len(points))), key=lambda i: scores[i],
                      reverse=True)[:K]
    base_ids = [points[i].id for i in base_idx]
    base_scores = [scores[i] for i in base_idx]   # aligned to base_ids
    max_rrf = max(base_idx) + 1 if base_idx else 0
    rank_headroom.append(max_rrf)

    # rerank-logit sign separates relevant(+) from not(-) (validation finding):
    # an ANSWERED query has a positive top-1; abstention queries fish deep on
    # near-tie NEGATIVE scores, so their "30-too-shallow" is noise, not signal.
    top1_score = base_scores[0] if base_scores else 0.0
    deep_score = scores[max(base_idx)] if base_idx else 0.0   # score at maxRRF
    answered = top1_score > 0

    row = (f"{subject:<8}{query[:38]:<40}{'+' if answered else '-':>2}"
           f"{max_rrf:>6}{top1_score:>8.2f}{deep_score:>8.2f}")
    for n in COMPARE:
        ids = topk_ids(points, scores, n, K)
        top3 = base_ids[:3] == ids[:3]
        jac = jaccard(base_ids, ids)
        lossless = max_rrf <= n
        # real harm: positive-scored base chunks the smaller pool DROPS
        lost_pos = sum(1 for bid, sc in zip(base_ids, base_scores)
                       if sc > 0 and bid not in ids)
        agg[n].append((query, top3, jac, lossless, lost_pos, answered))
        row += f"  {'Y' if top3 else 'n'} {jac:.2f} {lost_pos}"
    print(row)

print("\n(+/- = answered? per rerank-logit sign of top-1; topS/deepS = rerank score "
      "of rank-1 and of the deepest base chunk; per top_n: top3? jac lostPos)")

print("\n=== aggregate (ALL queries) ===")
print(f"{'top_n':>6}{'top3=all':>10}{'jac.min':>9}{'jac.mean':>10}{'lossless':>10}")
for n in COMPARE:
    rows = agg[n]
    if not rows:
        continue
    top3_all = sum(1 for _, t3, _, _, _, _ in rows if t3)
    jmin = min(j for _, _, j, _, _, _ in rows)
    jmean = sum(j for _, _, j, _, _, _ in rows) / len(rows)
    lossless = sum(1 for _, _, _, lo, _, _ in rows if lo)
    print(f"{n:>6}{top3_all:>6}/{len(rows):<3}{jmin:>9.2f}{jmean:>10.2f}"
          f"{lossless:>7}/{len(rows):<3}")

# The decisive view: ANSWERED queries only, counting positive chunks actually lost.
# This strips abstention noise — only here does "pool too shallow" mean lost recall.
print("\n=== aggregate (ANSWERED queries only — the real harm) ===")
print(f"{'top_n':>6}{'lossless':>11}{'posChunksLost':>15}{'qWithLoss':>11}")
ans_total = sum(1 for _, _, _, _, _, a in agg[COMPARE[0]] if a)
for n in COMPARE:
    ans = [(t3, j, lo, lp) for _, t3, j, lo, lp, a in agg[n] if a]
    if not ans:
        continue
    lossless = sum(1 for _, _, lo, _ in ans if lo)
    pos_lost = sum(lp for _, _, _, lp in ans)
    q_loss = sum(1 for _, _, _, lp in ans if lp > 0)
    print(f"{n:>6}{lossless:>8}/{len(ans):<3}{pos_lost:>15}{q_loss:>9}/{len(ans):<2}")
print(f"\nanswered queries: {ans_total}/{len(agg[COMPARE[0]])}  "
      f"(the rest are abstention — pool depth is moot for them)")

if rank_headroom:
    print(f"\nbaseline top-{K} drew from RRF ranks: min={min(rank_headroom)} "
          f"median={sorted(rank_headroom)[len(rank_headroom)//2]} "
          f"max={max(rank_headroom)}")
    print(f"  (any top_n >= {max(rank_headroom)} is LOSSLESS vs base; if max == pool "
          f"edge, depth has NOT converged — rerun deeper)")
print()

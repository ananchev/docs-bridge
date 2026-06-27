"""Diagnose WHICH first-stage channel is failing to surface the relevant chunk.

recall_topn.py showed real answers scatter to RRF rank 31/40, forcing a big (slow)
rerank pool. This asks WHY, by decomposing the hybrid retrieval. The reranker is our
relevance oracle (validated: +score = real answer, -score = abstention). For each
query we take the rerank-confirmed GOLD chunks (top positive-scored from a deep pool),
then look up where each gold chunk ranks in each first-stage channel:

  - dense-only   : BGE-M3 dense vector (semantic)
  - sparse-only  : BGE-M3 learned-sparse vector (neural lexical)
  - RRF-fused    : dense+sparse fused — what we ship today

Read it like this:
  gold high in DENSE, deep in FUSED   -> sparse is dragging it down via RRF; reweight
  gold high in SPARSE, deep in DENSE  -> semantic embed weak on domain terms; lexical
                                         boost (a real BM25 channel) would help
  gold deep in BOTH                    -> chunks/embeddings themselves miss it; re-chunk
  gold high in BOTH but deep in FUSED  -> RRF fusion math is the problem

Runs in-container like bench_search.py / recall_topn.py:

    sudo docker exec -i docs-bridge python - < tools/diag_channels.py

Env overrides: DIAG_DEPTH (how deep to search for gold, default 100),
DIAG_GOLD_POOL (rerank pool to PICK gold, default 50), DIAG_GOLD_K (golds/query, 3).
"""

from __future__ import annotations

import os

from qdrant_client import models as qm

from docs_bridge import config
from docs_bridge.qdrant_io import DENSE, SPARSE
from docs_bridge.search import Searcher

DEPTH = int(os.environ.get("DIAG_DEPTH", "100"))        # rank-lookup horizon
GOLD_POOL = int(os.environ.get("DIAG_GOLD_POOL", "50"))  # pool reranked to pick gold
GOLD_K = int(os.environ.get("DIAG_GOLD_K", "3"))         # # gold chunks per query

# Same user-confirmed answerable aig queries + the known-negative control as
# recall_topn.py (self-contained: the piped script can't import sibling tools).
QUERIES = [
    ("aig", "How do we read properties of a Teamcenter relation?"),
    ("aig", "How to read BOM data from Teamcenter in AIG"),
    ("aig", "How to send an email from T4X?"),
    ("aig", "How does SAP distinguish if a BOM line was created in SAP or via an external system?"),
    ("aig", "How do we update basic unit of measure in SAP?"),
    # exact-term "needle" lookups — where lexical/sparse is SUPPOSED to beat dense
    ("aig", "TC_transfer_area site preference"),
    ("aig", "EPM_attach_related_objects"),
    ("aig", "how to bake sourdough bread"),   # known-negative control
]

cfg = config.load()
s = Searcher(cfg)
PREFETCH = max(DEPTH, GOLD_POOL)


def encode(q):
    d, sp = s.embedder.encode([q])
    return d[0], sp[0]


def fused_ids(dense, sparse, collection, limit, fusion=qm.Fusion.RRF,
              with_payload=False):
    res = s.client.query_points(
        collection_name=collection,
        prefetch=[
            qm.Prefetch(query=dense, using=DENSE, limit=PREFETCH),
            qm.Prefetch(
                query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                using=SPARSE, limit=PREFETCH,
            ),
        ],
        query=qm.FusionQuery(fusion=fusion),
        limit=limit,
        with_payload=with_payload,
    )
    return res.points


def dense_ids(dense, collection, limit):
    res = s.client.query_points(collection_name=collection, query=dense,
                                using=DENSE, limit=limit, with_payload=False)
    return [p.id for p in res.points]


def sparse_ids(sparse, collection, limit):
    res = s.client.query_points(
        collection_name=collection,
        query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
        using=SPARSE, limit=limit, with_payload=False)
    return [p.id for p in res.points]


def rank_of(idlist, gid):
    """1-based rank, or None if beyond the lookup horizon."""
    return idlist.index(gid) + 1 if gid in idlist else None


def fmt(r):
    return f"{r:>4}" if r is not None else f"{'>'+str(DEPTH):>4}"


print(f"\nchannel diagnostic  |  depth={DEPTH}  gold_pool={GOLD_POOL}  gold_k={GOLD_K}")
print("for each rerank-confirmed gold chunk: its rank in each first-stage strategy\n")
print(f"{'query':<38}{'g#':>3}{'rerank':>8}{'dense':>6}{'sparse':>7}{'rrf':>5}{'dbsf':>6}")
print("-" * 73)

# rank of each gold chunk per strategy (None -> DEPTH+1 = 'worse than horizon')
chan = {"dense": [], "sparse": [], "rrf": [], "dbsf": []}

for subject, query in QUERIES:
    collection = cfg.subject(subject).collection
    if not s.client.collection_exists(collection):
        print(f"{query[:38]:<40}  (no/empty collection)")
        continue
    dense, sparse = encode(query)

    # pick gold: rerank the fused top GOLD_POOL, keep top-K positive-scored
    pool = fused_ids(dense, sparse, collection, GOLD_POOL, with_payload=True)
    texts = [(p.payload or {}).get("text", "") for p in pool]
    rr = s.reranker.score(query, texts)
    order = sorted(range(len(pool)), key=lambda i: rr[i], reverse=True)
    gold = [(pool[i].id, rr[i]) for i in order[:GOLD_K]]

    if gold and gold[0][1] <= 0:
        print(f"{query[:38]:<40}  ABSTENTION (top rerank {gold[0][1]:+.2f}) "
              f"-> not a recall target, skipped")
        continue

    d_ids = dense_ids(dense, collection, DEPTH)
    sp_ids = sparse_ids(sparse, collection, DEPTH)
    rrf_ids = [p.id for p in fused_ids(dense, sparse, collection, DEPTH, qm.Fusion.RRF)]
    dbsf_ids = [p.id for p in fused_ids(dense, sparse, collection, DEPTH, qm.Fusion.DBSF)]

    for gi, (gid, gscore) in enumerate(gold, 1):
        if gscore <= 0:
            continue  # only positive (real) chunks are recall targets
        rd, rs = rank_of(d_ids, gid), rank_of(sp_ids, gid)
        rr, rb = rank_of(rrf_ids, gid), rank_of(dbsf_ids, gid)
        chan["dense"].append(rd if rd else DEPTH + 1)
        chan["sparse"].append(rs if rs else DEPTH + 1)
        chan["rrf"].append(rr if rr else DEPTH + 1)
        chan["dbsf"].append(rb if rb else DEPTH + 1)
        q = query[:36] if gi == 1 else ""
        print(f"{q:<38}{gi:>3}{gscore:>+8.2f}{fmt(rd)}{fmt(rs)}{fmt(rr)}{fmt(rb)}")


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n else 0


print("\n=== per-strategy gold ranking (lower = better) ===")
print(f"{'strategy':<9}{'median':>7}{'worstHit':>10}{'<=10':>7}{'misses':>8}")
for name in ("dense", "sparse", "rrf", "dbsf"):
    xs = chan[name]
    if not xs:
        continue
    misses = sum(1 for r in xs if r > DEPTH)
    hits = [r for r in xs if r <= DEPTH]
    worst = max(hits) if hits else None
    best10 = sum(1 for r in xs if r <= 10)
    worst_s = str(worst) if worst is not None else "-"
    print(f"  {name:<7}{median(xs):>7}{worst_s:>10}{best10:>5}/{len(xs):<2}"
          f"{misses:>6}")

print(f"\nDECISION METRIC: 'worstHit' = the deepest a real gold chunk sits in that "
      f"strategy = the\nminimum rerank pool (top_n) needed to capture every validated "
      f"answer. Lower worstHit\n= smaller pool = faster search AT EQUAL recall. "
      f"'misses' = golds the strategy can't find\nwithin {DEPTH} at all (those it "
      f"would NEVER feed the reranker).")
print()

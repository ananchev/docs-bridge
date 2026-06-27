"""Measure docs-bridge /v1/search latency, broken down by stage.

Runs INSIDE the running docs-bridge container (it has the baked INT8 models, the
mounted config, and the `qdrant` network alias). It does NOT go through HTTP — it
drives the same Searcher the server uses, timing each stage separately so you can
see what the rerank actually costs before tuning `top_n`.

    sudo docker exec -i docs-bridge python - < tools/bench_search.py

Override via env (optional):
    sudo docker exec -i \
      -e BENCH_QUERY='Security Context on Log Channels and Log Lines' \
      -e BENCH_SUBJECT=aig -e BENCH_K=5 -e BENCH_RUNS=10 \
      docs-bridge python - < tools/bench_search.py

Reports: the cold first call (includes lazy model load), then warm per-stage
timings (embed / retrieve / rerank / full end-to-end), then a rerank-time sweep
over candidate counts so you can read the top_n latency/quality tradeoff directly.
"""

from __future__ import annotations

import os
import statistics
import time

from qdrant_client import models as qm

from docs_bridge import config
from docs_bridge.qdrant_io import DENSE, SPARSE
from docs_bridge.search import Searcher

SUBJECT = os.environ.get("BENCH_SUBJECT", "aig")
QUERY = os.environ.get("BENCH_QUERY", "Security Context on Log Channels and Log Lines")
K = int(os.environ.get("BENCH_K", "5"))
RUNS = int(os.environ.get("BENCH_RUNS", "10"))
SWEEP = [8, 16, 24, 32]

cfg = config.load()
s = Searcher(cfg)
collection = cfg.subject(SUBJECT).collection


def ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def embed():
    d, sp = s.embedder.encode([QUERY])
    return d[0], sp[0]


def retrieve(dense, sparse, limit):
    res = s.client.query_points(
        collection_name=collection,
        prefetch=[
            qm.Prefetch(query=dense, using=DENSE, limit=cfg.server.prefetch_limit),
            qm.Prefetch(
                query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                using=SPARSE,
                limit=cfg.server.prefetch_limit,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return res.points


# --- cold: first full search (triggers lazy model load) ----------------------
t0 = time.perf_counter()
s.search(SUBJECT, QUERY, K)
cold_ms = ms(t0)

have_reranker = s.reranker is not None

# --- warm: per-stage over RUNS -----------------------------------------------
t = {"embed": [], "retrieve": [], "rerank": [], "full": []}
for _ in range(RUNS):
    t0 = time.perf_counter()
    dense, sparse = embed()
    t["embed"].append(ms(t0))

    t0 = time.perf_counter()
    pts = retrieve(dense, sparse, cfg.rerank.top_n)
    t["retrieve"].append(ms(t0))

    if have_reranker:
        texts = [(p.payload or {}).get("text", "") for p in pts]
        t0 = time.perf_counter()
        s.reranker.score(QUERY, texts)
        t["rerank"].append(ms(t0))

    t0 = time.perf_counter()
    s.search(SUBJECT, QUERY, K)
    t["full"].append(ms(t0))

print(f"\nquery={QUERY!r}  subject={SUBJECT}  k={K}  runs={RUNS}")
print(
    f"config: prefetch_limit={cfg.server.prefetch_limit}  top_n={cfg.rerank.top_n}  "
    f"rerank.enabled={cfg.rerank.enabled}  int8={cfg.rerank.int8}  "
    f"max_length={cfg.rerank.max_length}"
)
print(f"\ncold first call (incl. model load): {cold_ms:8.1f} ms\n")
print(f"{'stage':<10}{'min':>9}{'median':>9}{'mean':>9}{'p95':>9}{'max':>9}   (ms)")
for stage in ("embed", "retrieve", "rerank", "full"):
    xs = t[stage]
    if not xs:
        print(f"{stage:<10}{'(disabled)':>45}")
        continue
    print(
        f"{stage:<10}{min(xs):9.1f}{statistics.median(xs):9.1f}"
        f"{statistics.mean(xs):9.1f}{pct(xs, 95):9.1f}{max(xs):9.1f}"
    )

# --- rerank-time vs candidate count (the top_n tradeoff) ---------------------
if have_reranker:
    dense, sparse = embed()
    pool = retrieve(dense, sparse, max(SWEEP))
    texts = [(p.payload or {}).get("text", "") for p in pool]
    print(f"\nrerank time vs candidate count (pool of {len(texts)}, median of 5):")
    print(f"{'top_n':>8}{'rerank ms':>12}")
    for n in SWEEP:
        if n > len(texts):
            continue
        runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            s.reranker.score(QUERY, texts[:n])
            runs.append(ms(t0))
        print(f"{n:>8}{statistics.median(runs):>12.1f}")
print()

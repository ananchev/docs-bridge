"""Measure C — the number of cores one search usefully consumes. Gate 1 §2.1.

Capacity for a multi-user deployment is total cores / cores-per-query (plan §0.3),
and C has never been measured: `OnnxReranker.__init__` accepts `threads=`
(rerank.py:33) but `search.py` never passes it, so every session grabs every core.
"More threads did nothing" on vhost2 tells us the curve is flat AT 10 — not whether
it flattened at 4 or at 10, and those imply very different fleets.

So: construct `OnnxReranker(model_dir, threads=N)` directly for a sweep of N and
score the SAME production-shaped batch (`rerank.top_n` real pairs) with each.
A fresh session per N is what makes this honest — ONNX Runtime's intra-op thread
pool belongs to the session, so it is sized at construction and cannot be changed
after. Each session is dropped before the next is built.

Runs INSIDE the running docs-bridge container (baked INT8 model, mounted config,
`qdrant` alias), like every other harness here:

    sudo podman exec -i docs-bridge python - < tools/eval/thread_scaling.py

C is read off the flattening point: the smallest N whose latency is within
TS_KNEE_TOL of the best latency in the sweep. Past C you are paying cores for
nothing — which is exactly the number `server.max_concurrent_searches` divides
into a box's vCPUs (plan §6.1).

Pairs come from a live retrieve of ONE query by default (self-contained). Point
TS_FIXTURE at a §2.2 fixture that has been copied into the container to score the
byte-identical batch instead.

Env:
    TS_THREADS     sweep, comma-separated     (default: 1,2,3,4,6,8,10 capped to nproc)
    TS_REPEATS     timed runs per N           (default: 3, cut short by TS_BUDGET_S)
    TS_BUDGET_S    per-N time budget, seconds (default: 60)
    TS_PAIRS       batch size                 (default: rerank.top_n — the real one)
    TS_SUBJECT     subject to retrieve from   (default: teamcenter)
    TS_QUERY       query to retrieve with     (default: the first validated query)
    TS_FIXTURE     path to a fixture.json inside the container (overrides retrieval)
    TS_KNEE_TOL    flatness tolerance         (default: 0.05 = within 5% of best)
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import time

from docs_bridge import config

THREADS = [int(x) for x in os.environ.get("TS_THREADS", "1,2,3,4,6,8,10").split(",")]
REPEATS = int(os.environ.get("TS_REPEATS", "3"))
BUDGET_S = float(os.environ.get("TS_BUDGET_S", "60"))
SUBJECT = os.environ.get("TS_SUBJECT", "teamcenter")
QUERY = os.environ.get("TS_QUERY", "How do we read properties of a Teamcenter relation?")
FIXTURE = os.environ.get("TS_FIXTURE")
KNEE_TOL = float(os.environ.get("TS_KNEE_TOL", "0.05"))

cfg = config.load()
PAIRS = int(os.environ.get("TS_PAIRS", str(cfg.rerank.top_n)))
NCPU = os.cpu_count() or 1
THREADS = sorted({n for n in THREADS if 1 <= n <= NCPU})

if not cfg.rerank.enabled:
    raise SystemExit("rerank is disabled — nothing to measure.")


def own_threads() -> int:
    """Threads in THIS process, from /proc — the sanity check that ORT honoured
    intra_op_num_threads at all (a silently-ignored knob would make C meaningless)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


# --- the batch ----------------------------------------------------------------
if FIXTURE:
    fx = json.load(open(FIXTURE))
    q0 = fx["queries"][0]
    QUERY = q0["query"]
    passages = [c["text"] for c in q0["candidates"]][:PAIRS]
    src = f"fixture {FIXTURE} (query 1)"
else:
    from qdrant_client import models as qm

    from docs_bridge.qdrant_io import DENSE, SPARSE
    from docs_bridge.search import Searcher

    s = Searcher(cfg)
    subject = cfg.subject(SUBJECT)
    if not s.client.collection_exists(subject.collection):
        raise SystemExit(f"collection {subject.collection!r} does not exist")
    d, sp = s.embedder.encode([QUERY])
    dense, sparse = d[0], sp[0]
    res = s.client.query_points(
        collection_name=subject.collection,
        prefetch=[
            qm.Prefetch(query=dense, using=DENSE, limit=cfg.server.prefetch_limit),
            qm.Prefetch(
                query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                using=SPARSE,
                limit=cfg.server.prefetch_limit,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=PAIRS,
        with_payload=True,
    )
    passages = [(p.payload or {}).get("text", "") for p in res.points]
    src = f"live retrieve, subject={SUBJECT} collection={subject.collection}"
    del s
    gc.collect()

if len(passages) < PAIRS:
    print(f"note: only {len(passages)} candidates available (asked {PAIRS})")
PAIRS = len(passages)
if not PAIRS:
    raise SystemExit("no candidate passages — cannot measure.")

print(f"\nthread scaling | host cores={NCPU} pairs={PAIRS} "
      f"max_length={cfg.rerank.max_length} int8={cfg.rerank.int8}")
print(f"  batch source: {src}")
print(f"  query: {QUERY[:70]}")
print(f"  sweep: {THREADS}  repeats<={REPEATS} budget={BUDGET_S:.0f}s/N")
env_keys = [k for k in sorted(os.environ) if k.startswith(("OMP_", "ORT_", "MKL_", "OPENBLAS_"))]
print(f"  thread env: {{{', '.join(f'{k}={os.environ[k]}' for k in env_keys) or 'none set'}}}")
print(f"  baseline threads in process (before any session): {own_threads()}\n")

from docs_bridge.rerank import OnnxReranker  # noqa: E402

hdr = (f"{'threads':>8}{'runs':>6}{'median s':>10}{'min s':>9}{'ms/pair':>9}"
       f"{'speedup':>9}{'marginal':>10}{'effic':>7}{'procThr':>9}{'RSS MB':>8}")
print(hdr)
print("-" * len(hdr))

rows = []
first_median = None
prev_median = None

for n in THREADS:
    rr = OnnxReranker(
        cfg.rerank.model_dir,
        use_int8=cfg.rerank.int8,
        max_length=cfg.rerank.max_length,
        threads=n,
    )
    rr.score(QUERY, passages[:4])                     # warm-up, discarded

    samples: list[float] = []
    spent = 0.0
    while len(samples) < REPEATS:
        t0 = time.perf_counter()
        scores = rr.score(QUERY, passages)
        dt = time.perf_counter() - t0
        samples.append(dt)
        spent += dt
        if spent > BUDGET_S and len(samples) >= 1:
            break

    med = statistics.median(samples)
    mn = min(samples)
    if first_median is None:
        first_median = med
    marginal = (prev_median / med - 1.0) * 100.0 if prev_median else float("nan")
    rows.append({
        "threads": n, "runs": len(samples), "median_s": med, "min_s": mn,
        "ms_per_pair": med / PAIRS * 1000.0,
        "speedup_vs_1": first_median / med,
        "marginal_gain_pct": None if prev_median is None else marginal,
        "efficiency": (first_median / med) / n,
        "process_threads": own_threads(), "rss_mb": rss_mb(),
        "top_score": max(scores) if scores else None,
    })
    r = rows[-1]
    print(f"{n:>8}{r['runs']:>6}{med:>10.2f}{mn:>9.2f}{r['ms_per_pair']:>9.1f}"
          f"{r['speedup_vs_1']:>9.2f}x"
          f"{('     n/a' if prev_median is None else f'{marginal:>9.1f}%'):>10}"
          f"{r['efficiency']:>7.2f}{r['process_threads']:>9}{r['rss_mb']:>8.0f}")
    prev_median = med

    del rr
    gc.collect()

# --- verdict: C ---------------------------------------------------------------
# Two independent readings of "where it flattens", because on a genuinely flat tail
# the within-tolerance-of-best rule is decided by measurement noise: (a) the smallest
# N already within KNEE_TOL of the best latency seen, and (b) the smallest N whose
# step to the NEXT tested N bought less than KNEE_TOL. They agree on a clean curve;
# when they disagree the curve is flat between them, and the honest answer is the
# range — so both get printed and recorded.
best = min(r["median_s"] for r in rows)
knee = next(r for r in rows if r["median_s"] <= best * (1.0 + KNEE_TOL))
C = knee["threads"]

C_marginal = rows[-1]["threads"]
for i, r in enumerate(rows[:-1]):
    nxt = rows[i + 1]["marginal_gain_pct"] or 0.0
    if nxt < KNEE_TOL * 100:
        C_marginal = r["threads"]
        break

print(f"\nbest latency in sweep: {best:.2f} s  (at "
      f"{min(rows, key=lambda r: r['median_s'])['threads']} threads)")
print(f"C = {C} cores  — smallest thread count within {KNEE_TOL * 100:.0f}% of best "
      f"({knee['median_s']:.2f} s, {knee['ms_per_pair']:.1f} ms/pair)")
print(f"    cross-check: first N whose next step gained < {KNEE_TOL * 100:.0f}% "
      f"=> C = {C_marginal}")
if C != C_marginal:
    print(f"    the two readings disagree, so the curve is flat across "
          f"{min(C, C_marginal)}..{max(C, C_marginal)} cores — size on the LARGER "
          f"({max(C, C_marginal)}) and treat the smaller as the optimistic bound.")
if C == max(THREADS):
    print("  NOTE: the curve had not flattened by the widest N tested — C may be "
          f"larger than {C}. Re-run on a box with more cores before sizing.")

print("\nimplied capacity at C (rerank only, ignoring embed+retrieve):")
print(f"{'vCPU':>6}{'permits':>9}{'concurrent':>12}{'searches/hr':>13}")
for vcpu in (4, 8, 16, 32):
    permits = max(1, vcpu // C)
    print(f"{vcpu:>6}{permits:>9}{permits:>12}{permits * 3600 / knee['median_s']:>13.0f}")

print("\njson: " + json.dumps({
    "host_cores": NCPU, "pairs": PAIRS, "top_n": cfg.rerank.top_n,
    "max_length": cfg.rerank.max_length, "int8": cfg.rerank.int8,
    "knee_tol": KNEE_TOL, "C": C, "C_marginal": C_marginal,
    "best_s": best, "at_C_s": knee["median_s"],
    "rows": rows,
}))

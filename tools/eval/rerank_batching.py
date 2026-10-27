"""Does length-bucketing the rerank batch cut latency for free? Gate 1 §3 addendum.

The §2.2 fixture measured the pair-length distribution on the real corpus: median
100 tokens, p95 420, and only 0.3% of pairs reaching the 512 cap. But `rerank.py`
scores all `top_n` pairs in ONE `session.run`, and its tokenizer pads to the
longest pair in that batch (`enable_padding` with no `length=` -> pad-to-longest).
So a single 500-token pair drags all 40 up to 500, and transformer attention is
quadratic in that padded length. The median pair is paying ~4-5x its own cost.

Sorting the candidates by token length and scoring them in sub-batches shrinks the
padding each pair sits in. It is a ~10-line change to `search.py`/`rerank.py`, needs
no new hardware, no new model and no architecture change — so if it pays, it is the
cheapest lever in the whole plan, cheaper even than §3's smaller model.

Quality must be checked, not assumed. Padding is mathematically inert (masked
attention), so a float model would return identical scores. This one is dynamically
quantized INT8: activation scales are computed per input tensor at runtime, so
changing what shares a batch CAN move scores. This harness therefore reports the
score delta AND whether the final top-k ranking moves — the latter is what actually
matters, and the former is only how we explain it.

Runs in a throwaway container with the §2.2 fixture mounted:

    sudo podman run --rm -i \
      -v /data/docker/docs-bridge/config/config.yaml:/config/config.yaml:ro,z \
      -v ~/gate1/out:/out:z -e RB_FIXTURE=/out/fixture.json \
      --entrypoint python docker.io/library/docs-bridge:latest - \
      < rerank_batching.py

Env:
    RB_FIXTURE   §2.2 fixture path inside the container (required)
    RB_BUCKETS   sub-batch sizes to sweep  (default: 4,8,16,40 — 40 = today)
    RB_REPEATS   timed runs per config     (default: 2)
"""

from __future__ import annotations

import json
import os
import statistics
import time

from docs_bridge import config
from docs_bridge.rerank import OnnxReranker

FIXTURE = os.environ.get("RB_FIXTURE", "/out/fixture.json")
BUCKETS = [int(x) for x in os.environ.get("RB_BUCKETS", "4,8,16,40").split(",")]
REPEATS = int(os.environ.get("RB_REPEATS", "2"))

cfg = config.load()
fx = json.load(open(FIXTURE))
K = fx["meta"]["default_k"]

# bucket=0: this script does its own bucketing over whole-batch scoring.
rr = OnnxReranker(
    cfg.rerank.model_dir,
    device=cfg.rerank.device,
    precision=cfg.rerank.precision,
    bucket=0,
    max_length=cfg.rerank.max_length,
)

# A second tokenizer WITHOUT truncation would over-count; use the production cap, so
# the lengths here are exactly the padded widths the model sees.
from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(os.path.join(cfg.rerank.model_dir, "tokenizer.json"))
tok.enable_truncation(max_length=cfg.rerank.max_length)


def lengths(query: str, passages: list[str]) -> list[int]:
    return [len(e.ids) for e in tok.encode_batch([(query, p) for p in passages])]


def padded_tokens(lens: list[int], bucket: int) -> int:
    """Token-positions the model actually computes: each sub-batch is a rectangle
    of (rows x longest row). This is the cost proxy bucketing attacks."""
    total = 0
    for i in range(0, len(lens), bucket):
        chunk = lens[i:i + bucket]
        total += len(chunk) * max(chunk)
    return total


def score_bucketed(query: str, passages: list[str], lens: list[int], bucket: int):
    """Sort by length, score each sub-batch, then restore the caller's order —
    exactly what a production change would do."""
    order = sorted(range(len(passages)), key=lambda i: lens[i])
    out = [0.0] * len(passages)
    for i in range(0, len(order), bucket):
        idx = order[i:i + bucket]
        sc = rr.score(query, [passages[j] for j in idx])
        for j, s in zip(idx, sc):
            out[j] = s
    return out


def topk(scores: list[float], k: int) -> list[int]:
    """search.py's selection: stable sort by score desc, keep k."""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


print(f"\nrerank batching | fixture={FIXTURE} top_n={fx['meta']['top_n']} k={K} "
      f"max_length={cfg.rerank.max_length} {cfg.rerank.device}/{cfg.rerank.precision}")
print(f"  queries={len(fx['queries'])} buckets={BUCKETS} repeats={REPEATS}")
print(f"  baseline recorded in fixture: "
      f"{fx['meta']['timing']['rerank_s_mean_per_query']:.2f} s/query "
      f"({fx['meta']['timing']['rerank_ms_per_pair']:.0f} ms/pair)\n")

# --- per-query baseline (one batch, production behaviour) ---------------------
work = []
for q in fx["queries"]:
    passages = [c["text"] for c in q["candidates"]]
    lens = lengths(q["query"], passages)
    rr.score(q["query"], passages[:4])            # warm-up, discarded
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        base_scores = rr.score(q["query"], passages)
        samples.append(time.perf_counter() - t0)
    work.append({
        "query": q["query"], "passages": passages, "lens": lens,
        "base_s": statistics.median(samples), "base_scores": base_scores,
        "base_top": topk(base_scores, K),
    })

tot_base = sum(w["base_s"] for w in work)
print(f"{'bucket':>7}{'s/query':>9}{'speedup':>9}{'padTok':>10}{'padRatio':>10}"
      f"{'maxDelta':>10}{'top-k moved':>13}{'top-3 moved':>13}")
print("-" * 81)

rows = []
for bucket in BUCKETS:
    per_query, deltas, moved_k, moved_3 = [], [], 0, 0
    pad_tot = 0
    for w in work:
        srt = sorted(w["lens"])
        pad_tot += padded_tokens(srt, bucket)
        samples = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            sc = score_bucketed(w["query"], w["passages"], w["lens"], bucket)
            samples.append(time.perf_counter() - t0)
        per_query.append(statistics.median(samples))
        deltas.append(max(abs(a - b) for a, b in zip(sc, w["base_scores"])))
        new_top = topk(sc, K)
        moved_k += int(new_top != w["base_top"])
        moved_3 += int(new_top[:3] != w["base_top"][:3])

    tot = sum(per_query)
    base_pad = sum(padded_tokens(sorted(w["lens"]), len(w["lens"])) for w in work)
    rows.append({
        "bucket": bucket, "s_per_query": tot / len(work),
        "speedup": tot_base / tot, "padded_tokens": pad_tot,
        "pad_ratio": pad_tot / base_pad, "max_delta": max(deltas),
        "topk_moved": moved_k, "top3_moved": moved_3, "queries": len(work),
    })
    r = rows[-1]
    print(f"{bucket:>7}{r['s_per_query']:>9.2f}{r['speedup']:>8.2f}x"
          f"{pad_tot:>10}{r['pad_ratio']:>10.2f}{r['max_delta']:>10.4f}"
          f"{f'{moved_k}/{len(work)}':>13}{f'{moved_3}/{len(work)}':>13}")

print(f"\none batch of {len(work[0]['lens'])} (today): "
      f"{tot_base / len(work):.2f} s/query")

best = min(rows, key=lambda r: r["s_per_query"])
safe = [r for r in rows if r["top3_moved"] == 0 and r["topk_moved"] == 0]
print(f"fastest bucket: {best['bucket']} at {best['s_per_query']:.2f} s/query "
      f"({best['speedup']:.2f}x), padding cut to {best['pad_ratio'] * 100:.0f}%")
if safe:
    b = min(safe, key=lambda r: r["s_per_query"])
    print(f"fastest bucket that moved NO ranking: {b['bucket']} at "
          f"{b['s_per_query']:.2f} s/query ({b['speedup']:.2f}x), "
          f"max score delta {b['max_delta']:.4f}")
    print(f"  => at top_n=40 a search's rerank drops "
          f"{tot_base / len(work):.1f}s -> {b['s_per_query']:.1f}s with no model, "
          f"no hardware and no quality change.")
else:
    print("every bucket size moved at least one ranking — INT8 dynamic activation "
          "scales are batch-sensitive here, so this lever is NOT free; adjudicate "
          "with rerank_ab.py's oracle before considering it.")

print("\njson: " + json.dumps({
    "top_n": fx["meta"]["top_n"], "k": K, "repeats": REPEATS,
    "base_s_per_query": tot_base / len(work), "rows": rows,
}))

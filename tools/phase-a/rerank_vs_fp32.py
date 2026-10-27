#!/usr/bin/env python3
"""Old INT8 runtime vs new INT8 runtime, both measured against the fp32 reference
(the A/B reference since 2026-09-28 — INT8 is not a judge). Stdlib only.

    rerank_vs_fp32.py <fixture.json> <gpu-fp32.json> <new-run.json>

The fixture's baseline_score column is the old runtime; the new run is rerank_ab.py
output on the new image. PASS = the new runtime is no further from fp32 IN RANKING
than the old one (top-6 order, overlap, #1, controls abstain), and not materially
further in score (mean |d| within 0.05). Two INT8 runtimes can disagree with EACH
OTHER by more than either does with fp32 — that is what the selftest measures, and
why it is reported, not gated.
"""
import json
import statistics as st
import sys

N, K = 40, 6
fx, fp, nw = (json.load(open(p))["queries"] for p in sys.argv[1:4])


def top(s, k=K):
    return [i for i, _ in sorted(enumerate(s), key=lambda t: -t[1])[:k]]


def jac(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


m = {k: {"d": [], "ord": 0, "jac": [], "top1": 0, "abstain": True} for k in ("old", "new")}
for f, r, n in zip(fx, fp, nw):
    assert f["query"] == r["query"] == n["query"], "fixture / reference / run disagree on queries"
    ref = r["scores"][:N]
    for k, s in (("old", [c["baseline_score"] for c in f["candidates"][:N]]), ("new", n["scores"][:N])):
        x = m[k]
        x["d"] += [abs(a - b) for a, b in zip(s, ref)]
        x["ord"] += top(s) == top(ref)
        x["jac"].append(jac(top(s), top(ref)))
        x["top1"] += top(s)[0] == top(ref)[0]
        if f.get("control"):
            x["abstain"] &= max(s) < 0

q = len(fx)
print(f"vs fp32, {q} queries x {N} pairs      old runtime   new runtime")
for label, fn in (("mean |d|", lambda x: f"{st.mean(x['d']):.3f}"),
                  ("max |d|", lambda x: f"{max(x['d']):.3f}"),
                  ("p95 |d|", lambda x: f"{sorted(x['d'])[int(.95 * len(x['d']))]:.3f}"),
                  ("top-6 in fp32's order", lambda x: f"{x['ord']}/{q}"),
                  ("top-6 overlap (mean)", lambda x: f"{st.mean(x['jac']):.3f}"),
                  ("same #1 as fp32", lambda x: f"{x['top1']}/{q}"),
                  ("controls abstain", lambda x: str(x["abstain"]))):
    print(f"  {label:30} {fn(m['old']):>11}   {fn(m['new']):>11}")
o, n = m["old"], m["new"]
ok = (n["ord"] >= o["ord"] and st.mean(n["jac"]) >= st.mean(o["jac"]) - 1e-9
      and n["top1"] >= o["top1"] and n["abstain"]
      and st.mean(n["d"]) <= st.mean(o["d"]) + 0.05)
print("VERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

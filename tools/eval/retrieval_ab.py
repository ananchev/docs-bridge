"""End-to-end retrieval A/B on a labelled question set — old image vs new image.

The Mac mini keeps its `teamcenter` collection across the 2026-09 toolchain upgrade
(plan §1.1), so the old and the new server image can search the SAME collection and
their answers can be compared chunk by chunk. rerank_ab.py already covers the
reranker's numerics on the fixture; this is the user-facing view: the real
`Searcher` (embed -> hybrid retrieve -> RRF -> rerank -> top-k), the real config,
the labelled questions in notes/gate2/questions-<corpus>.yaml.

Two steps. `run` executes INSIDE an image, against the live collection:

    sudo podman run --rm -i --network docs-bridge-net \\
      -v /data/docker/docs-bridge/config/config.yaml:/config/config.yaml:ro,z \\
      -v ~/phase-a/out:/out:z -v ~/phase-a/src/notes/gate2:/q:ro,z \\
      --entrypoint python <IMAGE> - run --questions /q/questions-teamcenter.yaml \\
      --subject teamcenter --label old --out /out/retrieval-old.json \\
      < tools/eval/retrieval_ab.py

and `compare` runs anywhere (stdlib only):

    python3 tools/eval/retrieval_ab.py compare \\
        --old retrieval-old.json --new retrieval-new.json

A retrieved chunk HITS a question when it matches any of the question's `expect`
entries (doc, section substring, text substring — see the YAML header). Per run it
reports hit@k and MRR over the real questions, and whether every control ABSTAINS
(best rerank score < 0, the tripwire Gate 1 validated). The comparison is chunk-ID
based, which is valid only because both runs search the same collection.

THE BAR for old -> new (compare): no question that hit in the old run misses in the
new one, and every control that abstained still abstains. Top-k overlap and score
shifts are reported, not gated: the rerank runtime changed, and a reshuffle among
correct answers is not a regression. Questions whose labels are still `draft` are
counted but flagged — their verdicts are only as good as the labels.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time


def _hits(result: dict, expect: list[dict]) -> bool:
    text = (result.get("text") or "").lower()
    section = result.get("section") or ""
    for e in expect:
        if result.get("doc_id") != e["doc"]:
            continue
        if e.get("section") and e["section"] not in section:
            continue
        if e.get("contains") and e["contains"].lower() not in text:
            continue
        return True
    return False


# --- run (inside an image) ------------------------------------------------------
def run(args) -> int:
    from importlib.metadata import version

    import yaml

    from docs_bridge import config
    from docs_bridge.search import Searcher

    qs = [q for q in (yaml.safe_load(open(args.questions)) or {})["questions"]
          if q.get("status") != "drop"]
    cfg = config.load(args.config)
    searcher = Searcher(cfg)
    k = args.k or cfg.server.default_k

    t0 = time.perf_counter()
    searcher.search(args.subject, "warm-up: load the embedder and the reranker", k=k)
    print(f"[run] warm-up {time.perf_counter() - t0:.1f} s (model load)", file=sys.stderr)

    rows = []
    for q in qs:
        t0 = time.perf_counter()
        res = searcher.search(args.subject, q["question"], k=k)
        dt = time.perf_counter() - t0
        top = [{"rank": i + 1, "chunk_id": r.get("chunk_id"), "doc_id": r.get("doc_id"),
                "section": r.get("section"), "score": r.get("score"),
                "hit": _hits(r, q.get("expect") or []),
                "snippet": (r.get("text") or "")[:160]} for i, r in enumerate(res)]
        first = next((t["rank"] for t in top if t["hit"]), None)
        rows.append({"id": q["id"], "question": q["question"], "kind": q["kind"],
                     "status": q.get("status", "draft"), "seconds": dt,
                     "first_hit_rank": first,
                     "max_score": max((t["score"] for t in top), default=None),
                     "top": top})
        mark = ("abstain" if (rows[-1]["max_score"] or 0) < 0 else "ANSWERS") \
            if q["kind"] == "control" else (f"hit@{first}" if first else "MISS")
        print(f"[run] {q['id']:<7} {dt:5.1f}s {mark:<8} {q['question'][:60]}",
              file=sys.stderr)

    out = {"label": args.label, "subject": args.subject, "k": k,
           "versions": {p: version(p) for p in ("onnxruntime", "tokenizers", "numpy")},
           "rerank": {"top_n": cfg.rerank.top_n, "int8": cfg.rerank.int8},
           "summary": summarize(rows, k), "questions": rows}
    json.dump(out, open(args.out, "w"), indent=2)
    print_summary(out)
    print(f"wrote {args.out}")
    return 0


def summarize(rows: list[dict], k: int) -> dict:
    real = [r for r in rows if r["kind"] != "control"]
    ctrl = [r for r in rows if r["kind"] == "control"]
    hits = [r for r in real if r["first_hit_rank"]]
    return {
        "questions": len(real), "controls": len(ctrl),
        "hit_at_k": len(hits) / len(real) if real else None,
        "mrr": (sum(1 / r["first_hit_rank"] for r in hits) / len(real)) if real else None,
        "controls_abstaining": sum(1 for r in ctrl if (r["max_score"] or 0) < 0),
        "draft_labels": sum(1 for r in rows if r["status"] == "draft"),
        "median_seconds": statistics.median(r["seconds"] for r in rows) if rows else None,
        "k": k,
    }


def print_summary(out: dict) -> None:
    s = out["summary"]
    print(f"\n{out['label']}: hit@{s['k']} {s['hit_at_k']:.2f} "
          f"({round(s['hit_at_k'] * s['questions'])}/{s['questions']}), MRR {s['mrr']:.2f}, "
          f"controls abstaining {s['controls_abstaining']}/{s['controls']}, "
          f"median {s['median_seconds']:.1f} s/query "
          f"[onnxruntime {out['versions']['onnxruntime']}, "
          f"tokenizers {out['versions']['tokenizers']}]")
    if s["draft_labels"]:
        print(f"  note: {s['draft_labels']} question(s) still have DRAFT labels")


# --- compare (stdlib only) ------------------------------------------------------
def compare(args) -> int:
    old, new = json.load(open(args.old)), json.load(open(args.new))
    print_summary(old)
    print_summary(new)
    by_old = {q["id"]: q for q in old["questions"]}
    fails, lost, gained, rows = [], [], [], []
    for n in new["questions"]:
        o = by_old.get(n["id"])
        if o is None:
            continue
        oi = [t["chunk_id"] for t in o["top"]]
        ni = [t["chunk_id"] for t in n["top"]]
        jac = len(set(oi) & set(ni)) / len(set(oi) | set(ni)) if (oi or ni) else 1.0
        dmax = (n["max_score"] or 0) - (o["max_score"] or 0)
        rows.append((n["id"], jac, oi[:3] == ni[:3], dmax))
        if n["kind"] == "control":
            if (o["max_score"] or 0) < 0 <= (n["max_score"] or 0):
                fails.append(f"{n['id']}: control abstained before, answers now "
                             f"(max {n['max_score']:+.2f})")
            continue
        if o["first_hit_rank"] and not n["first_hit_rank"]:
            lost.append(n)
            fails.append(f"{n['id']}: answered before (rank {o['first_hit_rank']}), "
                         f"missed now{' [draft label]' if n['status'] == 'draft' else ''}")
        elif n["first_hit_rank"] and not o["first_hit_rank"]:
            gained.append(n)

    jacs = [r[1] for r in rows]
    same_top3 = sum(1 for r in rows if r[2])
    print(f"\nold -> new over {len(rows)} questions: top-{new['k']} Jaccard mean "
          f"{statistics.mean(jacs):.3f} (identical sets on {sum(1 for j in jacs if j == 1)}), "
          f"top-3 identical on {same_top3}, max |score shift| "
          f"{max(abs(r[3]) for r in rows):.3f}")
    moved = [r for r in rows if r[1] < 1]
    for qid, jac, t3, d in sorted(moved, key=lambda r: r[1])[:10]:
        print(f"  {qid:<7} jaccard {jac:.2f}  top-3 {'same' if t3 else 'CHANGED'}  "
              f"max-score shift {d:+.3f}")
    print(f"\nnewly answered: {', '.join(q['id'] for q in gained) or 'none'}")
    print(f"no longer answered: {', '.join(q['id'] for q in lost) or 'none'}")

    verdict = "PASS" if not fails else "FAIL"
    print(f"\nVERDICT: {verdict}")
    for f in fails:
        print(f"  - {f}")
    if args.json_out:
        json.dump({"verdict": verdict, "failures": fails,
                   "old": old["summary"], "new": new["summary"],
                   "per_question": [{"id": r[0], "jaccard": r[1], "top3_same": r[2],
                                     "max_score_shift": r[3]} for r in rows]},
                  open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")
    return 0 if verdict == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ask every question through the real Searcher")
    r.add_argument("--config", default="/config/config.yaml")
    r.add_argument("--questions", required=True)
    r.add_argument("--subject", required=True)
    r.add_argument("--k", type=int, default=0, help="0 = server.default_k")
    r.add_argument("--label", required=True, help="e.g. old / new")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="old run vs new run")
    c.add_argument("--old", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--json", dest="json_out")
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())

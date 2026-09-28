"""Embed parity across a runtime upgrade — the blocking gate for the Mac mini.

`teamcenter` holds ~137k vectors embedded by the OLD runtime (onnxruntime 1.20.1,
tokenizers 0.19.1). The Mac mini keeps that collection, so after the 2026-09
toolchain upgrade its server embeds queries with the NEW runtime (onnxruntime 1.30,
tokenizers 0.23) against doc vectors made by the old one. This gate proves the two
runtimes put text in the same place. The model bytes are frozen (images/models/), so
only the runtime can differ — and this tool checks that too, rather than assume it.

`history` (runbook step 0b, BEFORE the freeze) asks a prior question: were all the
stored vectors made by one model in the first place? The collection was embedded
over several runs by worker images that each re-exported the model. See history().

Two steps. `dump` runs INSIDE an image (old, then new), against the live collection:

    sudo podman run --rm -i --network docs-bridge-net \\
      -v /data/docker/docs-bridge/config/config.yaml:/config/config.yaml:ro,z \\
      -v ~/phase-a/out:/out:z -v ~/phase-a/src/notes/gate2:/q:ro,z \\
      --entrypoint python <IMAGE> - dump --collection teamcenter --n 400 \\
      --queries /q/questions-teamcenter.yaml --out /out/embed-old.json \\
      < tools/eval/embed_upgrade_parity.py

(the new image adds `--ids-from /out/embed-old.json` so both embed the SAME chunks),
then `compare` runs anywhere with numpy — the new image is simplest:

    ... --entrypoint python <NEW IMAGE> - compare \\
        --old /out/embed-old.json --new /out/embed-new.json --json /out/parity.json

WHAT IS MEASURED, and the bar (PASS needs all of them):
  1. SAME MODEL — sha256 of the four model files identical in both images.
  2. SAME TOKENS — token ids identical for every text (reported; a mismatch is the
     usual cause of any drift below, so it is shown first).
  3. OLD vs NEW runtime, same texts, same batching (1 text per call, like a query):
     dense cosine >= 0.999 for EVERY text, sparse Jaccard >= 0.95 on average — the
     bar the INT8 adoption was cleared with (stack-upgrade-strategy.md §3).
  4. NO NEW DRIFT vs the STORED vectors: re-embedding never reproduces a stored
     vector exactly even on the old runtime — ingest embedded in batches, and
     dynamic INT8 scales depend on what shares a batch (Gate 1 test 4). So the old
     runtime's own distance to the stored vectors is the baseline, and the new
     runtime must not be further: mean dense cosine within 1e-4 of the old one's,
     and its worst text within 1e-3 of the old one's worst.
Questions (--queries) take part in 1-3; they have no stored vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

MODEL_FILES = ("model.int8.onnx", "sparse_linear.npz", "tokenizer.json", "meta.json")
DENSE_MIN = 0.999            # old vs new, per text
SPARSE_JACCARD_MIN = 0.95    # old vs new, mean
STORED_MEAN_TOL = 1e-4       # new-vs-stored mean may trail old-vs-stored by this
STORED_WORST_TOL = 1e-3      # ... and its worst text by this


# --- dump (inside an image) -----------------------------------------------------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_queries(path: str | None) -> list[str]:
    if not path:
        return []
    import yaml

    raw = yaml.safe_load(open(path)) or {}
    return [q["question"] for q in raw.get("questions", []) if q.get("question")]


def dump(args) -> int:
    from importlib.metadata import version

    from qdrant_client import QdrantClient, models

    from docs_bridge import config
    from docs_bridge.embed_onnx import OnnxEmbedder

    cfg = config.load(args.config)
    client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port, timeout=120)

    if args.ids_from:
        ids = [c["id"] for c in json.load(open(args.ids_from))["chunks"]]
        points = client.retrieve(args.collection, ids=ids, with_payload=True,
                                 with_vectors=True)
        by_id = {str(p.id): p for p in points}
        points = [by_id[i] for i in ids if i in by_id]
    else:
        # scroll returns points in id order; ids are uuid5(chunk_id), so the first n
        # are a pseudo-random spread across documents, and the order is stable.
        points, _ = client.scroll(args.collection, limit=args.n, with_payload=True,
                                  with_vectors=True)
    points = [p for p in points if (p.payload or {}).get("text", "").strip()]
    print(f"[dump] {len(points)} chunks from {args.collection}", file=sys.stderr)

    emb = OnnxEmbedder(cfg.onnx_model_dir, use_int8=cfg.embedding_int8)
    model = {f: _sha256(os.path.join(cfg.onnx_model_dir, f)) for f in MODEL_FILES
             if os.path.exists(os.path.join(cfg.onnx_model_dir, f))}

    def embed_one(text: str) -> dict:
        ids = emb.tokenizer.encode(text).ids
        d, s = emb.encode([text])          # one text per call: no batch effects
        return {
            "token_ids_sha1": hashlib.sha1(str(ids).encode()).hexdigest(),
            "n_tokens": len(ids),
            "dense": d[0],
            "sparse": {"indices": s[0].indices, "values": s[0].values},
        }

    chunks = []
    for i, p in enumerate(points):
        vec = p.vector or {}
        stored_sparse = vec.get("sparse")
        row = {"id": str(p.id), "doc_id": p.payload.get("doc_id"),
               "text": p.payload["text"],
               "stored_dense": vec.get("dense"),
               "stored_sparse": ({"indices": list(stored_sparse.indices),
                                  "values": list(stored_sparse.values)}
                                 if isinstance(stored_sparse, models.SparseVector)
                                 else None)}
        row.update(embed_one(p.payload["text"]))
        chunks.append(row)
        if (i + 1) % 50 == 0:
            print(f"[dump]   {i + 1}/{len(points)}", file=sys.stderr)
    queries = [{"text": q, **embed_one(q)} for q in _load_queries(args.queries)]

    out = {
        "runtime": {
            "python": platform.python_version(), "machine": platform.machine(),
            **{pkg: version(pkg) for pkg in ("onnxruntime", "tokenizers", "numpy",
                                             "qdrant-client")},
        },
        "model_dir": cfg.onnx_model_dir, "int8": cfg.embedding_int8, "model": model,
        "collection": args.collection, "chunks": chunks, "queries": queries,
    }
    json.dump(out, open(args.out, "w"))
    print(f"[dump] wrote {args.out}: {len(chunks)} chunks, {len(queries)} questions "
          f"| onnxruntime {out['runtime']['onnxruntime']}, "
          f"tokenizers {out['runtime']['tokenizers']}", file=sys.stderr)
    return 0


# --- compare (anywhere with numpy) ----------------------------------------------
def _cos(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _jaccard(a: dict | None, b: dict | None) -> float | None:
    if a is None or b is None:
        return None
    sa, sb = set(a["indices"]), set(b["indices"])
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def _stats(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": sum(xs) / len(xs), "min": min(xs)}


def compare(args) -> int:
    old, new = json.load(open(args.old)), json.load(open(args.new))
    fails: list[str] = []

    print("embed upgrade parity")
    for side, d in (("old", old), ("new", new)):
        r = d["runtime"]
        print(f"  {side}: onnxruntime {r['onnxruntime']}, tokenizers {r['tokenizers']}, "
              f"numpy {r['numpy']}, python {r['python']} [{r['machine']}]")

    # 1. same model bytes
    same_model = old["model"] == new["model"] and bool(old["model"])
    print(f"\n1. SAME MODEL: {'yes' if same_model else 'NO'}")
    for f in MODEL_FILES:
        o, n = old["model"].get(f), new["model"].get(f)
        print(f"     {f:<18} {'=' if o == n else '≠'} {(n or 'missing')[:16]}")
    if not same_model:
        fails.append("model files differ — this compares two MODELS, not two runtimes")

    old_c = {c["id"]: c for c in old["chunks"]}
    pairs = [(old_c[c["id"]], c) for c in new["chunks"] if c["id"] in old_c]
    if len(pairs) != len(old["chunks"]):
        fails.append(f"only {len(pairs)}/{len(old['chunks'])} chunks line up — "
                     f"run the new dump with --ids-from the old one")
    qpairs = [(o, n) for o, n in zip(old["queries"], new["queries"])
              if o["text"] == n["text"]]
    allpairs = pairs + qpairs

    # 2. same tokens
    tok_diff = [n.get("id", n["text"][:50]) for o, n in allpairs
                if o["token_ids_sha1"] != n["token_ids_sha1"]]
    print(f"\n2. SAME TOKENS: {len(allpairs) - len(tok_diff)}/{len(allpairs)} texts "
          f"tokenize identically")
    for t in tok_diff[:5]:
        print(f"     differs: {t}")

    # 3. old vs new runtime
    dense = [_cos(o["dense"], n["dense"]) for o, n in allpairs]
    jac = [_jaccard(o["sparse"], n["sparse"]) for o, n in allpairs]
    ds, js = _stats(dense), _stats(jac)
    worst = sorted(zip(dense, allpairs), key=lambda t: t[0])[:3]
    print(f"\n3. OLD vs NEW runtime ({len(pairs)} chunks + {len(qpairs)} questions, "
          f"1 text per call)")
    print(f"     dense cosine  mean {ds['mean']:.6f}  min {ds['min']:.6f}   "
          f"(bar: every text >= {DENSE_MIN})")
    print(f"     sparse Jacc.  mean {js['mean']:.4f}  min {js['min']:.4f}     "
          f"(bar: mean >= {SPARSE_JACCARD_MIN})")
    for c, (o, n) in worst:
        print(f"     worst: {c:.6f}  {(n.get('doc_id') or 'question')}: {n['text'][:60]!r}")
    if ds["min"] < DENSE_MIN:
        fails.append(f"old-vs-new dense cosine min {ds['min']:.6f} < {DENSE_MIN}")
    if js["mean"] < SPARSE_JACCARD_MIN:
        fails.append(f"old-vs-new sparse Jaccard mean {js['mean']:.4f} < {SPARSE_JACCARD_MIN}")

    # 4. drift vs the stored vectors: the old runtime is the baseline
    so = _stats([_cos(o["dense"], o["stored_dense"]) for o, _ in pairs if o["stored_dense"]])
    sn = _stats([_cos(n["dense"], n["stored_dense"]) for _, n in pairs if n["stored_dense"]])
    jo = _stats([_jaccard(o["sparse"], o["stored_sparse"]) for o, _ in pairs])
    jn = _stats([_jaccard(n["sparse"], n["stored_sparse"]) for _, n in pairs])
    print(f"\n4. vs the STORED vectors (baseline = the old runtime's own distance)")
    print(f"     dense  old->stored mean {so['mean']:.6f} min {so['min']:.6f} | "
          f"new->stored mean {sn['mean']:.6f} min {sn['min']:.6f}")
    if jo.get("n") and jn.get("n"):
        print(f"     sparse old->stored mean {jo['mean']:.4f}          | "
              f"new->stored mean {jn['mean']:.4f}")
    if sn["mean"] < so["mean"] - STORED_MEAN_TOL:
        fails.append(f"new runtime drifts further from the stored vectors: mean "
                     f"{sn['mean']:.6f} vs old {so['mean']:.6f}")
    if sn["min"] < so["min"] - STORED_WORST_TOL:
        fails.append(f"new runtime's worst stored match {sn['min']:.6f} vs old "
                     f"{so['min']:.6f}")

    verdict = "PASS" if not fails else "FAIL"
    print(f"\nVERDICT: {verdict}")
    for f in fails:
        print(f"  - {f}")
    if args.json_out:
        json.dump({"verdict": verdict, "failures": fails,
                   "runtime": {"old": old["runtime"], "new": new["runtime"]},
                   "same_model": same_model, "model": new["model"],
                   "token_mismatches": tok_diff,
                   "old_vs_new": {"dense": ds, "sparse_jaccard": js},
                   "vs_stored": {"old_dense": so, "new_dense": sn,
                                 "old_sparse_jaccard": jo, "new_sparse_jaccard": jn},
                   "n_chunks": len(pairs), "n_questions": len(qpairs)},
                  open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")
    return 0 if verdict == "PASS" else 1


# --- history (inside the CURRENT worker image) ------------------------------------
# Were all stored vectors made with the same model? The collection was embedded in
# several runs (Jun 24-25, Jun 27-28, Aug 31, ...) by worker images that were each
# rebuilt — and every rebuild re-exported the model. This re-embeds real chunks from
# each run with TODAY's worker and asks whether each run's stored vectors sit within
# the normal wobble of one ruler.
#
# The wobble: dynamic INT8 makes a vector depend on what shares its batch (Gate 1
# test 4), so two measurements with the SAME model differ a little. Each chunk is
# embedded twice — alone, and inside a window of up to 64 consecutive chunks in
# (doc_id, chunk_index) order, which is how ingest composed its batches — and the gap
# between the two is the wobble. A stored vector made by the same model should sit
# about that far from today's re-embedding; one made by a different model sits
# further, and a whole run of them shows up as a group that stands out.
HISTORY_OUTSIDE_FACTOR = 2.0   # outside = beyond 2x its OWN group's wobble p99 ...
HISTORY_OUTSIDE_FLOOR = 1e-5   # ... and never tighter than this cosine distance
# Per group, not pooled: a group whose texts happen to be batch-sensitive would widen
# a pooled limit for everyone, and a group made by another model does NOT wobble more
# between today's two measurements — it just sits further from its stored vectors.
HISTORY_MAX_OUTSIDE = 0.05     # a group is SUSPECT if >5% of its chunks are outside


def _doc_counts(client, collection: str) -> dict[str, int]:
    """doc_id -> chunk count, via the facet API (falls back to a payload scroll)."""
    try:
        res = client.facet(collection_name=collection, key="doc_id", limit=1_000_000,
                           exact=True)
        return {h.value: h.count for h in res.hits}
    except Exception:  # noqa: BLE001 — older client/server without facet
        counts: dict[str, int] = {}
        offset = None
        while True:
            pts, offset = client.scroll(collection, limit=5000, offset=offset,
                                        with_payload=["doc_id"], with_vectors=False)
            for p in pts:
                d = (p.payload or {}).get("doc_id")
                counts[d] = counts.get(d, 0) + 1
            if offset is None:
                return counts


def _group_of(doc_id: str, recent: set[str]) -> str:
    if doc_id in recent:
        return "reingested-aug31"
    if doc_id.lower().endswith(".pdf"):
        return "pdf-june"
    if "/SOA Reference/" in doc_id:
        return "soa-june"
    return "other-md"


def _doc_chunks(client, collection: str, doc_id: str) -> list:
    from qdrant_client import models

    flt = models.Filter(must=[models.FieldCondition(
        key="doc_id", match=models.MatchValue(value=doc_id))])
    out, offset = [], None
    while True:
        pts, offset = client.scroll(collection, scroll_filter=flt, limit=512,
                                    offset=offset, with_payload=True, with_vectors=True)
        out += pts
        if offset is None:
            break
    return sorted(out, key=lambda p: (p.payload or {}).get("chunk_index", 0))


def _pctl(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))] if s else float("nan")


def history(args) -> int:
    import random

    from qdrant_client import QdrantClient

    from docs_bridge import config
    from docs_bridge.embed_onnx import OnnxEmbedder

    cfg = config.load(args.config)
    client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port, timeout=300)
    recent: set[str] = set()
    for f in args.recent_docs or []:
        recent |= {ln.strip() for ln in open(f) if ln.strip()}
    counts = _doc_counts(client, args.collection)
    missing = sorted(recent - set(counts))
    if missing:
        print(f"[history] note: {len(missing)} listed doc(s) not in the collection, "
              f"e.g. {missing[0]!r}", file=sys.stderr)
    groups: dict[str, list[str]] = {}
    for d in sorted(counts):
        groups.setdefault(_group_of(d, recent), []).append(d)
    print("[history] groups: " + ", ".join(f"{g} {len(v)} docs / "
          f"{sum(counts[d] for d in v)} chunks" for g, v in groups.items()),
          file=sys.stderr)

    emb = OnnxEmbedder(cfg.onnx_model_dir, use_int8=cfg.embedding_int8)
    model_sha = _sha256(os.path.join(cfg.onnx_model_dir, "model.int8.onnx"))
    rng = random.Random(args.seed)
    rows = []
    for g, docs in groups.items():
        for w in range(args.windows):
            # a window = up to --batch consecutive chunks in (doc_id, chunk_index)
            # order starting at a random chunk of a random doc of this group
            start = rng.randrange(len(docs))
            window = []
            for d in docs[start:]:
                pts = _doc_chunks(client, args.collection, d)
                if not window:
                    pts = pts[rng.randrange(len(pts)):]
                window += [p for p in pts if (p.payload or {}).get("text", "").strip()]
                if len(window) >= args.batch:
                    break
            window = window[: args.batch]
            texts = [p.payload["text"] for p in window]
            batched, _ = emb.encode(texts)
            for p, b in zip(window, batched):
                alone, _ = emb.encode([p.payload["text"]])
                stored = (p.vector or {}).get("dense")
                rows.append({
                    "group": g, "doc_id": p.payload.get("doc_id"),
                    "chunk_index": p.payload.get("chunk_index"),
                    "wobble": 1 - _cos(alone[0], b),
                    "d_alone": 1 - _cos(alone[0], stored),
                    "d_batched": 1 - _cos(b, stored),
                })
            print(f"[history] {g}: window {w + 1}/{args.windows}, {len(window)} chunks "
                  f"from {len({p.payload.get('doc_id') for p in window})} doc(s)",
                  file=sys.stderr)

    wob = [r["wobble"] for r in rows]
    print(f"\nembed history | {args.collection} | today's worker model "
          f"{model_sha[:16]} | {len(rows)} chunks re-embedded")
    print(f"  normal wobble (same model, alone vs batch of {args.batch}), all groups: "
          f"cosine distance median {_pctl(wob, .5):.2e}  p99 {_pctl(wob, .99):.2e}")
    print(f"  a stored vector is OUTSIDE when it sits further from BOTH of today's "
          f"measurements of it than {HISTORY_OUTSIDE_FACTOR:g}x its group's wobble p99\n")
    print(f"  {'group':<18}{'chunks':>7}{'wobble med':>12}{'limit':>10}{'stored med':>12}"
          f"{'stored p95':>12}{'stored max':>12}{'outside':>9}")
    fails, report = [], {}
    for g in groups:
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            continue
        gw = [r["wobble"] for r in rs]
        limit = max(HISTORY_OUTSIDE_FACTOR * _pctl(gw, 0.99), HISTORY_OUTSIDE_FLOOR)
        best = [min(r["d_alone"], r["d_batched"]) for r in rs]
        out = [r for r, b in zip(rs, best) if b > limit]
        frac = len(out) / len(rs)
        print(f"  {g:<18}{len(rs):>7}{_pctl(gw, .5):>12.2e}{limit:>10.2e}"
              f"{_pctl(best, .5):>12.2e}{_pctl(best, .95):>12.2e}{max(best):>12.2e}"
              f"{frac:>8.0%}")
        report[g] = {"chunks": len(rs), "wobble_p99": _pctl(gw, .99), "limit": limit,
                     "stored_median": _pctl(best, .5),
                     "stored_p95": _pctl(best, .95), "stored_max": max(best),
                     "outside": frac,
                     "outside_docs": sorted({r["doc_id"] for r in out})}
        if frac > HISTORY_MAX_OUTSIDE:
            fails.append(f"{g}: {frac:.0%} of its stored vectors sit outside the wobble — "
                         f"embedded by a different model than today's worker; docs e.g. "
                         f"{report[g]['outside_docs'][:3]}")
    verdict = "CONSISTENT" if not fails else "SUSPECT"
    print(f"\nVERDICT: {verdict}")
    for f in fails:
        print(f"  - {f}")
    if args.json_out:
        json.dump({"verdict": verdict, "failures": fails, "model_sha256": model_sha,
                   "wobble": {"median": _pctl(wob, .5), "p99": _pctl(wob, .99)},
                   "groups": report, "rows": rows},
                  open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")
    return 0 if verdict == "CONSISTENT" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="embed a sample inside an image")
    d.add_argument("--config", default="/config/config.yaml")
    d.add_argument("--collection", default="teamcenter")
    d.add_argument("--n", type=int, default=400)
    d.add_argument("--ids-from", help="a previous dump: embed exactly its chunks")
    d.add_argument("--queries", help="questions YAML (notes/gate2/questions-*.yaml)")
    d.add_argument("--out", required=True)
    h = sub.add_parser("history", help="were all stored vectors made by one model?")
    h.add_argument("--config", default="/config/config.yaml")
    h.add_argument("--collection", default="teamcenter")
    h.add_argument("--recent-docs", nargs="*", help="doc-id lists of a later re-ingest "
                   "(one per line) — their group is the 'known recent' reference")
    h.add_argument("--windows", type=int, default=2, help="windows per group")
    h.add_argument("--batch", type=int, default=64, help="window size = ingest batch")
    h.add_argument("--seed", type=int, default=7)
    h.add_argument("--json", dest="json_out")
    c = sub.add_parser("compare", help="old dump vs new dump")
    c.add_argument("--old", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--json", dest="json_out")
    args = ap.parse_args()
    return {"dump": dump, "history": history, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

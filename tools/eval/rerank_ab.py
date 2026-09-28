"""A/B one reranker against the recorded baseline, on the §2.2 fixture.

Serves three jobs with one comparison, because they ask the identical question:
  * Gate 1 §2.3 — GPU probe:            --backend torch --device cuda --precision fp16
  * Gate 1 §3   — smaller reranker:     --backend onnx --model-dir /opt/bge-reranker-base-onnx
  * Gate 2 §5.2 — reranker A/B:         the same two, on the upgraded toolchain

The fixture (tools/eval/dump_rerank_fixture.py) carries the queries, the real RRF
candidate passages and the CPU-INT8 baseline scores, so this script needs NO
Qdrant, NO corpus and NO docs-bridge config — it runs on a bare GPU box with only
torch + transformers installed:

    python tools/eval/rerank_ab.py --fixture fixture.json \
        --backend torch --model-id BAAI/bge-reranker-v2-m3 \
        --device cuda --precision fp16 --json gpu-probe.json

The ONNX path deliberately re-implements OnnxReranker's ~15 lines instead of
importing it, for two reasons: a probe must not require an image rebuild to try a
model the image does not carry, and the GPU host has no docs_bridge install. The
one intentional difference is `token_type_ids`, taken from the tokenizer encoding
rather than hardcoded zeros — identical for XLM-R (v2-m3 and bge-reranker-base are
both XLM-R, type_vocab_size=1), but correct for a BERT-family reranker too.
Prove faithfulness before trusting any comparison:

    --backend onnx --model-dir <the baked baseline dir> --selftest

which must reproduce the fixture's baseline column to float noise.

THE BAR (plan §2.4): top-3 ordered-identical and top-6 Jaccard >= 0.83 against
the baseline. A DIFFERENCE is not automatically a regression — fp16 is strictly
more precise than INT8 — so every disagreement is adjudicated with the
reranker-as-oracle method from validate_union.py: the deeper `oracle_pool` defines
the global-best top-k under EACH scorer, and we ask which ranking contains more of
it. That is the difference between "the new model changed the answer" and "the new
model got it wrong".
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time


# --- scorers ------------------------------------------------------------------
class OnnxScorer:
    """Faithful stand-in for docs_bridge.rerank.OnnxReranker (see module docstring)."""

    kind = "onnx"

    def __init__(self, model_dir: str, int8: bool, max_length: int,
                 threads: int | None, device: str) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_file = "model.int8.onnx" if int8 else "model.onnx"
        path = os.path.join(model_dir, onnx_file)
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found")
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(path, sess_options=so, providers=providers)
        self.providers = self.session.get_providers()
        self._inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = self.tokenizer.token_to_id("<pad>")
        if pad_id is None:
            pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=0 if pad_id is None else pad_id,
                                      pad_token="<pad>" if pad_id else "[PAD]")
        self._len_tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self._len_tok.enable_truncation(max_length=max_length)
        meta_path = os.path.join(model_dir, "meta.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.desc = (f"onnx {'int8' if int8 else 'fp32'} on {self.providers[0]} "
                     f"({self.meta.get('model_id', model_dir)})")

    def lengths(self, query: str, passages: list[str]) -> list[int]:
        return [len(e.ids) for e in
                self._len_tok.encode_batch([(query, p) for p in passages])]

    def score(self, query: str, passages: list[str]) -> list[float]:
        import numpy as np

        encs = self.tokenizer.encode_batch([(query, p) for p in passages])
        feeds = {
            "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encs], dtype=np.int64),
        }
        feeds = {k: v for k, v in feeds.items() if k in self._inputs}
        logits = self.session.run(None, feeds)[0]
        return np.asarray(logits, dtype=np.float32).reshape(-1).tolist()


class TorchScorer:
    """§2.3's probe: the official cross-encoder weights, fp16 on CUDA. Tests physics,
    not packaging — no ONNX export, no CUDA/ORT build, nothing to rebuild."""

    kind = "torch"

    def __init__(self, model_id: str, device: str, precision: str,
                 max_length: int, threads: int | None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda but torch.cuda.is_available() is False "
                             "— check the driver, the container's --device flag, and "
                             "that this is a CUDA build of torch")
        if threads and device == "cpu":
            torch.set_num_threads(threads)
        self.device = device
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_id)
        m = AutoModelForSequenceClassification.from_pretrained(model_id)
        if precision == "fp16":
            m = m.half()
        elif precision == "bf16":
            m = m.bfloat16()
        self.model = m.to(device).eval()
        gpu = (torch.cuda.get_device_name(0) if device == "cuda"
               and torch.cuda.is_available() else platform.processor() or platform.machine())
        self.desc = f"torch {precision} on {device} [{gpu}] ({model_id})"

    def lengths(self, query: str, passages: list[str]) -> list[int]:
        enc = self.tok([query] * len(passages), passages, padding=False,
                       truncation=True, max_length=self.max_length)
        return [len(x) for x in enc["input_ids"]]

    def _sync(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def score(self, query: str, passages: list[str]) -> list[float]:
        with self.torch.no_grad():
            enc = self.tok([query] * len(passages), passages, padding=True,
                           truncation=True, max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            self._sync()
            return logits.reshape(-1).float().cpu().tolist()


class Bucketed:
    """Wraps a scorer to length-sort and score in sub-batches of `bucket`, restoring
    the caller's order. Padding is mathematically inert, so for a float model this
    is a pure speed change; under dynamic INT8 the per-tensor activation scales are
    computed over whatever shares a batch, so it can move scores — which is why it
    goes through the same adjudication as any other candidate reranker rather than
    being assumed free."""

    def __init__(self, inner, bucket: int) -> None:
        self.inner, self.bucket = inner, bucket
        self.kind = inner.kind
        self.desc = f"{inner.desc} + length-bucketed sub-batches of {bucket}"

    def score(self, query: str, passages: list[str]) -> list[float]:
        lens = self.inner.lengths(query, passages)
        order = sorted(range(len(passages)), key=lambda i: lens[i])
        out = [0.0] * len(passages)
        for i in range(0, len(order), self.bucket):
            idx = order[i:i + self.bucket]
            for j, sc in zip(idx, self.inner.score(query, [passages[j] for j in idx])):
                out[j] = sc
        return out


# --- ranking helpers ----------------------------------------------------------
def rank_ids(entries: list[dict], scores: list[float], k: int) -> list:
    """search.py's selection, verbatim: stable sort by score desc, keep k — so equal
    scores keep RRF order, exactly as production breaks ties."""
    order = sorted(range(len(entries)), key=lambda i: scores[i], reverse=True)[:k]
    return [entries[i]["id"] for i in order]


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default=os.environ.get("AB_FIXTURE", "fixture.json"))
    ap.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    ap.add_argument("--model-dir", help="onnx backend: the exported model dir")
    ap.add_argument("--model-id", default="BAAI/bge-reranker-v2-m3",
                    help="torch backend: HF model id")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--precision", choices=["int8", "fp32", "fp16", "bf16"],
                    default="int8")
    ap.add_argument("--threads", type=int, default=0, help="0 = every core")
    ap.add_argument("--max-length", type=int, default=0, help="0 = the fixture's")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per query")
    ap.add_argument("--jaccard-min", type=float, default=0.83)
    ap.add_argument("--bucket", type=int, default=0,
                    help="length-sort the candidates and score in sub-batches of "
                         "this size (0 = one batch, today's behaviour). Cuts padding "
                         "waste; under INT8 it can also move scores, so it is judged "
                         "on the same bar as a new model.")
    ap.add_argument("--no-oracle", action="store_true",
                    help="skip scoring the deep pool (faster; no adjudication)")
    ap.add_argument("--selftest", action="store_true",
                    help="assert this scorer reproduces the fixture baseline column")
    ap.add_argument("--json", dest="json_out", help="write the full result as JSON")
    args = ap.parse_args()

    fx = json.load(open(args.fixture))
    meta = fx["meta"]
    k = meta["default_k"]
    max_length = args.max_length or meta["baseline"]["max_length"]

    if args.backend == "onnx":
        model_dir = args.model_dir or meta["baseline"]["model_dir"]
        scorer = OnnxScorer(model_dir, int8=(args.precision == "int8"),
                            max_length=max_length,
                            threads=args.threads or None, device=args.device)
    else:
        scorer = TorchScorer(args.model_id, args.device, args.precision,
                             max_length, args.threads or None)

    if args.bucket:
        scorer = Bucketed(scorer, args.bucket)

    print(f"\nrerank A/B | fixture={args.fixture}")
    print(f"  baseline:  {meta['baseline']['kind']} "
          f"({meta['baseline']['model_meta'].get('model_id', '?')}) on "
          f"{meta['host']} [{meta['machine']}, {meta['cpu_count']} cores], "
          f"{meta['timing']['rerank_ms_per_pair']:.0f} ms/pair")
    print(f"  candidate: {scorer.desc}")
    print(f"  top_n={meta['top_n']} k={k} max_length={max_length} "
          f"oracle_pool={meta['oracle_pool']} queries={len(fx['queries'])}")
    print(f"  bar: top-{min(3, k)} ordered-identical AND top-{k} Jaccard >= "
          f"{args.jaccard_min}\n")

    # warm-up on the first query's batch: model load, kernel autotune, thread pool
    warm = fx["queries"][0]["candidates"][:8]
    t0 = time.perf_counter()
    scorer.score(fx["queries"][0]["query"], [c["text"] for c in warm])
    print(f"  warm-up (8 pairs, includes any lazy init): "
          f"{time.perf_counter() - t0:.2f} s\n")

    hdr = (f"{'query':<42}{'s':>7}{'ms/pr':>7}{'spdup':>7}{'top3':>6}"
           f"{'jacc':>6}{'goldA':>7}{'goldB':>7}{'maxS':>8}")
    print(hdr)
    print("-" * len(hdr))

    results, fails, flags = [], [], []
    for q in fx["queries"]:
        cands = q["candidates"]
        texts = [c["text"] for c in cands]
        base = [c["baseline_score"] for c in cands]

        samples = []
        for _ in range(max(1, args.repeats)):
            t0 = time.perf_counter()
            new = scorer.score(q["query"], texts)
            samples.append(time.perf_counter() - t0)
        med = statistics.median(samples)

        base_top = rank_ids(cands, base, k)
        new_top = rank_ids(cands, new, k)
        top3 = base_top[:3] == new_top[:3]
        jac = jaccard(base_top, new_top)

        # oracle: the deep pool defines the global-best top-k under EACH scorer
        gold_a_recall = gold_b_recall = None
        oracle = {}
        pool = q.get("oracle_pool") or []
        if pool and not args.no_oracle:
            pool_new = scorer.score(q["query"], [c["text"] for c in pool])
            pool_base = [c["baseline_score"] for c in pool]
            gold_a = rank_ids(pool, pool_base, k)          # baseline's deep view
            gold_b = rank_ids(pool, pool_new, k)           # candidate's deep view
            prod_ids = {c["id"] for c in cands}
            ga = [i for i in gold_a if i in prod_ids]
            gb = [i for i in gold_b if i in prod_ids]
            gold_a_recall = (len(set(ga) & set(new_top)) / len(ga)) if ga else None
            gold_b_recall = (len(set(gb) & set(new_top)) / len(gb)) if gb else None
            oracle = {
                "gold_baseline_deep": gold_a, "gold_candidate_deep": gold_b,
                "gold_baseline_in_prod": len(ga), "gold_candidate_in_prod": len(gb),
                "base_recall_goldA": (len(set(ga) & set(base_top)) / len(ga)) if ga else None,
                "base_recall_goldB": (len(set(gb) & set(base_top)) / len(gb)) if gb else None,
                "cand_recall_goldA": gold_a_recall, "cand_recall_goldB": gold_b_recall,
            }

        row = {
            "query": q["query"], "control": q["control"], "pairs": len(cands),
            "median_s": med, "ms_per_pair": med / len(cands) * 1000.0,
            "speedup_vs_baseline": q["rerank_s"] / med if med else None,
            "top3_identical": top3, "jaccard": jac,
            "baseline_top": base_top, "candidate_top": new_top,
            "baseline_max": max(base), "candidate_max": max(new),
            "scores": new, "oracle": oracle,
        }
        results.append(row)

        print(f"{q['query'][:41]:<42}{med:>7.2f}{row['ms_per_pair']:>7.1f}"
              f"{(row['speedup_vs_baseline'] or 0):>6.1f}x"
              f"{('yes' if top3 else 'NO'):>6}{jac:>6.2f}"
              f"{('  n/a' if gold_a_recall is None else f'{gold_a_recall:>7.2f}')}"
              f"{('  n/a' if gold_b_recall is None else f'{gold_b_recall:>7.2f}')}"
              f"{row['candidate_max']:>8.2f}")

        if q["control"]:
            # The known-negative must stay negative, or the score sign no longer
            # separates real answers from abstention noise on this hardware/model.
            # Only a sign the BASELINE got right and this scorer lost is a failure;
            # a baseline that was already positive is a fact about the fixture (the
            # corpus really does hold something bread-adjacent), so it is a warning.
            if row["candidate_max"] >= 0 > row["baseline_max"]:
                fails.append(f"control query scored POSITIVE ({row['candidate_max']:+.2f}) "
                             f"where the baseline was {row['baseline_max']:+.2f} — this "
                             f"scorer no longer separates answers from abstention")
            elif row["candidate_max"] >= 0:
                print(f"  note: control is positive for BOTH baseline "
                      f"({row['baseline_max']:+.2f}) and candidate "
                      f"({row['candidate_max']:+.2f}) — inspect the fixture, not the scorer")
            continue
        if not top3 or jac < args.jaccard_min:
            flags.append(row)

    # --- verdict --------------------------------------------------------------
    real = [r for r in results if not r["control"]]
    med_speedup = statistics.median([r["speedup_vs_baseline"] for r in real
                                     if r["speedup_vs_baseline"]])
    med_ms = statistics.median([r["ms_per_pair"] for r in real])
    print(f"\nspeed: {med_ms:.1f} ms/pair median ({med_speedup:.1f}x the baseline's "
          f"{meta['timing']['rerank_ms_per_pair']:.0f} ms/pair)")
    print(f"       => {med_ms * meta['top_n'] / 1000:.2f} s per search at "
          f"top_n={meta['top_n']}, "
          f"{med_ms * 60 / 1000:.2f} s at multi_top_n=60")

    if args.selftest:
        worst = 0.0
        for r, q in zip(results, fx["queries"]):
            for got, want in zip(r["scores"], [c["baseline_score"] for c in q["candidates"]]):
                worst = max(worst, abs(got - want))
        ok = worst < 1e-3
        print(f"\nSELFTEST: max |candidate - baseline| = {worst:.2e} "
              f"=> {'FAITHFUL' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"selftest mismatch {worst:.2e} — this scorer does not "
                         f"reproduce the baseline; no comparison below is trustworthy")

    print("")
    if flags:
        print(f"DIFFERS from baseline on {len(flags)}/{len(real)} queries — "
              f"adjudicating with the deep-pool oracle:")
        for r in flags:
            o = r["oracle"]
            print(f"\n  {r['query'][:70]}")
            print(f"    top-3 identical: {r['top3_identical']}  jaccard: {r['jaccard']:.2f}")
            if not o:
                print("    no oracle pool in fixture — cannot adjudicate")
                continue
            def pc(v):
                return "n/a" if v is None else f"{v:.2f}"

            print(f"    gold (baseline's deep top-{k}, {o['gold_baseline_in_prod']} of "
                  f"which are in the reranked {r['pairs']}): candidate recalls "
                  f"{pc(o['cand_recall_goldA'])} vs baseline {pc(o['base_recall_goldA'])}")
            print(f"    gold (candidate's deep top-{k}, {o['gold_candidate_in_prod']} in "
                  f"the reranked set): candidate recalls "
                  f"{pc(o['cand_recall_goldB'])} vs baseline {pc(o['base_recall_goldB'])}")
            better = ((o["cand_recall_goldA"] or 0) >= (o["base_recall_goldA"] or 0)
                      and (o["cand_recall_goldB"] or 0) >= (o["base_recall_goldB"] or 0))
            print(f"    -> {'IMPROVEMENT or tie (accept)' if better else 'REGRESSION (reject)'}")
            if not better:
                fails.append(f"oracle-confirmed regression on: {r['query'][:60]}")
    else:
        print(f"MATCHES baseline on all {len(real)} real queries "
              f"(top-3 identical, top-{k} Jaccard >= {args.jaccard_min}).")

    verdict = "PASS" if not fails else "FAIL"
    print(f"\nVERDICT: {verdict}")
    for f in fails:
        print(f"  - {f}")

    if args.json_out:
        json.dump({
            "candidate": scorer.desc, "backend": args.backend,
            "device": args.device, "precision": args.precision,
            "model_dir": getattr(args, "model_dir", None), "model_id": args.model_id,
            "bucket": args.bucket,
            "host": platform.node(), "machine": platform.machine(),
            "fixture_meta": meta, "verdict": verdict, "failures": fails,
            "median_ms_per_pair": med_ms, "median_speedup": med_speedup,
            # per-pair scores kept (8 x 40 floats) so two CANDIDATE runs can be
            # diffed against each other, not only against the baseline column —
            # e.g. fp16 bucketed vs fp16 unbucketed isolates what bucketing moved
            "queries": results,
        }, open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

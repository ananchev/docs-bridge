# Retrieval

The docs-bridge server answers `search(subject, query, k)` with a single
retrieve→rerank path (`search.Searcher`) shared by the MCP tool and the REST
endpoint. It returns the top `k` chunks with citations — never an LLM-synthesized
answer.

## The search path

```
query
  │  1. embed (same BGE-M3 ONNX model as ingest)
  ▼
dense vec + sparse vec
  │  2. per collection: dense prefetch ──┐
  │                     sparse prefetch ─┴─▶ RRF fusion ─▶ top candidates
  ▼
pooled candidates (capped at the rerank budget)
  │  3. cross-encoder rerank (BGE-reranker, scores each query–passage pair)
  ▼
top k chunks + citations
```

### 1. Embed the query

The query is embedded with `get_embedder()` — **the exact same model and backend
the ingest-worker used**. This is what guarantees the query vector lives in the
same space as the stored document vectors. There is no separate query encoder.

### 2. Hybrid retrieve + RRF fusion

For each target collection, the server issues one Qdrant `query_points` call with
two prefetches:

- a **dense** prefetch (`limit = server.prefetch_limit`)
- a **sparse** prefetch (`limit = server.prefetch_limit`)

Qdrant fuses the two candidate lists server-side with **Reciprocal Rank Fusion
(RRF)**. Because BGE-M3 produces both vectors from one model, the lexical (sparse)
and semantic (dense) signals are complementary by construction — no separate BM25
index. See [embeddings.md](embeddings.md).

### 3. Cross-encoder rerank

The fused candidates are re-scored by a **cross-encoder** (`rerank.OnnxReranker`,
BGE-reranker-v2-m3 as ONNX/INT8). Unlike the bi-encoder embedder, a cross-encoder
sees the query and passage *together*, so it is the real quality lever — and the
expensive one. That is why it runs only over the pooled top candidates, not the
whole collection. The top `k` by rerank score are returned.

If reranking is disabled (`rerank.enabled: false`), the server falls back to the
RRF fusion score for ordering.

## Multi-subject search

The `subject` argument accepts three shapes:

| Value | Meaning |
|---|---|
| `"teamcenter"` | a single subject |
| `["teamcenter", "power-query"]` | a list — span several related corpora |
| `"all"` | every configured subject |

When a search spans multiple pools:

1. Each collection is retrieved + RRF-fused independently, and every candidate stays
   tagged with its source subject.
2. The per-pool lists are **round-robin interleaved** (each pool's best candidate
   first, then each pool's second, …) and the total is capped at the candidate
   **budget**.
3. The whole pooled set is reranked **globally** — cross-encoder scores are
   comparable across pools, so the best chunks win regardless of which corpus they
   came from.
4. Each returned chunk carries its `subject` so the caller knows where it came from.

The candidate budget is `rerank.top_n` for a single pool and the larger
`rerank.multi_top_n` when spanning pools (so cross-pool hits are not squeezed out).
A bigger pool does **not** mean proportionally more cost: rerank work is bounded by
the budget, not by the number of pools. The single-subject path is byte-identical
to retrieving one collection's top-`budget`.

## Result shape

Each result is a dict (`Searcher._to_result`):

```json
{
  "score": 7.42,
  "subject": "teamcenter",
  "source_path": "/data/docs/teamcenter/install-guide.pdf",
  "section": "Installation > Prerequisites",
  "last_updated": "2026-05-01T12:00:00+00:00",
  "doc_id": "install-guide.pdf",
  "chunk_id": "install-guide.pdf:12",
  "snippet": "first 320 chars …",
  "text": "the full chunk text"
}
```

`score` is the rerank logit (higher = more relevant), or the RRF score when
reranking is off. `source_path`, `section`, and `last_updated` are the citation
fields a consuming LLM should surface to the user.

## Lazy model loading

The `Searcher` loads its Qdrant client, embedder, and reranker **lazily and once**.
`/healthz` and `list_subjects` stay light (no heavy models touched); the embedder
and reranker only load on the first real `search`. After that they are reused for
every subject and every request — load once, serve many.

## list_subjects

`list_subjects()` returns each configured subject with its description, its
collection name, a **live** point count from Qdrant, and a `populated` flag. It is
both a discovery aid for callers and the signal an LLM uses to pick the right
`subject` before searching. See [server.md](server.md).

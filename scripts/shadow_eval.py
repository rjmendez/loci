#!/usr/bin/env python3
"""Live retrieval shadow-eval — the gate for flipping the reranker/query_expand upgrades.

Wires the real Loci corpus into mcp/retrieval_eval.py: samples findings from Qdrant
(grouped by investigation_id as the proxy relevance signal), retrieves candidate pools by
dense search, and scores retrieval configurations so a flip is recommended only on evidence.

Usage:
  scripts/shadow_eval.py [--queries 60] [--pool 40] [--k 10] \
      [--rerankers cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-v2-m3] \
      [--query-expand]

- Baseline is the first --rerankers entry (default: current MiniLM). Each other entry is a
  candidate compared against it. --query-expand adds a bare-vs-expanded A/B on the baseline.
- Candidates download their model on first use (bge-reranker-v2-m3 ≈ 600 MB). Env:
  OLLAMA_BASE_URL/EMBED_MODEL/QDRANT_URL/QDRANT_API_KEY (same as the Loci server).
Fail-open: a query whose retrieval raises is skipped, not fatal.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))


def _load_corpus(limit_scan: int) -> list:
    import server as S
    client, col = S._get_qdrant()
    if not client:
        return []
    items, offset = [], None
    while len(items) < limit_scan:
        pts, offset = client.scroll(collection_name=col, limit=min(256, limit_scan - len(items)),
                                    with_payload=True, with_vectors=False, offset=offset)
        for p in pts:
            pl = p.payload or {}
            txt = pl.get("text") or ""
            if txt and pl.get("investigation_id"):
                items.append({"id": str(p.id), "text": str(txt)[:2000],
                              "group": pl.get("investigation_id")})
        if not offset:
            break
    return items


def _make_retrieve(pool: int):
    """retrieve_fn(query, exclude_id) -> [{id, text}] dense-search candidate pool."""
    import server as S
    client, col = S._get_qdrant()

    def retrieve(query: str, exclude_id):
        vec = S._embed(query)
        if not vec:
            return []
        res = client.query_points(collection_name=col, query=vec, using="dense",
                                  limit=pool + 1, with_payload=True).points
        out = []
        for r in res:
            rid = str(r.id)
            if rid == str(exclude_id):
                continue
            out.append({"id": rid, "text": str((r.payload or {}).get("text", ""))[:512]})
        return out[:pool]

    return retrieve


def _make_rerank(model_name: str):
    """rerank_fn(query, cands) using reranker.rerank under RERANK_MODEL=model_name."""
    import reranker

    def rerank(query, cands):
        os.environ["RERANK_MODEL"] = model_name
        ranked = reranker.rerank(query, [c["text"] for c in cands], top_k=None)
        # map reranked indices back to the original candidate dicts (preserve ids)
        return [cands[r["index"]] for r in ranked]

    return rerank


def _expand_retrieve(base_retrieve, pool: int):
    """Query-expanded retrieval: union the pools of the original + expanded queries."""
    import query_expand

    def retrieve(query, exclude_id):
        exp = query_expand.expand(query)
        queries = [query] + [q for q in exp.get("queries", []) if q and q != query]
        seen, merged = set(), []
        for q in queries[:4]:
            for c in base_retrieve(q, exclude_id):
                if c["id"] not in seen:
                    seen.add(c["id"]); merged.append(c)
        return merged[: pool * 2]

    return retrieve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rerankers", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--query-expand", action="store_true")
    ap.add_argument("--scan", type=int, default=4000)
    args = ap.parse_args()

    import retrieval_eval as RE
    corpus = _load_corpus(args.scan)
    if not corpus:
        print("no corpus (qdrant unavailable?)", file=sys.stderr)
        return 1
    qset = RE.build_query_set(corpus, min_relevant=1, limit=args.queries)
    print(f"corpus={len(corpus)} query_set={len(qset)} pool={args.pool} k={args.k}", file=sys.stderr)
    if not qset:
        print("no query set (no investigations with >=2 findings)", file=sys.stderr)
        return 1

    retrieve = _make_retrieve(args.pool)
    models = [m.strip() for m in args.rerankers.split(",") if m.strip()]
    results = {}

    # baseline = dense retrieval + first reranker
    base_model = models[0]
    results[f"rerank:{base_model}"] = RE.evaluate(qset, retrieve, _make_rerank(base_model), k=args.k)
    baseline_key = f"rerank:{base_model}"

    report = {"configs": {}, "comparisons": []}
    report["configs"][baseline_key] = results[baseline_key]

    for cand in models[1:]:
        key = f"rerank:{cand}"
        results[key] = RE.evaluate(qset, retrieve, _make_rerank(cand), k=args.k)
        report["configs"][key] = results[key]
        report["comparisons"].append(RE.compare(baseline_key, results[baseline_key], key, results[key]))

    if args.query_expand:
        exp_retrieve = _expand_retrieve(retrieve, args.pool)
        key = "query_expand+" + base_model
        results[key] = RE.evaluate(qset, exp_retrieve, _make_rerank(base_model), k=args.k)
        report["configs"][key] = results[key]
        report["comparisons"].append(RE.compare(baseline_key, results[baseline_key], key, results[key]))

    print(json.dumps(report, indent=2))
    for c in report["comparisons"]:
        print(f"[gate] {c['candidate']} vs {c['baseline']}: "
              f"flip={c['recommend_flip']} — {c['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

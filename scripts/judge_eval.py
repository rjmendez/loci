#!/usr/bin/env python3
"""Judge-based A/B retrieval eval — Claude as the relevance oracle on real queries.

Replaces shadow_eval's weak same-investigation proxy with real relevance judged by Claude, so
a reranker/query_expand flip is earned on actual relevance. Three steps:

  1. prep : sample queries, retrieve + rank with each config, emit the per-query pools to judge
       scripts/judge_eval.py --prep --out judge_prep.json [--queries 40 --pool 24 --k 10 --rerankers a,b --query-expand]
  2. judge: a workflow fans out one judge agent per query -> {query_id: [relevant_doc_ids]}
       (see .claude/workflows/judge-eval.js; run it on judge_prep.json -> judge_grades.json)
  3. score: real recall@k / MRR / nDCG@k per config vs the judge grades + the flip gate
       scripts/judge_eval.py --score judge_prep.json judge_grades.json

Reuses shadow_eval's live retrieval/rerank helpers. Env: OLLAMA_BASE_URL/EMBED_MODEL/QDRANT_*.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _prep(args) -> int:
    import shadow_eval as SE
    corpus = SE._load_corpus(args.scan)
    if not corpus:
        print("no corpus (qdrant unavailable?)", file=sys.stderr)
        return 1
    by_id = {c["id"]: c for c in corpus}
    # Query set: sample corpus items whose text is a real, non-trivial query.
    qs = [c for c in corpus if len(c.get("text", "")) > 40][: args.queries]
    retrieve = SE._make_retrieve(args.pool)
    models = [m.strip() for m in args.rerankers.split(",") if m.strip()]
    exp_retrieve = SE._expand_retrieve(retrieve, args.pool) if args.query_expand else None

    # 1. Retrieve the candidate pools per query FIRST (no rerank model -> no cache thrash).
    base_pool, exp_pool = {}, {}
    for c in qs:
        try:
            base_pool[c["id"]] = retrieve(c["text"], c["id"])
            if exp_retrieve is not None:
                exp_pool[c["id"]] = exp_retrieve(c["text"], c["id"])
        except Exception as exc:
            print(f"skip retrieve {c['id']}: {exc}", file=sys.stderr)

    # 2. Rank CONFIG-MAJOR: load each reranker ONCE, rank every query's pool with it. Avoids
    # reloading the 600MB bge model per query (the thrash bug that made this hang).
    rankings: dict = {c["id"]: {} for c in qs if c["id"] in base_pool}
    for m in models:
        rr = SE._make_rerank(m)
        for c in qs:
            if c["id"] not in base_pool:
                continue
            ranked = rr(c["text"], base_pool[c["id"]])
            rankings[c["id"]][f"rerank:{m}"] = [r["id"] for r in ranked][: args.k]
    if exp_retrieve is not None:
        rr = SE._make_rerank(models[0])
        for c in qs:
            if c["id"] not in exp_pool:
                continue
            ranked = rr(c["text"], exp_pool[c["id"]])
            rankings[c["id"]]["query_expand+" + models[0]] = [r["id"] for r in ranked][: args.k]

    # 3. Assemble each query's union pool (docs any config ranked in top-k) for the judge.
    out_queries = []
    for c in qs:
        rk = rankings.get(c["id"])
        if not rk:
            continue
        pool_ids: set = set()
        for ids in rk.values():
            pool_ids.update(ids)
        pool = [{"id": pid, "text": str(by_id.get(pid, {}).get("text", ""))[:500]}
                for pid in pool_ids if pid in by_id]
        out_queries.append({"id": c["id"], "query": c["text"][:800], "pool": pool, "rankings": rk})

    report = {"k": args.k, "baseline": f"rerank:{models[0]}", "queries": out_queries}
    with open(args.out, "w") as f:
        json.dump(report, f)
    print(f"prepped {len(out_queries)} queries, {sum(len(x['pool']) for x in out_queries)} pool docs "
          f"-> {args.out}", file=sys.stderr)
    return 0


def _score(prep_path: str, grades_path: str) -> int:
    import retrieval_eval as RE
    prep = json.load(open(prep_path))
    grades = json.load(open(grades_path))  # {query_id: [relevant_doc_ids]}
    k = prep["k"]
    baseline = prep["baseline"]
    per_query = []
    graded = 0
    for q in prep["queries"]:
        rel = grades.get(q["id"])
        if rel is None:
            continue
        graded += 1
        per_query.append(RE.score_configs(q["rankings"], set(rel), k))
    configs = RE.aggregate(per_query, k)
    comparisons = []
    for cfg, metrics in configs.items():
        if cfg == baseline:
            continue
        comparisons.append(RE.compare(baseline, configs[baseline], cfg, metrics))
    report = {"graded_queries": graded, "baseline": baseline, "configs": configs,
              "comparisons": comparisons}
    print(json.dumps(report, indent=2))
    for c in comparisons:
        print(f"[gate] {c['candidate']} vs {c['baseline']}: flip={c['recommend_flip']} — {c['reason']}",
              file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--score", nargs=2, metavar=("PREP_JSON", "GRADES_JSON"))
    ap.add_argument("--out", default="judge_prep.json")
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--scan", type=int, default=4000)
    ap.add_argument("--rerankers", default="cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-v2-m3")
    ap.add_argument("--query-expand", action="store_true")
    args = ap.parse_args()
    if args.score:
        return _score(args.score[0], args.score[1])
    if args.prep:
        return _prep(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

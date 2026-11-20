"""Retrieval shadow-eval — measure retrieval quality so gated upgrades flip on EVIDENCE.

The reranker upgrade (RERANK_MODEL -> bge-reranker-v2-m3) and query_expand are shipped
default-OFF behind a gate (mirroring the DAMA shadow-eval + canary discipline). This module
is that gate: it scores retrieval configurations on a held-out query set and recommends a flip
only when a candidate beats the baseline by a margin without regressing.

Default labels: no gold exists, so we use a self-supervised proxy: each
corpus item is a query whose "relevant" set is the OTHER items sharing its group (e.g. same
investigation) — a weak but real topical signal. Metrics are standard rank quality: recall@k,
MRR, nDCG@k. Gold mode (load_gold + evaluate_gold) scores against real graded labels from
mcp/gold_labels.py instead; selected by shadow_eval.py --gold.

Pure + injectable: `evaluate()` takes a `retrieve_fn` (query -> candidate pool) and an optional
`rerank_fn` (query, candidates -> reordered), so tests stub them and the live CLI
(scripts/shadow_eval.py) wires real Qdrant retrieval + reranker.rerank / query_expand. Fail-open:
a query whose retrieval raises is skipped, not fatal.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence


def build_query_set(items: Sequence[dict], min_relevant: int = 1,
                    limit: Optional[int] = None) -> list[dict]:
    """Build a proxy-labeled query set from corpus items.

    items: [{id, text, group}]. For each item, query=its text and relevant_ids = other items
    with the SAME non-empty group. Items with < min_relevant same-group neighbors are dropped
    (no signal). Returns [{query_id, query, relevant_ids:set}]. Deterministic order.
    """
    by_group: dict = {}
    for it in items:
        g = it.get("group")
        if g in (None, ""):
            continue
        by_group.setdefault(g, []).append(it)
    out: list[dict] = []
    for it in items:
        g = it.get("group")
        if g in (None, ""):
            continue
        rel = {o["id"] for o in by_group.get(g, []) if o["id"] != it["id"]}
        if len(rel) < min_relevant:
            continue
        out.append({"query_id": it["id"], "query": str(it.get("text", "")), "relevant_ids": rel})
        if limit is not None and len(out) >= limit:
            break
    return out


def recall_at_k(ranked_ids: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top = list(ranked_ids)[:k]
    return len(set(top) & relevant) / len(relevant)


def mrr(ranked_ids: Sequence, relevant: set) -> float:
    for i, rid in enumerate(ranked_ids):
        if rid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top = list(ranked_ids)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, rid in enumerate(top) if rid in relevant)
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg else 0.0


def evaluate(query_set: Sequence[dict],
             retrieve_fn: Callable[[str, object], Sequence[dict]],
             rerank_fn: Optional[Callable[[str, Sequence[dict]], Sequence[dict]]] = None,
             k: int = 10) -> dict:
    """Score a retrieval config over the query set. Returns aggregate
    {recall@k, mrr, ndcg@k, n, skipped}.

    retrieve_fn(query, exclude_id) -> [{id, text}] candidate pool (pre-rerank).
    rerank_fn(query, candidates) -> reordered candidates (None -> keep retrieval order).
    Fail-open: a query whose retrieve/rerank raises is skipped (counted), never fatal.
    """
    sums = {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    n = 0
    skipped = 0
    for q in query_set:
        try:
            cands = list(retrieve_fn(q["query"], q["query_id"]))
            if rerank_fn is not None:
                cands = list(rerank_fn(q["query"], cands))
        except Exception:
            skipped += 1
            continue
        ranked = [c.get("id") for c in cands]
        rel = q["relevant_ids"]
        sums["recall"] += recall_at_k(ranked, rel, k)
        sums["mrr"] += mrr(ranked, rel)
        sums["ndcg"] += ndcg_at_k(ranked, rel, k)
        n += 1
    denom = n or 1
    return {
        f"recall@{k}": round(sums["recall"] / denom, 4),
        "mrr": round(sums["mrr"] / denom, 4),
        f"ndcg@{k}": round(sums["ndcg"] / denom, 4),
        "n": n, "skipped": skipped, "k": k,
    }


def graded_ndcg_at_k(ranked_ids: Sequence, gains: dict, k: int) -> float:
    """nDCG@k with graded gain 2^grade - 1. `gains` maps id -> grade (0-3); an id absent from it
    counts as grade 0. The ideal ranking is built from the labeled grades only."""
    def g(i):
        return (2 ** int(gains.get(i, 0))) - 1
    dcg = sum(g(rid) / math.log2(i + 2) for i, rid in enumerate(list(ranked_ids)[:k]))
    ideal = sorted((2 ** int(v) - 1 for v in gains.values()), reverse=True)[:k]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def load_gold(path: str, min_grade: int = 1) -> list[dict]:
    """Query set from a gold_labels.py JSONL. Rows: {type:"query", query_id, query, exclude_id} and
    {type:"label", query_id, candidate_id, grade}. relevant_ids = candidates with grade >= min_grade;
    `gains` keeps every labeled grade. Queries without a single relevant candidate are kept with an
    empty relevant set (evaluate_gold skips and counts them). Unreadable lines are ignored."""
    import json
    heads: dict = {}
    gains: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not isinstance(r, dict):
                continue
            if r.get("type") == "query" and r.get("query_id"):
                heads[r["query_id"]] = r
            elif r.get("type") == "label" and r.get("query_id") is not None:
                gains.setdefault(r["query_id"], {})[str(r.get("candidate_id"))] = int(r.get("grade", 0))
    out = []
    for qid, h in heads.items():
        gn = gains.get(qid, {})
        out.append({"query_id": qid, "query": str(h.get("query", "")), "exclude_id": h.get("exclude_id"),
                    "relevant_ids": {c for c, v in gn.items() if v >= min_grade}, "gains": gn})
    return out


def evaluate_gold(gold_set: Sequence[dict],
                  retrieve_fn: Callable[[str, object], Sequence[dict]],
                  rerank_fn: Optional[Callable[[str, Sequence[dict]], Sequence[dict]]] = None,
                  k: int = 10) -> dict:
    """Like evaluate(), against load_gold() output: recall@k and MRR on grade >= min_grade, nDCG@k
    graded. Queries with no relevant label are excluded (`no_relevant`); a failing retrieve/rerank
    is skipped (`skipped`). Candidates the gold never saw count as grade 0."""
    sums = {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    n = skipped = no_rel = 0
    for q in gold_set:
        if not q["relevant_ids"]:
            no_rel += 1
            continue
        try:
            cands = list(retrieve_fn(q["query"], q.get("exclude_id")))
            if rerank_fn is not None:
                cands = list(rerank_fn(q["query"], cands))
        except Exception:
            skipped += 1
            continue
        ranked = [str(c.get("id")) for c in cands]
        sums["recall"] += recall_at_k(ranked, q["relevant_ids"], k)
        sums["mrr"] += mrr(ranked, q["relevant_ids"])
        sums["ndcg"] += graded_ndcg_at_k(ranked, q["gains"], k)
        n += 1
    d = n or 1
    return {f"recall@{k}": round(sums["recall"] / d, 4), "mrr": round(sums["mrr"] / d, 4),
            f"ndcg@{k}": round(sums["ndcg"] / d, 4), "n": n, "skipped": skipped,
            "no_relevant": no_rel, "k": k, "labels": "gold"}


def score_configs(rankings_by_config: dict, relevant: set, k: int) -> dict:
    """Per-query: score each config's ranking against a JUDGE-supplied relevant set — the
    real-relevance path that replaces the same-investigation proxy. rankings_by_config maps a
    config name to its ranked [doc_id, ...]. Returns {config -> {recall, mrr, ndcg}} (raw)."""
    out = {}
    for cfg, ranked in (rankings_by_config or {}).items():
        rel = set(relevant or ())
        out[cfg] = {"recall": recall_at_k(ranked, rel, k), "mrr": mrr(ranked, rel),
                    "ndcg": ndcg_at_k(ranked, rel, k)}
    return out


def aggregate(per_query_scores: list, k: int) -> dict:
    """Mean each config's metrics across queries. Input is the list of score_configs() dicts.
    Returns {config -> {recall@k, mrr, ndcg@k, n, k}} — the shape compare() consumes. Queries
    with an empty judge-relevant set contribute 0s (a config can't recall what has no relevant)."""
    agg, counts = {}, {}
    for pq in per_query_scores or ():
        for cfg, m in (pq or {}).items():
            a = agg.setdefault(cfg, {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0})
            for kk in ("recall", "mrr", "ndcg"):
                a[kk] += m.get(kk, 0.0)
            counts[cfg] = counts.get(cfg, 0) + 1
    out = {}
    for cfg, a in agg.items():
        n = counts[cfg] or 1
        out[cfg] = {f"recall@{k}": round(a["recall"] / n, 4), "mrr": round(a["mrr"] / n, 4),
                    f"ndcg@{k}": round(a["ndcg"] / n, 4), "n": counts[cfg], "k": k}
    return out


def compare(baseline_name: str, baseline: dict, candidate_name: str, candidate: dict,
            primary: str = "ndcg", min_rel_gain: float = 0.02) -> dict:
    """Compare two eval results and recommend whether to flip to the candidate.

    Flip only when the candidate improves the PRIMARY metric by >= min_rel_gain (relative)
    AND does not regress either other metric by more than a small tolerance — the disciplined
    gate: a candidate must clearly win, not just tie. Returns deltas + recommendation.
    """
    def _key(base_metric: str, d: dict) -> str:
        for kk in d:
            if kk.startswith(base_metric):
                return kk
        return base_metric

    metrics = ["recall", "mrr", "ndcg"]
    deltas = {}
    for m in metrics:
        bk, ck = _key(m, baseline), _key(m, candidate)
        b, c = baseline.get(bk, 0.0), candidate.get(ck, 0.0)
        rel = (c - b) / b if b else (0.0 if c == 0 else float("inf"))
        deltas[m] = {"baseline": b, "candidate": c, "abs": round(c - b, 4),
                     "rel": round(rel, 4) if rel != float("inf") else "inf"}

    prim = deltas[primary]
    prim_rel = prim["rel"] if prim["rel"] != "inf" else 1.0
    regressed = [m for m in metrics if m != primary and deltas[m]["abs"] < -0.01]
    flip = (isinstance(prim_rel, (int, float)) and prim_rel >= min_rel_gain and not regressed)
    if flip:
        reason = f"{candidate_name} improves {primary} by {prim['rel']} (>= {min_rel_gain}) with no regression"
    elif regressed:
        reason = f"{candidate_name} regresses {regressed} — hold"
    else:
        reason = f"{candidate_name} does not clearly beat {baseline_name} on {primary} — hold"
    return {"baseline": baseline_name, "candidate": candidate_name, "primary": primary,
            "deltas": deltas, "recommend_flip": bool(flip), "reason": reason}

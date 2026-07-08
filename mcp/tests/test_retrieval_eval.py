"""Tests for retrieval_eval — the shadow-eval metrics + gate logic (no live models)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import retrieval_eval as RE  # noqa: E402


def test_build_query_set_uses_same_group_as_relevant():
    items = [
        {"id": "a", "text": "qa", "group": "inv1"},
        {"id": "b", "text": "qb", "group": "inv1"},
        {"id": "c", "text": "qc", "group": "inv2"},   # lone in its group -> dropped
        {"id": "d", "text": "qd", "group": ""},        # no group -> dropped
    ]
    qs = RE.build_query_set(items, min_relevant=1)
    ids = {q["query_id"] for q in qs}
    assert ids == {"a", "b"}                      # only inv1 has neighbors
    a = next(q for q in qs if q["query_id"] == "a")
    assert a["relevant_ids"] == {"b"} and "a" not in a["relevant_ids"]


def test_metrics_math():
    # ranked ids, relevant = {x, y}
    ranked = ["z", "x", "w", "y"]
    rel = {"x", "y"}
    assert RE.recall_at_k(ranked, rel, 2) == 0.5      # only x in top-2
    assert RE.recall_at_k(ranked, rel, 4) == 1.0
    assert RE.mrr(ranked, rel) == 0.5                 # first relevant at rank 2
    # nDCG@4: hits at positions 2 and 4 (0-indexed 1,3)
    import math
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert abs(RE.ndcg_at_k(ranked, rel, 4) - dcg / idcg) < 1e-9
    assert RE.mrr(["a", "b"], {"z"}) == 0.0           # none relevant


def test_evaluate_with_stub_retriever_and_reranker():
    qs = [{"query_id": "a", "query": "qa", "relevant_ids": {"b"}}]
    # retriever returns b in the middle; a perfect reranker lifts b to the top.
    def retrieve(query, exclude_id):
        assert exclude_id == "a"
        return [{"id": "x"}, {"id": "b"}, {"id": "y"}]
    def rerank(query, cands):
        return sorted(cands, key=lambda c: 0 if c["id"] == "b" else 1)
    base = RE.evaluate(qs, retrieve, rerank_fn=None, k=10)
    reranked = RE.evaluate(qs, retrieve, rerank_fn=rerank, k=10)
    assert base["n"] == 1 and base["mrr"] == 0.5          # b at rank 2
    assert reranked["mrr"] == 1.0                          # b lifted to rank 1
    assert reranked["ndcg@10"] > base["ndcg@10"]


def test_evaluate_fail_open_skips_bad_query():
    qs = [{"query_id": "a", "query": "qa", "relevant_ids": {"b"}},
          {"query_id": "c", "query": "qc", "relevant_ids": {"d"}}]
    def retrieve(query, exclude_id):
        if exclude_id == "c":
            raise RuntimeError("boom")
        return [{"id": "b"}]
    r = RE.evaluate(qs, retrieve, k=5)
    assert r["n"] == 1 and r["skipped"] == 1              # one scored, one skipped, no raise


def test_compare_recommends_flip_only_on_clear_win():
    base = {"recall@10": 0.50, "mrr": 0.40, "ndcg@10": 0.45}
    better = {"recall@10": 0.55, "mrr": 0.44, "ndcg@10": 0.52}   # +15% ndcg, no regression
    tie = {"recall@10": 0.50, "mrr": 0.40, "ndcg@10": 0.455}      # +1% ndcg -> below margin
    regress = {"recall@10": 0.40, "mrr": 0.41, "ndcg@10": 0.52}   # ndcg up but recall down

    assert RE.compare("mini", base, "bge", better)["recommend_flip"] is True
    assert RE.compare("mini", base, "bge", tie, min_rel_gain=0.02)["recommend_flip"] is False
    out = RE.compare("mini", base, "bge", regress)
    assert out["recommend_flip"] is False and "regress" in out["reason"]

"""retrieval_eval gold mode: load_gold, graded nDCG, evaluate_gold."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import retrieval_eval as RE  # noqa: E402


def _gold_file(tmp_path):
    rows = [
        {"type": "query", "query_id": "q1", "query": "disk", "exclude_id": "ex1"},
        {"type": "label", "query_id": "q1", "candidate_id": "a", "grade": 3},
        {"type": "label", "query_id": "q1", "candidate_id": "b", "grade": 1},
        {"type": "label", "query_id": "q1", "candidate_id": "c", "grade": 0},
        {"type": "query", "query_id": "q2", "query": "none relevant", "exclude_id": None},
        {"type": "label", "query_id": "q2", "candidate_id": "a", "grade": 0},
    ]
    p = tmp_path / "gold.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    return str(p)


def test_load_gold_relevant_set_and_gains(tmp_path):
    g = {q["query_id"]: q for q in RE.load_gold(_gold_file(tmp_path))}
    assert g["q1"]["relevant_ids"] == {"a", "b"} and g["q1"]["gains"] == {"a": 3, "b": 1, "c": 0}
    assert g["q1"]["exclude_id"] == "ex1" and g["q2"]["relevant_ids"] == set()
    strict = {q["query_id"]: q for q in RE.load_gold(_gold_file(tmp_path), min_grade=2)}
    assert strict["q1"]["relevant_ids"] == {"a"}


def test_graded_ndcg_rewards_putting_the_best_grade_first():
    gains = {"a": 3, "b": 1, "c": 0}
    assert RE.graded_ndcg_at_k(["a", "b", "c"], gains, 3) == 1.0
    # dcg = 1/log2(2) + 0 + 7/log2(4) = 4.5 ; idcg = 7 + 1/log2(3)
    assert round(RE.graded_ndcg_at_k(["b", "c", "a"], gains, 3), 4) == 0.5897
    assert RE.graded_ndcg_at_k(["z"], gains, 3) == 0.0          # unlabeled = grade 0
    assert RE.graded_ndcg_at_k(["a"], {}, 3) == 0.0


def test_evaluate_gold_exact_metrics_skips_and_exclude_id(tmp_path):
    seen = []

    def retrieve(query, exclude_id):
        seen.append((query, exclude_id))
        return [{"id": "b"}, {"id": "c"}, {"id": "a"}]

    res = RE.evaluate_gold(RE.load_gold(_gold_file(tmp_path)), retrieve, k=3)
    assert res == {"recall@3": 1.0, "mrr": 1.0, "ndcg@3": 0.5897, "n": 1, "skipped": 0,
                   "no_relevant": 1, "k": 3, "labels": "gold"}
    assert seen == [("disk", "ex1")]                               # the no-relevant query was never retrieved
    res_k1 = RE.evaluate_gold(RE.load_gold(_gold_file(tmp_path)), retrieve, k=1)
    assert (res_k1["recall@1"], res_k1["ndcg@1"]) == (0.5, 0.1429)  # only b (grade 1): 1 / 7


def test_evaluate_gold_rerank_changes_the_score_and_failures_are_counted(tmp_path):
    g = RE.load_gold(_gold_file(tmp_path))

    def base(q, ex):
        return [{"id": "b"}, {"id": "c"}, {"id": "a"}]

    def rerank(q, cands):
        return sorted(cands, key=lambda c: c["id"])

    assert RE.evaluate_gold(g, base, rerank, k=3)["ndcg@3"] == 1.0
    assert RE.evaluate_gold(g, base, None, k=3)["ndcg@3"] == 0.5897

    def bad(q, ex):
        raise ConnectionError("down")

    res = RE.evaluate_gold(g, bad, k=3)
    assert (res["skipped"], res["n"]) == (1, 0)

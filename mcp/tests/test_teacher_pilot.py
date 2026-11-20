import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import teacher_pilot as P  # noqa: E402

Q = {"q1": "ollama port shadowed", "q2": "qdrant backup"}


def lab(q, c, grade, a=None, b=None):
    return {"type": "label", "query_id": q, "candidate_id": c, "grade": grade, "labels": {"label_a": a, "label_b": b}}


LABELS = [lab("q1", "c1", 3, 3, 3), lab("q1", "c2", 1, 0, 1), lab("q2", "c3", 2, 1, 2), lab("q2", "c4", 0, 0, 0)]
TEXTS = {"c1": "portproxy rules shadow ollama", "c2": "unrelated", "c3": "qdrant snapshot backup job"}


def test_read_gold_splits_queries_and_labels_and_skips_junk(tmp_path):
    p = tmp_path / "g.jsonl"
    rows = [{"type": "query", "query_id": "q1", "query": "ollama port shadowed"}, "not json at all",
            lab("q1", "c1", 3), {"type": "label", "query_id": "q1", "candidate_id": "c9", "grade": True},
            {"type": "label", "query_id": "q1", "grade": 2}]
    p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows), encoding="utf-8")
    queries, labels = P.read_gold(str(p))
    assert queries == {"q1": "ollama port shadowed"}
    assert [(x["query_id"], x["candidate_id"], x["grade"]) for x in labels] == [("q1", "c1", 3)]


def test_read_gold_missing_file_raises_pilot_error(tmp_path):
    with pytest.raises(P.PilotError, match="cannot read"):
        P.read_gold(str(tmp_path / "nope.jsonl"))


def test_distill_rows_label_threshold_and_teacher_label():
    rows = P.to_distill_rows(Q, LABELS, TEXTS)
    assert [(r["id"], r["label"], r.get("teacher_label")) for r in rows] == [
        ("q1:c1", 1, 1), ("q1:c2", 0, 0), ("q2:c3", 1, 0)]
    assert rows[0]["claim"] == "ollama port shadowed"
    assert rows[0]["evidence"] == "portproxy rules shadow ollama"


def test_distill_rows_skips_pairs_without_text_and_truncates_evidence():
    rows = P.to_distill_rows(Q, [lab("q2", "c4", 0), lab("q1", "big", 3)], {"big": "x" * 5000})
    assert [r["id"] for r in rows] == ["q1:big"]
    assert len(rows[0]["evidence"]) == P.EVIDENCE_CHARS


def test_distill_rows_omit_teacher_label_when_labeler_has_no_grade():
    rows = P.to_distill_rows(Q, [lab("q1", "c1", 3, None, 3)], TEXTS)
    assert "teacher_label" not in rows[0]


class Judge:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies[len(self.prompts) - 1]


def test_calibration_pairs_judge_and_gold_by_one_id_and_counts_the_rest():
    j = Judge(['{"correct": true}', '{"correct": false}', "garbage"])
    jr, gr, counts = P.to_calibration(Q, LABELS, TEXTS, j, model="m1")
    assert jr == [{"id": "q1:c1", "verdict": True, "model": "m1"}, {"id": "q1:c2", "verdict": False, "model": "m1"}]
    assert gr == [{"id": "q1:c1", "grade": 3}, {"id": "q1:c2", "grade": 1}]
    assert counts == {"pairs": 4, "judged": 2, "no_text": 1, "unparseable": 1}
    assert "ollama port shadowed" in j.prompts[0] and "portproxy rules shadow ollama" in j.prompts[0]


def test_calibration_limit_stops_early():
    j = Judge(['{"correct": true}'] * 4)
    jr, _, counts = P.to_calibration(Q, LABELS, TEXTS, j, limit=1)
    assert len(jr) == 1 and counts["pairs"] == 1 and len(j.prompts) == 1


def test_calibration_output_is_accepted_by_judge_calibration():
    import judge_calibration as C
    j = Judge(['{"correct": true}', '{"correct": false}', '{"correct": true}'])
    jr, gr, _ = P.to_calibration(Q, LABELS, TEXTS, j)
    rep = C.calibrate(jr, gr, threshold=2)
    assert rep["n"] == 3


@pytest.mark.parametrize("text,want", [('{"correct": true}', True), ('x {"correct": false} y', False),
                                       ('{"correct": "yes"}', None), ("", None), ("{bad", None)])
def test_parse_verdict(text, want):
    assert P.parse_verdict(text) is want


def test_parse_terms_dedupes_filters_and_caps():
    t = '{"terms": ["Portproxy", "portproxy", "", "x", "a" , "shadow", "ollama", "netsh", "iphlpsvc", "extra"]}'
    assert P.parse_terms(t) == ["Portproxy", "shadow", "ollama", "netsh", "iphlpsvc"]
    assert P.parse_terms('{"terms": "nope"}') == []
    assert P.parse_terms("none") == []


def test_query_expand_gold_keeps_only_terms_in_doc_and_not_in_query():
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return '{"terms": ["portproxy", "ollama", "kubernetes"]}'
    rows, counts = P.to_query_expand_gold(Q, LABELS, TEXTS, ask)
    # q1 target = c1 (grade 3): 'ollama' is already in the query, 'kubernetes' is not in the doc.
    assert rows[0] == {"id": "q1", "query": "ollama port shadowed", "expect": ["portproxy"]}
    assert counts["queries"] == 2 and len(asked) == 2


def test_query_expand_gold_refusal_stops_and_is_counted():
    calls = []

    def ask(prompt):
        calls.append(prompt)
        return None
    rows, counts = P.to_query_expand_gold(Q, LABELS, TEXTS, ask)
    assert rows == [] and counts["refused"] == 1 and len(calls) == 1


def test_query_expand_gold_drops_query_whose_terms_all_miss():
    rows, counts = P.to_query_expand_gold({"q1": "a b"}, [lab("q1", "c1", 3)], TEXTS, lambda p: '{"terms": ["zzz"]}')
    assert rows == [] and counts == {"queries": 1, "kept": 0, "no_terms": 1, "refused": 0}


def test_query_expand_gold_ignores_irrelevant_only_queries():
    rows, counts = P.to_query_expand_gold(Q, [lab("q1", "c2", 1)], TEXTS, lambda p: '{"terms": ["x1"]}')
    assert rows == [] and counts["queries"] == 0

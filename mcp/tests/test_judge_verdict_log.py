"""Every contradiction-judge decision is persisted, with ids and scores only.

Before this, _detect_conflicts judged up to ten neighbours per store and kept
none of the verdicts: only the first *conflict* reached conflicts.jsonl, without
its verdict. Agreements and same-topic verdicts, the negatives a pre-filter needs,
were thrown away.
"""
import json
import uuid
from unittest import mock

import pytest

import conflict_verify
import server


class _Point:
    def __init__(self, pid, score, payload):
        self.id = pid
        self.score = score
        self.payload = payload


class _Response:
    def __init__(self, points):
        self.points = points


class _Client:
    def __init__(self, points):
        self._points = points

    def query_points(self, **kwargs):
        return _Response(list(self._points))


NEW_TEXT = "the service exposes port 443 to the internet"
SECRET_REASON = "same port opposite polarity reason text"


def _neighbours():
    return [
        # The new finding itself comes back from the index and must be skipped.
        _Point("f-new", 0.99, {"id": "f-new", "record_type": "observed", "text": NEW_TEXT}),
        _Point("n-gap", 0.91234567, {"id": "n-gap", "record_type": "gap",
                                     "text": "unknown whether port 443 is exposed"}),
        _Point("n-obs", 0.8765, {"id": "n-obs", "record_type": "observed",
                                 "text": "port 443 is closed to the internet"}),
    ]


def _verdict_by_text(prompt, **kwargs):
    verdict = "contradict" if "closed" in prompt else "agree"
    return {"text": json.dumps({"verdict": verdict, "reason": SECRET_REASON}),
            "ok": True, "model": "judge-model:7b", "tier": "vllm"}


@pytest.fixture
def inv():
    return f"judge-log-{uuid.uuid4().hex[:8]}"


def _detect(inv, gen=_verdict_by_text, points=None):
    client = _Client(points if points is not None else _neighbours())
    with mock.patch.object(server, "_get_qdrant", lambda: (client, "loci_memory")), \
         mock.patch.object(server, "_embed", lambda _t: [0.1, 0.2, 0.3]), \
         mock.patch.object(conflict_verify, "_lazy_generate", gen):
        return server._detect_conflicts(
            inv, {"id": "f-new", "record_type": "observed", "text": NEW_TEXT})


def _log(inv):
    path = server._inv_dir(inv) / server.JUDGE_VERDICT_LOG_NAME
    return path, [json.loads(line) for line in path.read_text().splitlines()]


def test_every_judged_pair_is_logged_not_only_the_first_conflict(inv, monkeypatch):
    monkeypatch.delenv("LOCI_JUDGE_VERDICT_LOG", raising=False)
    conflicts = _detect(inv)
    # Both neighbours are conflicts (gap filled; judge says contradict)...
    assert [c["neighbor_id"] for c in conflicts] == ["n-gap", "n-obs"]
    _, rows = _log(inv)
    for row in rows:
        assert row.pop("ts")
    assert rows == [
        {"schema": 1, "event": "conflict_judge", "new_finding_id": "f-new",
         "neighbor_id": "n-gap", "neighbor_rank": 1, "similarity": 0.9123,
         "new_type": "observed", "neighbor_type": "gap",
         "heuristic_rule": "gap_filled", "verdict": "agree", "judge_ok": True,
         "model": "judge-model:7b", "tier": "vllm", "conflict_candidate": True},
        {"schema": 1, "event": "conflict_judge", "new_finding_id": "f-new",
         "neighbor_id": "n-obs", "neighbor_rank": 2, "similarity": 0.8765,
         "new_type": "observed", "neighbor_type": "observed",
         "heuristic_rule": None, "verdict": "contradict", "judge_ok": True,
         "model": "judge-model:7b", "tier": "vllm", "conflict_candidate": True},
    ]


def test_non_conflict_verdicts_are_logged_too(inv):
    points = [_Point("n-obs", 0.9, {"id": "n-obs", "record_type": "observed",
                                    "text": "port 443 answers with a TLS banner"})]
    assert _detect(inv, points=points) == []
    _, rows = _log(inv)
    assert [(r["neighbor_id"], r["verdict"], r["conflict_candidate"]) for r in rows] == [
        ("n-obs", "agree", False)]


def test_log_holds_no_finding_text_or_judge_reason(inv):
    _detect(inv)
    path, _ = _log(inv)
    raw = path.read_text()
    for forbidden in (NEW_TEXT, "unknown whether", "closed to the internet", SECRET_REASON):
        assert forbidden not in raw


def test_judge_failure_is_logged_as_no_verdict(inv):
    def _down(prompt, **kwargs):
        return {"text": "", "ok": False, "model": "judge-model:7b", "why": "backend down"}

    points = [_Point("n-obs", 0.9, {"id": "n-obs", "record_type": "observed", "text": "x"})]
    assert _detect(inv, gen=_down, points=points) == []
    _, rows = _log(inv)
    assert [(r["verdict"], r["judge_ok"], r["model"], r["tier"]) for r in rows] == [
        (None, False, "judge-model:7b", None)]


def test_appends_across_stores(inv):
    _detect(inv)
    _detect(inv)
    _, rows = _log(inv)
    assert len(rows) == 4


def test_flag_off_writes_nothing_and_changes_no_result(inv, monkeypatch):
    monkeypatch.setenv("LOCI_JUDGE_VERDICT_LOG", "0")
    conflicts = _detect(inv)
    assert [c["neighbor_id"] for c in conflicts] == ["n-gap", "n-obs"]
    assert not (server._inv_dir(inv) / server.JUDGE_VERDICT_LOG_NAME).exists()


def test_log_write_failure_never_breaks_detection(inv):
    with mock.patch("instrumentation_log.append_rows", side_effect=RuntimeError("disk")):
        conflicts = _detect(inv)
    assert [c["neighbor_id"] for c in conflicts] == ["n-gap", "n-obs"]


def test_judge_pair_reports_model_and_tier_without_changing_the_verdict():
    out = server._judge_conflict_pair(
        {"text": "a", "record_type": "observed"}, {"text": "b", "record_type": "observed"},
        gen_fn=lambda p, **k: {"text": '{"verdict":"agree","reason":"r"}', "ok": True,
                               "model": "m1"},
    )
    assert out == {"verdict": "agree", "reason": "r", "ok": True, "error": None,
                   "model": "m1", "tier": None}


def test_judge_pair_without_generation_reports_no_model():
    out = server._judge_conflict_pair({"text": ""}, {"text": "b"}, gen_fn=lambda p, **k: 1 / 0)
    assert out == {"verdict": None, "reason": "", "ok": True, "error": None,
                   "model": None, "tier": None}

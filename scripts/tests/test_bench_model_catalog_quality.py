from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bench_model_catalog_quality as B  # noqa: E402


def test_quality_cases_cover_required_roles_with_hard_minimum():
    B.validate_cases(min_per_role=15)
    counts = {role: len(cases) for role, cases in B.cases_by_role().items()}
    assert counts == {
        "cheap_fanout": 15,
        "escalation": 15,
        "guardian": 15,
        "synthesis": 15,
    }
    assert len({case["id"] for case in B.CASES}) == len(B.CASES)


def test_scoring_accepts_json_answer_and_rejects_forbidden_terms():
    case = next(case for case in B.CASES if case["id"] == "cheap-001")
    good = B.score_output(case, '{"answer":"LOCI-1842","rationale":"incident id only"}')
    bad = B.score_output(case, '{"answer":"LOCI-1842 east-7","rationale":"included shard"}')

    assert good.correct is True
    assert good.status == "correct"
    assert bad.correct is False
    assert "forbidden:east-7" in bad.reasons


def test_scoring_distinguishes_timeout_unavailable_and_wrong():
    case = next(case for case in B.CASES if case["id"] == "guardian-001")
    wrong = B.score_output(case, '{"verdict":"ALLOW","category":"benign"}')
    timeout = B.score_output(case, "", transport_status="timeout")
    unavailable = B.score_output(case, "not found", transport_status="unavailable")

    assert wrong.status == "wrong"
    assert timeout.status == "timeout"
    assert unavailable.status == "unavailable"
    assert not wrong.correct and not timeout.correct and not unavailable.correct


def test_guardian_scoring_accepts_native_safe_unsafe_outputs():
    block_case = next(case for case in B.CASES if case["id"] == "guardian-001")
    allow_case = next(case for case in B.CASES if case["id"] == "guardian-003")

    assert B.score_output(block_case, "unsafe\nS2").correct is True
    assert B.score_output(allow_case, "safe").correct is True


def test_quality_ranking_does_not_break_ties_by_latency():
    rows = [
        {"role": "cheap_fanout", "model": "slow", "correct": True, "score_status": "correct", "transport_status": "ok", "latency_ms": 2000},
        {"role": "cheap_fanout", "model": "fast", "correct": True, "score_status": "correct", "transport_status": "ok", "latency_ms": 10},
    ]
    summary = B.summarize(rows)
    B.attach_quality_winners(summary)

    winners = {row["model"] for row in summary if row["quality_rank"] == "winner"}
    assert winners == {"fast", "slow"}
    assert all(row["tie_policy"] == "latency ignored; equal accuracy remains a tie" for row in summary)

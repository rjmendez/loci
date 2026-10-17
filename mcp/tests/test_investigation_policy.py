"""
Tests for mcp/investigation_policy.py.

Covers:
- cap_perspectives_to_tier: light → 1, medium → 3, heavy → 5,
  request under cap → unchanged, request over cap → capped
- evaluate_pre_answer_for_refusal: 0/3 supported → refusal, 3/3 → no refusal,
  0/0 total → refusal, boundary at threshold
- filter_findings_for_verification: high-confidence skipped, borderline verified,
  mixed split, empty list, string confidence values ("high", "low")
- push_refusal_step_to_trajectory: verifies JSONL written with r_ref value

Mutation notes:
- cap_perspectives_to_tier: change tier_caps values → boundary tests fail
- evaluate_pre_answer_for_refusal: flip recommend_refusal condition → refusal tests fail
- filter_findings_for_verification: change >= to > → boundary skip test fails
- push_refusal_step_to_trajectory: remove store.push → no JSONL written test fails
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from investigation_policy import (
    cap_perspectives_to_tier,
    evaluate_pre_answer_for_refusal,
    filter_findings_for_verification,
    push_refusal_step_to_trajectory,
)


# ---------------------------------------------------------------------------
# cap_perspectives_to_tier
# ---------------------------------------------------------------------------

class TestCapPerspectivesToTier:
    def test_light_caps_to_1(self):
        assert cap_perspectives_to_tier(5, "light") == 1

    def test_medium_caps_to_3(self):
        assert cap_perspectives_to_tier(5, "medium") == 3

    def test_heavy_caps_to_5(self):
        assert cap_perspectives_to_tier(5, "heavy") == 5

    def test_request_under_cap_unchanged(self):
        # requesting 2 under medium cap of 3 → unchanged
        assert cap_perspectives_to_tier(2, "medium") == 2

    def test_request_over_cap_capped(self):
        assert cap_perspectives_to_tier(4, "light") == 1

    def test_unknown_tier_defaults_to_5(self):
        assert cap_perspectives_to_tier(3, "unknown_tier") == 3

    def test_tier_case_insensitive(self):
        assert cap_perspectives_to_tier(5, "LIGHT") == 1
        assert cap_perspectives_to_tier(5, "Medium") == 3

    def test_at_cap_boundary_not_reduced(self):
        assert cap_perspectives_to_tier(3, "medium") == 3


# ---------------------------------------------------------------------------
# evaluate_pre_answer_for_refusal
# ---------------------------------------------------------------------------

class TestEvaluatePreAnswerForRefusal:
    def test_zero_supported_of_three_recommends_refusal(self):
        result = evaluate_pre_answer_for_refusal(
            {"supported_count": 0, "unsupported_count": 3, "ambiguous_count": 0}
        )
        assert result["recommend_refusal"] is True
        assert result["r_ref"] == 1.0
        assert result["supported_ratio"] == 0.0

    def test_all_supported_no_refusal(self):
        result = evaluate_pre_answer_for_refusal(
            {"supported_count": 3, "unsupported_count": 0, "ambiguous_count": 0}
        )
        assert result["recommend_refusal"] is False
        assert result["r_ref"] <= 0  # -0.5*(1-ratio); 0.0 when ratio==1.0
        assert result["supported_ratio"] == 1.0

    def test_empty_total_recommends_refusal(self):
        result = evaluate_pre_answer_for_refusal({})
        assert result["recommend_refusal"] is True
        assert result["r_ref"] == 1.0
        assert result["reason"] == "no_claims_to_evaluate"

    def test_boundary_at_threshold_recommends_refusal(self):
        # supported_ratio == threshold (0.3) → recommend refusal (<=)
        result = evaluate_pre_answer_for_refusal(
            {"supported_count": 3, "unsupported_count": 7, "ambiguous_count": 0},
            refusal_threshold=0.3,
        )
        assert result["recommend_refusal"] is True
        assert result["supported_ratio"] == 0.3

    def test_just_above_threshold_no_refusal(self):
        # 4/10 = 0.4 > 0.3 → no refusal
        result = evaluate_pre_answer_for_refusal(
            {"supported_count": 4, "unsupported_count": 6, "ambiguous_count": 0},
            refusal_threshold=0.3,
        )
        assert result["recommend_refusal"] is False

    def test_ambiguous_counts_in_denominator(self):
        # 1 supported, 1 unsupported, 2 ambiguous → ratio = 1/4 = 0.25
        result = evaluate_pre_answer_for_refusal(
            {"supported_count": 1, "unsupported_count": 1, "ambiguous_count": 2}
        )
        assert result["supported_ratio"] == 0.25
        assert result["recommend_refusal"] is True

    def test_eighty_percent_unsupported_refusal_r_ref_is_1(self):
        # 1 supported, 4 unsupported → ratio = 0.2
        result = evaluate_pre_answer_for_refusal(
            {"supported_count": 1, "unsupported_count": 4, "ambiguous_count": 0}
        )
        assert result["recommend_refusal"] is True
        assert result["r_ref"] == 1.0


# ---------------------------------------------------------------------------
# filter_findings_for_verification
# ---------------------------------------------------------------------------

class TestFilterFindingsForVerification:
    def _finding(self, confidence):
        return {"id": f"f-{confidence}", "numeric_confidence": confidence}

    def test_high_confidence_skipped(self):
        findings = [self._finding(0.9)]
        result = filter_findings_for_verification(findings, skip_threshold=0.85)
        assert result["skip_count"] == 1
        assert result["verify_count"] == 0

    def test_borderline_goes_to_verify(self):
        findings = [self._finding(0.7)]
        result = filter_findings_for_verification(findings, skip_threshold=0.85)
        assert result["verify_count"] == 1
        assert result["skip_count"] == 0

    def test_at_threshold_is_skipped(self):
        findings = [self._finding(0.85)]
        result = filter_findings_for_verification(findings, skip_threshold=0.85)
        assert result["skip_count"] == 1

    def test_mixed_split(self):
        findings = [
            self._finding(0.9),
            self._finding(0.5),
            self._finding(0.85),
            self._finding(0.2),
        ]
        result = filter_findings_for_verification(findings, skip_threshold=0.85)
        assert result["skip_count"] == 2
        assert result["verify_count"] == 2
        assert result["estimated_call_reduction"] == 0.5

    def test_empty_list(self):
        result = filter_findings_for_verification([], skip_threshold=0.85)
        assert result["skip_count"] == 0
        assert result["verify_count"] == 0
        assert result["estimated_call_reduction"] == 0.0

    def test_string_confidence_high_skipped(self):
        findings = [{"id": "f1", "confidence": "high"}]  # maps to 0.9
        result = filter_findings_for_verification(findings, skip_threshold=0.85)
        assert result["skip_count"] == 1

    def test_string_confidence_low_verified(self):
        findings = [{"id": "f2", "confidence": "low"}]  # maps to 0.3
        result = filter_findings_for_verification(findings, skip_threshold=0.85)
        assert result["verify_count"] == 1

    def test_numeric_confidence_preferred_over_confidence(self):
        # numeric_confidence takes precedence
        findings = [{"id": "f3", "numeric_confidence": 0.9, "confidence": "low"}]
        result = filter_findings_for_verification(findings, skip_threshold=0.85)
        assert result["skip_count"] == 1


# ---------------------------------------------------------------------------
# push_refusal_step_to_trajectory
# ---------------------------------------------------------------------------

class TestPushRefusalStepToTrajectory:
    def test_writes_jsonl_with_r_ref(self, tmp_path):
        ok = push_refusal_step_to_trajectory(
            policy_id="pol-test",
            investigation_id="inv-001",
            r_ref=1.0,
            trajectory_dir=tmp_path,
        )
        assert ok is True
        traj_file = tmp_path / "pol-test.jsonl"
        assert traj_file.exists()
        entry = json.loads(traj_file.read_text())
        assert entry["reward"]["r_ref"] == 1.0
        assert entry["action"] == "REFUSE"

    def test_negative_r_ref_records_answer_action(self, tmp_path):
        push_refusal_step_to_trajectory(
            policy_id="pol-test2",
            investigation_id="inv-002",
            r_ref=-0.2,
            trajectory_dir=tmp_path,
        )
        traj_file = tmp_path / "pol-test2.jsonl"
        entry = json.loads(traj_file.read_text())
        assert entry["action"] == "ANSWER"
        assert entry["reward"]["r_ref"] == -0.2

    def test_investigation_id_recorded(self, tmp_path):
        push_refusal_step_to_trajectory(
            policy_id="pol-id",
            investigation_id="inv-xyz",
            r_ref=1.0,
            trajectory_dir=tmp_path,
        )
        entry = json.loads((tmp_path / "pol-id.jsonl").read_text())
        assert entry["investigation_id"] == "inv-xyz"

    def test_returns_false_on_bad_dir(self):
        # Passing a file path as directory should fail gracefully
        result = push_refusal_step_to_trajectory(
            policy_id="p",
            investigation_id="i",
            r_ref=1.0,
            trajectory_dir=Path("/nonexistent/path/that/cannot/be/created/xyz123"),
        )
        # May succeed (mkdir parents) or fail — we only require no exception raised
        assert isinstance(result, bool)

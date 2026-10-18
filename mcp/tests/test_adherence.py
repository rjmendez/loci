"""
Tests for mcp/adherence.py — Policy Adherence Confidence tracking.
"""

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from adherence import AdherenceStore, PolicyAdherenceRecord


def _make_record(policy_id="pol-1", call_id="call-1", was_downgraded=True,
                 r_cor=0.9, r_step=-0.01, tier_requested="light",
                 model_requested="claude-opus-4.8", model_used="claude-haiku-4.5") -> PolicyAdherenceRecord:
    return PolicyAdherenceRecord(
        policy_id=policy_id,
        call_id=call_id,
        tier_requested=tier_requested,
        model_requested=model_requested,
        model_used=model_used,
        was_downgraded=was_downgraded,
        r_cor=r_cor,
        r_step=r_step,
    )


class TestAdherenceStoreRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = AdherenceStore(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_and_read_roundtrip(self):
        rec = _make_record(call_id="call-abc", r_cor=0.85)
        self.store.record(rec)
        results = self.store.read("pol-1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].call_id, "call-abc")
        self.assertAlmostEqual(results[0].r_cor, 0.85)
        self.assertTrue(results[0].was_downgraded)

    def test_read_limit_respected(self):
        for i in range(20):
            self.store.record(_make_record(call_id=f"call-{i}"))
        results = self.store.read("pol-1", limit=5)
        self.assertEqual(len(results), 5)
        # Most recent 5 — call-15..call-19
        self.assertEqual(results[-1].call_id, "call-19")

    def test_read_empty_policy_returns_empty_list(self):
        results = self.store.read("nonexistent-policy-id")
        self.assertEqual(results, [])

    def test_multiple_policies_isolated(self):
        self.store.record(_make_record(policy_id="pol-A"))
        self.store.record(_make_record(policy_id="pol-B"))
        self.assertEqual(len(self.store.read("pol-A")), 1)
        self.assertEqual(len(self.store.read("pol-B")), 1)
        self.assertEqual(len(self.store.read("pol-C")), 0)

    def test_corrupt_line_skipped(self):
        path = Path(self.tmp.name) / "pol-1.jsonl"
        path.write_text('not-json\n{"policy_id":"pol-1","call_id":"ok","tier_requested":"light","model_requested":"a","model_used":"b","was_downgraded":true,"r_cor":0.9,"r_step":-0.01,"timestamp":"2026-01-01T00:00:00+00:00","context":{}}\n', encoding="utf-8")
        results = self.store.read("pol-1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].call_id, "ok")


class TestAdherenceComputeReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = AdherenceStore(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_insufficient_data_when_below_min_records(self):
        for i in range(3):
            self.store.record(_make_record(call_id=f"c{i}"))
        report = self.store.compute_report("pol-1", min_records=5)
        self.assertEqual(report["recommendation"], "insufficient_data")
        self.assertEqual(report["confidence"], 0.0)
        self.assertEqual(report["record_count"], 3)

    def test_effective_when_downgraded_rcor_gte_85pct_baseline(self):
        # baseline: 10 non-downgraded calls with r_cor=0.93
        for i in range(10):
            self.store.record(_make_record(call_id=f"base-{i}", was_downgraded=False, r_cor=0.93))
        # downgraded: 10 calls with r_cor=0.91 → effectiveness = 0.91/0.93 ≈ 0.978
        for i in range(10):
            self.store.record(_make_record(call_id=f"down-{i}", was_downgraded=True, r_cor=0.91))
        report = self.store.compute_report("pol-1")
        self.assertEqual(report["recommendation"], "effective")
        self.assertGreaterEqual(report["downgrade_effectiveness"], 0.85)
        self.assertGreater(report["adherence_rate"], 0.1)

    def test_ineffective_when_downgraded_rcor_lt_70pct_baseline(self):
        # baseline: 5 calls r_cor=1.0
        for i in range(5):
            self.store.record(_make_record(call_id=f"base-{i}", was_downgraded=False, r_cor=1.0))
        # downgraded: 5 calls r_cor=0.5 → effectiveness=0.5 < 0.7, and >3 downgraded
        for i in range(5):
            self.store.record(_make_record(call_id=f"down-{i}", was_downgraded=True, r_cor=0.5))
        report = self.store.compute_report("pol-1")
        self.assertEqual(report["recommendation"], "ineffective")
        self.assertLess(report["downgrade_effectiveness"], 0.7)

    def test_adherence_rate_calculation_correct(self):
        # 6 downgraded, 4 baseline → rate = 0.6
        for i in range(4):
            self.store.record(_make_record(call_id=f"base-{i}", was_downgraded=False, r_cor=0.9))
        for i in range(6):
            self.store.record(_make_record(call_id=f"down-{i}", was_downgraded=True, r_cor=0.88))
        report = self.store.compute_report("pol-1", min_records=5)
        self.assertAlmostEqual(report["adherence_rate"], 0.6, places=2)

    def test_confidence_scales_with_record_count(self):
        # 5 records (below 20 threshold) → confidence < 1.0
        for i in range(5):
            self.store.record(_make_record(call_id=f"c{i}", was_downgraded=True, r_cor=1.0))
        report5 = self.store.compute_report("pol-1", min_records=5)

        store2 = AdherenceStore(Path(self.tmp.name) / "pol2")
        for i in range(20):
            store2.record(_make_record(policy_id="pol-2", call_id=f"c{i}", was_downgraded=True, r_cor=1.0))
        report20 = store2.compute_report("pol-2", min_records=5)

        self.assertLess(report5["confidence"], report20["confidence"])

    def test_avg_r_step_included_in_report(self):
        for i in range(5):
            self.store.record(_make_record(call_id=f"c{i}", r_step=-0.02))
        report = self.store.compute_report("pol-1", min_records=5)
        self.assertIn("avg_r_step", report)
        self.assertAlmostEqual(report["avg_r_step"], -0.02, places=4)

    def test_all_baseline_no_downgraded(self):
        # adherence_rate=0 → monitoring (not effective, not ineffective)
        for i in range(5):
            self.store.record(_make_record(call_id=f"c{i}", was_downgraded=False, r_cor=0.9))
        report = self.store.compute_report("pol-1", min_records=5)
        self.assertEqual(report["adherence_rate"], 0.0)
        self.assertEqual(report["recommendation"], "monitoring")


if __name__ == "__main__":
    unittest.main()

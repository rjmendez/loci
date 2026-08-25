"""Reporting the audit lane's state, so verdicts about it can be read correctly.

run_provenance flags every observed finding without a matching receipt. That is
its documented contract and test_checks.py::test_observed_with_empty_audit_flagged
pins it deliberately. But with NO receipts it flags every observed finding it is
given, and a verdict firing on 100% of inputs carries no information.

Measured: 1 of 140 live investigations has an audit.jsonl, and nothing has
written a receipt since 2026-06-20. This reports that condition rather than
changing any verdict.
"""
import unittest

import server


def _obs(text="the relay dropped at 03:39"):
    return {"id": "f1", "type": "observed", "text": text, "source": "netscan"}


class AuditLaneStatusTest(unittest.TestCase):

    def test_empty_lane_is_reported_as_uninformative(self):
        lane = server._audit_lane_status([_obs()], [])
        self.assertEqual(lane["status"], "empty")
        self.assertEqual(lane["receipts"], 0)
        self.assertIs(lane["verdicts_informative"], False)
        self.assertIn("absence of evidence", lane["detail"])

    def test_the_detail_names_the_count_it_explains(self):
        lane = server._audit_lane_status([_obs(), dict(_obs(), id="f2")], [])
        self.assertEqual(lane["observed_findings"], 2)
        self.assertIn("2 observed", lane["detail"])

    def test_a_populated_lane_reports_present(self):
        lane = server._audit_lane_status([_obs()], [{"tool": "netscan", "text": "x"}])
        self.assertEqual(lane["status"], "present")
        self.assertEqual(lane["receipts"], 1)
        self.assertNotIn("verdicts_informative", lane)

    def test_non_observed_findings_are_not_counted(self):
        lane = server._audit_lane_status(
            [{"id": "a", "type": "inferred", "text": "probably the relay"}], [])
        self.assertEqual(lane["observed_findings"], 0)

    def test_textless_observed_findings_are_not_counted(self):
        """run_provenance skips them, so counting them would overstate the lane."""
        lane = server._audit_lane_status([{"id": "a", "type": "observed", "text": "  "}], [])
        self.assertEqual(lane["observed_findings"], 0)

    def test_record_type_is_accepted_as_well_as_type(self):
        lane = server._audit_lane_status(
            [{"id": "a", "record_type": "observed", "text": "the relay dropped"}], [])
        self.assertEqual(lane["observed_findings"], 1)

    def test_malformed_audit_entries_do_not_count_as_receipts(self):
        lane = server._audit_lane_status([_obs()], ["not a dict", None])
        self.assertEqual(lane["status"], "empty")


class ProvenanceContractUnchangedTest(unittest.TestCase):
    """The per-finding contract must be untouched by the lane reporting."""

    def test_an_observed_finding_with_no_receipt_is_still_flagged(self):
        from memcheck.checks.provenance import run_provenance
        results = run_provenance([_obs()], [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].verdict_type, "unsupported_observed")
        self.assertEqual(results[0].decision, "warn")


if __name__ == "__main__":
    unittest.main()

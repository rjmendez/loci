"""Adversarial verdicts have to reach a reader.

investigation_verify_all writes to a separate finding_verifications.jsonl so the
verdicts never bloat the findings scan. The side effect was that nothing read
them. Census of the tool-audit log: investigation_store 299 calls,
investigation_load 15, memory_self_check 0 — so the verdicts' only other
would-be consumer was itself never invoked. A skeptic refuting a stored finding
at 0.95 confidence landed in a file with no reader.

These verdicts are ADVISORY: investigation_verify_all documents that a verdict is
not a lifecycle state, so surfacing must not imply the finding was resolved.
"""
import json
import unittest
from unittest import mock

import investigation_tools as IT


def _v(fid, verdict, conf=0.9, degraded=False, ts="2026-08-26T00:00:00Z"):
    return {"record_type": "verification", "finding_id": fid, "verdict": verdict,
            "confidence": conf, "degraded": degraded, "ts": ts}


class VerificationSummaryTest(unittest.TestCase):

    def _summary(self, rows):
        with mock.patch.object(IT, "_read_jsonl", return_value=rows), \
             mock.patch.object(IT, "_inv_dir"):
            return IT._verification_summary("inv")

    def test_no_verdicts_returns_none(self):
        """The 140 investigations without verdicts must be unchanged."""
        self.assertIsNone(self._summary([]))

    def test_counts_and_refuted_ids(self):
        s = self._summary([_v("a", "refuted", 0.95), _v("b", "uncertain", 0.4),
                           _v("c", "confirmed", 0.8)])
        self.assertEqual(s["counts"]["refuted"], 1)
        self.assertEqual(s["counts"]["uncertain"], 1)
        self.assertEqual(s["counts"]["confirmed"], 1)
        self.assertEqual([r["finding_id"] for r in s["refuted"]], ["a"])

    def test_refuted_are_ordered_by_confidence(self):
        s = self._summary([_v("low", "refuted", 0.6), _v("high", "refuted", 0.95),
                           _v("mid", "refuted", 0.8)])
        self.assertEqual([r["finding_id"] for r in s["refuted"]], ["high", "mid", "low"])

    def test_a_degraded_verdict_is_never_a_refutation(self):
        """degraded means no model was reached — not a judgement about the finding."""
        s = self._summary([_v("a", "refuted", 0.9, degraded=True)])
        self.assertEqual(s["counts"]["degraded"], 1)
        self.assertEqual(s["counts"]["refuted"], 0)
        self.assertNotIn("refuted", s)

    def test_last_verdict_per_finding_wins(self):
        """Append-only log, same last-write-wins rule as finding_updates."""
        s = self._summary([_v("a", "refuted", 0.9, ts="2026-08-01T00:00:00Z"),
                           _v("a", "confirmed", 0.7, ts="2026-08-26T00:00:00Z")])
        self.assertEqual(s["verified_findings"], 1)
        self.assertEqual(s["counts"]["confirmed"], 1)
        self.assertEqual(s["counts"]["refuted"], 0)

    def test_the_hint_says_advisory_not_resolved(self):
        s = self._summary([_v("a", "refuted", 0.95)])
        self.assertIn("advisory", s["hint"].lower())
        self.assertIn("do NOT change", s["hint"])

    def test_a_malformed_row_does_not_break_the_summary(self):
        s = self._summary([_v("a", "refuted", 0.9), {"no": "finding_id"},
                           _v("b", "refuted", "not-a-number")])
        self.assertEqual(s["counts"]["refuted"], 2)
        self.assertEqual([r["confidence"] for r in s["refuted"]], [0.9, 0.0])

    def test_an_unreadable_log_returns_none_rather_than_raising(self):
        with mock.patch.object(IT, "_read_jsonl", side_effect=OSError("gone")), \
             mock.patch.object(IT, "_inv_dir"):
            self.assertIsNone(IT._verification_summary("inv"))


class VerificationReachesInvestigationLoadTest(unittest.TestCase):
    """End to end: investigation_verify_all -> finding_verifications.jsonl ->
    investigation_load(). Only the model verifier (a dependency) is scripted."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        import server
        self.server = server
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        self.server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def _seed(self):
        s = self.server
        s.investigation_start(investigation_id="verify-e2e", title="verify surfacing")
        ids = {}
        for text in ("The cron job runs hourly.", "The queue is backed by Redis."):
            ids[text] = json.loads(s.investigation_store(
                investigation_id="verify-e2e", finding_type="observed", text=text,
                source="test", confidence="high"))["finding_id"]
        return ids

    def test_refuted_verdict_is_attached_to_investigation_load(self):
        import verify

        ids = self._seed()
        verdicts = {"The cron job runs hourly.": ("refuted", 0.95),
                    "The queue is backed by Redis.": ("confirmed", 0.8)}
        seen = []

        def _fake_verify(text, **_kw):
            seen.append(text)
            verdict, conf = verdicts[text]
            return {"verdict": verdict, "confidence": conf, "degraded": False}

        with mock.patch.object(verify, "verify_finding", _fake_verify):
            out = json.loads(self.server.investigation_verify_all(investigation_id="verify-e2e"))
        self.assertEqual(out["verified"], 2)
        self.assertEqual(sorted(seen), sorted(verdicts))

        loaded = json.loads(self.server.investigation_load(investigation_id="verify-e2e"))
        v = loaded["verifications"]
        self.assertEqual(v["verified_findings"], 2)
        self.assertEqual(v["counts"]["refuted"], 1)
        self.assertEqual(v["counts"]["confirmed"], 1)
        self.assertEqual([r["finding_id"] for r in v["refuted"]],
                         [ids["The cron job runs hourly."]])
        # Advisory: the refuted finding is still open, not resolved.
        by_id = {f["id"]: f for f in loaded["recent_findings"]}
        self.assertEqual(by_id[ids["The cron job runs hourly."]]["resolution"], "open")

    def test_load_without_verdicts_has_no_verifications_block(self):
        """Negative twin on the same investigation shape."""
        self._seed()
        loaded = json.loads(self.server.investigation_load(investigation_id="verify-e2e"))
        self.assertNotIn("verifications", loaded)


if __name__ == "__main__":
    unittest.main()

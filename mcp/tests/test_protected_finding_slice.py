"""investigation_load's finding window was findings[-N:] — recency, silently.

Two failures in one line. A `gap` recorded early in a long investigation is an
open obligation that nobody has closed, and dropping it on age means the longer
it stays open the less likely anyone sees it again. And the payload said nothing
about having selected at all: a caller got `total_findings` and a list, with
nothing stating the list was not the whole of it, which is the same shape as a
compressor that trims its summary and does not say so.

_select_findings lets protected types claim slots from the oldest routine ones,
capped so recency cannot be squeezed out entirely, and reports the remainder.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from investigation_tools import _select_findings  # noqa: E402


def _f(i, rt="observed"):
    return {"id": f"f{i:03d}", "record_type": rt, "text": f"finding {i}"}


class SelectFindingsTest(unittest.TestCase):
    def test_short_investigation_returns_everything_and_reports_no_omission(self):
        findings = [_f(i) for i in range(5)]
        selected, omitted = _select_findings(findings, 20)
        self.assertEqual(len(selected), 5)
        self.assertEqual(omitted, {}, "nothing was dropped, so say nothing")

    def test_plain_window_is_still_the_most_recent(self):
        findings = [_f(i) for i in range(50)]
        selected, omitted = _select_findings(findings, 10)
        self.assertEqual([f["id"] for f in selected], [f"f{i:03d}" for i in range(40, 50)])
        self.assertEqual(omitted["count"], 40)
        self.assertEqual(omitted["by_record_type"], {"observed": 40})
        self.assertEqual(omitted["promoted_past_the_window"], 0)

    def test_an_old_gap_is_pulled_into_the_window(self):
        """The case the change exists for: a gap at index 0 of a 50-finding
        investigation must not vanish from every full-fidelity load."""
        findings = [_f(0, rt="gap")] + [_f(i) for i in range(1, 50)]
        selected, omitted = _select_findings(findings, 10)
        self.assertIn("f000", [f["id"] for f in selected])
        self.assertEqual(omitted["promoted_past_the_window"], 1)

    def test_promotion_evicts_the_oldest_routine_finding_not_a_recent_one(self):
        findings = [_f(0, rt="gap")] + [_f(i) for i in range(1, 50)]
        selected, _ = _select_findings(findings, 10)
        ids = [f["id"] for f in selected]
        self.assertNotIn("f040", ids, "the oldest in-window routine finding is the victim")
        self.assertIn("f049", ids, "the newest finding must always survive")

    def test_selection_stays_in_chronological_order(self):
        findings = [_f(0, rt="gap")] + [_f(i) for i in range(1, 30)]
        selected, _ = _select_findings(findings, 10)
        self.assertEqual([f["id"] for f in selected], sorted(f["id"] for f in selected))

    def test_assumptions_are_protected_too(self):
        findings = [_f(0, rt="assumed")] + [_f(i) for i in range(1, 40)]
        selected, _ = _select_findings(findings, 10)
        self.assertIn("f000", [f["id"] for f in selected])

    def test_protected_records_take_at_most_half_the_window(self):
        """An investigation that is mostly gaps must not squeeze recency out: a
        caller asking for 10 still gets 5 genuinely recent findings."""
        findings = [_f(i, rt="gap") for i in range(40)] + [_f(i) for i in range(40, 50)]
        selected, omitted = _select_findings(findings, 10)
        self.assertEqual(omitted["promoted_past_the_window"], 5)
        recent = [f for f in selected if int(f["id"][1:]) >= 40]
        self.assertEqual(len(recent), 5)

    def test_newest_protected_wins_a_contested_slot(self):
        # limit 10 -> at most 5 promotions, but six old gaps compete for them.
        findings = [_f(i, rt="gap") for i in range(6)] + [_f(i) for i in range(6, 40)]
        selected, omitted = _select_findings(findings, 10)
        ids = [f["id"] for f in selected]
        self.assertEqual(omitted["promoted_past_the_window"], 5)
        self.assertNotIn("f000", ids, "the oldest gap is the one that loses")
        for i in range(1, 6):
            self.assertIn(f"f{i:03d}", ids)

    def test_small_windows_do_not_promote_at_all(self):
        """grounding.py loads a case with last_n_findings=6 and takes [:3]. Since
        selection is chronological and promoted records are older than the window,
        promoting into a 6-window would hand that lane three gaps and no recent
        findings. Measured on a real 51-finding case before this guard existed."""
        findings = [_f(i, rt="gap") for i in range(4)] + [_f(i) for i in range(4, 51)]
        selected, omitted = _select_findings(findings, 6)
        self.assertEqual(omitted["promoted_past_the_window"], 0)
        self.assertEqual([f["id"] for f in selected],
                         [f"f{i:03d}" for i in range(45, 51)])
        self.assertNotIn("gap", [f["record_type"] for f in selected[:3]])

    def test_omission_report_breaks_down_by_record_type(self):
        findings = ([_f(i) for i in range(20)]
                    + [_f(i, rt="inferred") for i in range(20, 25)]
                    + [_f(i) for i in range(25, 30)])
        _, omitted = _select_findings(findings, 5)
        self.assertEqual(omitted["count"], 25)
        self.assertEqual(omitted["by_record_type"], {"observed": 20, "inferred": 5})
        self.assertIn("investigation_search", omitted["note"])

    def test_untyped_records_are_counted_not_skipped(self):
        findings = [{"id": f"f{i}", "text": "t"} for i in range(30)]
        _, omitted = _select_findings(findings, 5)
        self.assertEqual(omitted["by_record_type"], {"(untyped)": 25})

    def test_legacy_type_key_is_honoured_for_protection(self):
        findings = [{"id": "f000", "type": "gap", "text": "t"}] + [_f(i) for i in range(1, 30)]
        selected, _ = _select_findings(findings, 10)
        self.assertIn("f000", [f["id"] for f in selected])

    def test_zero_or_negative_limit_returns_everything(self):
        findings = [_f(i) for i in range(5)]
        for limit in (0, -1):
            selected, omitted = _select_findings(findings, limit)
            self.assertEqual(len(selected), 5)
            self.assertEqual(omitted, {})


if __name__ == "__main__":
    unittest.main()

"""investigation_load's summary fidelities are entirely model-authored.

summary_l1/summary_l2 come from investigation_reflect, so at fidelity="summary"
every word a caller reads about a set of findings is generated — nothing states,
from the data, that all twenty were high-confidence or that one of them is a gap.
_field_invariants is the deterministic floor under that: counts the model cannot
invent, and which survive the summary being absent, stale, or wrong.

The shapes are borrowed from how a log compressor describes an elided run
(constant / enumeration / identifier), with one deliberate difference: nothing
here is budget-trimmed, so there is never a shed to disclose. A summary that
silently drops facts is worse than one that never had them, because a reader
takes a field's absence as evidence the field did not hold.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from investigation_tools import (  # noqa: E402
    _INVARIANT_MAX_VALUES,
    _field_invariants,
)


def _f(**kw):
    base = {"record_type": "observed", "text": "t", "confidence": "high"}
    base.update(kw)
    return base


class FieldInvariantsTest(unittest.TestCase):
    def test_empty_set_reports_nothing_rather_than_guessing(self):
        got = _field_invariants([])
        self.assertEqual(got["n"], 0)
        self.assertEqual(got["constant"], {})
        self.assertEqual(got["varies"], {})

    def test_a_field_the_whole_set_agrees_on_is_a_constant(self):
        got = _field_invariants([_f(), _f(), _f()])
        self.assertEqual(got["n"], 3)
        self.assertEqual(got["constant"]["confidence"], "high")
        self.assertEqual(got["constant"]["record_type"], "observed")
        self.assertNotIn("confidence", got["varies"])

    def test_a_handful_of_values_is_an_enumeration_with_counts(self):
        findings = [_f(record_type="observed")] * 4 + [_f(record_type="gap")]
        got = _field_invariants(findings)
        self.assertEqual(got["varies"]["record_type"], {"observed": 4, "gap": 1})
        self.assertNotIn("record_type", got["constant"])

    def test_enumeration_is_ordered_by_descending_count(self):
        findings = [_f(confidence="low")] + [_f(confidence="high")] * 3
        self.assertEqual(list(_field_invariants(findings)["varies"]["confidence"]),
                         ["high", "low"])

    def test_a_field_with_too_many_values_reports_only_a_count(self):
        """Listing an identifier restates the data instead of describing it — and
        reporting the count is itself the disclosure that values were not listed."""
        findings = [_f(source=f"tool_{i}") for i in range(_INVARIANT_MAX_VALUES + 3)]
        got = _field_invariants(findings)
        self.assertEqual(got["distinct_only"]["source"], _INVARIANT_MAX_VALUES + 3)
        self.assertNotIn("source", got["varies"])
        self.assertNotIn("source", got["constant"])

    def test_a_field_only_some_findings_carry_is_not_reported_as_constant(self):
        """The dangerous read. Two of three findings resolved 'fixed' must never
        render as `all resolution=fixed` — absence is counted, not ignored."""
        findings = [_f(resolution="fixed"), _f(resolution="fixed"), _f()]
        got = _field_invariants(findings)
        self.assertNotIn("resolution", got["constant"])
        self.assertEqual(got["varies"]["resolution"], {"fixed": 2, "(absent)": 1})

    def test_a_field_no_finding_carries_is_silent(self):
        got = _field_invariants([_f(), _f()])
        self.assertNotIn("resolution", got["constant"])
        self.assertNotIn("resolution", got["varies"])
        self.assertNotIn("resolution", got["distinct_only"])

    def test_legacy_type_key_counts_as_record_type(self):
        """Older records carry "type" where newer ones carry "record_type"; a set
        mixing them is one class, not two."""
        findings = [{"record_type": "observed", "text": "a"}, {"type": "observed", "text": "b"}]
        self.assertEqual(_field_invariants(findings)["constant"]["record_type"], "observed")

    def test_tags_report_the_shared_set_and_the_universe(self):
        findings = [_f(tags=["dama", "clock"]), _f(tags=["dama", "audio"])]
        got = _field_invariants(findings)
        self.assertEqual(got["tags_on_every_finding"], ["dama"])
        self.assertEqual(got["tags_distinct"], 3)

    def test_one_untagged_finding_empties_the_shared_tag_set(self):
        findings = [_f(tags=["dama"]), _f(tags=[])]
        got = _field_invariants(findings)
        self.assertEqual(got["tags_on_every_finding"], [])
        self.assertEqual(got["tags_distinct"], 1)

    def test_output_is_bounded_so_there_is_never_a_shed_to_disclose(self):
        """The property that makes this safe to trust: a big, wildly varied set
        still renders a small fixed-shape object, so nothing is ever dropped to
        fit. Any future budget added here would need a disclosure alongside it."""
        findings = [_f(record_type=f"t{i}", confidence=f"c{i}", source=f"s{i}",
                       tags=[f"tag{i}"]) for i in range(500)]
        got = _field_invariants(findings)
        self.assertEqual(got["n"], 500)
        for bucket in ("constant", "varies", "distinct_only"):
            for field, value in got[bucket].items():
                if bucket == "varies":
                    self.assertLessEqual(len(value), _INVARIANT_MAX_VALUES, field)
        self.assertEqual(got["distinct_only"]["record_type"], 500)


if __name__ == "__main__":
    unittest.main()

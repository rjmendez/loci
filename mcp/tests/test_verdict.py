"""Tests for memcheck/verdict.py — pure stdlib, no mocking needed."""

import sys
import os
import unittest

# Run from mcp/ directory; this path resolves whether pytest is launched from
# mcp/ or from the repo root with --rootdir.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memcheck.verdict import Verdict, make_signature, new_verdict, redact_excerpt


class TestMakeSignature(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(
            make_signature("claim", "hello world"),
            make_signature("claim", "hello world"),
        )

    def test_whitespace_collapsed(self):
        self.assertEqual(
            make_signature("claim", "hello  world"),
            make_signature("claim", "hello world"),
        )

    def test_case_insensitive(self):
        self.assertEqual(
            make_signature("claim", "Hello World"),
            make_signature("claim", "hello world"),
        )

    def test_kind_differentiates(self):
        self.assertNotEqual(
            make_signature("claim", "text"),
            make_signature("code", "text"),
        )

    def test_empty_inputs(self):
        sig = make_signature("", "")
        self.assertIsInstance(sig, str)
        self.assertEqual(len(sig), 64)  # SHA-256 hex

    def test_unicode(self):
        sig = make_signature("claim", "café résumé naïve")
        self.assertIsInstance(sig, str)
        self.assertEqual(len(sig), 64)

    def test_leading_trailing_whitespace(self):
        self.assertEqual(
            make_signature("x", "  hello  "),
            make_signature("x", "hello"),
        )


class TestRedactExcerpt(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(redact_excerpt(None), "")

    def test_empty_string_unchanged(self):
        self.assertEqual(redact_excerpt(""), "")

    def test_short_string_unchanged(self):
        self.assertEqual(redact_excerpt("hello"), "hello")

    def test_at_limit_unchanged(self):
        text = "a" * 512
        self.assertEqual(redact_excerpt(text), text)

    def test_over_limit_truncated(self):
        text = "a" * 600
        result = redact_excerpt(text)
        self.assertEqual(len(result), 512)

    def test_custom_max_chars(self):
        self.assertEqual(redact_excerpt("hello world", max_chars=5), "hello")

    def test_custom_max_at_boundary(self):
        self.assertEqual(redact_excerpt("hello", max_chars=5), "hello")

    def test_unicode_boundary_safe(self):
        # Python slicing is codepoint-based; multibyte chars should not cause issues
        text = "é" * 600
        result = redact_excerpt(text)
        self.assertEqual(len(result), 512)


class TestNewVerdict(unittest.TestCase):
    def _make(self, **overrides):
        defaults = dict(
            subject_kind="claim",
            subject_signature="abc123",
            subject_excerpt="some text",
            verdict_type="contradiction",
            decision="flag",
            confidence=0.9,
            rationale="contradicts prior",
            source="rule",
        )
        defaults.update(overrides)
        return new_verdict(**defaults)

    def test_basic_fields(self):
        v = self._make()
        self.assertEqual(v.decision, "flag")
        self.assertEqual(v.occurrences, 1)
        self.assertEqual(v.source, "rule")

    def test_id_is_set(self):
        v = self._make()
        self.assertIsNotNone(v.id)
        self.assertGreater(len(v.id), 0)

    def test_ids_unique(self):
        self.assertNotEqual(self._make().id, self._make().id)

    def test_refs_defaults_to_empty_list(self):
        v = self._make()
        self.assertEqual(v.refs, [])

    def test_refs_none_becomes_empty_list(self):
        v = self._make(refs=None)
        self.assertEqual(v.refs, [])

    def test_refs_preserved(self):
        v = self._make(refs=["ref1", "ref2"])
        self.assertEqual(v.refs, ["ref1", "ref2"])

    def test_long_excerpt_truncated(self):
        v = self._make(subject_excerpt="x" * 600)
        self.assertLessEqual(len(v.subject_excerpt), 512)

    def test_timestamps_set(self):
        v = self._make()
        self.assertIn("T", v.first_seen)
        self.assertIn("T", v.last_seen)


class TestVerdictSerialization(unittest.TestCase):
    def _make_verdict(self) -> Verdict:
        return new_verdict(
            subject_kind="claim",
            subject_signature="sig123",
            subject_excerpt="test excerpt",
            verdict_type="unsupported_observed",
            decision="warn",
            confidence=0.75,
            rationale="no audit receipt",
            source="rule",
            refs=["finding:0", "finding:1"],
        )

    def test_to_payload_roundtrip(self):
        v = self._make_verdict()
        payload = v.to_payload()
        v2 = Verdict.from_payload(payload)
        self.assertEqual(v.id, v2.id)
        self.assertEqual(v.decision, v2.decision)
        self.assertEqual(v.refs, v2.refs)
        self.assertEqual(v.confidence, v2.confidence)

    def test_to_payload_json_safe(self):
        import json
        v = self._make_verdict()
        payload = v.to_payload()
        dumped = json.dumps(payload)  # should not raise
        self.assertIsInstance(dumped, str)

    def test_from_payload_ignores_extra_keys(self):
        payload = self._make_verdict().to_payload()
        payload["unknown_future_field"] = "some value"
        v = Verdict.from_payload(payload)  # should not raise
        self.assertEqual(v.decision, "warn")

    def test_from_payload_missing_refs_defaults_to_empty(self):
        payload = self._make_verdict().to_payload()
        del payload["refs"]
        v = Verdict.from_payload(payload)
        self.assertEqual(v.refs, [])

    def test_from_payload_refs_none_becomes_empty(self):
        payload = self._make_verdict().to_payload()
        payload["refs"] = None
        v = Verdict.from_payload(payload)
        self.assertEqual(v.refs, [])

    def test_from_payload_missing_occurrences_defaults_to_one(self):
        payload = self._make_verdict().to_payload()
        del payload["occurrences"]
        v = Verdict.from_payload(payload)
        self.assertEqual(v.occurrences, 1)

    def test_from_payload_confidence_as_string(self):
        payload = self._make_verdict().to_payload()
        payload["confidence"] = "0.85"
        v = Verdict.from_payload(payload)
        self.assertAlmostEqual(v.confidence, 0.85)

    def test_from_payload_missing_optional_timestamps(self):
        payload = self._make_verdict().to_payload()
        del payload["first_seen"]
        del payload["last_seen"]
        v = Verdict.from_payload(payload)
        self.assertEqual(v.first_seen, "")
        self.assertEqual(v.last_seen, "")


if __name__ == "__main__":
    unittest.main()

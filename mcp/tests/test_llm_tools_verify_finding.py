"""Tests for llm_tools.verify_finding threading stored provenance into
verify.verify_finding's provenance firewall.

The public verify_finding MCP wrapper accepts finding_id but previously
called verify.verify_finding without candidate_provenance_tier/evidence_rows,
so a model_asserted finding with no independent evidence could still reach
the model verifier and be confirmed on its own say-so. This module checks
_finding_provenance_context() and the wrapper's threading of its result.

Run: pytest mcp/tests/test_llm_tools_verify_finding.py -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import llm_tools  # noqa: E402
import server  # noqa: E402


def _json(result: str) -> dict:
    return json.loads(result)


class FindingProvenanceContextTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig_dir
        self._tmp.cleanup()

    def test_no_investigation_or_finding_id_returns_none_none(self):
        self.assertEqual(llm_tools._finding_provenance_context(None, None), (None, None))
        self.assertEqual(llm_tools._finding_provenance_context("inv", None), (None, None))
        self.assertEqual(llm_tools._finding_provenance_context(None, "fid"), (None, None))

    def test_unknown_finding_id_fails_open_to_none_none(self):
        inv_id = "prov-ctx-unknown"
        server.investigation_start(investigation_id=inv_id, title="t")
        tier, rows = llm_tools._finding_provenance_context(inv_id, "does-not-exist")
        self.assertIsNone(tier)
        self.assertIsNone(rows)

    def test_finds_own_tier_and_only_linked_evidence(self):
        # Evidence is what supports THIS claim, not every other finding: an
        # unrelated tool_verified row must not be handed to the firewall.
        inv_id = "prov-ctx-1"
        server.investigation_start(investigation_id=inv_id, title="t")
        related = _json(server.investigation_store(
            inv_id, "observed", "nginx 1.24 serves port 443 on host alpha", "unit-test",
            confidence="high", metadata={"evidence_provenance_tier": "tool_verified"},
        ))["finding_id"]
        unrelated = _json(server.investigation_store(
            inv_id, "observed", "disk usage on host bravo is 41 percent", "unit-test",
            confidence="high", metadata={"evidence_provenance_tier": "tool_verified"},
        ))["finding_id"]
        target = _json(server.investigation_store(
            inv_id, "inferred", "host alpha serves nginx 1.24 on port 443", "unit-test",
            confidence="high", metadata={"evidence_provenance_tier": "model_asserted"},
        ))["finding_id"]

        tier, rows = llm_tools._finding_provenance_context(inv_id, target)
        self.assertEqual(tier, "model_asserted")
        ids = {r["evidence_id"] for r in rows}
        self.assertIn(related, ids)
        self.assertNotIn(unrelated, ids)
        self.assertNotIn(target, ids)


class VerifyFindingWrapperThreadingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig_dir
        self._tmp.cleanup()

    def test_model_only_finding_is_gated_uncertain_without_calling_model(self):
        inv_id = "prov-wrap-1"
        server.investigation_start(investigation_id=inv_id, title="t")
        fid = _json(server.investigation_store(
            inv_id, "inferred", "The reflection loop is fixed by model agreement.",
            "unit-test", confidence="high",
            metadata={"evidence_provenance_tier": "model_asserted"},
        ))["finding_id"]

        def _should_not_call(*args, **kwargs):
            raise AssertionError("model verifier must not run on a circular model-only finding")

        with mock.patch("verify._lazy_generate", side_effect=_should_not_call):
            result = _json(llm_tools.verify_finding(
                "The reflection loop is fixed by model agreement.",
                investigation_id=inv_id,
                finding_id=fid,
            ))
        self.assertEqual(result["verdict"], "uncertain")
        self.assertEqual(result["confidence"], 0.0)
        self.assertFalse(result["provenance_firewall"]["allowed"])

    def test_model_only_finding_with_tool_verified_support_reaches_model(self):
        inv_id = "prov-wrap-2"
        server.investigation_start(investigation_id=inv_id, title="t")
        # The support must be about the claim; an unrelated tool row is not support.
        _json(server.investigation_store(
            inv_id, "observed", "Reflection loop replay run: loop fixed, model agreement 5/5.",
            "unit-test", confidence="high",
            metadata={"evidence_provenance_tier": "tool_verified"},
        ))
        fid = _json(server.investigation_store(
            inv_id, "inferred", "The reflection loop is fixed by model agreement.",
            "unit-test", confidence="high",
            metadata={"evidence_provenance_tier": "model_asserted"},
        ))["finding_id"]

        def _confirmed_gen(prompt, *, fmt=None, max_tokens=256):
            return {"ok": True, "text": json.dumps(
                {"verdict": "confirmed", "refutation": "", "confidence": 0.7})}

        with mock.patch("verify._lazy_generate", side_effect=_confirmed_gen):
            result = _json(llm_tools.verify_finding(
                "The reflection loop is fixed by model agreement.",
                investigation_id=inv_id,
                finding_id=fid,
            ))
        self.assertEqual(result["verdict"], "confirmed")

    def test_no_finding_id_preserves_legacy_ungated_behavior(self):
        # Ad hoc claims with no stored finding must not be gated: there is no
        # provenance tier to gate on, so the old behavior (call the model) holds.
        def _confirmed_gen(prompt, *, fmt=None, max_tokens=256):
            return {"ok": True, "text": json.dumps(
                {"verdict": "confirmed", "refutation": "", "confidence": 0.6})}

        with mock.patch("verify._lazy_generate", side_effect=_confirmed_gen):
            result = _json(llm_tools.verify_finding("an ad hoc claim with no finding_id"))
        self.assertEqual(result["verdict"], "confirmed")


if __name__ == "__main__":
    unittest.main()

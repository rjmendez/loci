"""investigation_store provenance gate: a model claim cannot be stored as verified fact.

The tier is only raised above model_asserted with a citation the server can check:
  tool_verified          metadata.receipt_ids naming receipts audit_log actually recorded;
  deterministic_derived  derived_from naming findings already stored in the investigation.
Everything untagged is model_asserted. Temp MEMORY_DIR and findings only; the live store is never touched.

Mutation runs (each reverted after):
  - untagged default back to tool_verified       -> test_untagged_write_is_model_asserted FAILED
  - removed the receipt existence check          -> test_made_up_receipt_is_refused FAILED
  - removed the receipt requirement              -> test_verified_claim_without_receipt_is_refused FAILED

Run: pytest mcp/tests/test_store_provenance_gate.py -v
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

import server  # noqa: E402

GLOBAL_RECEIPT = "rcpt-20261031-aaaaaaaaaaaa"
INV_RECEIPT = "rcpt-20261031-bbbbbbbbbbbb"


class StoreProvenanceGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.mem = base / "mem"
        self.root = base / "inv"
        self.root.mkdir()
        (base / "audit").mkdir()
        (base / "audit" / "20261031.jsonl").write_text(
            json.dumps({"receipt_id": GLOBAL_RECEIPT, "tool": "redis-cli"}) + "\n", encoding="utf-8")
        (self.root / "audit.jsonl").write_text(
            json.dumps({"receipt_id": INV_RECEIPT, "tool": "nmap"}) + "\n", encoding="utf-8")
        (self.root / "findings.jsonl").write_text(json.dumps({"id": "f-real-1", "text": "stored"}) + "\n",
                                                  encoding="utf-8")
        self.mem_patch = mock.patch.object(server, "MEMORY_DIR", self.mem)
        self.mem_patch.start()
        self.addCleanup(self.mem_patch.stop)
        inv_patch = mock.patch.object(server, "_inv_dir", lambda _iid: self.root)
        inv_patch.start()
        self.addCleanup(inv_patch.stop)
        self.addCleanup(self._tmp.cleanup)

    def _resolve(self, tier, derived_from=None, metadata=None):
        return server._resolve_write_provenance("inv", tier, derived_from, metadata)

    def test_untagged_write_is_model_asserted(self) -> None:
        self.assertEqual(self._resolve(None), ("model_asserted", None))

    def test_blank_tier_is_treated_as_untagged(self) -> None:
        self.assertEqual(self._resolve("   "), ("model_asserted", None))

    def test_explicit_model_asserted_passes(self) -> None:
        self.assertEqual(self._resolve("model_asserted"), ("model_asserted", None))

    def test_verified_claim_without_receipt_is_refused(self) -> None:
        tier, err = self._resolve("tool_verified")
        self.assertIsNone(tier)
        self.assertIn("needs metadata", json.loads(err)["error"])

    def test_made_up_receipt_is_refused(self) -> None:
        tier, err = self._resolve("tool_verified", metadata={"receipt_ids": ["rcpt-20261031-fabricated00"]})
        self.assertIsNone(tier)
        self.assertIn("no recorded audit receipt", json.loads(err)["error"])

    def test_receipt_recorded_in_the_global_log_is_accepted(self) -> None:
        # positive twin: a receipt audit_log wrote to the daily global log is real evidence
        self.assertEqual(self._resolve("tool_verified", metadata={"receipt_ids": [GLOBAL_RECEIPT]}),
                         ("tool_verified", None))

    def test_receipt_recorded_in_the_investigation_log_is_accepted(self) -> None:
        self.assertEqual(self._resolve("tool_verified", metadata={"receipt_ids": [INV_RECEIPT]}),
                         ("tool_verified", None))

    def test_receipt_ids_as_json_string_are_parsed(self) -> None:
        self.assertEqual(self._resolve("tool_verified", metadata=json.dumps({"receipt_ids": [GLOBAL_RECEIPT]})),
                         ("tool_verified", None))

    def test_derived_claim_without_parents_is_refused(self) -> None:
        tier, err = self._resolve("deterministic_derived")
        self.assertIsNone(tier)
        self.assertIn("needs derived_from", json.loads(err)["error"])

    def test_derived_claim_citing_an_invented_finding_is_refused(self) -> None:
        tier, err = self._resolve("deterministic_derived", ["f-invented-999"])
        self.assertIsNone(tier)
        self.assertIn("names no stored finding", json.loads(err)["error"])

    def test_derived_claim_citing_a_stored_finding_passes(self) -> None:
        self.assertEqual(self._resolve("deterministic_derived", ["f-real-1"]), ("deterministic_derived", None))

    def test_unknown_tier_is_refused(self) -> None:
        tier, err = self._resolve("definitely_true")
        self.assertIsNone(tier)
        self.assertIn("evidence_provenance_tier must be one of", json.loads(err)["error"])


if __name__ == "__main__":
    unittest.main()

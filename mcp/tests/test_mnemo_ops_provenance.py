"""Provenance threading through the Mnemosyne fan-out/read-back path.

_store_index() (server.py) sends finding metadata to Mnemosyne via
_mnemo_remember(); _mnemo_recall() (mnemo_ops.py) reads it back for
investigation_search. Both must carry evidence_provenance_tier so a
model_asserted finding doesn't silently read back as the legacy
tool_verified default with no way to tell the two apart.

Run: pytest mcp/tests/test_mnemo_ops_provenance.py -v
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import mnemo_ops  # noqa: E402
import server  # noqa: E402


def _json(result: str) -> dict:
    import json
    return json.loads(result)


class StoreIndexMnemoMetadataTest(unittest.TestCase):
    """server._store_index must pass the finding's provenance tier into
    Mnemosyne metadata, not just record_type/source/confidence."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig_dir
        self._tmp.cleanup()

    def test_model_asserted_finding_forwards_provenance_to_mnemo_metadata(self):
        inv_id = "mnemo-prov-1"
        server.investigation_start(investigation_id=inv_id, title="mnemo provenance test")

        captured = {}

        def _fake_remember(content, *, importance=0.6, metadata=None):
            captured["metadata"] = metadata
            return True

        with mock.patch.object(server, "_mnemo_remember", side_effect=_fake_remember):
            res = _json(server.investigation_store(
                investigation_id=inv_id,
                finding_type="inferred",
                text="The reflection loop is fixed by model agreement.",
                source="test-model",
                confidence="high",
                metadata={"evidence_provenance_tier": "model_asserted"},
            ))
        self.assertTrue(res.get("stored"), res)
        self.assertEqual(captured["metadata"].get("evidence_provenance_tier"), "model_asserted")
        self.assertFalse(captured["metadata"].get("provenance_defaulted"))

    def test_legacy_untagged_finding_forwards_defaulted_provenance(self):
        inv_id = "mnemo-prov-2"
        server.investigation_start(investigation_id=inv_id, title="mnemo legacy provenance test")

        captured = {}

        def _fake_remember(content, *, importance=0.6, metadata=None):
            captured["metadata"] = metadata
            return True

        with mock.patch.object(server, "_mnemo_remember", side_effect=_fake_remember):
            res = _json(server.investigation_store(
                investigation_id=inv_id,
                finding_type="observed",
                text="Legacy finding with no provenance tag.",
                source="test-tool",
                confidence="high",
            ))
        self.assertTrue(res.get("stored"), res)
        self.assertEqual(captured["metadata"].get("evidence_provenance_tier"), "tool_verified")
        self.assertTrue(captured["metadata"].get("provenance_defaulted"))


class MnemoRecallProvenanceTest(unittest.TestCase):
    """mnemo_ops._mnemo_recall must surface the stored provenance tier instead
    of dropping it (which would let it silently normalize to tool_verified with
    no way to tell a real tool_verified row from an untagged legacy one)."""

    def test_recall_surfaces_model_asserted_tier_explicitly(self):
        raw = [{
            "content": "The reflection loop is fixed by model agreement.",
            "metadata": {
                "investigation_id": "inv-x",
                "record_type": "inferred",
                "source": "test-model",
                "evidence_provenance_tier": "model_asserted",
                "provenance_defaulted": False,
            },
            "score": 0.9,
        }]
        with mock.patch.object(mnemo_ops, "_get_mnemo_funcs", return_value=(None, lambda **k: raw)):
            rows = mnemo_ops._mnemo_recall("query", investigation_id="inv-x")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["evidence_provenance_tier"], "model_asserted")
        self.assertFalse(rows[0]["provenance_defaulted"])

    def test_recall_flags_untagged_legacy_row_as_defaulted(self):
        raw = [{
            "content": "Legacy finding with no provenance tag.",
            "metadata": {
                "investigation_id": "inv-x",
                "record_type": "observed",
                "source": "test-tool",
            },
            "score": 0.9,
        }]
        with mock.patch.object(mnemo_ops, "_get_mnemo_funcs", return_value=(None, lambda **k: raw)):
            rows = mnemo_ops._mnemo_recall("query", investigation_id="inv-x")
        self.assertEqual(len(rows), 1)
        # Fails open to the legacy default...
        self.assertEqual(rows[0]["evidence_provenance_tier"], "tool_verified")
        # ...but the defaulted flag says this was never actually asserted as
        # tool_verified, so a caller can tell the two apart.
        self.assertTrue(rows[0]["provenance_defaulted"])


if __name__ == "__main__":
    unittest.main()

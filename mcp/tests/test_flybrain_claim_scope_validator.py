"""Validation/preservation tests for flybrain claim_scope metadata."""

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


def _json(result: str) -> dict:
    return json.loads(result)


def _valid_claim_scope() -> dict:
    return {
        "dataset": "fw",
        "dataset_version": "flywire783",
        "sex": "female",
        "life_stage": "adult",
        "annotation_completeness": 0.95,
        "circuit_class": "mushroom_body",
        "experience_window": "naive",
    }


class FlybrainClaimScopeValidatorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_non_flybrain_metadata_unchanged(self):
        inv_id = "fb-scope-non-flybrain"
        _json(server.investigation_start(investigation_id=inv_id, title="non flybrain metadata"))

        res = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="non flybrain finding",
            source="unit-test",
            metadata={"kind": "other"},
        ))
        self.assertTrue(res.get("stored"), res)

        loaded = _json(server.investigation_load(inv_id))
        finding = loaded["recent_findings"][0]
        self.assertEqual(finding.get("metadata", {}).get("kind"), "other")

    def test_rejects_missing_required_claim_scope_keys(self):
        inv_id = "fb-scope-missing-keys"
        _json(server.investigation_start(investigation_id=inv_id, title="missing claim scope keys"))

        claim_scope = _valid_claim_scope()
        claim_scope.pop("dataset_version")

        res = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="flybrain claim with incomplete scope",
            source="virtual-fly-brain-query_connectivity",
            metadata={
                "flybrain_provenance": {"tool_name": "query_connectivity"},
                "claim_scope": claim_scope,
            },
        ))
        self.assertIn("error", res)
        self.assertIn("dataset_version", res["error"])

    def test_rejects_flybrain_provenance_without_claim_scope(self):
        inv_id = "fb-scope-missing-object"
        _json(server.investigation_start(investigation_id=inv_id, title="missing claim_scope object"))

        res = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="flybrain claim missing claim_scope",
            source="virtual-fly-brain-query_connectivity",
            metadata={"flybrain_provenance": {"tool_name": "query_connectivity"}},
        ))
        self.assertIn("error", res)
        self.assertIn("claim_scope", res["error"])

    def test_valid_scope_is_preserved_and_forwarded_to_mnemo(self):
        inv_id = "fb-scope-valid"
        _json(server.investigation_start(investigation_id=inv_id, title="valid claim_scope"))
        captured = {}

        def _fake_remember(content, *, importance=0.6, metadata=None):
            captured["metadata"] = metadata or {}
            return True

        with mock.patch.object(server, "_mnemo_remember", side_effect=_fake_remember):
            res = _json(server.investigation_store(
                investigation_id=inv_id,
                finding_type="observed",
                text="flybrain claim with complete scope tuple",
                source="virtual-fly-brain-query_connectivity",
                metadata={
                    "flybrain_provenance": {"tool_name": "query_connectivity"},
                    "claim_scope": _valid_claim_scope(),
                },
            ))
        self.assertTrue(res.get("stored"), res)
        self.assertEqual(captured["metadata"].get("claim_scope"), _valid_claim_scope())

        loaded = _json(server.investigation_load(inv_id))
        finding = loaded["recent_findings"][0]
        self.assertEqual(finding.get("metadata", {}).get("claim_scope"), _valid_claim_scope())
        self.assertEqual(
            finding.get("metadata", {}).get("flybrain_provenance", {}).get("claim_scope"),
            _valid_claim_scope(),
        )

    def test_nested_claim_scope_json_string_is_accepted_and_normalized(self):
        inv_id = "fb-scope-nested-json"
        _json(server.investigation_start(investigation_id=inv_id, title="nested claim_scope json"))

        metadata = json.dumps({
            "flybrain_provenance": {
                "tool_name": "query_connectivity",
                "claim_scope": _valid_claim_scope(),
            }
        })
        res = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="flybrain claim from JSON metadata",
            source="virtual-fly-brain-query_connectivity",
            metadata=metadata,
        ))
        self.assertTrue(res.get("stored"), res)

        loaded = _json(server.investigation_load(inv_id))
        finding = loaded["recent_findings"][0]
        self.assertEqual(finding.get("metadata", {}).get("claim_scope"), _valid_claim_scope())

    def test_rejects_invalid_scope_value_shapes_for_hardened_fields(self):
        inv_id = "fb-scope-invalid-shapes"
        _json(server.investigation_start(investigation_id=inv_id, title="invalid claim_scope values"))

        invalid_cases = (
            ("dataset_version", "  "),
            ("annotation_completeness", "partial"),
            ("annotation_completeness", 1.2),
            ("life_stage", "adult!"),
            ("experience_window", "after-training"),
        )
        for key, value in invalid_cases:
            claim_scope = _valid_claim_scope()
            claim_scope[key] = value
            res = _json(server.investigation_store(
                investigation_id=inv_id,
                finding_type="observed",
                text=f"flybrain invalid {key}",
                source="virtual-fly-brain-query_connectivity",
                metadata={
                    "flybrain_provenance": {"tool_name": "query_connectivity"},
                    "claim_scope": claim_scope,
                },
            ))
            self.assertIn("error", res)
            self.assertIn(key, res["error"])

    def test_normalizes_life_stage_and_percentage_annotation(self):
        inv_id = "fb-scope-normalized"
        _json(server.investigation_start(investigation_id=inv_id, title="normalized claim_scope values"))

        claim_scope = _valid_claim_scope()
        claim_scope["life_stage"] = "Larva"
        claim_scope["annotation_completeness"] = "95%"
        claim_scope["experience_window"] = "sleep deprived"

        res = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="flybrain claim with normalizable scope values",
            source="virtual-fly-brain-query_connectivity",
            metadata={
                "flybrain_provenance": {"tool_name": "query_connectivity"},
                "claim_scope": claim_scope,
            },
        ))
        self.assertTrue(res.get("stored"), res)

        loaded = _json(server.investigation_load(inv_id))
        stored_scope = loaded["recent_findings"][0]["metadata"]["claim_scope"]
        self.assertEqual(stored_scope["life_stage"], "larval")
        self.assertEqual(stored_scope["experience_window"], "sleep_deprived")
        self.assertEqual(stored_scope["annotation_completeness"], 0.95)


if __name__ == "__main__":
    unittest.main()

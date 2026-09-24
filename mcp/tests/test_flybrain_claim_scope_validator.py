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
from flybrain_scope_engine import comparative_claim_tier_engine, hemibrain_flywire_compat_layer  # noqa: E402


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

    def test_comparative_scope_engine_is_deterministic_without_llm(self):
        payload = {
            "dataset": "hb",
            "dataset_version": "neuprint_JRC_Hemibrain_1point2point1",
            "sex": "female",
            "life_stage": "adult",
            "annotation_completeness": 0.93,
            "circuit_class": "mushroom_body",
            "experience_window": "naive",
            "evidence_strength": "validated",
            "confidence": 0.9,
        }
        tier = comparative_claim_tier_engine.tier(payload)
        self.assertIn(tier["tier"], {"T2", "T3"})
        self.assertGreaterEqual(tier["score"], 0.7)

        normalized = hemibrain_flywire_compat_layer.normalize_scope({
            "dataset": "hb",
            "dataset_version": "neuprint_JRC_Hemibrain_1point2point1",
            "sex": "female",
            "life_stage": "adult",
            "annotation_completeness": 0.93,
            "circuit_class": "mushroom_body",
            "experience_window": "naive",
        })
        self.assertEqual(normalized["dataset"], "hemibrain")
        self.assertEqual(normalized["dataset_version"], "neuprint_JRC_Hemibrain_1point2point1")

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

    def test_rejects_unsupported_cross_stage_generalization_in_plasticity_scope(self):
        inv_id = "fb-scope-cross-stage-unsupported"
        _json(server.investigation_start(investigation_id=inv_id, title="cross-stage unsupported"))
        claim_scope = _valid_claim_scope()
        claim_scope["life_stage"] = "cross-stage developmental"
        claim_scope["experience_window"] = "experience-dependent plasticity"

        res = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="cross-stage plasticity claim without stability support",
            source="virtual-fly-brain-query_connectivity",
            metadata={
                "flybrain_provenance": {"tool_name": "query_connectivity"},
                "claim_scope": claim_scope,
            },
        ))
        self.assertIn("error", res)
        self.assertIn("cross-stage generalization", res["error"])

    def test_allows_cross_stage_generalization_with_explicit_support(self):
        inv_id = "fb-scope-cross-stage-supported"
        _json(server.investigation_start(investigation_id=inv_id, title="cross-stage supported"))
        claim_scope = _valid_claim_scope()
        claim_scope["life_stage"] = "cross-stage developmental"
        claim_scope["experience_window"] = "experience-dependent plasticity"

        res = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="cross-stage plasticity claim with support",
            source="virtual-fly-brain-query_connectivity",
            metadata={
                "flybrain_provenance": {
                    "tool_name": "query_connectivity",
                    "generalization_support": {"cross_stage_validated": True},
                },
                "claim_scope": claim_scope,
            },
        ))
        self.assertTrue(res.get("stored"), res)

    def test_tier_engine_downgrades_unsupported_cross_sex_generalization(self):
        payload = {
            "dataset": "fw",
            "dataset_version": "flywire783",
            "sex": "both male/female",
            "life_stage": "adult",
            "annotation_completeness": 0.98,
            "circuit_class": "mushroom_body",
            "experience_window": "naive",
            "evidence_strength": "generalizable",
            "confidence": 0.98,
        }
        tier = comparative_claim_tier_engine.tier(payload)
        self.assertEqual(tier["tier"], "T1")
        self.assertIn("unsupported_cross_sex_generalization", tier["ceiling_reasons"])


if __name__ == "__main__":
    unittest.main()

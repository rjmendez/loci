import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

if "fcntl" not in sys.modules:
    sys.modules["fcntl"] = types.SimpleNamespace(
        LOCK_EX=1,
        LOCK_SH=2,
        LOCK_NB=4,
        LOCK_UN=8,
        flock=lambda *_args, **_kwargs: None,
    )

import server  # noqa: E402
import slow_neuromod  # noqa: E402


def _json(result: str) -> dict:
    return json.loads(result)


class TestSlowNeuromodulationLayer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_memory_dir = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig_memory_dir
        self._tmp.cleanup()

    def test_state_observe_and_policies_are_fail_open_and_bounded(self):
        state = slow_neuromod.load_state(server.MEMORY_DIR)
        self.assertEqual(state["routing_tone"], 0.0)
        self.assertEqual(state["confidence_tone"], 0.0)
        self.assertEqual(state["consolidation_tone"], 0.0)

        out = slow_neuromod.observe(
            server.MEMORY_DIR,
            event="test-observe",
            routing_signal=1.0,
            confidence_signal=0.5,
            consolidation_signal=-1.0,
        )
        self.assertIn("state", out)
        observed = out["state"]
        self.assertGreater(observed["routing_tone"], 0.0)
        self.assertGreater(observed["confidence_tone"], 0.0)
        self.assertLess(observed["consolidation_tone"], 0.0)

        route = slow_neuromod.routing_policy(observed, mnemo_top_k=20, qdrant_limit=30)
        self.assertGreaterEqual(route["mnemo_top_k"], 1)
        self.assertGreaterEqual(route["qdrant_limit"], 1)

        confidence = slow_neuromod.confidence_policy(observed, confidence=0.5)
        self.assertGreaterEqual(confidence["confidence"], 0.0)
        self.assertLessEqual(confidence["confidence"], 1.0)

        consolidation = slow_neuromod.consolidation_policy(observed, default_min_findings=3)
        self.assertGreaterEqual(consolidation["min_findings_for_causal"], 2)
        self.assertLessEqual(consolidation["min_findings_for_causal"], 5)

    def test_invariant_gates_fail_closed_on_invalid_policy_shapes(self):
        with self.assertRaises(ValueError):
            slow_neuromod.assert_routing_policy_invariants({"mnemo_top_k": 10}, minimum_top_k=1)
        with self.assertRaises(ValueError):
            slow_neuromod.assert_confidence_policy_invariants({"confidence": 0.5})
        with self.assertRaises(ValueError):
            slow_neuromod.assert_consolidation_policy_invariants({"min_findings_for_causal": 3})

    def test_investigation_search_uses_slow_routing_bias(self):
        slow_neuromod.save_state(
            server.MEMORY_DIR,
            {
                "schema_version": 1,
                "routing_tone": -0.4,  # bias toward mnemo, away from qdrant
                "confidence_tone": 0.0,
                "consolidation_tone": 0.0,
                "events_seen": 1,
                "last_event": "seed",
                "updated_at": "",
            },
        )

        captured = {"mnemo_top_k": None, "qdrant_limit": None}

        def _fake_mnemo_recall(query, *, top_k, investigation_id=None):  # noqa: ARG001
            captured["mnemo_top_k"] = top_k
            return [{
                "score": 0.8,
                "investigation_id": "inv-a",
                "record_type": "observed",
                "source": "test",
                "confidence": "high",
                "text": "memory hit",
            }]

        def _fake_qdrant(query, investigation_id=None, limit=0, rerank_top_k=0, min_confidence=None):  # noqa: ARG001
            captured["qdrant_limit"] = limit
            return {"ok": True, "reason": "semantic", "results": []}

        with mock.patch.object(server, "_mnemo_recall", side_effect=_fake_mnemo_recall), \
             mock.patch.object(server, "_qdrant_similarity_search", side_effect=_fake_qdrant), \
             mock.patch.object(server, "_get_mnemo_funcs", return_value=(None, object())), \
             mock.patch.object(server, "_search_resolution_maps", return_value=({}, {})):
            payload = _json(server.investigation_search(query="routing bias check", limit=10))

        self.assertEqual(payload["mode"], "mnemo+semantic")
        self.assertEqual(captured["mnemo_top_k"], 48)   # base 40 * 1.2
        self.assertEqual(captured["qdrant_limit"], 24)  # base 30 * 0.8

    def test_memory_consolidate_applies_slow_threshold(self):
        class _FakeMnemo:
            def __init__(self):
                self.beam = type("Beam", (), {"conn": type("Conn", (), {"cursor": lambda *_: None})()})()

            def sleep_all_sessions(self, dry_run=False):  # noqa: ARG002
                return {"items_consolidated": 1, "session_results": []}

        calls = {"causal": 0}

        def _fake_run_causal(_inv_id, _findings):
            calls["causal"] += 1
            return 1

        with mock.patch.object(server, "_load_mnemosyne_class", return_value=_FakeMnemo), \
             mock.patch.object(server, "_snapshot_sleep_consolidation_rowid", return_value=None), \
             mock.patch.object(server, "_run_consolidation_quality_audit", return_value=None), \
             mock.patch.object(server, "_find_most_recent_investigation", return_value=("inv-a", [{"id": "f1"}, {"id": "f2"}])), \
             mock.patch.object(server, "_run_causal_inference", side_effect=_fake_run_causal), \
             mock.patch.object(server, "consolidation_policy", return_value={"consolidation_tone": 0.2, "min_findings_for_causal": 2}), \
             mock.patch.object(server, "assert_consolidation_policy_invariants", return_value=None):
            payload = _json(server.memory_consolidate(dry_run=False))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["causal_edges_inferred"], 1)
        self.assertEqual(calls["causal"], 1)
        self.assertEqual(payload["consolidation_aggregation"]["applied_min_findings_for_causal"], 2)
        self.assertEqual(
            payload["consolidation_aggregation"]["modulation"]["provenance"],
            "deterministic_derived",
        )

    def test_memory_confidence_surfaces_provenance_preserving_aggregation(self):
        fake_results = [{
            "score": 0.9,
            "text": "alpha confidence hit",
            "source": "test-src",
            "investigation_id": "inv-a",
            "confidence": "high",
            "finding_id": "f-1",
        }]
        with mock.patch.object(server, "_confidence_retrieve", return_value=(fake_results, None)), \
             mock.patch.object(server, "_confidence_cues", return_value={
                 "fluency": 0.9,
                 "accessibility": 0.9,
                 "source_div": 1,
                 "corroboration": 1.0,
                 "trust": 1.0,
                 "top_text": "alpha confidence hit",
             }), \
             mock.patch.object(server, "_confidence_verdict", return_value=(0.8, "recollection", "ok")), \
             mock.patch.object(server, "_confidence_llm_entailment", return_value=None):
            payload = _json(server.memory_confidence("alpha"))
        agg = payload["confidence_aggregation"]
        self.assertEqual(agg["method"], "cue_verdict_plus_slow_modulation")
        self.assertEqual(agg["modulation"]["provenance"], "deterministic_derived")
        self.assertEqual(agg["evidence_refs"][0]["finding_id"], "f-1")


if __name__ == "__main__":
    unittest.main(verbosity=2)

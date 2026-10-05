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
        self.assertEqual(state, slow_neuromod.neutral_state())

        out = slow_neuromod.observe(
            server.MEMORY_DIR,
            event="test-observe",
            routing_signal=1.0,
            confidence_signal=0.5,
            consolidation_signal=-1.0,
        )
        observed = out["state"]
        self.assertIs(out["degraded"], False)
        # One slow EMA step (alpha 0.08) from neutral.
        self.assertAlmostEqual(observed["routing_tone"], 0.08)
        self.assertAlmostEqual(observed["confidence_tone"], 0.04)
        self.assertAlmostEqual(observed["consolidation_tone"], -0.08)
        self.assertEqual((observed["events_seen"], observed["last_event"]), (1, "test-observe"))
        self.assertEqual(slow_neuromod.load_state(server.MEMORY_DIR), observed)  # persisted

        # A second event folds into the first; out-of-range signals are clamped to +-1.
        again = slow_neuromod.observe(server.MEMORY_DIR, event="e2", routing_signal=10.0)["state"]
        self.assertAlmostEqual(again["routing_tone"], 0.92 * 0.08 + 0.08 * 1.0)
        self.assertAlmostEqual(again["confidence_tone"], 0.92 * 0.04)
        self.assertEqual(again["events_seen"], 2)

        route = slow_neuromod.routing_policy(observed, mnemo_top_k=20, qdrant_limit=30)
        self.assertEqual((route["mnemo_top_k"], route["qdrant_limit"]), (19, 31))  # x0.96, x1.04
        neutral_route = slow_neuromod.routing_policy({}, mnemo_top_k=20, qdrant_limit=30)
        self.assertEqual((neutral_route["mnemo_top_k"], neutral_route["qdrant_limit"]), (20, 30))

        confidence = slow_neuromod.confidence_policy(observed, confidence=0.5)
        self.assertAlmostEqual(confidence["delta"], 0.12 * 0.04)
        self.assertAlmostEqual(confidence["confidence"], 0.5048)
        self.assertEqual(slow_neuromod.confidence_policy({"confidence_tone": 0.4}, confidence=0.99)["confidence"], 1.0)

        consolidation = slow_neuromod.consolidation_policy(observed, default_min_findings=3)
        self.assertEqual(consolidation["min_findings_for_causal"], 3)   # |tone| < 0.2: unchanged
        for tone, expected in ((0.2, 2), (0.19, 3), (-0.2, 4), (-0.4, 4)):
            got = slow_neuromod.consolidation_policy({"consolidation_tone": tone}, default_min_findings=3)
            self.assertEqual(got["min_findings_for_causal"], expected, tone)

    def test_stored_tones_are_clamped_and_corrupt_state_is_neutral(self):
        path = server.MEMORY_DIR / "_slow_neuromodulation" / "state.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"routing_tone": 5.0, "confidence_tone": -5.0}))
        state = slow_neuromod.load_state(server.MEMORY_DIR)
        self.assertEqual((state["routing_tone"], state["confidence_tone"]), (0.4, -0.4))
        path.write_text("{not json")
        self.assertEqual(slow_neuromod.load_state(server.MEMORY_DIR), slow_neuromod.neutral_state())

    def test_invariant_gates_fail_closed_on_invalid_policy_shapes(self):
        S = slow_neuromod
        with self.assertRaisesRegex(ValueError, "missing qdrant_limit"):
            S.assert_routing_policy_invariants({"routing_tone": 0.0, "mnemo_top_k": 10}, minimum_top_k=1)
        with self.assertRaisesRegex(ValueError, "missing confidence_tone"):
            S.assert_confidence_policy_invariants({"confidence": 0.5})
        with self.assertRaisesRegex(ValueError, "missing consolidation_tone"):
            S.assert_consolidation_policy_invariants({"min_findings_for_causal": 3})
        # out-of-bounds values, one check each
        bad_routing = [
            ({"routing_tone": 0.5, "mnemo_top_k": 5, "qdrant_limit": 5}, "routing_tone out of bounds"),
            ({"routing_tone": 0.0, "mnemo_top_k": 0, "qdrant_limit": 5}, "mnemo_top_k below minimum"),
            ({"routing_tone": 0.0, "mnemo_top_k": 5, "qdrant_limit": 0}, "qdrant_limit below minimum"),
        ]
        for policy, msg in bad_routing:
            with self.assertRaisesRegex(ValueError, msg):
                S.assert_routing_policy_invariants(policy, minimum_top_k=1)
        bad_conf = [
            ({"confidence_tone": 0.5, "delta": 0.0, "confidence": 0.5}, "confidence_tone out of bounds"),
            ({"confidence_tone": 0.0, "delta": 0.3, "confidence": 0.5}, "delta out of bounds"),
            ({"confidence_tone": 0.0, "delta": 0.0, "confidence": 1.1}, "confidence out of range"),
            ({"confidence_tone": 0.0, "delta": 0.0, "confidence": float("nan")}, "confidence must be finite"),
        ]
        for policy, msg in bad_conf:
            with self.assertRaisesRegex(ValueError, msg):
                S.assert_confidence_policy_invariants(policy)
        for policy, msg in (({"consolidation_tone": -0.5, "min_findings_for_causal": 3}, "tone out of bounds"),
                            ({"consolidation_tone": 0.0, "min_findings_for_causal": 9}, "out of range"),
                            ({"consolidation_tone": 0.0, "min_findings_for_causal": 0}, "out of range")):
            with self.assertRaisesRegex(ValueError, msg):
                S.assert_consolidation_policy_invariants(policy)
        # positive twins: in-bounds policies at the edges pass
        S.assert_routing_policy_invariants({"routing_tone": 0.4, "mnemo_top_k": 1, "qdrant_limit": 1}, minimum_top_k=1)
        S.assert_confidence_policy_invariants({"confidence_tone": -0.4, "delta": 0.2, "confidence": 1.0})
        S.assert_consolidation_policy_invariants({"consolidation_tone": 0.4, "min_findings_for_causal": 8})

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
        state = slow_neuromod.neutral_state()
        state["confidence_tone"] = 0.4           # delta = 0.12 * 0.4 = +0.048
        slow_neuromod.save_state(server.MEMORY_DIR, state)
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
        self.assertEqual(agg["modulation"], {"confidence_tone": 0.4, "delta": 0.048,
                                             "provenance": "deterministic_derived"})
        self.assertEqual((agg["base_confidence"], agg["adjusted_confidence"]), (0.8, 0.848))
        self.assertEqual(payload["confidence"], 0.848)
        self.assertEqual(agg["evidence_refs"][0]["finding_id"], "f-1")


if __name__ == "__main__":
    unittest.main(verbosity=2)

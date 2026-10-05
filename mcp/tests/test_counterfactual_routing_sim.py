import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

if "fcntl" not in sys.modules:
    sys.modules["fcntl"] = types.SimpleNamespace(
        LOCK_EX=1,
        LOCK_SH=2,
        LOCK_NB=4,
        LOCK_UN=8,
        flock=lambda *_args, **_kwargs: None,
    )

import server


class TestCounterfactualRoutingSimulation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_memory_dir = server.MEMORY_DIR
        self._orig_get_qdrant = server._get_qdrant
        self._orig_search = server._qdrant_search_collection
        self._orig_collect_global = server._collect_recent_global_audit
        self._orig_load_manifest = server._load_manifest
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig_memory_dir
        server._get_qdrant = self._orig_get_qdrant
        server._qdrant_search_collection = self._orig_search
        server._collect_recent_global_audit = self._orig_collect_global
        server._load_manifest = self._orig_load_manifest
        self._tmp.cleanup()

    def test_memory_route_include_trace_captures_candidates(self):
        def _fake_get_qdrant():
            return object(), "loci_memory"

        def _fake_search(query, collection_name, limit):  # noqa: ARG001
            return [
                {
                    "finding_id": "f-1",
                    "id": "f-1",
                    "investigation_id": "inv-1",
                    "text": "auth failure on cluster alpha",
                    "source": "test",
                    "score": 0.91,
                },
                {
                    "finding_id": "f-2",
                    "id": "f-2",
                    "investigation_id": "inv-2",
                    "text": "auth failure on cluster beta",
                    "source": "test",
                    "score": 0.83,
                },
            ]

        server._get_qdrant = _fake_get_qdrant
        server._qdrant_search_collection = _fake_search
        server._load_manifest = lambda _inv_id: {"title": "Sim Test"}

        payload = json.loads(server.memory_route(
            query="auth failure",
            top_k=2,
            deduplicate=False,
            include_trace=True,
        ))
        self.assertEqual(payload["count"], 2)
        self.assertIn("routing_trace", payload)
        trace = payload["routing_trace"]
        self.assertEqual(trace["version"], 1)
        self.assertEqual(len(trace["candidate_hits"]), 2)
        self.assertEqual(trace["policy"]["top_k"], 2)
        self.assertEqual(trace["policy"]["deduplicate"], False)
        self.assertEqual(trace["policy"]["drive_state"]["hunger"], 0.0)
        self.assertIn("slow_modulation", trace["policy"])
        self.assertEqual(trace["policy"]["slow_modulation"]["provenance"], "deterministic_derived")
        self.assertEqual(trace["metrics"]["exploration_slots"], 0)
        self.assertIn("routing_aggregation", payload)
        self.assertEqual(payload["routing_aggregation"]["source_counts"]["test"], 2)
        self.assertEqual(len(payload["routing_aggregation"]["selected_refs"]), 2)
        self.assertIn("aggregation", trace)

    def test_memory_route_hunger_drive_increases_exploration(self):
        def _fake_get_qdrant():
            return object(), "loci_memory"

        def _fake_search(query, collection_name, limit):  # noqa: ARG001
            self.assertEqual(limit, 9)  # top_k=2 with hunger=1.0 raises candidate breadth
            return [
                {
                    "finding_id": "f-1",
                    "id": "f-1",
                    "investigation_id": "inv-1",
                    "text": "first",
                    "source": "test",
                    "score": 0.95,
                },
                {
                    "finding_id": "f-2",
                    "id": "f-2",
                    "investigation_id": "inv-1",
                    "text": "second",
                    "source": "test",
                    "score": 0.90,
                },
                {
                    "finding_id": "f-3",
                    "id": "f-3",
                    "investigation_id": "inv-1",
                    "text": "third",
                    "source": "test",
                    "score": 0.80,
                },
            ]

        server._get_qdrant = _fake_get_qdrant
        server._qdrant_search_collection = _fake_search
        server._load_manifest = lambda _inv_id: {"title": "Sim Test"}

        payload = json.loads(server.memory_route(
            query="auth failure",
            top_k=2,
            deduplicate=False,
            include_trace=True,
            drive_state={"hunger": 1.0},
        ))
        trace = payload["routing_trace"]
        self.assertEqual(trace["policy"]["drive_state"]["hunger"], 1.0)
        self.assertEqual(trace["metrics"]["priority_slots"], 1)
        self.assertEqual(trace["metrics"]["exploration_slots"], 1)
        self.assertEqual([row["finding_id"] for row in payload["routed"]], ["f-1", "f-3"])
        selected = payload["routing_aggregation"]["selected_refs"]
        self.assertEqual([row["finding_id"] for row in selected], ["f-1", "f-3"])

    def test_counterfactual_simulate_replays_audited_decision(self):
        audit_entry = {
            "ts": "2026-09-22T16:00:00Z",
            "tool": "memory_route",
            "inputs": json.dumps({"query": "auth failure", "top_k": 2, "deduplicate": True}),
            "output": json.dumps({
                "query": "auth failure",
                "routed": [
                    {"finding_id": "f-1", "text": "alpha auth failure"},
                    {"finding_id": "f-2", "text": "beta auth failure"},
                ],
                "count": 2,
                "routing_trace": {
                    "version": 1,
                    "policy": {
                        "top_k": 2,
                        "deduplicate": True,
                        "dedup_threshold": 0.8,
                        "drive_state": {"hunger": 1.0, "fatigue": 0.0, "urgency": 0.0},
                        "slow_modulation": {"routing_tone": 0.0, "qdrant_limit": 6, "provenance": "deterministic_derived"},
                    },
                    "candidate_hits": [
                        {"finding_id": "f-1", "id": "f-1", "text": "alpha auth failure", "score": 0.95},
                        {"finding_id": "f-2", "id": "f-2", "text": "beta auth failure", "score": 0.89},
                        {"finding_id": "f-3", "id": "f-3", "text": "gamma auth failure", "score": 0.70},
                    ],
                },
            }),
        }

        server._collect_recent_global_audit = lambda limit=200, days=3: [audit_entry]  # noqa: ARG005
        server._load_manifest = lambda _inv_id: {"title": "Sim Test"}

        payload = json.loads(server.memory_route_counterfactual_simulate(
            limit=5,
            top_k=1,
            deduplicate=True,
            dedup_threshold=0.8,
        ))
        self.assertEqual(payload["simulated"], 1)
        self.assertEqual(payload["changed"], 1)
        first = payload["results"][0]
        self.assertEqual(first["baseline_count"], 2)
        self.assertEqual(first["counterfactual"]["count"], 1)
        self.assertEqual(first["counterfactual"]["policy"]["drive_state"]["hunger"], 1.0)
        self.assertEqual(first["counterfactual"]["policy"]["slow_modulation"]["provenance"], "deterministic_derived")
        self.assertIn("aggregation", first["counterfactual"])
        self.assertEqual(len(first["counterfactual"]["aggregation"]["selected_refs"]), 1)
        # hunger=1.0, top_k=1: no priority slot, the single exploration slot takes
        # the tail of the candidate pool.
        self.assertEqual(first["counterfactual"]["removed_finding_ids"], ["f-1", "f-2"])
        self.assertEqual(first["counterfactual"]["added_finding_ids"], ["f-3"])
        self.assertEqual(first["counterfactual"]["overlap_count"], 0)

    def _replay(self, *, hunger, top_k, candidates, baseline_ids, **kw):
        audit_entry = {
            "ts": "2026-09-22T16:00:00Z",
            "tool": "memory_route",
            "inputs": json.dumps({"query": "auth failure", "top_k": 2, "deduplicate": False}),
            "output": json.dumps({
                "query": "auth failure",
                "routed": [{"finding_id": fid, "text": fid} for fid in baseline_ids],
                "count": len(baseline_ids),
                "routing_trace": {
                    "version": 1,
                    "policy": {
                        "top_k": 2,
                        "deduplicate": False,
                        "dedup_threshold": 0.8,
                        "drive_state": {"hunger": hunger, "fatigue": 0.0, "urgency": 0.0},
                    },
                    "candidate_hits": candidates,
                },
            }),
        }
        server._collect_recent_global_audit = lambda limit=200, days=3: [audit_entry]  # noqa: ARG005
        server._load_manifest = lambda _inv_id: {"title": "Sim Test"}
        payload = json.loads(server.memory_route_counterfactual_simulate(
            limit=5, top_k=top_k, deduplicate=False, **kw))
        self.assertEqual(payload["simulated"], 1)
        return payload

    @staticmethod
    def _candidates(n):
        # Distinct texts (no dedup), strictly decreasing score: rank == id order.
        return [{"finding_id": f"f-{i}", "id": f"f-{i}", "text": f"hit number {i} word{i}",
                 "score": round(1.0 - i * 0.05, 2)} for i in range(1, n + 1)]

    def test_counterfactual_simulate_priority_keeps_the_top_hit(self):
        """No hunger: every slot is a priority slot, so top_k=1 keeps the best-
        scored baseline hit and removes exactly the other one."""
        payload = self._replay(hunger=0.0, top_k=1, candidates=self._candidates(3),
                               baseline_ids=["f-1", "f-2"])
        cf = payload["results"][0]["counterfactual"]
        self.assertEqual([r["finding_id"] for r in cf["aggregation"]["selected_refs"]], ["f-1"])
        self.assertEqual(cf["removed_finding_ids"], ["f-2"])
        self.assertEqual(cf["added_finding_ids"], [])
        self.assertEqual(cf["overlap_count"], 1)
        self.assertEqual(payload["changed"], 1)

    def test_counterfactual_simulate_unchanged_policy_reports_no_change(self):
        payload = self._replay(hunger=0.0, top_k=2, candidates=self._candidates(3),
                               baseline_ids=["f-1", "f-2"])
        cf = payload["results"][0]["counterfactual"]
        self.assertEqual((cf["removed_finding_ids"], cf["added_finding_ids"]), ([], []))
        self.assertEqual(cf["overlap_count"], 2)
        self.assertEqual(payload["changed"], 0)

    def test_counterfactual_simulate_accepts_routing_tone_override(self):
        audit_entry = {
            "ts": "2026-09-22T16:00:00Z",
            "tool": "memory_route",
            "inputs": json.dumps({"query": "auth failure", "top_k": 2, "deduplicate": True}),
            "output": json.dumps({
                "query": "auth failure",
                "routed": [{"finding_id": "f-1", "text": "alpha auth failure"}],
                "count": 1,
                "routing_trace": {
                    "version": 1,
                    "policy": {
                        "top_k": 2,
                        "deduplicate": True,
                        "dedup_threshold": 0.8,
                        "drive_state": {"hunger": 0.0, "fatigue": 0.0, "urgency": 0.0},
                    },
                    "candidate_hits": [
                        {"finding_id": "f-1", "id": "f-1", "text": "alpha auth failure", "score": 0.95},
                        {"finding_id": "f-2", "id": "f-2", "text": "beta auth failure", "score": 0.89},
                        {"finding_id": "f-3", "id": "f-3", "text": "gamma auth failure", "score": 0.70},
                    ],
                },
            }),
        }

        server._collect_recent_global_audit = lambda limit=200, days=3: [audit_entry]  # noqa: ARG005
        server._load_manifest = lambda _inv_id: {"title": "Sim Test"}

        payload = json.loads(server.memory_route_counterfactual_simulate(
            limit=5,
            top_k=2,
            deduplicate=False,
            dedup_threshold=0.8,
            routing_tone_override=-0.4,
        ))
        self.assertEqual(payload["counterfactual_overrides"]["routing_tone_override"], -0.4)
        first = payload["results"][0]
        self.assertEqual(first["counterfactual"]["policy"]["slow_modulation"]["routing_tone"], -0.4)

    def test_counterfactual_routing_tone_changes_the_replayed_selection(self):
        """The tone must drive the replay, not just be echoed. hunger=1.0, top_k=2:
        candidate_limit 9; tone 0 keeps all 9 candidates so the exploration slot
        takes f-9, tone -0.4 scales the Qdrant limit to round(9*0.8)=7 -> f-7."""
        neutral = self._replay(hunger=1.0, top_k=2, candidates=self._candidates(9),
                               baseline_ids=["f-1", "f-9"], routing_tone_override=0.0)
        damped = self._replay(hunger=1.0, top_k=2, candidates=self._candidates(9),
                              baseline_ids=["f-1", "f-9"], routing_tone_override=-0.4)
        n_cf = neutral["results"][0]["counterfactual"]
        d_cf = damped["results"][0]["counterfactual"]
        self.assertEqual(n_cf["policy"]["slow_modulation"]["qdrant_limit"], 9)
        self.assertEqual(d_cf["policy"]["slow_modulation"]["qdrant_limit"], 7)
        self.assertEqual([r["finding_id"] for r in n_cf["aggregation"]["selected_refs"]], ["f-1", "f-9"])
        self.assertEqual([r["finding_id"] for r in d_cf["aggregation"]["selected_refs"]], ["f-1", "f-7"])
        self.assertEqual((d_cf["removed_finding_ids"], d_cf["added_finding_ids"]), (["f-9"], ["f-7"]))

    def test_counterfactual_simulate_reports_missing_trace(self):
        audit_entry = {
            "ts": "2026-09-22T16:00:00Z",
            "tool": "memory_route",
            "inputs": json.dumps({"query": "auth failure"}),
            "output": json.dumps({
                "query": "auth failure",
                "routed": [{"finding_id": "f-1", "text": "alpha auth failure"}],
                "count": 1,
            }),
        }
        server._collect_recent_global_audit = lambda limit=200, days=3: [audit_entry]  # noqa: ARG005
        payload = json.loads(server.memory_route_counterfactual_simulate(limit=5))
        self.assertEqual(payload["simulated"], 0)
        self.assertEqual(payload["skipped_missing_trace"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

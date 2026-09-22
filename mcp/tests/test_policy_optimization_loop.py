import json
import tempfile
import unittest
from pathlib import Path

import server


def _route_audit_entry(*, ts: str, top_k: int, baseline_ids: list[str]) -> dict:
    candidates = [
        {"finding_id": "f-1", "id": "f-1", "text": "alpha auth failure", "score": 0.95},
        {"finding_id": "f-2", "id": "f-2", "text": "beta auth failure", "score": 0.91},
        {"finding_id": "f-3", "id": "f-3", "text": "gamma auth failure", "score": 0.83},
    ]
    routed = [{"finding_id": fid, "text": f"{fid} auth failure"} for fid in baseline_ids]
    return {
        "ts": ts,
        "tool": "memory_route",
        "output": json.dumps({
            "query": "auth failure",
            "routed": routed,
            "count": len(routed),
            "routing_trace": {
                "version": 1,
                "policy": {"top_k": top_k, "deduplicate": True, "dedup_threshold": 0.8},
                "candidate_hits": candidates,
            },
        }),
    }


class TestPolicyOptimizationLoop(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_memory_dir = server.MEMORY_DIR
        self._orig_collect_global = server._collect_recent_global_audit
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig_memory_dir
        server._collect_recent_global_audit = self._orig_collect_global
        self._tmp.cleanup()

    def test_policy_optimization_recommends_conservative_change(self):
        # Baseline traces claim top_k=2 but output only one row; optimizer should
        # learn that top_k=1 reproduces observed decisions with less drift.
        entries = [
            _route_audit_entry(ts="2026-09-22T16:00:00Z", top_k=2, baseline_ids=["f-1"]),
            _route_audit_entry(ts="2026-09-22T16:01:00Z", top_k=2, baseline_ids=["f-1"]),
            _route_audit_entry(ts="2026-09-22T16:02:00Z", top_k=2, baseline_ids=["f-1"]),
            {
                "ts": "2026-09-22T16:03:00Z",
                "tool": "memory_consolidate",
                "output": json.dumps({
                    "status": "ok",
                    "consolidation_quality_audit": {"sampled": 4, "flagged": [{"id": "a"}], "degraded": False},
                }),
            },
        ]
        server._collect_recent_global_audit = lambda limit=200, days=7: entries  # noqa: ARG005

        payload = json.loads(server.memory_route_policy_optimize(limit=10, min_decisions=3))
        self.assertTrue(payload["optimized"])
        self.assertEqual(payload["recommendation"]["top_k"], 1)
        self.assertEqual(payload["decision_count"], 3)
        self.assertEqual(payload["consolidation_outcomes"]["flagged"], 1)
        self.assertIn("provenance_aggregation", payload)
        self.assertEqual(payload["provenance_aggregation"]["decision_count"], 3)

    def test_policy_optimization_requires_minimum_decisions(self):
        entries = [
            _route_audit_entry(ts="2026-09-22T16:00:00Z", top_k=2, baseline_ids=["f-1"]),
        ]
        server._collect_recent_global_audit = lambda limit=200, days=7: entries  # noqa: ARG005

        payload = json.loads(server.memory_route_policy_optimize(limit=10, min_decisions=3))
        self.assertFalse(payload["optimized"])
        self.assertEqual(payload["reason"], "insufficient_decisions")
        self.assertIsNone(payload["recommendation"])

    def test_policy_optimization_persisted_report_excludes_raw_evidence_text(self):
        entries = [
            _route_audit_entry(ts="2026-09-22T16:00:00Z", top_k=2, baseline_ids=["f-1"]),
            _route_audit_entry(ts="2026-09-22T16:01:00Z", top_k=2, baseline_ids=["f-1"]),
            _route_audit_entry(ts="2026-09-22T16:02:00Z", top_k=2, baseline_ids=["f-1"]),
        ]
        server._collect_recent_global_audit = lambda limit=200, days=7: entries  # noqa: ARG005

        payload = json.loads(server.memory_route_policy_optimize(limit=10, min_decisions=3, persist=True))
        self.assertTrue(payload.get("persisted"))
        policy_path = Path(self._tmp.name) / "_policy" / "route_policy_optimization.jsonl"
        self.assertTrue(policy_path.exists())
        line = policy_path.read_text().strip().splitlines()[-1]
        record = json.loads(line)
        self.assertIn("top_candidates", record)
        self.assertIn("provenance_aggregation", record)
        self.assertNotIn("candidate_hits", record)
        self.assertNotIn("alpha auth failure", line)

    def test_policy_optimization_fails_closed_on_invariant_violation(self):
        bad = _route_audit_entry(ts="2026-09-22T16:00:00Z", top_k=2, baseline_ids=["f-missing"])
        entries = [bad, dict(bad, ts="2026-09-22T16:01:00Z"), dict(bad, ts="2026-09-22T16:02:00Z")]
        server._collect_recent_global_audit = lambda limit=200, days=7: entries  # noqa: ARG005

        payload = json.loads(server.memory_route_policy_optimize(limit=10, min_decisions=3))
        self.assertFalse(payload["optimized"])
        self.assertEqual(payload["reason"], "invariant_gate_failed")
        self.assertIn("invariant_gate", payload)
        self.assertFalse(payload["invariant_gate"]["ok"])
        self.assertTrue(payload["invariant_gate"]["violations"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

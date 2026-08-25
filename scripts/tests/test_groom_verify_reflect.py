"""The two passes that wire up #194's dormant features.

Neither feature was broken — investigation_verify_all and the reflection loop
were correct, exposed code that had never been invoked. Scheduling them is the
fix, but only with the guards below: a fail-open verifier and a cron job are a
bad combination without them.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import loci_groom as g


class PassVerifyGateTest(unittest.TestCase):
    """verify_finding is fail-open: no model -> uncertain/0.0 for EVERY finding.

    investigation_verify_all writes those to finding_verifications.jsonl, so an
    unattended run against a dead backend fills an empty lane with records that
    carry no information. Measured on the first real run: 5 records, all
    uncertain/0.0. The pass must refuse instead.
    """

    def test_a_dead_generation_backend_degrades_and_writes_nothing(self):
        called = []
        rep = g.pass_verify(gen_probe=lambda: False,
                            verify_fn=lambda **kw: called.append(kw) or "{}")
        self.assertEqual(rep["status"], "degraded")
        self.assertIn("generation backend", rep["detail"])
        self.assertEqual(called, [], "must not verify anything with no model")

    def test_a_raising_probe_also_degrades(self):
        rep = g.pass_verify(gen_probe=lambda: (_ for _ in ()).throw(OSError("refused")))
        self.assertEqual(rep["status"], "degraded")

    def test_a_live_backend_proceeds(self):
        seen = []

        def fake_verify(investigation_id, limit):
            seen.append(investigation_id)
            return json.dumps({"results": [{"finding_id": "x", "verdict": "confirmed",
                                            "confidence": 0.8, "degraded": False}]})

        rep = g.pass_verify(gen_probe=lambda: True, verify_fn=fake_verify, limit=2)
        self.assertEqual(rep["status"], "ok")
        self.assertGreaterEqual(rep["investigations_checked"], 0)


class PassReflectReportingTest(unittest.TestCase):
    """The backlog number has to be true.

    reflection_loop_tick emits `remaining_queue`; it only emits `queue_size` on
    the empty-queue early return. Reading queue_size with a default of 0 reported
    a backlog of 0 while 262 items were still queued.
    """

    def test_the_remaining_backlog_is_reported_not_defaulted_to_zero(self):
        rep = g.pass_reflect(
            seed_fn=lambda: json.dumps({"queued": 260}),
            tick_fn=lambda **kw: json.dumps({"processed_items": 3,
                                             "findings_written": 1,
                                             "remaining_queue": 257}),
        )
        self.assertEqual(rep["status"], "ok")
        self.assertEqual(rep["remaining_queue"], 257)
        self.assertNotIn("queue_size", rep)

    def test_an_absent_backlog_key_is_omitted_rather_than_reported_as_zero(self):
        rep = g.pass_reflect(
            seed_fn=lambda: json.dumps({"queued": 5}),
            tick_fn=lambda **kw: json.dumps({"processed_items": 1, "findings_written": 0}),
        )
        self.assertNotIn("remaining_queue", rep,
                         "a missing backlog must not be reported as an empty one")

    def test_a_failing_seed_degrades_without_ticking(self):
        ticked = []
        rep = g.pass_reflect(
            seed_fn=lambda: (_ for _ in ()).throw(RuntimeError("no sources")),
            tick_fn=lambda **kw: ticked.append(kw) or "{}",
        )
        self.assertEqual(rep["status"], "degraded")
        self.assertEqual(ticked, [])

    def test_a_failing_tick_degrades(self):
        rep = g.pass_reflect(
            seed_fn=lambda: json.dumps({"queued": 5}),
            tick_fn=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        self.assertEqual(rep["status"], "degraded")


class PassRegistryTest(unittest.TestCase):
    def test_both_passes_are_registered_and_not_applyable(self):
        for name in ("verify", "reflect"):
            self.assertIn(name, g.PASSES)
            self.assertFalse(g.PASSES[name]["applyable"])


if __name__ == "__main__":
    unittest.main()

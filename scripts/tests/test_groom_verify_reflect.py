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

    def _memory_dir(self, *names, verified=()):
        """A memory dir holding one investigation per name, each with findings;
        the ones in ``verified`` already carry a finding_verifications.jsonl."""
        import tempfile
        from pathlib import Path
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for name in names:
            (root / name).mkdir()
            (root / name / "findings.jsonl").write_text('{"id": "f1"}\n', encoding="utf-8")
            if name in verified:
                (root / name / "finding_verifications.jsonl").write_text("", encoding="utf-8")
        return root

    def _recording_verify(self, per_inv=None, fail=()):
        calls = []

        def fake_verify(investigation_id, limit):
            calls.append((investigation_id, limit))
            if investigation_id in fail:
                raise RuntimeError(f"verifier down for {investigation_id}")
            n = (per_inv or {}).get(investigation_id, 1)
            return json.dumps({"results": [{"finding_id": f"x{i}", "verdict": "confirmed",
                                            "confidence": 0.8, "degraded": False}
                                           for i in range(n)]})
        return fake_verify, calls

    def test_a_live_backend_proceeds(self):
        root = self._memory_dir("inv-a", "inv-b", "inv-c")
        fake_verify, calls = self._recording_verify(per_inv={"inv-a": 2, "inv-b": 3})

        rep = g.pass_verify(gen_probe=lambda: True, verify_fn=fake_verify, limit=2,
                            memory_dir=root)

        self.assertEqual(calls, [("inv-a", g.VERIFY_PER_INVESTIGATION),
                                 ("inv-b", g.VERIFY_PER_INVESTIGATION)])
        self.assertEqual(rep, {"pass": "verify", "status": "ok", "investigations_checked": 2,
                               "findings_verified": 5, "errors": 0, "budget": 2})

    def test_already_verified_investigations_are_skipped_so_runs_advance(self):
        root = self._memory_dir("inv-a", "inv-b", "inv-c", verified=("inv-a",))
        fake_verify, calls = self._recording_verify()

        rep = g.pass_verify(gen_probe=lambda: True, verify_fn=fake_verify, limit=5,
                            memory_dir=root)

        self.assertEqual([inv for inv, _ in calls], ["inv-b", "inv-c"])
        self.assertEqual(rep["investigations_checked"], 2)
        self.assertEqual(rep["findings_verified"], 2)

    def test_the_default_budget_is_the_per_run_cap(self):
        from unittest import mock
        root = self._memory_dir("inv-a", "inv-b", "inv-c")
        fake_verify, calls = self._recording_verify()

        with mock.patch.object(g, "VERIFY_MAX_PER_RUN", 1):
            rep = g.pass_verify(gen_probe=lambda: True, verify_fn=fake_verify, memory_dir=root)

        self.assertEqual([inv for inv, _ in calls], ["inv-a"])
        self.assertEqual((rep["budget"], rep["investigations_checked"]), (1, 1))

    def test_every_attempt_failing_is_degraded(self):
        root = self._memory_dir("inv-a", "inv-b")
        fake_verify, calls = self._recording_verify(fail=("inv-a", "inv-b"))

        rep = g.pass_verify(gen_probe=lambda: True, verify_fn=fake_verify, limit=5,
                            memory_dir=root)

        self.assertEqual(len(calls), 2)
        self.assertEqual(rep["status"], "degraded")
        self.assertEqual(rep["detail"], "every attempt failed (2)")
        self.assertEqual((rep["investigations_checked"], rep["errors"]), (0, 2))

    def test_one_failure_among_successes_is_counted_not_degraded(self):
        root = self._memory_dir("inv-a", "inv-b", "inv-c")
        fake_verify, calls = self._recording_verify(fail=("inv-b",))

        rep = g.pass_verify(gen_probe=lambda: True, verify_fn=fake_verify, limit=5,
                            memory_dir=root)

        self.assertEqual([inv for inv, _ in calls], ["inv-a", "inv-b", "inv-c"])
        self.assertEqual(rep["status"], "ok")
        self.assertEqual((rep["investigations_checked"], rep["findings_verified"], rep["errors"]),
                         (2, 2, 1))


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

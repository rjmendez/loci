"""The summaries pass, and the access-record filter it depends on.

The summary ladder inside investigation_reflect was correct code that nothing
called: 6 of 142 manifests carried a summary while 7 of 15 real
investigation_load calls asked for fidelity=summary or brief.

It also could not have produced a good one. findings.jsonl is a mixed append log
and 3,681 of its 6,610 records (55.7%) are text-less access rows written on every
read. Being the newest rows, they filled findings[-20:] completely for some
investigations, so the model was handed twenty empty bullets plus a title and
invented a plausible topic from the title alone.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import loci_groom as g


class HasLlmSummaryTest(unittest.TestCase):
    """The deterministic stub must not count as done, or one run with the backend
    down would permanently mark every investigation as summarised."""

    def test_model_authored_summary_counts(self):
        self.assertTrue(g._has_llm_summary(
            {"summary_l1": ["a point"], "summary_l2": "A real paragraph about it."}))

    def test_deterministic_stub_does_not_count(self):
        self.assertFalse(g._has_llm_summary(
            {"summary_l1": ["raw finding text"],
             "summary_l2": "Investigation with 12 findings. Latest: something."}))

    def test_singular_stub_does_not_count(self):
        self.assertFalse(g._has_llm_summary(
            {"summary_l1": ["x"], "summary_l2": "Investigation with 1 finding."}))

    def test_missing_pieces_do_not_count(self):
        self.assertFalse(g._has_llm_summary({}))
        self.assertFalse(g._has_llm_summary({"summary_l1": [], "summary_l2": "text"}))
        self.assertFalse(g._has_llm_summary({"summary_l1": ["x"], "summary_l2": "   "}))


class PassSummariesGateTest(unittest.TestCase):
    def test_a_dead_backend_degrades_and_reflects_nothing(self):
        called = []
        rep = g.pass_summaries(gen_probe=lambda: False,
                               reflect_fn=lambda **kw: called.append(kw))
        self.assertEqual(rep["status"], "degraded")
        self.assertIn("generation backend", rep["detail"])
        self.assertEqual(called, [], "must not stamp a stub with no model")

    def test_a_raising_probe_also_degrades(self):
        rep = g.pass_summaries(
            gen_probe=lambda: (_ for _ in ()).throw(OSError("refused")))
        self.assertEqual(rep["status"], "degraded")


class PassSummariesWorkTest(unittest.TestCase):
    def _inv(self, tmp, name, manifest, findings=None):
        d = tmp / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(manifest))
        recs = findings if findings is not None else [
            {"record_type": "observed", "text": "a real finding to summarise"}]
        (d / "findings.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in recs))
        return d

    def setUp(self):
        import tempfile, pathlib
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_skips_investigations_that_already_have_one(self):
        self._inv(self.root, "done", {"investigation_id": "done",
                                      "summary_l1": ["p"], "summary_l2": "Real summary."})
        seen = []
        rep = g.pass_summaries(memory_dir=self.root, gen_probe=lambda: True,
                               reflect_fn=lambda **kw: seen.append(kw))
        self.assertEqual(seen, [])
        self.assertEqual(rep["already_had"], 1)
        self.assertEqual(rep["summarised"], 0)

    def test_summarises_one_that_lacks_it(self):
        d = self._inv(self.root, "todo", {"investigation_id": "todo"})

        def reflect(investigation_id):
            m = json.loads((d / "manifest.json").read_text())
            m["summary_l1"] = ["a point"]
            m["summary_l2"] = "A real paragraph."
            (d / "manifest.json").write_text(json.dumps(m))

        rep = g.pass_summaries(memory_dir=self.root, gen_probe=lambda: True,
                               reflect_fn=reflect)
        self.assertEqual(rep["summarised"], 1)
        self.assertEqual(rep["errors"], 0)

    def test_a_stub_written_by_reflect_counts_as_an_error_not_a_success(self):
        """Otherwise a transient model failure would be recorded as done and the
        investigation would never be retried."""
        d = self._inv(self.root, "todo", {"investigation_id": "todo"})

        def reflect(investigation_id):
            m = json.loads((d / "manifest.json").read_text())
            m["summary_l1"] = ["raw text"]
            m["summary_l2"] = "Investigation with 3 findings."
            (d / "manifest.json").write_text(json.dumps(m))

        rep = g.pass_summaries(memory_dir=self.root, gen_probe=lambda: True,
                               reflect_fn=reflect)
        self.assertEqual(rep["summarised"], 0)
        self.assertEqual(rep["errors"], 1)
        self.assertEqual(rep["status"], "degraded")

    def test_budget_bounds_the_run(self):
        for i in range(5):
            self._inv(self.root, f"inv{i}", {"investigation_id": f"inv{i}"})

        def reflect(investigation_id):
            p = self.root / investigation_id / "manifest.json"
            m = json.loads(p.read_text())
            m["summary_l1"] = ["p"]
            m["summary_l2"] = "Real."
            p.write_text(json.dumps(m))

        rep = g.pass_summaries(memory_dir=self.root, limit=2,
                               gen_probe=lambda: True, reflect_fn=reflect)
        self.assertEqual(rep["summarised"], 2)


if __name__ == "__main__":
    unittest.main()


class EmptyInvestigationTest(unittest.TestCase):
    """Once the backlog is clear the only ones left are those with nothing to
    summarise. Counting them as errors made the pass report degraded on every
    single run, which is how a real failure goes unnoticed."""

    def setUp(self):
        import tempfile, pathlib
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _inv(self, name, lines):
        d = self.root / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"investigation_id": name}))
        (d / "findings.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        return d

    def test_no_findings_file_is_not_an_error(self):
        d = self.root / "bare"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"investigation_id": "bare"}))
        seen = []
        rep = g.pass_summaries(memory_dir=self.root, gen_probe=lambda: True,
                               reflect_fn=lambda **kw: seen.append(kw))
        self.assertEqual(seen, [], "must not reflect an investigation with no findings")
        self.assertEqual(rep["nothing_to_say"], 1)
        self.assertEqual(rep["errors"], 0)
        self.assertEqual(rep["status"], "ok")

    def test_access_rows_alone_are_not_summarisable(self):
        self._inv("acc", [{"record_type": "access", "query": "q"}] * 5)
        rep = g.pass_summaries(memory_dir=self.root, gen_probe=lambda: True,
                               reflect_fn=lambda **kw: None)
        self.assertEqual(rep["nothing_to_say"], 1)
        self.assertEqual(rep["errors"], 0)

    def test_one_real_finding_is_summarisable(self):
        d = self._inv("real", [{"record_type": "access"},
                               {"record_type": "observed", "text": "a real one"}])

        def reflect(investigation_id):
            p = d / "manifest.json"
            m = json.loads(p.read_text())
            m["summary_l1"] = ["p"]
            m["summary_l2"] = "Real."
            p.write_text(json.dumps(m))

        rep = g.pass_summaries(memory_dir=self.root, gen_probe=lambda: True,
                               reflect_fn=reflect)
        self.assertEqual(rep["summarised"], 1)
        self.assertEqual(rep["nothing_to_say"], 0)

    def test_a_cleared_backlog_reports_ok_not_degraded(self):
        self._inv("empty", [{"record_type": "access"}])
        d = self.root / "done"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(
            {"investigation_id": "done", "summary_l1": ["p"], "summary_l2": "Real."}))
        rep = g.pass_summaries(memory_dir=self.root, gen_probe=lambda: True,
                               reflect_fn=lambda **kw: None)
        self.assertEqual(rep["status"], "ok")
        self.assertEqual(rep["summarised"], 0)

    def test_a_retracted_only_investigation_has_nothing_to_say(self):
        """investigation_reflect drops retracted findings, so an investigation
        whose only finding is retracted reports 'Investigation with 0 findings'
        forever. dtl-mnemo-probe is exactly this case."""
        d = self._inv("retracted", [{"id": "f1", "record_type": "observed",
                                     "text": "a finding that was retracted"}])
        (d / "retractions.jsonl").write_text(
            json.dumps({"finding_id": "f1", "active": True}) + "\n")
        seen = []
        rep = g.pass_summaries(memory_dir=self.root, gen_probe=lambda: True,
                               reflect_fn=lambda **kw: seen.append(kw))
        self.assertEqual(seen, [])
        self.assertEqual(rep["nothing_to_say"], 1)
        self.assertEqual(rep["errors"], 0)
        self.assertEqual(rep["status"], "ok")

    def test_an_inactive_retraction_does_not_hide_a_finding(self):
        d = self._inv("unretracted", [{"id": "f1", "record_type": "observed",
                                       "text": "still a live finding"}])
        (d / "retractions.jsonl").write_text(
            json.dumps({"finding_id": "f1", "active": False}) + "\n")

        def reflect(investigation_id):
            p = d / "manifest.json"
            m = json.loads(p.read_text())
            m["summary_l1"] = ["p"]; m["summary_l2"] = "Real."
            p.write_text(json.dumps(m))

        rep = g.pass_summaries(memory_dir=self.root, gen_probe=lambda: True,
                               reflect_fn=reflect)
        self.assertEqual(rep["summarised"], 1)

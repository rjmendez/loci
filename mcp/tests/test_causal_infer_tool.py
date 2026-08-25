"""causal_infer: producing edges is a thing you can do (#192).

Before this, the only caller was memory_consolidate — behind `not dry_run` and
`len(findings) >= 3`, on whichever investigation happened to be most recent. So
causal_edges.jsonl held 0 records across the entire corpus, and
causal_edges_list answered "nothing is known" and "this never ran" with the same
empty shape.

Verified live on dama-imu-physiological-sensing-2026-08: 0 edges before, 46
after, all method=declared_lineage, and causal_edges_list then reported 46 — the
first non-zero that tool has ever returned.
"""
import json
import unittest
from unittest import mock

import server


class CausalInferTest(unittest.TestCase):

    def test_a_missing_investigation_errors_and_does_not_create_one(self):
        """_inv_dir() mkdirs, so guarding on it would both fail to guard AND
        conjure an empty investigation from a typo. _load_manifest is read-only."""
        name = "no-such-investigation-xyzzy"
        out = json.loads(server.causal_infer(investigation_id=name))
        self.assertIn("error", out)
        self.assertFalse((server.MEMORY_DIR / name).exists(),
                         "a lookup miss must not create an investigation")

    def test_too_few_findings_says_so_rather_than_reporting_zero_edges(self):
        """'nothing to relate' must not look like 'inference found nothing'."""
        with mock.patch.object(server, "_load_manifest", return_value={"id": "inv"}), \
             mock.patch.object(server, "_inv_dir"), \
             mock.patch.object(server, "_read_jsonl", return_value=[{"id": "a", "text": "x"}]), \
             mock.patch.object(server, "_load_retracted_ids", return_value=set()):
            out = json.loads(server.causal_infer(investigation_id="inv"))
        self.assertEqual(out["status"], "too_few_findings")
        self.assertEqual(out["edges_written"], 0)

    def test_it_reports_what_it_wrote(self):
        findings = [{"id": "a", "text": "the base went down"},
                    {"id": "b", "text": "rtk dropped", "derived_from": ["a"]}]
        with mock.patch.object(server, "_load_manifest", return_value={"id": "inv"}), \
             mock.patch.object(server, "_inv_dir"), \
             mock.patch.object(server, "_read_jsonl", return_value=findings), \
             mock.patch.object(server, "_load_retracted_ids", return_value=set()), \
             mock.patch.object(server, "_run_causal_inference", return_value=7) as run:
            out = json.loads(server.causal_infer(investigation_id="inv"))
        self.assertEqual(out["edges_written"], 7)
        self.assertEqual(out["findings_considered"], 2)
        self.assertEqual(out["status"], "ok")
        run.assert_called_once()

    def test_retracted_findings_are_excluded(self):
        findings = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}, {"id": "c", "text": "z"}]
        seen = {}
        with mock.patch.object(server, "_load_manifest", return_value={"id": "inv"}), \
             mock.patch.object(server, "_inv_dir"), \
             mock.patch.object(server, "_read_jsonl", return_value=findings), \
             mock.patch.object(server, "_load_retracted_ids", return_value={"b"}), \
             mock.patch.object(server, "_run_causal_inference",
                              side_effect=lambda i, f: seen.setdefault("f", f) and 0 or 0):
            server.causal_infer(investigation_id="inv")
        self.assertEqual([f["id"] for f in seen["f"]], ["a", "c"])

    def test_textless_findings_are_excluded(self):
        """The producers key off text; a blank finding contributes nothing."""
        findings = [{"id": "a", "text": "x"}, {"id": "b", "text": "   "}, {"id": "c", "text": "z"}]
        seen = {}
        with mock.patch.object(server, "_load_manifest", return_value={"id": "inv"}), \
             mock.patch.object(server, "_inv_dir"), \
             mock.patch.object(server, "_read_jsonl", return_value=findings), \
             mock.patch.object(server, "_load_retracted_ids", return_value=set()), \
             mock.patch.object(server, "_run_causal_inference",
                              side_effect=lambda i, f: seen.setdefault("f", f) and 0 or 0):
            server.causal_infer(investigation_id="inv")
        self.assertEqual([f["id"] for f in seen["f"]], ["a", "c"])

    def test_limit_takes_the_newest(self):
        findings = [{"id": str(i), "text": f"f{i}"} for i in range(10)]
        seen = {}
        with mock.patch.object(server, "_load_manifest", return_value={"id": "inv"}), \
             mock.patch.object(server, "_inv_dir"), \
             mock.patch.object(server, "_read_jsonl", return_value=findings), \
             mock.patch.object(server, "_load_retracted_ids", return_value=set()), \
             mock.patch.object(server, "_run_causal_inference",
                              side_effect=lambda i, f: seen.setdefault("f", f) and 0 or 0):
            server.causal_infer(investigation_id="inv", limit=3)
        self.assertEqual([f["id"] for f in seen["f"]], ["7", "8", "9"])


if __name__ == "__main__":
    unittest.main()

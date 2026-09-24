"""scripts/backfill_provenance_tiers.py on a temp store, never the live one.

The script must tag a finding only when a rule makes its tier unambiguous,
leave everything else untagged, never touch a row that already has a tier,
write nothing on a dry run, back up before appending, and be safe to rerun.
"""
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("backfill_provenance_tiers",
                                                  _SCRIPTS / "backfill_provenance_tiers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bf = _load()

ROWS = [
    {"id": "gap-1", "record_type": "gap", "type": "gap", "text": "unverified wiring", "source": "manual"},
    {"id": "reason-1", "record_type": "inferred", "text": "[reasoned] redis pool exhaustion", "source": "investigation_reason"},
    {"id": "ideate-1", "record_type": "inferred", "text": "idea", "source": "dt://dt-loci-003/ideate/writer"},
    {"id": "audit-ok", "record_type": "audit", "text": "rc=0", "source": "audit_log", "tool": "pytest"},
    {"id": "audit-llm", "record_type": "audit", "text": "summary", "source": "audit_log", "tool": "llm_local"},
    {"id": "audit-notool", "record_type": "audit", "text": "?", "source": "audit_log"},
    {"id": "human-1", "record_type": "observed", "text": "I saw it", "source": "manual", "authored_by": "human:rj"},
    {"id": "plain-1", "record_type": "observed", "text": "port 443 open", "source": "hunt://x"},
    {"id": "tagged-1", "record_type": "gap", "text": "x", "source": "manual", "evidence_provenance_tier": "tool_verified"},
    {"id": "conflict-1", "record_type": "gap", "text": "x", "source": "manual", "authored_by": "human"},
    {"id": "access-1", "record_type": "access", "query": "q"},
]


class BackfillTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._td.name) / "memory-sessions"
        self.inv = self.root / "case-a"
        self.inv.mkdir(parents=True)
        lines = [json.dumps(r) for r in ROWS] + ['{"id": "torn", "text": "half']
        (self.inv / "findings.jsonl").write_text("\n".join(lines) + "\n")
        self.findings_before = (self.inv / "findings.jsonl").read_bytes()
        undef = self.root / "undefined"
        undef.mkdir()
        (undef / "findings.jsonl").write_text(json.dumps(ROWS[0]) + "\n")

    def tearDown(self):
        self._td.cleanup()

    def _run(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = bf.main(["--memory-dir", str(self.root), "--json", *args])
        self.assertEqual(rc, 0)
        return json.loads(buf.getvalue())

    def _tags(self):
        path = self.inv / bf.UPDATES
        if not path.exists():
            return {}
        return {r["finding_id"]: r for r in map(json.loads, path.read_text().splitlines())
                if r.get("record_type") == "provenance_tier"}

    def test_rules_are_unambiguous_only(self):
        verdicts = {r["id"]: bf.classify(r) for r in ROWS if r["record_type"] != "access"}
        tier = lambda fid: verdicts[fid].get("tier")  # noqa: E731
        self.assertEqual(tier("gap-1"), "model_asserted")
        self.assertEqual(tier("reason-1"), "model_asserted")
        self.assertEqual(set(verdicts["reason-1"]["rules"]), {"model_writer_source", "reasoned_text_marker"})
        self.assertEqual(tier("ideate-1"), "model_asserted")
        self.assertEqual(tier("audit-ok"), "tool_verified")
        self.assertEqual(tier("audit-llm"), "model_asserted")
        self.assertEqual(tier("human-1"), "human_authored")
        self.assertEqual(verdicts["audit-notool"], {"action": "skip", "reason": "no_unambiguous_rule"})
        self.assertEqual(verdicts["plain-1"], {"action": "skip", "reason": "no_unambiguous_rule"})
        self.assertEqual(verdicts["tagged-1"], {"action": "skip", "reason": "explicit_tier"})
        self.assertEqual(verdicts["conflict-1"]["reason"], "conflict")

    def test_dry_run_reports_per_rule_counts_and_writes_nothing(self):
        report = self._run()
        self.assertEqual(report["mode"], "dry-run")
        self.assertEqual(report["would_tag"], 6)
        self.assertEqual(report["per_rule"]["finding_type_assumed_or_gap"], 1)
        self.assertEqual(report["per_rule"]["model_writer_source"], 1)
        self.assertEqual(report["per_rule"]["reasoned_text_marker"], 1)
        self.assertEqual(report["per_rule"]["deep_think_ideate_source"], 1)
        self.assertEqual(report["per_rule"]["audit_receipt_tool"], 2)
        self.assertEqual(report["per_rule"]["explicit_human_author"], 1)
        self.assertEqual(report["per_tier"], {"model_asserted": 4, "tool_verified": 1, "human_authored": 1})
        self.assertEqual(report["skipped"], {"no_unambiguous_rule": 2, "explicit_tier": 1, "conflict": 1})
        self.assertEqual([d["investigation"] for d in report["ignored_dirs"]], ["undefined"])
        self.assertFalse((self.inv / bf.UPDATES).exists())
        self.assertFalse((self.root.parent / "backups").exists())
        self.assertEqual((self.inv / "findings.jsonl").read_bytes(), self.findings_before)

    def test_apply_appends_backs_up_and_never_rewrites_findings(self):
        (self.inv / bf.UPDATES).write_text(json.dumps({"record_type": "other", "x": 1}) + "\n")
        prior = (self.inv / bf.UPDATES).read_text()
        backup = pathlib.Path(self._td.name) / "bk"
        report = self._run("--apply", "--backup-dir", str(backup))
        self.assertEqual(report["applied"]["appended"], 6)
        self.assertEqual((backup / "case-a" / bf.UPDATES).read_text(), prior)
        self.assertEqual(len((backup / "case-a" / "planned.jsonl").read_text().splitlines()), 6)
        self.assertIn("ROLLBACK", (backup / "ROLLBACK.txt").read_text().upper())
        text = (self.inv / bf.UPDATES).read_text()
        self.assertTrue(text.startswith(prior), "existing lines must be kept, append-only")
        tags = self._tags()
        self.assertEqual(tags["audit-ok"]["evidence_provenance_tier"], "tool_verified")
        self.assertEqual(tags["gap-1"]["rule"], "finding_type_assumed_or_gap")
        self.assertNotIn("plain-1", tags)
        self.assertNotIn("tagged-1", tags)
        self.assertEqual((self.inv / "findings.jsonl").read_bytes(), self.findings_before)
        self.assertFalse((self.root / "undefined" / bf.UPDATES).exists())

    def test_rerun_is_a_no_op(self):
        self._run("--apply", "--backup-dir", str(pathlib.Path(self._td.name) / "bk1"))
        before = (self.inv / bf.UPDATES).read_text()
        report = self._run("--apply", "--backup-dir", str(pathlib.Path(self._td.name) / "bk2"))
        self.assertEqual(report["would_tag"], 0)
        self.assertEqual(report["skipped"].get("already_backfilled"), 6)
        self.assertEqual((self.inv / bf.UPDATES).read_text(), before)
        self.assertNotIn("applied", report)

    def test_records_are_read_back_by_the_server_fold(self):
        sys.path.insert(0, str(_SCRIPTS.parent / "mcp"))
        import inv_store

        self._run("--apply", "--backup-dir", str(pathlib.Path(self._td.name) / "bk"))
        orig = inv_store._get_memory_dir
        inv_store._get_memory_dir = lambda: self.root
        try:
            rows = inv_store._fold_provenance_overrides(
                inv_store._read_jsonl(self.inv / "findings.jsonl"), "case-a")
        finally:
            inv_store._get_memory_dir = orig
        by_id = {r.get("id"): r for r in rows}
        self.assertEqual(by_id["reason-1"]["evidence_provenance_tier"], "model_asserted")
        self.assertFalse(by_id["reason-1"]["provenance_defaulted"])
        self.assertNotIn("provenance_source", by_id["plain-1"])
        self.assertEqual(by_id["tagged-1"]["evidence_provenance_tier"], "tool_verified")


if __name__ == "__main__":
    unittest.main()

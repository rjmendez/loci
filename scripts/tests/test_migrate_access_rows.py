"""migrate_access_rows moves legacy access rows from findings.jsonl to access.jsonl.

Every case runs against a temp store, never the live one. The script must:
report without writing by default, back up before it writes, keep every
non-access line byte-for-byte (unparseable ones included), and be safe to rerun,
including after a crash between writing access.jsonl and findings.jsonl.
"""
import importlib.util
import json
import pathlib
import tempfile
import unittest

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("migrate_access_rows", _SCRIPTS / "migrate_access_rows.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mig = _load()

F1 = json.dumps({"id": "f1", "record_type": "observed", "text": "real one"})
F2 = json.dumps({"id": "f2", "type": "inferred", "text": "real two"})
A1 = json.dumps({"id": "f1", "investigation_id": "inv", "record_type": "access",
                 "last_accessed": 1, "query": "q"})
A2 = json.dumps({"id": "f2", "investigation_id": "inv", "record_type": "access",
                 "last_accessed": 2, "query": "q2"})
TORN = '{"id": "torn", "text": "half'


class MigrateAccessRowsTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._td.name)
        self.inv = self.root / "inv"
        self.inv.mkdir()
        self.findings = self.inv / "findings.jsonl"
        self.findings.write_text("\n".join([F1, A1, TORN, F2, A2]) + "\n")
        self.original = self.findings.read_text()
        clean = self.root / "clean"
        clean.mkdir()
        (clean / "findings.jsonl").write_text(F1 + "\n")

    def tearDown(self):
        self._td.cleanup()

    def _files(self):
        return sorted(p.name for p in self.inv.iterdir())

    def test_dry_run_is_the_default_and_writes_nothing(self):
        before = self._files()
        self.assertEqual(mig.main(["--memory-dir", str(self.root)]), 0)
        self.assertEqual(self.findings.read_text(), self.original)
        self.assertEqual(self._files(), before)
        report = mig.run(self.root, apply=False)
        self.assertEqual(report["mode"], "dry-run")
        self.assertEqual(report["access_rows"], 2)
        self.assertEqual(report["investigations_with_access_rows"], 1)

    def test_apply_moves_rows_backs_up_and_keeps_other_lines_verbatim(self):
        report = mig.run(self.root, apply=True)
        self.assertEqual(report["access_rows"], 2)
        self.assertEqual(self.findings.read_text().splitlines(), [F1, TORN, F2])
        self.assertEqual((self.inv / "access.jsonl").read_text().splitlines(), [A1, A2])
        backups = [p for p in self.inv.iterdir() if p.name.startswith("findings.jsonl.bak-")]
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), self.original)
        self.assertEqual((self.root / "clean" / "findings.jsonl").read_text(), F1 + "\n")
        self.assertFalse((self.root / "clean" / "access.jsonl").exists())

    def test_rerun_is_a_no_op(self):
        mig.run(self.root, apply=True)
        snapshot = {p.name: p.read_text() for p in self.inv.iterdir() if p.is_file()}
        report = mig.run(self.root, apply=True)
        self.assertEqual(report["access_rows"], 0)
        self.assertEqual({p.name: p.read_text() for p in self.inv.iterdir() if p.is_file()}, snapshot)

    def test_rerun_after_crash_between_writes_does_not_duplicate(self):
        # Simulate a run that wrote access.jsonl and then died before rewriting findings.jsonl.
        (self.inv / "access.jsonl").write_text(A1 + "\n" + A2 + "\n")
        mig.run(self.root, apply=True)
        self.assertEqual((self.inv / "access.jsonl").read_text().splitlines(), [A1, A2])
        self.assertEqual(self.findings.read_text().splitlines(), [F1, TORN, F2])

    def test_appends_after_existing_access_log(self):
        newer = json.dumps({"id": "f1", "record_type": "access", "last_accessed": 9, "query": "new"})
        (self.inv / "access.jsonl").write_text(newer + "\n")
        mig.run(self.root, apply=True)
        self.assertEqual((self.inv / "access.jsonl").read_text().splitlines(), [newer, A1, A2])

    def test_single_investigation_and_bad_id(self):
        report = mig.run(self.root, apply=True, investigation="clean")
        self.assertEqual(report["access_rows"], 0)
        self.assertEqual(self.findings.read_text(), self.original)
        self.assertEqual(mig.main(["--memory-dir", str(self.root), "--investigation", "../x"]), 2)


if __name__ == "__main__":
    unittest.main()

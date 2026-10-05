"""The global audit log of a test's temp store stays inside the hermetic root.

server.audit_log writes the daily global log to MEMORY_DIR.parent / "audit".
Dozens of tests point MEMORY_DIR at a bare tempfile.mkdtemp(), whose parent was
the machine-wide /tmp, so every run appended to /tmp/audit/<date>.jsonl on the
shared host -- outside any cleanup, and read back by the audit lane of later
runs. loci_hermetic now moves the process temp dir under its own root.
"""
import json
import os
import sys
import tempfile
from pathlib import Path


def _hermetic_root() -> Path:
    return Path(sys.modules["loci_hermetic"].root()).resolve()


def test_mkdtemp_lands_under_the_hermetic_root():
    made = Path(tempfile.mkdtemp()).resolve()
    assert made.parent == _hermetic_root() / "tmp"
    assert Path(os.environ["TMPDIR"]).resolve() == _hermetic_root() / "tmp"


def test_audit_log_for_a_mkdtemp_store_is_written_inside_the_hermetic_root(monkeypatch):
    import server

    store = Path(tempfile.mkdtemp())
    monkeypatch.setattr(server, "MEMORY_DIR", store)

    out = json.loads(server.audit_log("hermetic_probe", "{}", "probe output"))

    assert out["logged"] is True
    day_files = sorted((store.parent / "audit").glob("*.jsonl"))
    assert day_files, "audit_log wrote no global log next to the store"
    rows = [json.loads(line) for f in day_files for line in f.read_text().splitlines() if line]
    assert any(row.get("tool") == "hermetic_probe" for row in rows)
    for f in day_files:
        assert f.resolve().is_relative_to(_hermetic_root()), f

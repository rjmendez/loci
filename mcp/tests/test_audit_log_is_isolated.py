"""The audit log a test writes must never be the operator's.

mcp/memcheck defaults its audit log to ~/.hermes/memcheck-audit.jsonl, and
test_daemon.py drives the writer with no isolation of its own. Harmless while
the writer only appended; a size-based rotation added to that writer on
2026-09-01 made one `pytest mcp/tests` run take the real log from 9,588,755
bytes / 49,412 records back to 2026-08-23 -- about 38,700 records over 62 days,
with no backup, on a machine where that log is the only record of what memcheck
has seen.

A clean runner has no such file, so nothing about this is visible in CI.
"""
import os
import pathlib

from memcheck import cli as memcheck_cli  # noqa: E402  (conftest puts mcp/ on sys.path)


def test_the_audit_path_is_not_the_operators():
    home = pathlib.Path.home().resolve()
    path = memcheck_cli.audit_log_path().resolve()
    assert not str(path).startswith(str(home / ".hermes")), (
        f"tests resolve the audit log to {path}, inside the operator's home"
    )


def test_the_isolation_is_an_env_override_not_a_monkeypatched_internal():
    """Pins the mechanism: memcheck resolves the path per call from the
    environment, so a subprocess or a re-import stays isolated too."""
    assert os.environ.get("MEMCHECK_AUDIT_LOG"), (
        "the autouse fixture in conftest.py is gone; every test that reaches "
        "_append_audit_line is writing to ~/.hermes again"
    )


def test_writing_an_audit_line_lands_in_the_isolated_file():
    memcheck_cli._append_audit_line({"ts": "x", "tool_name": "test"})
    written = pathlib.Path(os.environ["MEMCHECK_AUDIT_LOG"])
    assert written.exists() and "test" in written.read_text()

"""scripts/mnemosyne_activity_check.py — a broken probe must break the silence.

The script's contract is 'output only when working_memory grew; silent when
idle', and a cron agent reads that silence as 'nothing to do'. Both probes used
to return 0 on failure, which is a real count that can never exceed the stored
high-water mark — so a dead mnemosyne CLI or a moved bank DB rendered as idle,
forever, because the high-water mark is only refreshed inside the growth branch.

The module does all of its work at import time, so each test execs a fresh copy
under a controlled HOME. Stdlib only: the test-scripts CI job installs pytest,
pytest-timeout and aiohttp and nothing else.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import sqlite3
import sys
from unittest import mock

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "mnemosyne_activity_check.py"


def run_script(home: pathlib.Path) -> str:
    """Exec a fresh copy of the script with HOME redirected; return its stdout."""
    out = io.StringIO()
    with mock.patch.dict(os.environ, {"HOME": str(home)}, clear=False):
        spec = importlib.util.spec_from_file_location("_mnem_activity_uut", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(out):
            spec.loader.exec_module(mod)
    sys.modules.pop("_mnem_activity_uut", None)
    return out.getvalue()


def _make_bank(home: pathlib.Path, rows: int) -> None:
    """Create the dama-gotchi bank DB the script probes, with `rows` entries."""
    db = home / ".hermes/mnemosyne/data/banks/dama-gotchi/mnemosyne.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE working_memory (id INTEGER PRIMARY KEY)")
    conn.executemany("INSERT INTO working_memory (id) VALUES (?)",
                     [(i,) for i in range(rows)])
    conn.commit()
    conn.close()


def _state(home: pathlib.Path, name: str, value: int) -> None:
    p = home / ".hermes/mnemosyne" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(value))


def test_unreadable_banks_warn_instead_of_reporting_idle(tmp_path):
    """Neither probe can run (no mnemosyne CLI, no bank DB under this HOME) and
    both high-water marks are positive. Reporting 0 for both left this silent."""
    _state(tmp_path, "last_wm_count.txt", 1413)
    _state(tmp_path, "last_wm_count_dama-gotchi.txt", 52)

    out = run_script(tmp_path)

    assert out.strip(), "a probe that could not run must not read as 'idle'"
    assert "WARNING" in out
    assert "default" in out and "dama-gotchi" in out
    assert "NOT idle" in out


def test_one_dead_probe_does_not_suppress_the_other_bank_growth(tmp_path):
    """The mnemosyne CLI is missing but the dama-gotchi bank grew: the growth
    line must still be emitted, alongside the warning for the dead probe."""
    _make_bank(tmp_path, rows=60)
    _state(tmp_path, "last_wm_count_dama-gotchi.txt", 52)

    out = run_script(tmp_path)

    assert "working_memory grew: dama-gotchi: 52->60 (+8)" in out
    assert "Run mnemosyne_sleep now." in out
    assert "WARNING" in out and "default" in out


def test_a_measured_zero_is_still_silent(tmp_path):
    """0 keeps meaning 'the bank really is empty'. The bank DB is present and
    holds nothing, the mark is 0 — that is genuinely idle, and stays quiet."""
    _make_bank(tmp_path, rows=0)
    _state(tmp_path, "last_wm_count_dama-gotchi.txt", 0)
    # Give the default bank a working CLI so it has nothing to say either.
    cli = tmp_path / ".hermes/hermes-agent/venv/bin/mnemosyne"
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text("#!/bin/sh\necho '  Working memory: 7'\n")
    cli.chmod(0o755)
    _state(tmp_path, "last_wm_count.txt", 7)

    assert run_script(tmp_path) == ""


def test_growth_is_reported_and_the_high_water_mark_is_advanced(tmp_path):
    _make_bank(tmp_path, rows=99)
    _state(tmp_path, "last_wm_count_dama-gotchi.txt", 90)
    cli = tmp_path / ".hermes/hermes-agent/venv/bin/mnemosyne"
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text("#!/bin/sh\necho '  Working memory: 2010'\n")
    cli.chmod(0o755)
    _state(tmp_path, "last_wm_count.txt", 1413)

    out = run_script(tmp_path)

    assert "default: 1413->2010 (+597)" in out
    assert "dama-gotchi: 90->99 (+9)" in out
    assert "WARNING" not in out
    assert (tmp_path / ".hermes/mnemosyne/last_wm_count.txt").read_text() == "2010"
    # Second run over the same state is idle again.
    assert run_script(tmp_path) == ""

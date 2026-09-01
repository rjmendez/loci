"""glymphatic_sweep.sweep_orphans — the orphan TTL must not depend on host timezone.

working_memory.created_at is naive UTC (SQLite CURRENT_TIMESTAMP), but it is
compared against time.time(), a UTC epoch. Taking .timestamp() on the naive
datetime reads it as local wall-clock, so the DELETE gate is wrong by the host's
UTC offset: rows survive past their TTL west of UTC and are deleted early east
of it. These tests force both signs of offset so a UTC CI host cannot hide it.
"""

import importlib.util
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_sweep():
    spec = importlib.util.spec_from_file_location(
        "glymphatic_sweep_under_test", REPO / "scripts" / "glymphatic_sweep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sweep():
    return _load_sweep()


@pytest.fixture
def host_tz():
    """Pin the host timezone for one test, then restore it."""
    original = os.environ.get("TZ")

    def _set(posix_tz):
        os.environ["TZ"] = posix_tz
        time.tzset()

    yield _set

    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


def _make_db(path, age_hours):
    """One un-recalled, un-linked row created `age_hours` ago, in sqlite's format."""
    created = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE working_memory ("
        " id TEXT PRIMARY KEY, content TEXT, importance REAL,"
        " created_at TEXT, recall_count INTEGER)"
    )
    conn.execute("CREATE TABLE graph_edges (source TEXT, target TEXT)")
    conn.execute(
        "INSERT INTO working_memory VALUES (?,?,?,?,?)",
        ("row-1", "c", 0.5, created.strftime("%Y-%m-%d %H:%M:%S"), 0),
    )
    conn.commit()
    conn.close()
    return str(path)


def _surviving_ids(db_path):
    conn = sqlite3.connect(db_path)
    ids = [r[0] for r in conn.execute("SELECT id FROM working_memory")]
    conn.close()
    return ids


def test_row_inside_ttl_survives_east_of_utc(sweep, host_tz, tmp_path, monkeypatch):
    # UTC+10. Reading naive UTC as local makes the row look 10h OLDER than it is,
    # so a row 5h short of its TTL is deleted early -- silent data loss.
    host_tz("XXX-10")
    db = _make_db(tmp_path / "east.db", age_hours=7 * 24 - 5)
    monkeypatch.setattr(sweep, "DB_PATH", db)
    monkeypatch.setattr(sweep, "ORPHAN_TTL_DAYS", 7.0)

    assert sweep.sweep_orphans(dry_run=False) == 0
    assert _surviving_ids(db) == ["row-1"]


def test_row_past_ttl_is_removed_west_of_utc(sweep, host_tz, tmp_path, monkeypatch):
    # UTC-4. Reading naive UTC as local makes the row look 4h YOUNGER, so a row
    # 2h past its TTL is kept and the sweep silently under-collects.
    host_tz("XXX4")
    db = _make_db(tmp_path / "west.db", age_hours=7 * 24 + 2)
    monkeypatch.setattr(sweep, "DB_PATH", db)
    monkeypatch.setattr(sweep, "ORPHAN_TTL_DAYS", 7.0)

    assert sweep.sweep_orphans(dry_run=False) == 1
    assert _surviving_ids(db) == []


def test_offset_aware_created_at_still_parses(sweep, host_tz, tmp_path, monkeypatch):
    # Rows written with an explicit offset must keep working unchanged.
    host_tz("XXX-10")
    created = datetime.now(timezone.utc) - timedelta(days=30)
    db = tmp_path / "aware.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE working_memory ("
        " id TEXT PRIMARY KEY, content TEXT, importance REAL,"
        " created_at TEXT, recall_count INTEGER)"
    )
    conn.execute("CREATE TABLE graph_edges (source TEXT, target TEXT)")
    conn.execute("INSERT INTO working_memory VALUES (?,?,?,?,?)",
                 ("row-1", "c", 0.5, created.isoformat(), 0))
    conn.commit()
    conn.close()

    monkeypatch.setattr(sweep, "DB_PATH", str(db))
    monkeypatch.setattr(sweep, "ORPHAN_TTL_DAYS", 7.0)

    assert sweep.sweep_orphans(dry_run=False) == 1

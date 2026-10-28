"""A corrupt LadybugDB database or write-ahead log must never kill the Loci server again.

2026-10-26 04:39: an interrupted checkpoint left a corrupt ``graph.ladybug.wal``; ladybug 0.20.4 then died natively
(0xC0000409 in ``_lbug``) on every open, 24 crashes in a watchdog loop. These tests pin the four defences:

1. recovery open: a read-only open refused with "checkpoint is in progress" is repaired by one read-write open;
2. WAL quarantine: a failed open with a WAL on disk moves the WAL aside (never deletes) and retries once;
3. subprocess probe: the real database is opened read-only in a child first; a native crash or hang there means the
   database is never opened in this process;
4. health: the probe state, crash reason and exit code, quarantine and recovery counts, engine version.

The engine is a recording fake (``make_engine``); the clock, the probe runner, the lease and the UTC time are injected.
One test drives the real engine and kills a real writer mid-write.

Mutation runs (each applied to ladybug_store.py or server.py, confirmed red, reverted):
- _heal_open: drop the ``_is_checkpoint_error`` match (checkpoint = False) -> test_checkpoint_error_is_repaired_by_one_read_write_open red
- _heal_open: skip the exclusive-lease acquisition -> test_recovery_never_opens_read_write_while_another_process_holds_the_lease red
- _heal_open: drop the RECOVERY_MIN_INTERVAL_S check -> test_recovery_is_limited_to_one_per_interval red
- _heal_open: remove the ``_recovery_episode`` guard -> test_recovery_is_logged_once_per_episode red
- _heal_open: drop ``_is_lock_error`` from the quarantine guard -> test_a_foreign_file_lock_is_not_quarantined red
- _quarantine_wal: shutil.move -> shutil.copy2 (WAL stays in place) -> test_failed_open_with_a_wal_quarantines_it_and_retries red
- _quarantine_wal: swallow the move error silently (no last_quarantine_error) -> test_a_failed_move_is_reported_not_swallowed red
- _prune_quarantine: keep 6 -> test_only_the_newest_five_quarantine_sets_are_kept red
- _native_guard: ignore the cache -> test_a_clean_probe_is_cached_for_exactly_its_ttl red
- _native_guard: ``return True`` when crashed -> test_a_crashing_probe_blocks_the_in_process_open red
- _native_guard: skip the re-probe after the quarantine -> test_a_quarantine_that_cures_the_crash_unblocks_the_store red
- _is_native_crash: drop the 0xC0000409 member -> test_exit_codes_are_classified_and_shown_signed red
- _probe_once: treat a timeout as clean -> test_a_hung_probe_blocks_the_in_process_open red
- _native_guard: probe even when no database exists -> test_a_missing_database_is_not_probed red
- available(): drop ``and self._native_guard()`` -> test_a_crashing_probe_blocks_the_in_process_open red
- server._ladybug_health_state: drop the crashed branch -> test_health_state_reports_a_crashed_store red
"""
import datetime as dt
import json
import logging
import os
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph import ladybug_store as K  # noqa: E402

CKPT = "Runtime exception: Cannot open database in read-only mode while checkpoint is in progress"
CRASH = -1073740791  # 0xC0000409 as Windows tools print it
NOW = dt.datetime(2026, 10, 28, 12, 0, 0, 123456, tzinfo=dt.timezone.utc)
STAMP = "20261028T120000123456Z"


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_engine(behavior=None):
    """Recording stand-in for ``ladybug``. ``behavior(read_only, n)`` may raise for the n-th Database open."""
    calls = []

    class Database:
        def __init__(self, path, read_only=False):
            calls.append(("open", str(path), bool(read_only)))
            if behavior:
                behavior(read_only, len(calls))
            self.path = path

        def close(self):
            calls.append(("close", str(self.path)))

    class Connection:
        def __init__(self, db):
            self.db = db

        def execute(self, query, params=None):
            calls.append(("exec", query))

        def close(self):
            pass

    return types.SimpleNamespace(Database=Database, Connection=Connection), calls


def opens(calls):
    return [(c[2]) for c in calls if c[0] == "open"]  # read_only flag of each open, in order


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    def build(behavior=None, probe=False, wal=None, db=True):
        fake, calls = make_engine(behavior)
        monkeypatch.setattr(K, "ladybug", fake)
        monkeypatch.setattr(K, "_HAS_LADYBUG", True)
        path = tmp_path / "graph.ladybug"
        if db:
            path.write_bytes(b"db")
        for suffix, data in (wal or {}).items():
            (tmp_path / f"graph.ladybug{suffix}").write_bytes(data)
        store = K.LadybugStore(str(path))
        store.probe_enabled = probe
        store._now = lambda: NOW
        store._clock = FakeClock()
        return store, calls
    return build


def read_session(store):
    with store._session(write=False) as conn:
        return conn


def write_session(store):
    with store._session(write=True) as conn:
        return conn


# --------------------------------------------------------------------------- #
# 1. recovery open
# --------------------------------------------------------------------------- #
def ckpt_until_recovered():
    """Read-only opens raise the checkpoint error until a read-write open has happened."""
    state = {"recovered": False}

    def behavior(read_only, n):
        if not read_only:
            state["recovered"] = True
        elif not state["recovered"]:
            raise RuntimeError(CKPT)
    return behavior


def test_checkpoint_error_is_repaired_by_one_read_write_open(store_factory):
    store, calls = store_factory(ckpt_until_recovered())
    assert read_session(store) is not None
    assert opens(calls) == [True, False, True]   # RO refused, RW recovery, RO retry
    assert store.recovery_count == 1
    assert store.health_report()["recovery_count"] == 1


def test_a_read_that_needs_no_recovery_opens_once(store_factory):
    store, calls = store_factory()
    assert read_session(store) is not None
    assert opens(calls) == [True]
    assert store.recovery_count == 0


def test_another_read_error_is_not_a_recovery(store_factory):
    def behavior(read_only, n):
        raise RuntimeError("Runtime exception: disk is full")
    store, calls = store_factory(behavior)
    assert read_session(store) is None
    assert opens(calls) == [True]
    assert store.recovery_count == 0
    assert store.last_open_error == "RuntimeError('Runtime exception: disk is full')"


def test_recovery_never_opens_read_write_while_another_process_holds_the_lease(store_factory):
    store, calls = store_factory(ckpt_until_recovered())
    leases = []

    def lease(fd, exclusive, timeout_s=None):
        leases.append(exclusive)
        return not exclusive      # a live writer refuses every exclusive request, readers share
    store._acquire_lease = lease
    assert read_session(store) is None
    assert opens(calls) == [True]                       # no read-write open happened
    assert leases == [False, True, False]               # shared, exclusive refused, shared taken back
    assert store.recovery_count == 0
    # twin: the same store recovers once the exclusive lease is available
    store._acquire_lease = lambda fd, exclusive, timeout_s=None: True
    assert read_session(store) is not None
    assert opens(calls) == [True, True, False, True]
    assert store.recovery_count == 1


def test_recovery_is_limited_to_one_per_interval(store_factory):
    def behavior(read_only, n):
        if read_only:
            raise RuntimeError(CKPT)   # recovery never cures it, so every read re-enters the recovery path
    store, calls = store_factory(behavior)
    clock = store._clock
    assert read_session(store) is None
    assert opens(calls) == [True, False, True]
    clock.t += 59.9
    assert read_session(store) is None
    assert opens(calls) == [True, False, True, True]           # inside the interval: the read fails, no recovery
    clock.t += 0.1                                              # 60.0 s after the first recovery
    assert read_session(store) is None
    assert opens(calls) == [True, False, True, True, True, False, True]
    assert store.recovery_count == 2


def test_recovery_is_logged_once_per_episode(store_factory, caplog):
    flags = {"broken": True}

    def behavior(read_only, n):
        if read_only and flags["broken"]:
            raise RuntimeError(CKPT)
    store, calls = store_factory(behavior)
    caplog.set_level(logging.WARNING, logger="loci-mcp.ladybug")
    for _ in range(3):
        read_session(store)
        store._clock.t += 61
    episode = [r for r in caplog.records if "leftover checkpoint" in r.getMessage()]
    assert len(episode) == 1
    flags["broken"] = False                          # a read opens: the episode is over
    assert read_session(store) is not None
    flags["broken"] = True                           # a new episode logs again
    read_session(store)
    episode = [r for r in caplog.records if "leftover checkpoint" in r.getMessage()]
    assert len(episode) == 2


# --------------------------------------------------------------------------- #
# 2. WAL quarantine
# --------------------------------------------------------------------------- #
def fail_while_wal_present(tmp_path):
    def behavior(read_only, n):
        if (tmp_path / "graph.ladybug.wal").exists():
            raise RuntimeError("Runtime exception: corrupted wal file")
    return behavior


def test_failed_open_with_a_wal_quarantines_it_and_retries(tmp_path, store_factory, caplog):
    store, calls = store_factory(fail_while_wal_present(tmp_path),
                                 wal={".wal": b"1234567", ".wal.checkpoint": b"123"})
    caplog.set_level(logging.WARNING, logger="loci-mcp.ladybug")
    assert write_session(store) is not None
    assert opens(calls) == [False, False]
    q = tmp_path / "quarantine"
    assert sorted(p.name for p in q.iterdir()) == [f"graph.ladybug.wal.{STAMP}", f"graph.ladybug.wal.checkpoint.{STAMP}"]
    assert (q / f"graph.ladybug.wal.{STAMP}").read_bytes() == b"1234567"      # moved, not deleted or truncated
    assert (q / f"graph.ladybug.wal.checkpoint.{STAMP}").read_bytes() == b"123"
    assert not (tmp_path / "graph.ladybug.wal").exists()
    assert not (tmp_path / "graph.ladybug.wal.checkpoint").exists()
    assert store.quarantine_count == 1
    msg = [r.getMessage() for r in caplog.records if "WAL quarantined" in r.getMessage()]
    assert len(msg) == 1 and "'graph.ladybug.wal': 7" in msg[0] and "'graph.ladybug.wal.checkpoint': 3" in msg[0]


def test_a_failed_open_without_a_wal_quarantines_nothing(tmp_path, store_factory):
    def behavior(read_only, n):
        raise RuntimeError("Runtime exception: corrupted wal file")
    store, calls = store_factory(behavior)
    assert write_session(store) is None
    assert opens(calls) == [False]
    assert store.quarantine_count == 0
    assert not (tmp_path / "quarantine").exists()


def test_a_read_session_quarantines_too(tmp_path, store_factory):
    store, calls = store_factory(fail_while_wal_present(tmp_path), wal={".wal": b"xx"})
    assert read_session(store) is not None
    assert opens(calls) == [True, True]
    assert store.quarantine_count == 1
    assert [p.name for p in (tmp_path / "quarantine").iterdir()] == [f"graph.ladybug.wal.{STAMP}"]


def test_a_foreign_file_lock_is_not_quarantined(tmp_path, store_factory):
    """Another process holding the database's own lock has a LIVE wal; moving it would lose its writes."""
    def behavior(read_only, n):
        raise RuntimeError("IO exception: Could not set lock on file : graph.ladybug")
    store, calls = store_factory(behavior, wal={".wal": b"live"})
    assert write_session(store) is None
    assert opens(calls) == [False]
    assert (tmp_path / "graph.ladybug.wal").read_bytes() == b"live"
    assert store.quarantine_count == 0


def test_only_the_newest_five_quarantine_sets_are_kept(tmp_path, store_factory):
    q = tmp_path / "quarantine"
    q.mkdir()
    old = [f"2026100{d}T000000000000Z" for d in range(1, 7)]          # six older sets
    for stamp in old:
        (q / f"graph.ladybug.wal.{stamp}").write_bytes(b"old")
    store, calls = store_factory(fail_while_wal_present(tmp_path), wal={".wal": b"new"})
    assert write_session(store) is not None
    assert sorted(p.name for p in q.iterdir()) == sorted(
        [f"graph.ladybug.wal.{STAMP}"] + [f"graph.ladybug.wal.{s}" for s in old[2:]])


def test_a_failed_move_is_reported_not_swallowed(tmp_path, store_factory, monkeypatch, caplog):
    store, calls = store_factory(fail_while_wal_present(tmp_path), wal={".wal": b"abc"})
    moves = []

    def move(src, dst):
        moves.append((src, dst))
        raise OSError("disk full")
    monkeypatch.setattr(K.shutil, "move", move)
    caplog.set_level(logging.ERROR, logger="loci-mcp.ladybug")
    assert write_session(store) is None
    assert len(moves) == 1
    assert store.last_quarantine_error == "OSError('disk full')"
    assert store.quarantine_count == 0
    assert (tmp_path / "graph.ladybug.wal").read_bytes() == b"abc"          # still there, untouched
    assert store.health_report()["last_quarantine_error"] == "OSError('disk full')"
    assert any("quarantine FAILED" in r.getMessage() and "disk full" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# 3. subprocess probe of the real database
# --------------------------------------------------------------------------- #
class Runner:
    """Recording probe runner: answers from a queue of (exit code, output), then repeats the last answer."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, db_path, timeout_s):
        self.calls.append((db_path, timeout_s))
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def test_a_clean_probe_is_cached_for_exactly_its_ttl(store_factory):
    store, calls = store_factory(probe=True)
    store._probe_runner = runner = Runner((0, "5"))
    assert read_session(store) is not None
    assert runner.calls == [(store.db_path, 60.0)]
    assert store.probe_state == "ok"
    store._clock.t += 299.9
    assert read_session(store) is not None
    assert len(runner.calls) == 1                       # cached
    store._clock.t += 0.1                               # 300.0 s
    assert read_session(store) is not None
    assert len(runner.calls) == 2


def test_the_probe_timeout_is_injectable(store_factory):
    store, calls = store_factory(probe=True)
    store.probe_timeout_s = 7.5
    store._probe_runner = runner = Runner((0, "5"))
    assert read_session(store) is not None
    assert runner.calls == [(store.db_path, 7.5)]


def test_a_crashing_probe_blocks_the_in_process_open(store_factory):
    store, calls = store_factory(probe=True)
    store._probe_runner = runner = Runner((CRASH, ""))
    assert read_session(store) is None
    assert write_session(store) is None
    assert [c for c in calls if c[0] == "open"] == []   # never opened in this process
    assert store.probe_state == "crashed"
    assert store.last_crash_exit_code == CRASH
    assert store.last_crash_reason == "engine crashed opening the database (exit -1073740791)"
    assert store.last_open_error == store.last_crash_reason
    assert store.available() is False
    assert len(runner.calls) == 1                       # no WAL to quarantine -> no re-probe; result cached


def test_a_clean_probe_lets_the_store_open_and_be_available(store_factory):
    store, calls = store_factory(probe=True)
    store._probe_runner = Runner((0, "5"))
    assert store.available() is True
    assert read_session(store) is not None
    assert opens(calls) == [False, True]                # engine_probe's throwaway database, then the real read


def test_a_quarantine_that_cures_the_crash_unblocks_the_store(tmp_path, store_factory):
    store, calls = store_factory(probe=True, wal={".wal": b"bad"})
    store._probe_runner = runner = Runner((CRASH, ""), (0, "5"))
    assert read_session(store) is not None
    assert len(runner.calls) == 2                       # crash, quarantine, re-probe
    assert store.quarantine_count == 1
    assert store.probe_state == "ok"
    assert [p.name for p in (tmp_path / "quarantine").iterdir()] == [f"graph.ladybug.wal.{STAMP}"]


def test_a_quarantine_that_does_not_cure_it_stays_blocked(tmp_path, store_factory):
    store, calls = store_factory(probe=True, wal={".wal": b"bad"})
    store._probe_runner = runner = Runner((CRASH, ""))
    assert read_session(store) is None
    assert len(runner.calls) == 2
    assert store.quarantine_count == 1
    assert [c for c in calls if c[0] == "open"] == []
    assert store.probe_state == "crashed"


def test_a_crash_is_reprobed_after_its_shorter_ttl(store_factory):
    store, calls = store_factory(probe=True)
    store._probe_runner = runner = Runner((CRASH, ""), (0, "5"))
    assert read_session(store) is None
    store._clock.t += 19.9
    assert read_session(store) is None
    assert len(runner.calls) == 1
    store._clock.t += 0.1                               # 20.0 s: the engine was repaired meanwhile
    assert read_session(store) is not None
    assert len(runner.calls) == 2
    assert store.probe_state == "ok"


@pytest.mark.parametrize("code,shown", [
    (CRASH, -1073740791),                # signed, as Windows tools print it
    (0xC0000409, -1073740791),           # unsigned, as subprocess returns it on Windows
    (0xC0000005, -1073741819),           # access violation
    (-11, -11),                          # SIGSEGV on POSIX
])
def test_exit_codes_are_classified_and_shown_signed(store_factory, code, shown):
    store, calls = store_factory(probe=True)
    store._probe_runner = Runner((code, ""))
    assert read_session(store) is None
    assert store.last_crash_exit_code == shown
    assert store.last_crash_reason == f"engine crashed opening the database (exit {shown})"


def test_a_hung_probe_blocks_the_in_process_open(store_factory):
    store, calls = store_factory(probe=True)
    store._probe_runner = Runner((None, "no answer"))
    assert read_session(store) is None
    assert store.probe_state == "crashed"
    assert store.last_crash_exit_code is None
    assert store.last_crash_reason == "engine hung opening the database (no answer within 60 s)"
    assert [c for c in calls if c[0] == "open"] == []


def test_a_clean_probe_failure_keeps_the_fail_open_behaviour(store_factory):
    """A clean error in the child (lock held, empty schema) is not a crash: the in-process open decides."""
    store, calls = store_factory(probe=True)
    store._probe_runner = Runner((1, "RuntimeError: Could not set lock on file"))
    assert read_session(store) is not None
    assert opens(calls) == [True]
    assert store.probe_state == "error"
    assert store.last_probe_error == "probe exit 1: RuntimeError: Could not set lock on file"
    assert store.last_crash_reason == ""


def test_a_missing_database_is_not_probed(store_factory):
    store, calls = store_factory(probe=True, db=False)
    store._probe_runner = runner = Runner((CRASH, ""))
    assert read_session(store) is not None              # nothing on disk to crash on: the engine creates it
    assert runner.calls == []
    assert store.probe_state == "no-database"
    # twin: the same store with a database file is probed
    with open(store.db_path, "wb") as f:
        f.write(b"db")
    store._clock.t += 1
    assert read_session(store) is None
    assert len(runner.calls) == 1


def test_a_wal_without_a_database_is_probed(store_factory):
    store, calls = store_factory(probe=True, db=False, wal={".wal": b"x"})
    store._probe_runner = runner = Runner((0, "0"))
    assert read_session(store) is not None
    assert len(runner.calls) == 1


def test_the_real_probe_child_is_isolated_and_gets_the_dll_dirs(monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return types.SimpleNamespace(returncode=0, stdout="5\n", stderr="")
    monkeypatch.setattr(K.subprocess, "run", run)
    monkeypatch.setattr(K, "_DLL_DIRS", ["C:/a", "C:/b"])
    monkeypatch.setattr(K, "ladybug", types.SimpleNamespace(__file__=os.path.join("site", "ladybug", "__init__.py")))
    assert K._run_probe_child("db-path", 12.0) == (0, "5")
    assert seen["cmd"][:3] == [sys.executable, "-I", "-c"] and seen["cmd"][-1] == "db-path"
    assert "read_only=True" in seen["cmd"][3] and "MATCH (n) RETURN count(*)" in seen["cmd"][3]
    assert seen["kw"]["timeout"] == 12.0
    assert seen["kw"]["env"]["LOCI_LADYBUG_DLL_DIRS"] == os.pathsep.join(["C:/a", "C:/b"])
    assert seen["kw"]["env"]["LOCI_LADYBUG_SYS_PATH"] == "site"


def test_the_real_probe_child_reports_a_timeout_as_none(monkeypatch):
    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(K.subprocess, "run", run)
    assert K._run_probe_child("db-path", 3.0) == (None, "no answer within 3 s")


# --------------------------------------------------------------------------- #
# 4. health
# --------------------------------------------------------------------------- #
def test_health_report_of_a_fresh_store(store_factory):
    store, calls = store_factory()
    assert store.health_report() == {
        "engine_version": K._engine_version(),
        "probe_state": "unprobed",
        "last_crash_reason": "",
        "last_crash_exit_code": None,
        "last_probe_error": "",
        "quarantine_count": 0,
        "last_quarantine_error": "",
        "recovery_count": 0,
    }


def test_health_report_after_a_crash_and_a_quarantine(store_factory):
    store, calls = store_factory(probe=True, wal={".wal": b"bad"})
    store._probe_runner = Runner((CRASH, ""))
    read_session(store)
    report = store.health_report()
    assert report["probe_state"] == "crashed"
    assert report["last_crash_reason"] == "engine crashed opening the database (exit -1073740791)"
    assert report["last_crash_exit_code"] == CRASH
    assert report["quarantine_count"] == 1


def test_engine_version_comes_from_package_metadata(monkeypatch):
    from importlib import metadata
    monkeypatch.setattr(metadata, "version", lambda dist: "9.9.9")
    assert K._engine_version() == "9.9.9"

    def missing(dist):
        raise metadata.PackageNotFoundError(dist)
    monkeypatch.setattr(metadata, "version", missing)
    assert K._engine_version() == "unknown"


def test_health_state_reports_a_crashed_store(monkeypatch):
    import server

    class Store:
        def __init__(self, state):
            self.probe_state = state
            self.probes = []

        def readable_probe(self):
            self.probes.append(1)
            return True

    crashed = Store("crashed")
    monkeypatch.setattr(server, "_ladybug_store", crashed)
    assert server._ladybug_health_state() == "crashed"
    ok = Store("ok")
    monkeypatch.setattr(server, "_ladybug_store", ok)
    assert server._ladybug_health_state() == "available"
    assert ok.probes == [1]


def test_loci_health_carries_the_store_detail_and_fails_on_a_crash(monkeypatch):
    import server

    detail = {"engine_version": "0.22.0", "probe_state": "crashed", "quarantine_count": 2, "recovery_count": 1,
              "last_crash_reason": "engine crashed opening the database (exit -1073740791)",
              "last_crash_exit_code": CRASH}
    store = types.SimpleNamespace(health_report=lambda: detail, probe_state="crashed")
    monkeypatch.setattr(server, "_ladybug_store", store)
    monkeypatch.setattr(server, "_ladybug_health_state", lambda: "crashed")
    monkeypatch.setattr(server, "_ladybug_last_error", "")
    out = json.loads(server.loci_health())
    assert out["ladybug"] == "crashed"
    assert out["ladybug_detail"] == detail
    assert out["status"] != "ok"
    assert ("ladybug: engine crashed opening the database (exit -1073740791); "
            "the database is not opened in-process (WAL quarantines: 2)") in out["failures"]
    # twin: a healthy store adds the detail and no failure
    healthy = dict(detail, probe_state="ok", last_crash_reason="", last_crash_exit_code=None)
    monkeypatch.setattr(server, "_ladybug_store", types.SimpleNamespace(health_report=lambda: healthy))
    monkeypatch.setattr(server, "_ladybug_health_state", lambda: "available")
    out = json.loads(server.loci_health())
    assert out["ladybug_detail"] == healthy
    assert not any("ladybug" in f for f in out.get("failures", []))


# --------------------------------------------------------------------------- #
# Real engine: kill a writer mid-write, reopen through the store
# --------------------------------------------------------------------------- #
WRITER = """
import os, sys, time
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from graph.ladybug_store import LadybugStore
s = LadybugStore(sys.argv[3])
n = 0
while True:
    rows = [{"id": "k-%d-%d" % (n, i), "investigation": "kt", "type": "observed", "text": "x" * 400,
             "confidence": "high", "source": "kt", "ts": 1} for i in range(300)]
    s.upsert_findings_batch(rows)
    n += 1
    if n == 1:
        open(sys.argv[4], "w").write("ready")
    if n % 4 == 0:
        try:
            s._exec("CHECKPOINT")
        except Exception:
            pass
"""


@pytest.mark.skipif(not K._HAS_LADYBUG, reason="import ladybug failed: the real-engine kill test needs the engine")
def test_killed_writer_leaves_a_store_that_reads_or_reports_cleanly(tmp_path, monkeypatch):
    import time
    mcp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db = str(tmp_path / "graph.ladybug")
    ready = tmp_path / "ready"
    env = dict(os.environ, LOCI_HOME=str(tmp_path / "home"), LOCI_LADYBUG_PROBE="0")
    proc = subprocess.Popen([sys.executable, "-c", WRITER, mcp_dir, os.path.dirname(mcp_dir), db, str(ready)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 120
        while not ready.exists():
            assert proc.poll() is None, f"writer died early (exit {proc.returncode})"
            assert time.monotonic() < deadline, "writer never committed a batch"
            time.sleep(0.05)
        time.sleep(0.4)          # land inside a later batch or checkpoint
    finally:
        proc.kill()
        proc.wait(timeout=30)
    store = K.LadybugStore(db)
    store.probe_enabled = True
    rows = store._rows("MATCH (f:Finding) RETURN count(f)")      # this process survives the reopen: the point
    report = store.health_report()
    if rows:
        assert isinstance(rows[0][0], int) and rows[0][0] >= 300  # readable: at least the committed first batch
        assert report["probe_state"] in ("ok", "error")
    else:
        assert store.last_open_error != ""                         # a clean, reported error
        assert report["probe_state"] in ("crashed", "error", "ok")

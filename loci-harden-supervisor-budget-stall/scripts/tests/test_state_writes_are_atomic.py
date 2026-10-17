"""A write that dies part-way must not destroy the file it was replacing.

Three scripts persisted state by truncating the live file and writing into it:
the A2A context bridge (~/.hermes/bridge_state.json, rewritten by a 10-minute
systemd timer), ua-watch (~/.ua-watch-state.json, rewritten once per project
inside the scan loop) and event_log.compact (the append-only audit log itself).
Every one of their readers treats a truncated file as "nothing here yet" and
returns an empty default without a word, so the loss is silent and the next run
redoes work it had already done — re-broadcasting to peers, re-ingesting into
Qdrant, or reporting that no memory mutation was ever recorded.

Each test kills the write half-way and asserts the previous contents survived.
Against truncate-in-place they fail; against temp + os.replace they pass.
"""

import importlib.util
import json
import os
import pathlib
import time

import pytest

os.environ.setdefault("LOCI_ENV_FILE", "/nonexistent-env-file-for-tests")
os.environ.setdefault("LOCI_A2A_TOKEN", "t")

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, _SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _half_write_text(self, data, *args, **kwargs):
    """Path.write_text that truncates, writes half the payload and then dies."""
    with open(self, "w") as fh:
        fh.write(data[: len(data) // 2])
    raise OSError("simulated crash mid-write")


class _DiesAfterFirstLine:
    """File object that lets one write() through, then raises."""

    def __init__(self, fh):
        self._fh = fh
        self._writes = 0

    def write(self, s):
        self._writes += 1
        if self._writes > 1:
            raise OSError("simulated crash mid-write")
        return self._fh.write(s)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._fh.close()
        return False


# ── scripts/a2a_context_bridge.py ────────────────────────────────────────────

def test_bridge_state_survives_a_crash_mid_save(tmp_path, monkeypatch):
    bridge = _load("bridge_atomic_uut", "a2a_context_bridge.py")
    state_file = tmp_path / "bridge_state.json"
    monkeypatch.setattr(bridge, "STATE_FILE", str(state_file))

    good = {"last_run": "2026-08-30T12:00:00+00:00", "sent_ids": ["a", "b", "c"]}
    bridge._save_state(good)
    assert bridge._load_state() == good

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pathlib.Path, "write_text", _half_write_text)
        with pytest.raises(OSError):
            bridge._save_state({"last_run": "2026-08-30T12:10:00+00:00",
                                "sent_ids": ["a", "b", "c", "d", "e"]})

    # The watermark and the dedup list are still the ones that were committed.
    assert bridge._load_state() == good


# ── scripts/ua-watch.py ──────────────────────────────────────────────────────

def test_ua_watch_state_survives_a_crash_mid_save(tmp_path, monkeypatch):
    ua = _load("ua_watch_atomic_uut", "ua-watch.py")
    state_file = tmp_path / "ua-watch-state.json"
    monkeypatch.setattr(ua, "STATE_FILE", state_file)

    good = {"/repo/one": {"git_hash": "abc123", "fg_mtime": 1.0}}
    ua.save_state(good)
    assert ua.load_state() == good

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pathlib.Path, "write_text", _half_write_text)
        with pytest.raises(OSError):
            ua.save_state({"/repo/one": {"git_hash": "abc123", "fg_mtime": 1.0},
                           "/repo/two": {"git_hash": "def456", "fg_mtime": 2.0}})

    # Both skip gates still see the recorded hash, so /repo/one is not re-ingested.
    assert ua.load_state() == good


# ── scripts/event_log.py ─────────────────────────────────────────────────────

def test_compact_leaves_the_log_intact_when_the_rewrite_dies(tmp_path, monkeypatch):
    ev = _load("event_log_atomic_uut", "event_log.py")
    log = tmp_path / "event_log.jsonl"
    now = time.time()
    old = {"ts": now - 200 * 86400, "op": "store", "id": "old"}
    recent = [{"ts": now - i, "op": "store", "id": f"r{i}"} for i in range(3)]
    log.write_text("".join(json.dumps(e) + "\n" for e in [old] + recent))
    before = log.read_text()

    real_open = open

    def crashing_open(file, mode="r", *args, **kwargs):
        fh = real_open(file, mode, *args, **kwargs)
        return _DiesAfterFirstLine(fh) if "w" in mode else fh

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "open", crashing_open, raising=False)
        with pytest.raises(OSError):
            ev.compact(before_ts=now - 86400,
                       archive_dir=str(tmp_path / "archive"),
                       log_path=str(log))

    # Every event is still there — including the three the rewrite meant to keep.
    assert log.read_text() == before
    assert len(ev.replay(log_path=str(log))) == 4

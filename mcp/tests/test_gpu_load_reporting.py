"""GPU load signal: Windows-safe path, reported (never silent) degraded states, sidecar parsing."""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gpu_load  # noqa: E402
import gpu_load_sidecar  # noqa: E402

LOGGER = "loci.gpu_load"


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Zero the module's counters/state so each test sees only its own reads."""
    monkeypatch.setattr(gpu_load, "_status", {
        "state": "unread", "path": "", "detail": "",
        "counts": {"ok": 0, "absent": 0, "stale": 0, "unreadable": 0},
        "last_warned": {},
    })


def _snapshot(path: Path, age_s: float = 0.0, util: float = 80.0) -> None:
    path.write_text(json.dumps({
        "timestamp": time.time() - age_s,
        "gpus": [{"index": 0, "name": "RTX", "util_pct": util,
                  "vram_used_mb": 1000, "vram_total_mb": 12000}],
    }), encoding="utf-8")


def _warnings(caplog):
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]


# ---- F10: the default path must exist on a host without os.getuid -------------------------------

def test_live_signal_path_without_getuid_is_in_the_tempdir(monkeypatch):
    monkeypatch.delattr(os, "getuid", raising=False)
    assert gpu_load.live_signal_path() == Path(tempfile.gettempdir()) / "loci_gpu_load.json"


def test_live_signal_path_with_getuid_is_the_per_user_tmpfs(monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: 1234, raising=False)
    assert gpu_load.live_signal_path() == Path("/run/user/1234/loci_gpu_load.json")


def test_signal_path_without_getuid_does_not_raise(monkeypatch):
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.delenv("LOCI_GPU_LOAD_PATH", raising=False)
    assert gpu_load._signal_path() == Path(tempfile.gettempdir()) / "loci_gpu_load.json"


def test_write_then_read_roundtrip_over_an_existing_file(monkeypatch, tmp_path):
    """The sidecar rewrites the file every few seconds; Path.rename would fail on Windows the 2nd time."""
    target = tmp_path / "sig" / "loci_gpu_load.json"
    monkeypatch.setenv("LOCI_GPU_LOAD_PATH", str(target))
    gpu_load.write_gpu_load([gpu_load.GpuEntry(0, "RTX", 10.0, 1000.0, 12000.0)])
    gpu_load.write_gpu_load([gpu_load.GpuEntry(0, "RTX", 90.0, 2000.0, 12000.0)])
    load = gpu_load.read_gpu_load()
    assert load is not None
    assert [(g.util_pct, g.vram_used_mb) for g in load.gpus] == [(90.0, 2000.0)]
    assert not (target.parent / "loci_gpu_load.tmp").exists()


# ---- F10: degraded states are explicit ----------------------------------------------------------

def test_absent_signal_is_reported_counted_and_warned_once(monkeypatch, tmp_path, caplog):
    target = tmp_path / "none.json"
    monkeypatch.setenv("LOCI_GPU_LOAD_PATH", str(target))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert gpu_load.read_gpu_load() is None
        assert gpu_load.read_gpu_load() is None
        health = gpu_load.gpu_load_health()
    assert health == {
        "gpu_load_signal": "absent",
        "path": str(target),
        "detail": "file does not exist; no sidecar is writing it",
        "counts": {"ok": 0, "absent": 3, "stale": 0, "unreadable": 0},
        "age_s": None,
    }
    warned = _warnings(caplog)
    assert len(warned) == 1
    assert warned[0].getMessage() == (
        f"GPU load signal absent at {target} (file does not exist; no sidecar is writing it): "
        "load-aware routing is inactive and Ollama-first routing applies unchanged. "
        "Start mcp/gpu_load_sidecar.py or set LOCI_GPU_LOAD_PATH."
    )


def test_fresh_signal_is_present_with_no_warning(monkeypatch, tmp_path, caplog):
    target = tmp_path / "sig.json"
    _snapshot(target, age_s=0.0)
    monkeypatch.setenv("LOCI_GPU_LOAD_PATH", str(target))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        health = gpu_load.gpu_load_health()
    assert health["gpu_load_signal"] == "present"
    assert health["path"] == str(target)
    assert health["detail"] == ""
    assert health["counts"] == {"ok": 1, "absent": 0, "stale": 0, "unreadable": 0}
    assert 0.0 <= health["age_s"] < 5.0
    assert _warnings(caplog) == []


def test_stale_signal_is_reported_stale(monkeypatch, tmp_path, caplog):
    target = tmp_path / "sig.json"
    _snapshot(target, age_s=60.0)
    monkeypatch.setenv("LOCI_GPU_LOAD_PATH", str(target))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert gpu_load.read_gpu_load() is None
    assert gpu_load._status["state"] == "stale"
    assert gpu_load._status["counts"]["stale"] == 1
    assert gpu_load._status["detail"].startswith("snapshot is 60s old, limit 30s")
    assert len(_warnings(caplog)) == 1


def test_signal_just_inside_the_age_limit_is_present(monkeypatch, tmp_path):
    target = tmp_path / "sig.json"
    _snapshot(target, age_s=20.0)
    monkeypatch.setenv("LOCI_GPU_LOAD_PATH", str(target))
    load = gpu_load.read_gpu_load()
    assert load is not None and load.max_util_pct == 80.0
    assert gpu_load._status["state"] == "ok"


def test_corrupt_signal_is_reported_unreadable(monkeypatch, tmp_path, caplog):
    target = tmp_path / "sig.json"
    target.write_text("not json {{", encoding="utf-8")
    monkeypatch.setenv("LOCI_GPU_LOAD_PATH", str(target))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert gpu_load.read_gpu_load() is None
    assert gpu_load._status["state"] == "unreadable"
    assert gpu_load._status["counts"] == {"ok": 0, "absent": 0, "stale": 0, "unreadable": 1}
    assert gpu_load._status["detail"].startswith("JSONDecodeError(")
    assert len(_warnings(caplog)) == 1


def test_unresolvable_path_is_reported_not_raised(monkeypatch, caplog):
    def boom():
        raise RuntimeError("no home")
    monkeypatch.setattr(gpu_load, "_signal_path", boom)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert gpu_load.read_gpu_load() is None
    assert gpu_load._status["state"] == "unreadable"
    assert gpu_load._status["detail"] == "signal path unresolvable: RuntimeError('no home')"
    assert len(_warnings(caplog)) == 1


def test_warning_is_rate_limited_per_state_then_repeats(monkeypatch, tmp_path, caplog):
    target = tmp_path / "sig.json"
    monkeypatch.setenv("LOCI_GPU_LOAD_PATH", str(target))
    now = [1000.0]
    monkeypatch.setattr(gpu_load, "_monotonic", lambda: now[0])

    with caplog.at_level(logging.INFO, logger=LOGGER):
        gpu_load.read_gpu_load()                      # absent -> WARNING #1
        _snapshot(target)
        gpu_load.read_gpu_load()                      # ok -> INFO "recovered"
        target.unlink()
        now[0] += 10
        gpu_load.read_gpu_load()                      # absent again inside 300 s -> suppressed
        assert len(_warnings(caplog)) == 1
        _snapshot(target)
        gpu_load.read_gpu_load()
        target.unlink()
        now[0] += 289                                 # t=1299, 299 s after the warning -> still suppressed
        gpu_load.read_gpu_load()
        assert len(_warnings(caplog)) == 1
        _snapshot(target)
        gpu_load.read_gpu_load()
        target.unlink()
        now[0] += 1                                   # t=1300, exactly the interval -> allowed again
        gpu_load.read_gpu_load()
    assert len(_warnings(caplog)) == 2
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert infos == [f"GPU load signal recovered at {target}"] * 3


# ---- llm_local: the reader itself being unusable is counted and warned once ---------------------

def test_llm_local_reports_an_unusable_reader_once(monkeypatch, caplog):
    import llm_local

    monkeypatch.setattr(llm_local, "_gpu_load_unavailable", 0)
    monkeypatch.setitem(sys.modules, "gpu_load", None)  # `from gpu_load import ...` now raises ImportError
    with caplog.at_level(logging.WARNING, logger="loci-mcp.llm_local"):
        assert llm_local._read_gpu_load() is None
        assert llm_local._read_gpu_load() is None
    assert llm_local._gpu_load_unavailable == 2
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(msgs) == 1
    assert msgs[0] == (
        "GPU load reader unavailable (ModuleNotFoundError('import of gpu_load halted; None in sys.modules')): "
        "load-aware routing is inactive until it imports"
    )


def test_llm_local_working_reader_is_not_counted(monkeypatch, tmp_path, caplog):
    import llm_local

    target = tmp_path / "sig.json"
    _snapshot(target)
    monkeypatch.setenv("LOCI_GPU_LOAD_PATH", str(target))
    monkeypatch.setattr(llm_local, "_gpu_load_unavailable", 0)
    with caplog.at_level(logging.WARNING, logger="loci-mcp.llm_local"):
        load = llm_local._read_gpu_load()
    assert load is not None and load.max_util_pct == 80.0
    assert llm_local._gpu_load_unavailable == 0
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


# ---- sidecar writer side: parsing canned nvidia-smi output --------------------------------------

CANNED = (
    "0, NVIDIA GeForce RTX 4070 Ti, 37, 5120, 12282\r\n"
    "1, NVIDIA GeForce RTX 2080 Ti, 5, 612, 11264\r\n"
)


def test_parse_nvidia_smi_csv_reads_both_cards():
    got = gpu_load_sidecar.parse_nvidia_smi_csv(CANNED)
    assert [(g.index, g.name, g.util_pct, g.vram_used_mb, g.vram_total_mb) for g in got] == [
        (0, "NVIDIA GeForce RTX 4070 Ti", 37.0, 5120.0, 12282.0),
        (1, "NVIDIA GeForce RTX 2080 Ti", 5.0, 612.0, 11264.0),
    ]


def test_parse_skips_na_and_short_rows_with_a_warning(caplog):
    text = CANNED + "2, Mystery, [N/A], 1, 2\n3, Short\n"
    with caplog.at_level(logging.WARNING, logger="loci.gpu_load_sidecar"):
        got = gpu_load_sidecar.parse_nvidia_smi_csv(text)
    assert [g.index for g in got] == [0, 1]
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 2
    assert msgs[0].startswith("skipping unparseable nvidia-smi row '2, Mystery, [N/A], 1, 2'")
    assert msgs[1] == "skipping short nvidia-smi row '3, Short'"


def test_parse_empty_output_is_empty_without_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="loci.gpu_load_sidecar"):
        assert gpu_load_sidecar.parse_nvidia_smi_csv("\n") == []
    assert caplog.records == []


def test_parsed_entries_round_trip_through_the_signal_file(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCI_GPU_LOAD_PATH", str(tmp_path / "sig.json"))
    gpu_load.write_gpu_load(gpu_load_sidecar.parse_nvidia_smi_csv(CANNED))
    load = gpu_load.read_gpu_load()
    assert load is not None
    assert [(g.index, g.util_pct) for g in load.gpus] == [(0, 37.0), (1, 5.0)]
    assert load.is_loaded() is False


def test_sidecar_failure_warning_is_first_then_every_sixtieth(caplog):
    with caplog.at_level(logging.WARNING, logger="loci.gpu_load_sidecar"):
        n = 0
        for _ in range(120):
            n = gpu_load_sidecar._note_failure(n, "no GPU rows from nvidia-smi")
    assert n == 120
    assert [r.getMessage() for r in caplog.records] == [
        f"GPU load signal not updated ({k} consecutive failed polls): no GPU rows from nvidia-smi"
        for k in (1, 60, 120)
    ]


# ---- pinned by independent mutation runs: the exact staleness boundary and the llm_local reader's log-once rule ----

def test_a_signal_exactly_at_the_age_limit_is_fresh_and_just_over_it_is_stale(monkeypatch, tmp_path):
    path = tmp_path / "gpu.json"
    path.write_text(json.dumps({"timestamp": 1000.0, "gpus": [
        {"index": 0, "name": "RTX", "util_pct": 55.0, "vram_used_mb": 1000, "vram_total_mb": 12000}]}), encoding="utf-8")
    monkeypatch.setenv("LOCI_GPU_LOAD_PATH", str(path))
    assert gpu_load._SIGNAL_MAX_AGE_S == 30.0
    monkeypatch.setattr(gpu_load.time, "time", lambda: 1029.999)
    assert gpu_load.read_gpu_load() is not None and gpu_load._status["state"] == "ok"      # positive twin
    monkeypatch.setattr(gpu_load.time, "time", lambda: 1030.0)
    assert gpu_load.read_gpu_load() is not None                                            # exactly 30 s: still fresh
    monkeypatch.setattr(gpu_load.time, "time", lambda: 1030.5)
    assert gpu_load.read_gpu_load() is None and gpu_load._status["state"] == "stale"


def test_llm_local_reports_an_unusable_load_reader_once_and_counts_every_call(monkeypatch, caplog):
    import llm_local

    def broken():
        raise RuntimeError("nvml gone")
    monkeypatch.setattr(gpu_load, "read_gpu_load", broken)
    monkeypatch.setattr(llm_local, "_gpu_load_unavailable", 0)
    with caplog.at_level(logging.WARNING, logger=llm_local._LOG.name):
        assert llm_local._read_gpu_load() is None
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1       # on the FIRST failure, not later
        assert llm_local._read_gpu_load() is None
        assert llm_local._read_gpu_load() is None
    assert llm_local._gpu_load_unavailable == 3
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warns == ["GPU load reader unavailable (RuntimeError('nvml gone')): load-aware routing is inactive until it imports"]
    monkeypatch.setattr(gpu_load, "read_gpu_load", lambda: "fine")                         # twin: a working reader is untouched
    assert llm_local._read_gpu_load() == "fine" and llm_local._gpu_load_unavailable == 3

"""Shared GPU load signal for load-aware LLM routing in Loci.

The sidecar (gpu_load_sidecar.py) polls nvidia-smi every few seconds and writes a JSON
snapshot to a tmpfs path. This module reads that snapshot at call time so generate() can
make routing decisions before committing to a backend.

Callers always fail open: if the file is absent, stale, or unreadable, None is returned
and normal Ollama-first routing applies unchanged. The fail-open is never silent: each
degraded state is counted, logged once at WARNING (rate-limited) and reported by
gpu_load_health(), which is the hook-ready dict for loci_health.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_SIGNAL_MAX_AGE_S = 30.0   # treat signal as absent if older than this
_LOAD_UTIL_THRESHOLD = 75  # GPU util% at or above this -> loaded
_LOAD_VRAM_THRESHOLD = 85  # VRAM used% at or above this -> loaded
_URGENT_UTIL_THRESHOLD = 92  # util% at or above this -> urgent (prefer cloud fallback)


SIGNAL_PATH_VAR = "LOCI_GPU_LOAD_PATH"
_SIGNAL_NAME = "loci_gpu_load.json"
_LOG = logging.getLogger("loci.gpu_load")
_WARN_INTERVAL_S = 300.0  # at most one WARNING per degraded state per this long

_monotonic = time.monotonic  # injectable clock for the warning rate limit


def live_signal_path() -> Path:
    """The default path the sidecar writes: a per-user tmpfs path where os.getuid exists (POSIX),
    else <tempdir>/loci_gpu_load.json (Windows, which has no getuid). The reader and the sidecar
    must resolve the same tempdir, so on Windows set LOCI_GPU_LOAD_PATH for both when they run
    under different accounts."""
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return Path(tempfile.gettempdir()) / _SIGNAL_NAME
    return Path(f"/run/user/{getuid()}/{_SIGNAL_NAME}")


def _signal_path() -> Path:
    """Path of the GPU load snapshot: $LOCI_GPU_LOAD_PATH when set, else the live tmpfs path.

    The override exists so the test harness (testsupport/loci_hermetic.py) can point
    every reader away from the live signal: generate() reads it on each call, and a
    busy workstation otherwise changes routing -- and test outcomes -- under test.
    """
    override = os.environ.get(SIGNAL_PATH_VAR, "").strip()
    if override:
        return Path(override)
    return live_signal_path()


@dataclass
class GpuEntry:
    index: int
    name: str
    util_pct: float
    vram_used_mb: float
    vram_total_mb: float

    @property
    def vram_pct(self) -> float:
        if self.vram_total_mb <= 0:
            return 0.0
        return 100.0 * self.vram_used_mb / self.vram_total_mb


@dataclass
class GpuLoad:
    gpus: List[GpuEntry]
    timestamp: float
    age_s: float = field(init=False)

    def __post_init__(self) -> None:
        self.age_s = time.time() - self.timestamp

    @property
    def max_util_pct(self) -> float:
        return max((g.util_pct for g in self.gpus), default=0.0)

    @property
    def max_vram_pct(self) -> float:
        return max((g.vram_pct for g in self.gpus), default=0.0)

    def is_loaded(
        self,
        util_threshold: int = _LOAD_UTIL_THRESHOLD,
        vram_threshold: int = _LOAD_VRAM_THRESHOLD,
    ) -> bool:
        """True when any GPU is above the utilisation or VRAM threshold."""
        return self.max_util_pct >= util_threshold or self.max_vram_pct >= vram_threshold

    def is_urgent(self) -> bool:
        """True when utilisation is so high that even a short Ollama attempt is inadvisable."""
        return self.max_util_pct >= _URGENT_UTIL_THRESHOLD

    def summary(self) -> str:
        parts = [f"GPU{g.index} util={g.util_pct:.0f}% vram={g.vram_pct:.0f}%" for g in self.gpus]
        return " | ".join(parts) if parts else "no gpus"


_BAD_STATES = ("absent", "stale", "unreadable")
_lock = threading.Lock()
_status: dict = {
    "state": "unread",
    "path": "",
    "detail": "",
    "counts": {"ok": 0, "absent": 0, "stale": 0, "unreadable": 0},
    "last_warned": {},
}


def _record(state: str, path: str, detail: str = "") -> None:
    """Count one read outcome; on a change into a degraded state log one WARNING (rate-limited per state)."""
    with _lock:
        previous = _status["state"]
        _status["counts"][state] += 1
        _status["state"] = state
        _status["path"] = path
        _status["detail"] = detail
        warn = False
        if state in _BAD_STATES and state != previous:
            now = _monotonic()
            last = _status["last_warned"].get(state)
            if last is None or now - last >= _WARN_INTERVAL_S:
                _status["last_warned"][state] = now
                warn = True
        recovered = state == "ok" and previous in _BAD_STATES
    if warn:
        _LOG.warning(
            "GPU load signal %s at %s (%s): load-aware routing is inactive and Ollama-first routing "
            "applies unchanged. Start mcp/gpu_load_sidecar.py or set %s.",
            state, path, detail or "no detail", SIGNAL_PATH_VAR,
        )
    elif recovered:
        _LOG.info("GPU load signal recovered at %s", path)


def read_gpu_load(max_age_s: float = _SIGNAL_MAX_AGE_S) -> Optional[GpuLoad]:
    """Read the GPU load snapshot. Returns None if absent, stale, or unreadable; the reason is
    counted, logged once at WARNING and visible in gpu_load_health()."""
    try:
        path = _signal_path()
    except Exception as exc:  # noqa: BLE001 - never raise into generate()
        _record("unreadable", "", f"signal path unresolvable: {exc!r}")
        return None
    shown = str(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        timestamp = float(raw["timestamp"])
        gpus = [
            GpuEntry(
                index=int(g["index"]),
                name=str(g.get("name", "")),
                util_pct=float(g.get("util_pct", 0)),
                vram_used_mb=float(g.get("vram_used_mb", 0)),
                vram_total_mb=float(g.get("vram_total_mb", 1)),
            )
            for g in raw.get("gpus", [])
        ]
    except FileNotFoundError:
        _record("absent", shown, "file does not exist; no sidecar is writing it")
        return None
    except Exception as exc:  # noqa: BLE001 - fail open, but reported
        _record("unreadable", shown, repr(exc))
        return None
    age = time.time() - timestamp
    if age > max_age_s:
        _record("stale", shown, f"snapshot is {age:.0f}s old, limit {max_age_s:.0f}s")
        return None
    _record("ok", shown)
    return GpuLoad(gpus=gpus, timestamp=timestamp)


def gpu_load_health() -> dict:
    """Hook-ready dict for loci_health: re-reads the signal and reports its state.

    ``gpu_load_signal`` is "present" when a fresh snapshot was read, else "absent", "stale" or
    "unreadable"; ``counts`` are the cumulative outcomes of every read in this process."""
    load = read_gpu_load()
    with _lock:
        state = _status["state"]
        out = {
            "gpu_load_signal": "present" if state == "ok" else state,
            "path": _status["path"],
            "detail": _status["detail"],
            "counts": dict(_status["counts"]),
        }
    out["age_s"] = None if load is None else round(load.age_s, 3)
    return out


def write_gpu_load(gpus: List[GpuEntry]) -> None:
    """Atomically write a GPU load snapshot. Called only by the sidecar."""
    path = _signal_path()
    payload = {
        "timestamp": time.time(),
        "gpus": [
            {
                "index": g.index,
                "name": g.name,
                "util_pct": g.util_pct,
                "vram_used_mb": g.vram_used_mb,
                "vram_total_mb": g.vram_total_mb,
            }
            for g in gpus
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)  # Path.rename refuses to overwrite on Windows

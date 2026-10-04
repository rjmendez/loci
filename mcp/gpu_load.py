"""Shared GPU load signal for load-aware LLM routing in Loci.

The sidecar (gpu_load_sidecar.py) polls nvidia-smi every few seconds and writes a JSON
snapshot to a tmpfs path. This module reads that snapshot at call time so generate() can
make routing decisions before committing to a backend.

Callers always fail open: if the file is absent, stale, or unreadable, None is returned
and normal Ollama-first routing applies unchanged.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_SIGNAL_MAX_AGE_S = 30.0   # treat signal as absent if older than this
_LOAD_UTIL_THRESHOLD = 75  # GPU util% at or above this -> loaded
_LOAD_VRAM_THRESHOLD = 85  # VRAM used% at or above this -> loaded
_URGENT_UTIL_THRESHOLD = 92  # util% at or above this -> urgent (prefer cloud fallback)


SIGNAL_PATH_VAR = "LOCI_GPU_LOAD_PATH"


def live_signal_path() -> Path:
    """The default tmpfs path the sidecar writes, scoped to the current user."""
    uid = os.getuid()
    return Path(f"/run/user/{uid}/loci_gpu_load.json")


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


def read_gpu_load(max_age_s: float = _SIGNAL_MAX_AGE_S) -> Optional[GpuLoad]:
    """Read the GPU load snapshot. Returns None if absent, stale, or unreadable."""
    try:
        raw = json.loads(_signal_path().read_text(encoding="utf-8"))
        timestamp = float(raw["timestamp"])
        if time.time() - timestamp > max_age_s:
            return None
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
        return GpuLoad(gpus=gpus, timestamp=timestamp)
    except Exception:
        return None


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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.rename(path)

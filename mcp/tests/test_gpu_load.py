"""Tests for gpu_load.py — the shared GPU load signal reader/writer."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpu_load import (
    GpuEntry,
    GpuLoad,
    _LOAD_UTIL_THRESHOLD,
    _LOAD_VRAM_THRESHOLD,
    _URGENT_UTIL_THRESHOLD,
    read_gpu_load,
    write_gpu_load,
)


def _make_entry(util=0.0, vram_used=1000.0, vram_total=12000.0, index=0) -> GpuEntry:
    return GpuEntry(index=index, name="Test GPU", util_pct=util,
                    vram_used_mb=vram_used, vram_total_mb=vram_total)


class TestGpuEntry:
    def test_vram_pct_normal(self):
        e = _make_entry(vram_used=6000, vram_total=12000)
        assert e.vram_pct == pytest.approx(50.0)

    def test_vram_pct_zero_total(self):
        e = GpuEntry(index=0, name="X", util_pct=0, vram_used_mb=0, vram_total_mb=0)
        assert e.vram_pct == 0.0


class TestGpuLoad:
    def _load(self, gpus, ago=0.0) -> GpuLoad:
        return GpuLoad(gpus=gpus, timestamp=time.time() - ago)

    def test_max_util_single(self):
        gl = self._load([_make_entry(util=60)])
        assert gl.max_util_pct == 60.0

    def test_max_util_multiple(self):
        gl = self._load([_make_entry(util=40), _make_entry(util=80, index=1)])
        assert gl.max_util_pct == 80.0

    def test_max_vram_pct(self):
        gl = self._load([_make_entry(vram_used=10000, vram_total=12000)])
        assert gl.max_vram_pct == pytest.approx(83.33, abs=0.1)

    def test_not_loaded_low(self):
        gl = self._load([_make_entry(util=30, vram_used=2000, vram_total=12000)])
        assert not gl.is_loaded()

    def test_loaded_by_util(self):
        gl = self._load([_make_entry(util=_LOAD_UTIL_THRESHOLD)])
        assert gl.is_loaded()

    def test_loaded_by_vram(self):
        # 85% of 12000 = 10200 MB
        gl = self._load([_make_entry(util=5, vram_used=10200, vram_total=12000)])
        assert gl.is_loaded()

    def test_not_urgent_below_threshold(self):
        gl = self._load([_make_entry(util=_URGENT_UTIL_THRESHOLD - 1)])
        assert not gl.is_urgent()

    def test_urgent_at_threshold(self):
        gl = self._load([_make_entry(util=_URGENT_UTIL_THRESHOLD)])
        assert gl.is_urgent()

    def test_no_gpus_returns_zero(self):
        gl = self._load([])
        assert gl.max_util_pct == 0.0
        assert not gl.is_loaded()

    def test_summary_format(self):
        gl = self._load([_make_entry(util=70, vram_used=9000, vram_total=12000)])
        s = gl.summary()
        assert "GPU0" in s
        assert "util=" in s
        assert "vram=" in s


class TestReadWriteRoundtrip:
    def test_roundtrip(self, tmp_path):
        signal_file = tmp_path / "loci_gpu_load.json"
        gpus = [_make_entry(util=80, vram_used=10000, vram_total=12000)]

        with patch("gpu_load._signal_path", return_value=signal_file):
            write_gpu_load(gpus)
            result = read_gpu_load()

        assert result is not None
        assert len(result.gpus) == 1
        assert result.gpus[0].util_pct == 80.0
        assert result.gpus[0].vram_used_mb == 10000.0
        assert result.is_loaded()

    def test_stale_returns_none(self, tmp_path):
        signal_file = tmp_path / "loci_gpu_load.json"
        payload = {
            "timestamp": time.time() - 60,  # 60s old, well past the 30s max
            "gpus": [{"index": 0, "name": "X", "util_pct": 90,
                      "vram_used_mb": 5000, "vram_total_mb": 12000}],
        }
        signal_file.write_text(json.dumps(payload))

        with patch("gpu_load._signal_path", return_value=signal_file):
            result = read_gpu_load()

        assert result is None

    def test_missing_file_returns_none(self, tmp_path):
        missing = tmp_path / "nonexistent.json"
        with patch("gpu_load._signal_path", return_value=missing):
            assert read_gpu_load() is None

    def test_corrupt_file_returns_none(self, tmp_path):
        bad = tmp_path / "loci_gpu_load.json"
        bad.write_text("not json {{{{")
        with patch("gpu_load._signal_path", return_value=bad):
            assert read_gpu_load() is None

    def test_atomic_write_uses_tmp_then_rename(self, tmp_path):
        signal_file = tmp_path / "loci_gpu_load.json"
        gpus = [_make_entry(util=50)]
        with patch("gpu_load._signal_path", return_value=signal_file):
            write_gpu_load(gpus)
        assert signal_file.exists()
        assert not (tmp_path / "loci_gpu_load.tmp").exists()

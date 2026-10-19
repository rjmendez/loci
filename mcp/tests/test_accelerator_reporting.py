"""loci.accelerator: the cupy -> numpy fallback is logged, reported and retried, never silent."""
from __future__ import annotations

import logging
import sys
import types

import pytest

from loci import accelerator

LOGGER = "loci.accelerator"
_GLOBALS = ("_BACKEND", "_XP", "_REASON", "_FALLBACK", "_PERMANENT", "_RETRY_AT", "_RETRY_DELAY_S", "_RETRIES")


class FakeCupy(types.ModuleType):
    """Recording stand-in for cupy: ``array`` raises while ``failure`` is set; counts every probe."""

    def __init__(self, devices: int = 2, failure: Exception | None = None):
        super().__init__("cupy")
        self.failure = failure
        self.probes = 0
        self._devices = devices
        runtime = types.SimpleNamespace(getDeviceCount=self._device_count)
        self.cuda = types.SimpleNamespace(runtime=runtime)

    def _device_count(self) -> int:
        return self._devices

    def array(self, values):
        self.probes += 1
        if self.failure is not None:
            raise self.failure
        return list(values)


@pytest.fixture
def clock(monkeypatch):
    for name in _GLOBALS:  # register the originals so teardown restores whatever _detect assigns
        monkeypatch.setattr(accelerator, name, getattr(accelerator, name))
    now = [1000.0]
    monkeypatch.setattr(accelerator, "_monotonic", lambda: now[0])
    monkeypatch.setattr(accelerator, "_RETRIES", 0)
    monkeypatch.setattr(accelerator, "_RETRY_DELAY_S", accelerator._RETRY_INITIAL_S)
    monkeypatch.delenv("LOCI_ACCELERATOR", raising=False)
    return now


def _msgs(caplog, level=None):
    return [(r.levelno, r.getMessage()) for r in caplog.records
            if r.name == LOGGER and (level is None or r.levelno == level)]


def test_cpu_override_is_a_choice_not_a_fallback(monkeypatch, caplog, clock):
    monkeypatch.setenv("LOCI_ACCELERATOR", "cpu")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        accelerator._detect()
    assert accelerator.accelerator_info() == {
        "accelerator": "cpu", "backend": "numpy", "device_count": 0,
        "fallback": False, "reason": "LOCI_ACCELERATOR=cpu", "retries": 0,
    }
    assert _msgs(caplog) == [(logging.INFO, "accelerator backend=cpu (numpy): LOCI_ACCELERATOR=cpu")]


def test_working_cupy_reports_cuda_and_device_count(monkeypatch, caplog, clock):
    fake = FakeCupy(devices=2)
    monkeypatch.setitem(sys.modules, "cupy", fake)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        accelerator._detect()
    assert accelerator.get_array_module() is fake
    assert accelerator.accelerator_info() == {
        "accelerator": "cuda", "backend": "cupy", "device_count": 2,
        "fallback": False, "reason": "", "retries": 0,
    }
    assert _msgs(caplog) == [(logging.INFO, "accelerator backend=cuda (cupy)")]
    assert fake.probes == 1


def test_missing_cupy_is_a_reported_permanent_fallback(monkeypatch, caplog, clock):
    monkeypatch.setitem(sys.modules, "cupy", None)  # `import cupy` raises ModuleNotFoundError
    with caplog.at_level(logging.INFO, logger=LOGGER):
        accelerator._detect()
    reason = "cupy not importable: ModuleNotFoundError('import of cupy halted; None in sys.modules')"
    assert accelerator.accelerator_info() == {
        "accelerator": "cpu", "backend": "numpy", "device_count": 0,
        "fallback": True, "reason": reason, "retries": 0,
    }
    assert _msgs(caplog) == [(logging.INFO, f"accelerator backend=cpu (numpy): {reason}")]

    # permanent: even once cupy becomes importable, no retry happens inside the process
    fake = FakeCupy()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    clock[0] += 10_000
    assert accelerator.get_array_module().__name__ == "numpy"
    assert fake.probes == 0
    assert accelerator.accelerator_info()["retries"] == 0


def test_transient_cupy_failure_warns_once_and_retries_with_backoff(monkeypatch, caplog, clock):
    fake = FakeCupy(failure=RuntimeError("CUDA driver busy"))
    monkeypatch.setitem(sys.modules, "cupy", fake)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        accelerator._detect()
        reason = "cupy failed: RuntimeError('CUDA driver busy'); retrying with backoff"
        assert accelerator.accelerator_info()["reason"] == reason
        assert accelerator.accelerator_info()["fallback"] is True
        assert _msgs(caplog) == [(logging.WARNING, f"accelerator backend=cpu (numpy): {reason}")]

        clock[0] += 29                                   # before the first retry (30 s)
        assert accelerator.get_array_module().__name__ == "numpy"
        assert fake.probes == 1

        clock[0] += 1                                    # at 30 s: retry, still failing, no new log line
        assert accelerator.get_array_module().__name__ == "numpy"
        assert fake.probes == 2
        assert accelerator.accelerator_info()["retries"] == 1
        assert len(_msgs(caplog)) == 1

        clock[0] += 59                                   # the delay doubled to 60 s
        accelerator.get_array_module()
        assert fake.probes == 2
        clock[0] += 1
        accelerator.get_array_module()
        assert fake.probes == 3
        assert accelerator.accelerator_info()["retries"] == 2

        fake.failure = None                              # the transient fault clears
        clock[0] += 120
        assert accelerator.get_array_module() is fake
    info = accelerator.accelerator_info()
    assert (info["accelerator"], info["fallback"], info["reason"], info["retries"]) == ("cuda", False, "", 3)
    assert _msgs(caplog)[-1] == (logging.INFO, "accelerator backend=cuda (cupy) (recovered after retry)")


def test_retry_delay_is_capped(monkeypatch, clock):
    fake = FakeCupy(failure=RuntimeError("busy"))
    monkeypatch.setitem(sys.modules, "cupy", fake)
    accelerator._detect()
    for _ in range(12):
        clock[0] += 10_000
        accelerator.get_array_module()
    assert accelerator._RETRY_DELAY_S == accelerator._RETRY_MAX_S == 3600.0


def test_device_count_failure_is_reported_not_hidden(monkeypatch, clock):
    fake = FakeCupy()

    def broken():
        raise RuntimeError("no driver")
    fake.cuda.runtime.getDeviceCount = broken
    monkeypatch.setitem(sys.modules, "cupy", fake)
    accelerator._detect()
    info = accelerator.accelerator_info()
    assert info["accelerator"] == "cuda"
    assert info["device_count"] is None
    assert info["cuda_error"] == "no driver"


def test_health_wraps_the_info_dict(monkeypatch, clock):
    monkeypatch.setenv("LOCI_ACCELERATOR", "cpu")
    accelerator._detect()
    assert accelerator.accelerator_health() == {"accelerator": accelerator.accelerator_info()}
    assert accelerator.accelerator_health()["accelerator"]["reason"] == "LOCI_ACCELERATOR=cpu"

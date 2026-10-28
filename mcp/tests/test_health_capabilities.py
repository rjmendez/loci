"""loci_health reports the silent-degradation surfaces from the 2026-10-19 fail-open audit.

Before: loci_health said status ok / no warnings while the GPU load signal could never be read (Windows had no os.getuid),
the accelerator ran on numpy after a cupy failure, and tier lookups fell back to defaults after a pool exception. Each of
those now shows in ``capabilities`` and as a warning. Warnings only: every one of these is fail-open by design.

Mutation runs (each applied to mcp/server.py, confirmed red, reverted):
- drop the gpu_load warning branch -> test_an_absent_or_stale_gpu_load_signal_is_a_warning red
- warn when the signal is present too (``!=`` -> ``==``) -> test_a_present_gpu_load_signal_adds_no_warning red
- drop the accelerator fallback warning -> test_an_accelerator_that_fell_back_is_a_warning red
- test ``"loci.accelerator" in sys.modules`` as always true -> test_the_accelerator_is_not_probed_unless_something_already_loaded_it red
- drop the pool-failure or policy-degraded warning -> the two counter tests red
- let a probe exception escape -> test_a_failing_probe_never_breaks_health red
"""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server  # noqa: E402
import gpu_load  # noqa: E402
from loci import model_selector, policy  # noqa: E402


def _signal(state="ok", detail="", path="/tmp/loci_gpu_load.json"):
    return {"gpu_load_signal": "present" if state == "ok" else state, "path": path, "detail": detail,
            "counts": {"ok": 0, "absent": 0, "stale": 0, "unreadable": 0}, "age_s": None}


@pytest.fixture
def health(monkeypatch):
    """loci_health with every capability input stubbed to its healthy value; tests override one input at a time."""
    monkeypatch.setattr(server, "_ladybug_health_state", lambda: "available")
    monkeypatch.setattr(gpu_load, "gpu_load_health", lambda: _signal("ok"))
    monkeypatch.setattr(model_selector, "pool_failure_counts", lambda: {})
    monkeypatch.setattr(policy, "policy_degraded_counts", lambda: {})
    monkeypatch.delitem(sys.modules, "loci.accelerator", raising=False)

    def call():
        out = json.loads(server.loci_health())
        return out, out.get("warnings", [])
    return call


def _fake_accelerator(monkeypatch, info):
    mod = types.ModuleType("loci.accelerator")
    mod.accelerator_info = lambda: info
    monkeypatch.setitem(sys.modules, "loci.accelerator", mod)


def test_a_present_gpu_load_signal_adds_no_warning(health):
    out, warnings = health()
    assert out["capabilities"]["gpu_load"]["gpu_load_signal"] == "present"
    assert not [w for w in warnings if "GPU load" in w]


@pytest.mark.parametrize("state,detail", [("absent", "no file at /tmp/loci_gpu_load.json"),
                                          ("stale", "snapshot is 60s old, limit 30s"),
                                          ("unreadable", "JSONDecodeError")])
def test_an_absent_or_stale_gpu_load_signal_is_a_warning(health, monkeypatch, state, detail):
    monkeypatch.setattr(gpu_load, "gpu_load_health", lambda: _signal(state, detail))
    out, warnings = health()
    assert f"GPU load signal {state}: load-aware model routing is inactive ({detail})" in warnings
    assert out["capabilities"]["gpu_load"]["gpu_load_signal"] == state


def test_an_absent_signal_without_detail_names_the_path(health, monkeypatch):
    monkeypatch.setattr(gpu_load, "gpu_load_health", lambda: _signal("absent", "", path="C:/Temp/loci_gpu_load.json"))
    _, warnings = health()
    assert "GPU load signal absent: load-aware model routing is inactive (C:/Temp/loci_gpu_load.json)" in warnings


def test_the_accelerator_is_not_probed_unless_something_already_loaded_it(health):
    out, _ = health()                      # the fixture removed loci.accelerator from sys.modules
    assert "accelerator" not in out["capabilities"]
    assert "loci.accelerator" not in sys.modules      # health did not import it (importing it runs the cupy probe)


def test_an_accelerator_that_fell_back_is_a_warning(health, monkeypatch):
    _fake_accelerator(monkeypatch, {"accelerator": "cpu", "backend": "numpy", "device_count": 0, "fallback": True,
                                    "reason": "cupy failed: CUDA path could not be detected", "retries": 2})
    out, warnings = health()
    assert "accelerator fell back to numpy: cupy failed: CUDA path could not be detected (2 retries)" in warnings
    assert out["capabilities"]["accelerator"]["accelerator"] == "cpu"


def test_a_working_accelerator_and_a_deliberate_cpu_choice_are_not_warnings(health, monkeypatch):
    for info in ({"accelerator": "cuda", "backend": "cupy", "device_count": 2, "fallback": False, "reason": "", "retries": 0},
                 {"accelerator": "cpu", "backend": "numpy", "device_count": 0, "fallback": False,
                  "reason": "LOCI_ACCELERATOR=cpu", "retries": 0}):
        _fake_accelerator(monkeypatch, info)
        out, warnings = health()
        assert out["capabilities"]["accelerator"] == info
        assert not [w for w in warnings if "accelerator" in w]


def test_failed_pool_lookups_are_a_warning_with_the_counts(health, monkeypatch):
    monkeypatch.setattr(model_selector, "pool_failure_counts", lambda: {"tier_light": 2})
    out, warnings = health()
    assert "model pool lookups failed {'tier_light': 2}: tier candidates fell back to the policy map" in warnings
    assert out["capabilities"]["pool_lookup_failures"] == {"tier_light": 2}


def test_a_degraded_tier_policy_is_a_warning_with_the_counts(health, monkeypatch):
    monkeypatch.setattr(policy, "policy_degraded_counts", lambda: {"unparseable LOCI_TIER_MODEL_MAP": 1})
    _, warnings = health()
    assert ("tier policy ran degraded {'unparseable LOCI_TIER_MODEL_MAP': 1} (unparseable tier map or failing pool)"
            in warnings)


def test_nothing_degraded_means_no_capability_warning_and_empty_counter_blocks(health):
    out, warnings = health()
    assert out["capabilities"]["pool_lookup_failures"] == {} and out["capabilities"]["policy_degraded"] == {}
    assert not [w for w in warnings if w.startswith(("GPU load", "accelerator", "model pool", "tier policy"))]


def test_a_failing_probe_never_breaks_health(health, monkeypatch):
    def boom():
        raise RuntimeError("nvml exploded")
    monkeypatch.setattr(gpu_load, "gpu_load_health", boom)
    out, _ = health()
    assert "capabilities" not in out and out["code_version"]          # the rest of the report is intact

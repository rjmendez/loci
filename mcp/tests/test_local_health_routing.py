"""Measured-health routing: the snapshot decision, and what llm_local does with it.

No network: requests.post is faked, the cloud tier is replaced by a recorder, and the snapshot is a file in
tmp_path. The point is that failure evidence moves work off local only when it is recent, organic and
sufficient, and that a request is never refused on health alone.
"""
import json
import os
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import llm_local as L  # noqa: E402
import local_health as H  # noqa: E402

NOW = 1_800_000_000.0


def snap(group="medium-tier", **kw):
    g = {"n": 40, "n_organic": 40, "basis": "organic", "ok_rate": 1.0, "good_rate": 1.0, "empty_rate": 0.0, "p95_s": 1.0}
    g.update(kw)
    return {"version": 1, "as_of": NOW, "window_s": 3600, "min_organic": 20, "groups": {group: g}}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("LOCI_LEDGER_HEALTH_FILE", "LOCI_LOCAL_HEALTH_ROUTING", "LOCI_LLM_VIA_LITELLM",
              "LOCI_LOCAL_HEALTH_MAX_AGE_S", "LOCI_LOCAL_HEALTH_MIN_N", "LOCI_LOCAL_HEALTH_MIN_GOOD",
              "LOCI_LOCAL_HEALTH_MAX_EMPTY"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("snapshot,now,degraded,fragment", [
    (snap(), NOW, False, "meets"),
    (snap(good_rate=0.5), NOW, True, "good rate"),
    (snap(empty_rate=0.4), NOW, True, "empty rate"),
    (snap(good_rate=0.5), NOW + 7200, False, "old"),                       # stale is not evidence
    (snap(good_rate=0.5, n=5), NOW, False, "only 5"),                      # thin is not evidence
    (snap(good_rate=0.5, basis="all", n_organic=3), NOW, False, "organic"),  # synthetic is not evidence
    ({"version": 2}, NOW, False, "no usable"),
    ({}, NOW, False, "no usable"),
])
def test_check_is_conservative(snapshot, now, degraded, fragment):
    got, why, _ = H.check(snapshot, now=now)
    assert got is degraded and fragment in why


def test_it_judges_the_group_with_the_most_organic_attempts():
    s = snap(good_rate=1.0, n=30)
    s["groups"]["complex-tier"] = {"n": 90, "n_organic": 90, "basis": "organic", "ok_rate": 0.5,
                                    "good_rate": 0.5, "empty_rate": 0.0, "p95_s": 9}
    got, why, group = H.check(s, now=NOW)
    assert got is True and group == "complex-tier"


def test_thresholds_are_tunable(monkeypatch):
    monkeypatch.setenv("LOCI_LOCAL_HEALTH_MIN_GOOD", "0.4")
    assert H.check(snap(good_rate=0.5), now=NOW)[0] is False


def test_missing_or_corrupt_file_is_not_degraded(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_LEDGER_HEALTH_FILE", str(tmp_path / "nope.json"))
    assert H.check(now=NOW)[0] is False
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    monkeypatch.setenv("LOCI_LEDGER_HEALTH_FILE", str(bad))
    assert H.check(now=NOW)[0] is False


# --- through llm_local -------------------------------------------------------------------------

class Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


@pytest.fixture
def stack(monkeypatch, tmp_path):
    calls = {"ollama": 0, "cloud": 0}

    def post(url, json=None, headers=None, timeout=None):  # noqa: A002
        assert url.endswith("/api/generate"), url
        calls["ollama"] += 1
        return Resp({"response": "from local"})

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=post))
    monkeypatch.setattr(L, "_gen_env", lambda: "http://fake-ollama:11434")
    monkeypatch.setattr(L, "_read_gpu_load", lambda: None)
    import backends
    monkeypatch.setattr(backends, "ollama_gen_model", lambda: "qwen2.5:3b")
    monkeypatch.setattr(L, "_heuristic_route", lambda p: {"role": "triage"})
    path = tmp_path / "health.json"
    monkeypatch.setenv("LOCI_LEDGER_HEALTH_FILE", str(path))
    monkeypatch.setenv("LOCI_LOCAL_HEALTH_ROUTING", "1")

    cloud_result = {"value": {"text": "from cloud", "ok": True, "model": "cloud-m", "tier": "cloud-x"}}

    def fake_cloud(prompt, **kw):
        calls["cloud"] += 1
        return cloud_result["value"]

    monkeypatch.setattr(L, "_try_cloud_tier", fake_cloud)

    def write(snapshot):
        path.write_text(json.dumps(snapshot))

    return types.SimpleNamespace(calls=calls, write=write, cloud=cloud_result)


def degraded_now():
    s = snap(good_rate=0.4, empty_rate=0.0)
    s["as_of"] = time.time()
    return s


def test_degraded_local_gets_the_cloud_tier_first(stack):
    stack.write(degraded_now())
    out = L.generate("hello there, a long enough prompt")
    assert out["text"] == "from cloud" and out["health_routed"] is True
    assert stack.calls == {"ollama": 0, "cloud": 1}


def test_the_flag_off_ignores_the_snapshot(stack, monkeypatch):
    monkeypatch.delenv("LOCI_LOCAL_HEALTH_ROUTING")
    stack.write(degraded_now())
    out = L.generate("hello there, a long enough prompt")
    assert out["text"] == "from local" and stack.calls == {"ollama": 1, "cloud": 0}


def test_a_healthy_snapshot_stays_local(stack):
    s = snap()
    s["as_of"] = time.time()
    stack.write(s)
    assert L.generate("hello there, a long enough prompt")["text"] == "from local"
    assert stack.calls["cloud"] == 0


def test_a_refused_cloud_tier_falls_back_to_local_not_to_failure(stack):
    stack.write(degraded_now())
    stack.cloud["value"] = {"text": "", "ok": False, "tier": "cloud-refused", "why": "prompt export disabled"}
    out = L.generate("hello there, a long enough prompt")
    assert out["text"] == "from local" and not out.get("health_routed")
    assert stack.calls == {"ollama": 1, "cloud": 1}


def test_no_cloud_tier_available_falls_back_to_local(stack):
    stack.write(degraded_now())
    stack.cloud["value"] = None
    assert L.generate("hello there, a long enough prompt")["text"] == "from local"


def test_an_explicit_model_pin_ignores_health_routing(stack):
    stack.write(degraded_now())
    out = L.generate("hello there, a long enough prompt", model="granite3-guardian:2b")
    assert out["text"] == "from local" and stack.calls["cloud"] == 0


def test_a_broken_health_module_does_not_break_generation(stack, monkeypatch):
    monkeypatch.setattr(H, "check", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert L.generate("hello there, a long enough prompt")["text"] == "from local"

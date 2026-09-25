"""llm_local.generate() is capped by one total deadline across every tier.

Before: 120s on the configured model + 120s on a discovered model + 45s supervisor
+ vLLM + cloud, all while the MCP server was frozen (see test_tool_offload).
A fake clock advances by each request's timeout, so no test sleeps.
"""
from __future__ import annotations

from unittest import mock

import pytest

import llm_local


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


class _TimeoutPost:
    """requests.post stand-in: every call 'times out' after its full timeout."""

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.timeouts: list[float] = []
        self.models: list[str] = []

    def __call__(self, url, json=None, timeout=None):  # noqa: A002
        self.timeouts.append(timeout)
        self.models.append((json or {}).get("model", ""))
        self.clock.now += timeout
        raise TimeoutError(f"read timed out ({timeout})")


def _tags_response(names):
    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    resp.json = mock.Mock(return_value={"models": [{"name": n} for n in names]})
    return resp


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(llm_local.time, "monotonic", c.monotonic)
    monkeypatch.setattr(llm_local, "_gen_env", lambda: "http://ollama")
    monkeypatch.setattr(llm_local, "_TIMEOUT", 120.0)
    monkeypatch.delenv("LOCI_LLM_DEADLINE_S", raising=False)
    return c


def test_retry_gets_only_the_remaining_budget_then_stops(clock):
    post = _TimeoutPost(clock)
    get_calls: list[float] = []

    def fake_get(url, timeout=None):
        get_calls.append(timeout)
        return _tags_response(["other-model:latest"])

    calls: list[str] = []

    def record(name):
        def _f(*a, **k):
            calls.append(name)
            return None
        return _f

    with mock.patch("requests.post", post), mock.patch("requests.get", fake_get), \
         mock.patch.object(llm_local, "_supervisor_route", record("supervisor")), \
         mock.patch.object(llm_local, "_try_vllm", record("vllm")), \
         mock.patch.object(llm_local, "_try_cloud_tier", record("cloud")):
        out = llm_local.generate("hello", model="qwen3.8:latest", max_tokens=8)

    assert post.timeouts == [120.0, 30.0]
    assert post.models == ["qwen3.8:latest", "other-model:latest"]
    assert get_calls == [30.0]
    assert calls == []
    assert out["ok"] is False
    assert out["deadline_exceeded"] is True
    assert out["why"].startswith("LLM deadline 150s exhausted after 150.0s")


def test_budget_smaller_than_one_attempt_skips_discovery_and_fallbacks(clock, monkeypatch):
    monkeypatch.setenv("LOCI_LLM_DEADLINE_S", "60")
    post = _TimeoutPost(clock)
    get = mock.Mock()
    calls: list[str] = []
    with mock.patch("requests.post", post), mock.patch("requests.get", get), \
         mock.patch.object(llm_local, "_supervisor_route", lambda *a, **k: calls.append("supervisor")), \
         mock.patch.object(llm_local, "_try_vllm", lambda *a, **k: calls.append("vllm")), \
         mock.patch.object(llm_local, "_try_cloud_tier", lambda *a, **k: calls.append("cloud")):
        out = llm_local.generate("hello", model="m", max_tokens=8)

    assert post.timeouts == [60.0]
    assert get.call_count == 0
    assert calls == []
    assert out["deadline_exceeded"] is True


def test_fast_failure_still_reaches_fallbacks_with_bounded_supervisor(clock):
    def refused(url, json=None, timeout=None):  # noqa: A002
        clock.now += 1.0
        raise ConnectionError("refused")

    seen: dict = {}

    def supervisor(prompt, *, fmt, max_tokens, timeout=None):
        seen["supervisor_timeout"] = timeout
        return {"role": "triage", "provider": "openrouter"}

    vllm_result = {"text": "ok", "ok": True, "model": "v", "tier": "vllm"}
    with mock.patch("requests.post", refused), \
         mock.patch("requests.get", side_effect=OSError("no tags")), \
         mock.patch.object(llm_local, "_supervisor_route", supervisor), \
         mock.patch.object(llm_local, "_try_vllm", return_value=vllm_result):
        out = llm_local.generate("hello", max_tokens=8)

    assert out == vllm_result
    assert seen["supervisor_timeout"] == 120.0


def test_success_path_uses_the_per_request_timeout_and_is_not_marked(clock):
    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    resp.json = mock.Mock(return_value={"response": "hi"})
    post = mock.Mock(return_value=resp)
    with mock.patch("requests.post", post):
        out = llm_local.generate("hello", model="m", max_tokens=8)

    assert out == {"text": "hi", "ok": True, "model": "m"}
    assert post.call_args.kwargs["timeout"] == 120.0


@pytest.mark.parametrize("raw, expected", [("", 150.0), ("30", 30.0), ("0", 150.0),
                                           ("-5", 150.0), ("soon", 150.0)])
def test_deadline_env_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("LOCI_LLM_DEADLINE_S", raw)
    assert llm_local._deadline_s() == expected

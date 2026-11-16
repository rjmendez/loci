"""Command offload lane: a role mapped to a lane name is served by a configured command.

subprocess is replaced by a recording stub, the clock by a fake; no tmux, no network.
"""
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import backends  # noqa: E402
import llm_local as L  # noqa: E402
import offload_lane  # noqa: E402
import server  # noqa: E402

CMD = ["tool-bin", "offload.py"]


class _Proc:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


class _Runner:
    """Records every subprocess call; replies per subcommand."""

    def __init__(self):
        self.calls = []
        self.health = 0
        self.ask_rc = 0
        self.ask_out = "hello"
        self.ask_exc = None
        self.health_exc = None

    def __call__(self, cmd, **kw):
        self.calls.append({"cmd": list(cmd), "input": kw.get("input"), "timeout": kw.get("timeout")})
        if cmd[-1] == "health":
            if self.health_exc:
                raise self.health_exc
            return _Proc(self.health)
        if self.ask_exc:
            raise self.ask_exc
        return _Proc(self.ask_rc, self.ask_out)

    def asks(self):
        return [c for c in self.calls if "ask" in c["cmd"]]

    def healths(self):
        return [c for c in self.calls if c["cmd"][-1] == "health"]


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def env(monkeypatch):
    runner = _Runner()
    clock = _Clock()
    offload_lane.reset_cache()
    monkeypatch.setattr(offload_lane, "_run", runner)
    monkeypatch.setattr(offload_lane, "_clock", clock)
    monkeypatch.setattr(backends, "tmux_offload_enabled", lambda: True)
    monkeypatch.setattr(backends, "tmux_offload_role_sessions", lambda: {"triage": "copilot"})
    monkeypatch.setattr(backends, "tmux_offload_expensive_roles", lambda: set())
    monkeypatch.setattr(backends, "tmux_offload_require_mapped_session", lambda: False)
    monkeypatch.setattr(backends, "_config", lambda: {"offload_lane": {"copilot": {"command": CMD}}})
    monkeypatch.setattr(L, "_gen_env", lambda _v="http://fake-ollama:11434": _v)
    monkeypatch.setattr(backends, "ollama_gen_model", lambda: "qwen2.5:3b")
    yield runner, clock
    offload_lane.reset_cache()


def _set_lane(monkeypatch, **extra):
    cfg = {"command": CMD}
    cfg.update(extra)
    monkeypatch.setattr(backends, "_config", lambda: {"offload_lane": {"copilot": cfg}})


def _tmux_recorder(monkeypatch, live):
    seen = []

    def fake(name):
        seen.append(name)
        return live
    monkeypatch.setattr(L, "_tmux_session_available", fake)
    return seen


def _fake_ollama(monkeypatch):
    """The normal fallback path: requests.post is a recording stub answering 'ollama'."""
    posts = []

    class _R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "ollama"}

    def post(url, json=None, timeout=None):
        posts.append(url)
        return _R()
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=post, get=lambda *a, **k: _R()))
    return posts


# --- config ---------------------------------------------------------------

def test_offload_lane_defaults(monkeypatch):
    monkeypatch.setattr(backends, "_config", lambda: {"offload_lane": {"copilot": {"command": CMD}}})
    assert backends.offload_lane("copilot") == {
        "command": CMD, "allow_cloud": False, "health_timeout_s": 30.0,
        "timeout_s": 180.0, "cache_ttl_s": 60.0, "max_chars": 20000,
    }


def test_offload_lane_none_without_config_or_command(monkeypatch):
    monkeypatch.setattr(backends, "_config", lambda: {})
    assert backends.offload_lane("copilot") is None
    monkeypatch.setattr(backends, "_config", lambda: {"offload_lane": {"copilot": {"command": []}}})
    assert backends.offload_lane("copilot") is None
    monkeypatch.setattr(backends, "_config", lambda: {"offload_lane": {"copilot": {"command": CMD}}})
    assert backends.offload_lane("other") is None
    assert backends.offload_lane("copilot")["command"] == CMD


def test_offload_lane_reads_overrides(monkeypatch):
    monkeypatch.setattr(backends, "_config", lambda: {"offload_lane": {"copilot": {
        "command": CMD, "allow_cloud": True, "timeout_s": 5, "max_chars": 10}}})
    lane = backends.offload_lane("copilot")
    assert (lane["allow_cloud"], lane["timeout_s"], lane["max_chars"]) == (True, 5.0, 10)


# --- availability + cache -------------------------------------------------

def test_available_true_when_health_exits_zero(env):
    runner, _ = env
    lane = backends.offload_lane("copilot")
    assert offload_lane.available("copilot", lane) is True
    assert runner.calls == [{"cmd": CMD + ["health"], "input": None, "timeout": 30.0}]


def test_available_false_when_health_nonzero_or_raises(env):
    runner, clock = env
    lane = backends.offload_lane("copilot")
    runner.health = 1
    assert offload_lane.available("copilot", lane) is False
    clock.t += 61
    runner.health, runner.health_exc = 0, subprocess.TimeoutExpired(CMD, 30)
    assert offload_lane.available("copilot", lane) is False
    clock.t += 61
    runner.health_exc = None
    assert offload_lane.available("copilot", lane) is True  # positive twin


def test_available_cached_until_ttl(env):
    runner, clock = env
    lane = backends.offload_lane("copilot")
    offload_lane.available("copilot", lane)
    clock.t += 59
    offload_lane.available("copilot", lane)
    assert len(runner.healths()) == 1
    clock.t += 2  # 61 s since the probe
    offload_lane.available("copilot", lane)
    assert len(runner.healths()) == 2


# --- policy ---------------------------------------------------------------

def test_policy_uses_command_lane_and_never_asks_tmux(env, monkeypatch):
    seen = _tmux_recorder(monkeypatch, True)
    policy = L._tmux_offload_policy("triage")
    assert policy == {
        "enabled": True, "role": "triage", "priority": "normal", "session": "copilot",
        "available": True, "require_mapped_session": False,
        "reason": "command lane available", "lane": "command",
    }
    assert seen == []


def test_policy_unavailable_lane_reported(env):
    runner, _ = env
    runner.health = 1
    policy = L._tmux_offload_policy("triage")
    assert (policy["available"], policy["reason"]) == (False, "command lane unavailable")


def test_tmux_path_unchanged_without_lane_config(env, monkeypatch):
    runner, _ = env
    monkeypatch.setattr(backends, "_config", lambda: {})
    seen = _tmux_recorder(monkeypatch, True)
    policy = L._tmux_offload_policy("triage")
    assert policy == {
        "enabled": True, "role": "triage", "priority": "normal", "session": "copilot",
        "available": True, "require_mapped_session": False, "reason": "session available",
    }
    assert seen == ["copilot"]
    assert runner.calls == []


def test_strict_refusal_text_unchanged_when_lane_down(env, monkeypatch):
    runner, _ = env
    runner.health = 1
    monkeypatch.setattr(backends, "tmux_offload_require_mapped_session", lambda: True)
    r = L.generate("triage this", role="triage")
    assert r["ok"] is False
    assert r["tier"] == "tmux-offload-refused"
    assert r["why"] == "tmux offload required for role 'triage' but session 'copilot' is unavailable"
    assert runner.asks() == []


def test_strict_mode_served_when_lane_up(env, monkeypatch):
    runner, _ = env
    runner.ask_out = "lane says hi"
    monkeypatch.setattr(backends, "tmux_offload_require_mapped_session", lambda: True)
    r = L.generate("triage this", role="triage")
    assert (r["ok"], r["text"], r["tier"]) == (True, "lane says hi", "command-lane")


# --- dispatch -------------------------------------------------------------

def test_dispatch_serves_text_with_prompt_on_stdin_local_only(env):
    runner, _ = env
    runner.ask_out = "the answer"
    r = L.generate("what is 2+2", role="triage")
    assert r["ok"] is True
    assert r["text"] == "the answer"
    assert r["model"] == "command-lane:copilot"
    assert r["tier"] == "command-lane"
    assert runner.asks() == [{"cmd": CMD + ["ask", "--prompt", "-", "--local-only"],
                              "input": "what is 2+2", "timeout": 180.0}]


def test_dispatch_omits_local_only_only_when_allow_cloud(env, monkeypatch):
    runner, _ = env
    _set_lane(monkeypatch, allow_cloud=True)
    L.generate("p", role="triage")
    assert runner.asks()[0]["cmd"] == CMD + ["ask", "--prompt", "-"]


def test_dispatch_unwraps_offload_json_answer(env):
    runner, _ = env
    runner.ask_out = json.dumps({"answer": "inner", "model": "m", "note": "n"})
    assert L.generate("p", role="triage")["text"] == "inner"


def test_dispatch_caps_output(env, monkeypatch):
    runner, _ = env
    _set_lane(monkeypatch, max_chars=5)
    runner.ask_out = "0123456789"
    assert L.generate("p", role="triage")["text"] == "01234"


@pytest.mark.parametrize("what", ["timeout", "nonzero", "empty"])
def test_lane_failure_falls_open_to_local(env, monkeypatch, what):
    runner, _ = env
    posts = _fake_ollama(monkeypatch)
    if what == "timeout":
        runner.ask_exc = subprocess.TimeoutExpired(CMD, 180)
    elif what == "nonzero":
        runner.ask_rc = 2
    else:
        runner.ask_out = "  \n"
    r = L.generate("p", role="triage")
    assert len(runner.asks()) == 1
    assert (r["ok"], r["text"], r["model"]) == (True, "ollama", "qwen2.5:3b")
    assert len(posts) == 1


def test_ask_reasons(env):
    runner, _ = env
    lane = backends.offload_lane("copilot")
    runner.ask_exc = subprocess.TimeoutExpired(CMD, 180)
    assert offload_lane.ask(lane, "p") == {"ok": False, "text": "", "why": "command lane timed out after 180.0s"}
    runner.ask_exc, runner.ask_rc = None, 3
    assert offload_lane.ask(lane, "p")["why"] == "command lane exited 3"
    runner.ask_rc, runner.ask_out = 0, ""
    assert offload_lane.ask(lane, "p")["why"] == "command lane returned no text"
    runner.ask_out = "ok"
    assert offload_lane.ask(lane, "p") == {"ok": True, "text": "ok", "why": ""}  # positive twin


def test_json_fmt_invalid_reply_falls_back(env, monkeypatch):
    runner, _ = env
    posts = _fake_ollama(monkeypatch)
    runner.ask_out = "not json"
    r = L.generate("p", role="triage", fmt="json")
    assert len(runner.asks()) == 1
    assert len(posts) == 1
    assert r["model"] == "qwen2.5:3b"
    assert "tier" not in r


def test_json_fmt_valid_reply_is_served(env, monkeypatch):
    runner, _ = env
    posts = _fake_ollama(monkeypatch)
    runner.ask_out = '{"a": 1}'
    r = L.generate("p", role="triage", fmt="json")
    assert (r["text"], r["tier"], posts) == ('{"a": 1}', "command-lane", [])


def test_unavailable_lane_never_asks(env, monkeypatch):
    runner, _ = env
    runner.health = 1
    posts = _fake_ollama(monkeypatch)
    L.generate("p", role="triage")
    assert runner.asks() == []
    assert len(posts) == 1


# --- loci_health ----------------------------------------------------------

class _Tmux:
    stdout = "claude: 1 windows (created ...)\n"
    returncode = 0


def _health(monkeypatch, required):
    if required:
        monkeypatch.setenv("LOCI_TMUX_COMPANION_REQUIRED", "1")
    else:
        monkeypatch.delenv("LOCI_TMUX_COMPANION_REQUIRED", raising=False)
    monkeypatch.setenv("LOCI_TMUX_COMPANION_SESSIONS", "claude,copilot")
    monkeypatch.setattr(backends, "_alive", lambda url, timeout=1.0: True)
    monkeypatch.setattr(backends, "_http_probe", lambda *a, **k: (True, {}))
    monkeypatch.setattr(backends, "ollama_url", lambda *a, **k: "http://localhost:11434")
    monkeypatch.setattr(backends, "ollama_gen_url", lambda *a, **k: "http://localhost:11434")
    monkeypatch.setattr(backends, "qdrant", lambda: ("http://localhost:6333", ""))
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: _Tmux())
    return json.loads(server.loci_health())


def test_health_healthy_lane_replaces_missing_copilot(env, monkeypatch):
    out = _health(monkeypatch, required=True)
    assert out["status"] == "ok"
    assert out["tmux_companion_command_lanes"] == ["copilot"]
    assert "tmux_companion_missing" not in out


def test_health_unhealthy_lane_still_reports_missing(env, monkeypatch):
    runner, _ = env
    runner.health = 1
    out = _health(monkeypatch, required=True)
    assert out["status"] == "unhealthy"
    assert out["tmux_companion_missing"] == ["copilot"]
    assert "tmux_companion_command_lanes" not in out


def test_health_without_lane_config_unchanged(env, monkeypatch):
    monkeypatch.setattr(backends, "_config", lambda: {})
    out = _health(monkeypatch, required=False)
    assert out["tmux_companion_optional_down"] == ["copilot"]
    assert out["tmux_companion_missing"] == ["copilot"]

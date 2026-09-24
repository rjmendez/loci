"""route_audit: writes where it says, reports honestly, never logs prompt text."""
import json
import os
import stat
import sys

import pytest

import route_audit


def _events(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_creates_missing_parent_dir_and_writes(tmp_path):
    log = tmp_path / "fresh" / "nested" / "event_log.jsonl"
    assert route_audit.record_route_event(
        tier="local", route="ollama", reason="generate_success",
        prompt="hunter2 is the password", model="m", log_path=str(log),
    ) is True
    [ev] = _events(log)
    assert ev["op"] == "route_event"
    assert ev["route"] == "ollama"
    assert ev["prompt_len"] == len("hunter2 is the password")
    assert "hunter2" not in log.read_text()


def test_default_path_follows_env_at_call_time(tmp_path, monkeypatch):
    log = tmp_path / "later" / "event_log.jsonl"
    monkeypatch.setenv("LOCI_EVENT_LOG", str(log))
    assert route_audit.record_route_event(tier="t", route="r", reason="x") is True
    assert len(_events(log)) == 1


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0,
                    reason="needs POSIX permissions and a non-root user")
def test_reports_false_when_write_fails(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert route_audit.record_route_event(
            tier="t", route="r", reason="x", log_path=str(ro / "event_log.jsonl"),
        ) is False
    finally:
        ro.chmod(stat.S_IRWXU)


def test_opt_out_env_disables_writes(tmp_path, monkeypatch):
    log = tmp_path / "event_log.jsonl"
    monkeypatch.setenv("LOCI_ROUTE_AUDIT", "0")
    assert route_audit.record_route_event(
        tier="t", route="r", reason="x", log_path=str(log),
    ) is False
    assert not log.exists()


def test_llm_local_generate_emits_events_without_prompt_text(tmp_path, monkeypatch):
    import llm_local

    log = tmp_path / "event_log.jsonl"
    monkeypatch.setenv("LOCI_EVENT_LOG", str(log))

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "fine"}

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    monkeypatch.setenv("LOCI_OLLAMA_GEN_URL", "http://127.0.0.1:9")
    out = llm_local.generate("top secret prompt", model="qwen2.5:3b")
    assert out["ok"] is True
    text = log.read_text()
    assert "top secret" not in text
    reasons = [e["reason"] for e in _events(log)]
    assert reasons == ["generate_request", "generate_success"]

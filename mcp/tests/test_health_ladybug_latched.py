"""loci_health must not read 'ok' while the graph store is permanently latched (#422)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server  # noqa: E402


def _health(monkeypatch, state, last_error="", since=""):
    monkeypatch.setattr(server, "_ladybug_health_state", lambda: state)
    monkeypatch.setattr(server, "_ladybug_last_error", last_error)
    monkeypatch.setattr(server, "_ladybug_since", since)
    return json.loads(server.loci_health())


def test_latched_graph_store_is_a_failure_with_reason_and_time(monkeypatch):
    out = _health(monkeypatch, "latched", "graph module missing: ImportError('x')", "2026-10-04T10:00:00+00:00")
    assert out["ladybug"] == "latched"
    assert out["status"] != "ok", out
    assert any(f.startswith("ladybug: graph store latched") and "2026-10-04T10:00:00" in f and "ImportError" in f
               for f in out["failures"]), out["failures"]
    assert out["ladybug_last_error"].startswith("graph module missing")
    assert out["ladybug_failing_since"] == "2026-10-04T10:00:00+00:00"


def test_available_graph_store_adds_no_failure_or_error_fields(monkeypatch):
    out = _health(monkeypatch, "available")
    assert not any("ladybug" in f for f in out.get("failures", [])), out
    assert "ladybug_last_error" not in out


def test_transient_contention_alone_is_not_a_failure(monkeypatch):
    out = _health(monkeypatch, "contended")
    assert not any("ladybug" in f for f in out.get("failures", [])), out


def test_note_failure_keeps_the_first_time_and_latest_reason(monkeypatch):
    monkeypatch.setattr(server, "_ladybug_last_error", "")
    monkeypatch.setattr(server, "_ladybug_since", "")
    server._ladybug_note_failure("first")
    first_since = server._ladybug_since
    assert first_since and server._ladybug_last_error == "first"
    server._ladybug_note_failure("second " + "x" * 400)
    assert server._ladybug_since == first_since
    assert server._ladybug_last_error.startswith("second") and len(server._ladybug_last_error) == 200

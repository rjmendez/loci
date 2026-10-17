"""Tests for the `ground` MCP tool — the warm-server wrapper over grounding.ground()."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server  # noqa: E402 — must follow the path setup above


def test_ground_empty_title_guard():
    out = json.loads(server.ground("   "))
    assert out["degraded"] is True
    assert out["chars"] == 0
    assert out["sources"] == []
    assert "error" in out


def test_ground_delegates_and_serializes(monkeypatch):
    captured = {}

    def fake_ground(task, opts):
        captured["task"] = task
        captured["opts"] = opts
        return {"block": "## X", "sources": ["case:c1"], "chars": 4, "degraded": False}

    import grounding
    monkeypatch.setattr(grounding, "ground", fake_ground)

    out = json.loads(server.ground(
        "decompose X", focus="split it", case_ids=["c1"], entities=["e1"],
        code_refs=["foo"], budget_chars=1234, allow_keyword=True, graph_available=True,
    ))
    assert out == {"block": "## X", "sources": ["case:c1"], "chars": 4, "degraded": False}
    # task/opts are mapped from the tool's snake_case params to grounding's camelCase keys
    assert captured["task"] == {
        "title": "decompose X", "focus": "split it",
        "caseIds": ["c1"], "entities": ["e1"], "codeRefs": ["foo"],
    }
    assert captured["opts"] == {
        "budgetChars": 1234, "allowKeyword": True, "graphAvailable": True,
    }


def test_ground_defaults_are_conservative(monkeypatch):
    captured = {}
    import grounding
    monkeypatch.setattr(grounding, "ground",
                        lambda task, opts: captured.update(task=task, opts=opts) or
                        {"block": "", "sources": [], "chars": 0, "degraded": False})
    server.ground("t")
    # keyword + code-graph lanes default OFF; empty lists, not None
    assert captured["opts"]["allowKeyword"] is False
    assert captured["opts"]["graphAvailable"] is False
    assert captured["task"]["caseIds"] == [] and captured["task"]["entities"] == []

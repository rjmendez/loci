"""Tests for scripts/graph_facts.py — the deterministic code-graph fact producer."""
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import graph_facts as GF  # noqa: E402


def test_summarize_impact_found():
    r = {"resolved": [{"id": "m.py::foo", "name": "foo"}],
         "direct_callers": ["a", "b", "a"], "transitive_caller_count": 3,
         "co_referenced": [{"name": "bar"}], "referencing_finding_count": 2,
         "investigations": [{"id": "inv1"}]}
    t = GF._summarize_impact("foo", r)
    assert "resolved to 1 symbol" in t
    assert "a, b" in t and t.count("a,") == 1  # de-duped
    assert "transitive callers: 3" in t
    assert "co-referenced" in t and "bar" in t
    assert "2 finding" in t and "inv1" in t


def test_summarize_impact_not_found():
    assert "not found" in GF._summarize_impact("ghost", {"resolved": []})


def test_summarize_subsystem_and_dead():
    s = GF._summarize_subsystem("x/", {"files": ["x/a.py"], "symbol_count": 5,
                                       "kinds": {"function": 5},
                                       "hotspot_symbols": [{"name": "h", "findings": 3}]})
    assert "1 file" in s and "h:3" in s
    assert "no files matched" in GF._summarize_subsystem("y/", {"files": []})
    d = GF._summarize_dead({"candidates": [{"name": "dead1"}, {"name": "dead2"}]})
    assert "dead1" in d and "dead2" in d and "(2)" in d
    assert "no dead-code" in GF._summarize_dead({"candidates": []})


def test_main_no_graph_is_fail_open(monkeypatch, capsys):
    monkeypatch.setattr(GF, "_find_graph", lambda: None)
    monkeypatch.setattr(sys, "argv", ["graph_facts.py", '[{"key":"k1","impact":"foo"}]'])
    rc = GF.main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["k1"]["kind"] == "unavailable"  # degrades, does not crash


def test_main_dispatches_per_request(monkeypatch, capsys):
    monkeypatch.setattr(GF, "_find_graph", lambda: "/fake/graph.kuzu")

    class FakeKS:
        def __init__(self, *a, **k):
            pass

    fake_analytics = type("A", (), {
        "impact_report": staticmethod(lambda ks, s: {"resolved": [{"id": f"m::{s}", "name": s}],
                                                     "direct_callers": []}),
        "subsystem_report": staticmethod(lambda ks, p: {"files": []}),
        "dead_code_candidates": staticmethod(lambda ks: {"candidates": []}),
    })
    import graph.kuzu_store as kstore
    import graph.analytics as analytics
    monkeypatch.setattr(kstore, "KuzuStore", FakeKS)
    for name in ("impact_report", "subsystem_report", "dead_code_candidates"):
        monkeypatch.setattr(analytics, name, getattr(fake_analytics, name))

    monkeypatch.setattr(sys, "argv", ["graph_facts.py",
                        '[{"key":"a","impact":"foo"},{"key":"b","subsystem":"x/"},{"key":"c","deadCode":true}]'])
    GF.main()
    out = json.loads(capsys.readouterr().out)
    assert out["a"]["kind"] == "impact" and "foo" in out["a"]["text"]
    assert out["b"]["kind"] == "subsystem"
    assert out["c"]["kind"] == "dead_code"

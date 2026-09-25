"""ground() must say when a lane failed, and must return within its deadline.

Regressions (audit ids ground-degraded-false-on-failed-lanes,
ground-degraded-false-when-lanes-fail, ground-unbounded-latency):

- rag_context_search reports total failure as ``mode='rag_failed'`` with
  ``qdrant_available=True``; ground() only looked at ``qdrant_available`` and
  returned ``{block:'', degraded:False}``, byte-identical to "nothing stored".
- The case / resolved / entity / keyword lanes swallowed exceptions without
  touching ``degraded``.
- ground() had no time budget, so a slow generation backend (query expansion,
  120s per request) held the call past the MCP client's 300s idle abort.
"""
import json
import sys
import time
import types

import grounding as G


def _rag_failed(q, **k):
    # The exact shape server.rag_context_search returns when every collection fails to embed.
    return json.dumps({
        "query": q, "context": "", "sources": [], "total_chars": 0, "truncated": False,
        "result_count": 0, "mode": "rag_failed",
        "collections_searched": ["loci_memory"], "collections_failed": ["loci_memory"],
        "qdrant_available": True, "collection_errors": ["loci_memory: embedding_unavailable"],
    })


def _healthy_fake():
    fake = types.ModuleType("server")
    fake.investigation_load = lambda cid, **k: {"manifest": {}, "recent_findings": []}
    fake.investigation_entity_lookup = lambda ent, **k: {"total_findings": 0}
    fake.rag_context_search = lambda q, **k: {"context": "", "result_count": 0,
                                               "qdrant_available": True, "mode": "rag_hybrid",
                                               "collections_failed": []}
    fake.investigation_search = lambda q, **k: {"results": []}
    fake.impact_report = lambda ref, **k: {"resolved": [], "partial": False}
    return fake


def _lanes(r):
    return {d["lane"] for d in r["degraded_lanes"]}


def test_rag_failed_on_every_collection_is_degraded(monkeypatch):
    fake = _healthy_fake()
    fake.rag_context_search = _rag_failed
    monkeypatch.setitem(sys.modules, "server", fake)
    r = G.ground({"title": "uwb ranging drift"}, {"memoryDir": ""})
    assert r["chars"] == 0
    assert r["degraded"] is True, "a RAG lane that consulted nothing must not read as full coverage"
    assert "rag" in _lanes(r)


def test_partial_rag_failure_keeps_context_but_degrades(monkeypatch):
    fake = _healthy_fake()
    fake.rag_context_search = lambda q, **k: {
        "context": "prior note about uwb", "result_count": 1, "mode": "rag_degraded",
        "qdrant_available": True, "collections_failed": ["code_chunks"]}
    monkeypatch.setitem(sys.modules, "server", fake)
    r = G.ground({"title": "uwb ranging drift"}, {"memoryDir": ""})
    assert "prior note about uwb" in r["block"]
    assert r["degraded"] is True and "rag" in _lanes(r)


def test_raising_case_and_entity_lanes_are_degraded_and_named(monkeypatch):
    def _boom(*a, **k):
        raise OSError("store unreadable")

    fake = _healthy_fake()
    fake.investigation_load = _boom
    fake.investigation_entity_lookup = _boom
    monkeypatch.setitem(sys.modules, "server", fake)
    r = G.ground({"title": "uwb ranging drift", "caseIds": ["uwb-case"], "entities": ["10.0.0.5"]},
                 {"memoryDir": ""})
    assert r["degraded"] is True
    assert {"case", "resolved", "entity"} <= _lanes(r)
    assert any("store unreadable" in d["reason"] for d in r["degraded_lanes"])


def test_all_lanes_failing_live_shape(monkeypatch):
    """The auditors' live repro: sentinel case id raising, entity index down, RAG failed."""
    fake = _healthy_fake()

    def _load(cid, **k):
        raise ValueError("Invalid investigation_id: %r is a missing-value sentinel" % cid)

    def _ent(ent, **k):
        raise RuntimeError("entity index unavailable")

    fake.investigation_load = _load
    fake.investigation_entity_lookup = _ent
    fake.rag_context_search = _rag_failed
    monkeypatch.setitem(sys.modules, "server", fake)
    r = G.ground({"title": "deep think telemetry audit", "caseIds": ["undefined"],
                  "entities": ["telemetry.deepthink.internal"]}, {"budgetChars": 1500, "memoryDir": ""})
    assert r["degraded"] is True
    assert {"case", "entity", "rag"} <= _lanes(r)


def test_missing_case_is_an_honest_empty_not_a_failure(monkeypatch):
    fake = _healthy_fake()
    fake.investigation_load = lambda cid, **k: {
        "error": f"Investigation '{cid}' not found. Call investigation_start first."}
    monkeypatch.setitem(sys.modules, "server", fake)
    r = G.ground({"title": "t", "caseIds": ["nope"]}, {"memoryDir": ""})
    assert r["degraded"] is False and r["degraded_lanes"] == []


def test_healthy_empty_run_is_not_degraded(monkeypatch):
    monkeypatch.setitem(sys.modules, "server", _healthy_fake())
    r = G.ground({"title": "t", "caseIds": ["c1"], "entities": ["e"]}, {"memoryDir": ""})
    assert r["degraded"] is False and r["degraded_lanes"] == []


def test_error_payloads_from_entity_code_and_keyword_lanes_degrade(monkeypatch):
    fake = _healthy_fake()
    fake.investigation_entity_lookup = lambda ent, **k: {"error": "entity index offline"}
    fake.impact_report = lambda ref, **k: {"error": "LadybugDB graph store unavailable."}
    fake.investigation_search = lambda q, **k: {"error": "fts down", "results": []}
    monkeypatch.setitem(sys.modules, "server", fake)
    r = G.ground({"title": "t", "entities": ["e"], "codeRefs": ["f"]},
                 {"memoryDir": "", "graphAvailable": True, "allowKeyword": True})
    assert {"entity", "code_graph", "keyword"} <= _lanes(r)


def test_partial_impact_report_degrades(monkeypatch):
    fake = _healthy_fake()
    fake.impact_report = lambda ref, **k: {"resolved": [{"id": "f", "name": "f", "kind": "fn"}],
                                           "transitive_caller_count": 0, "partial": True,
                                           "queries_failed": 2, "co_referenced": []}
    monkeypatch.setitem(sys.modules, "server", fake)
    r = G.ground({"title": "t", "codeRefs": ["f"]}, {"memoryDir": "", "graphAvailable": True})
    assert "code_graph" in _lanes(r)


def test_slow_lane_hits_deadline_and_is_marked(monkeypatch):
    fake = _healthy_fake()

    def _slow_rag(q, **k):
        time.sleep(5)
        return {"context": "late", "result_count": 1, "qdrant_available": True}

    fake.rag_context_search = _slow_rag
    monkeypatch.setitem(sys.modules, "server", fake)
    t0 = time.monotonic()
    r = G.ground({"title": "t"}, {"memoryDir": "", "deadlineS": 0.3})
    elapsed = time.monotonic() - t0
    assert elapsed < 2.5, f"ground ignored its deadline ({elapsed:.1f}s)"
    assert r["degraded"] is True
    assert {"lane": "rag", "reason": "timeout"} in r["degraded_lanes"]
    assert "late" not in r["block"]


def test_deadline_is_env_configurable(monkeypatch):
    fake = _healthy_fake()

    def _slow_load(cid, **k):
        time.sleep(5)
        return {"manifest": {}, "recent_findings": []}

    fake.investigation_load = _slow_load
    monkeypatch.setitem(sys.modules, "server", fake)
    monkeypatch.setenv("LOCI_GROUND_DEADLINE_S", "0.3")
    t0 = time.monotonic()
    r = G.ground({"title": "t", "caseIds": ["c1", "c2"]}, {"memoryDir": ""})
    assert time.monotonic() - t0 < 2.5
    assert "case" in _lanes(r) and r["degraded"] is True


def test_deadline_defaults_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("LOCI_GROUND_DEADLINE_S", raising=False)
    assert G._deadline_seconds({}) == G._GROUND_DEADLINE_DEFAULT_S
    monkeypatch.setenv("LOCI_GROUND_DEADLINE_S", "0")
    assert G._deadline_seconds({}) is None
    monkeypatch.setenv("LOCI_GROUND_DEADLINE_S", "garbage")
    assert G._deadline_seconds({}) == G._GROUND_DEADLINE_DEFAULT_S


def test_real_rag_context_search_embed_outage_degrades_ground(monkeypatch):
    """End to end through the real server.rag_context_search (the audit repro)."""
    import server

    monkeypatch.setenv("LOCI_RAG_EXPAND", "0")
    monkeypatch.setattr(server, "_get_qdrant", lambda: (object(), "loci_memory"))

    def _embed_down(*a, **k):
        raise RuntimeError("embedding_unavailable")

    monkeypatch.setattr(server, "_qdrant_search_collection", _embed_down)
    # Keep the test offline: loading the reranker probes CUDA, which can hang on some hosts.
    monkeypatch.setattr(server, "_rag_cross_encode", lambda *a, **k: None)
    monkeypatch.setattr(server, "_rag_record_access", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "server", server)
    r = G.ground({"title": "uwb ranging drift"}, {"memoryDir": "", "deadlineS": 30})
    assert r["degraded"] is True and "rag" in _lanes(r)

"""Every read path must drop a retracted finding, not just investigation_search and rag_context_search.

2026-10-05 audit (S1-01, S2-03, S3-04): docs_search/docs_recall, procedure_search, memory_hints, the entity tools and
memory_confidence read findings.jsonl or the index directly and returned a finding after memory_retract had removed it.
Each test stores a doc/finding, proves the tool returns it, retracts it, and proves it is gone while a second finding
that was not retracted still comes back (so the filter is not simply returning nothing). Temp store, side effects stubbed.
"""
import json
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import inv_store  # noqa: E402
import mnemo_ops  # noqa: E402
import server  # noqa: E402


def _j(s: str) -> dict:
    return json.loads(s)


@pytest.fixture
def store(tmp_path, monkeypatch):
    mem = tmp_path / "mem"
    mem.mkdir()
    monkeypatch.setattr(server, "MEMORY_DIR", mem)
    monkeypatch.setenv("LOCI_CODE_ROOT", str(tmp_path / "code"))
    monkeypatch.setenv("LOCI_RAG_EXPAND", "0")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setattr(inv_store, "_STORE_LOCK_TIMEOUT_S", 0.5)
    inv_store._manifest_cache.clear()
    server._session_hints.clear()
    no = lambda *a, **k: None  # noqa: E731
    for name, value in {
        "_get_qdrant": lambda *a, **k: (None, None),
        "_qdrant_upsert": no,
        "_mnemo_remember": lambda *a, **k: False,
        "_event_log_append": no,
        "_mirror_finding_to_ladybug": no,
        "_autolink_finding_to_ladybug": no,
        "_retract_quarantine_verdict": lambda *a, **k: False,
        "_forget_finding_verdicts": lambda *a, **k: 0,
        "_semantic_neighbor_ids": lambda *a, **k: [],
        "_get_cross_encoder": lambda *a, **k: None,
        "_entity_lookup_ladybug": lambda *a, **k: [],
    }.items():
        monkeypatch.setattr(server, name, value, raising=False)
    no_mnemo = lambda: (None, None)  # noqa: E731
    monkeypatch.setattr(server, "_get_mnemo_funcs", no_mnemo)
    monkeypatch.setattr(mnemo_ops, "_get_mnemo_funcs", no_mnemo)
    yield mem
    inv_store._manifest_cache.clear()
    server._session_hints.clear()


def _start(inv):
    assert _j(server.investigation_start(investigation_id=inv, title="read paths")).get("status") == "created"
    return inv


def _store(inv, text, ftype="observed", **kw):
    r = _j(server.investigation_store(inv, ftype, text, "unit-test", **kw))
    assert r.get("stored"), r
    return r["finding_id"]


def _retract(inv, fid):
    r = _j(server.memory_retract(inv, fid, reason="hallucination", dry_run=False, scope_semantic=False))
    assert r.get("applied") is True, r


# -------------------------------------------------------------------------------------------------- docs

def test_docs_search_and_docs_recall_drop_a_retracted_doc(store):
    inv = _start("docs-case")
    gone = _store(inv, "gizmo calibration steps for the old firmware", tags="docs",
                  metadata={"source_path": "gizmo_old.md", "title": "Gizmo calibration old"})
    kept = _store(inv, "gizmo calibration steps for the new firmware", tags="docs",
                  metadata={"source_path": "gizmo_new.md", "title": "Gizmo calibration new"})
    ids = lambda out: {r["finding_id"] for r in _j(out)["results"]}  # noqa: E731
    assert ids(server.docs_search("gizmo calibration", investigation_id=inv)) == {gone, kept}
    _retract(inv, gone)
    assert ids(server.docs_search("gizmo calibration", investigation_id=inv)) == {kept}
    assert gone not in ids(server.docs_recall("gizmo calibration", investigation_id=inv))


# -------------------------------------------------------------------------------------------- procedures

def test_procedure_search_drops_a_retracted_procedure(store):
    inv = _start("proc-case")
    kw = dict(procedure_preconditions="pre", procedure_steps="1. do it", procedure_postconditions="post")
    gone = _store(inv, "restart the widget service safely", ftype="procedure", **kw)
    kept = _store(inv, "restart the widget cache safely", ftype="procedure", **kw)
    ids = lambda: {p["finding_id"] for p in _j(server.procedure_search("restart the widget", investigation_id=inv))["procedures"]}  # noqa: E731
    assert ids() == {gone, kept}
    _retract(inv, gone)
    assert ids() == {kept}


# ------------------------------------------------------------------------------------------------- hints

@pytest.mark.parametrize("path", ["ring_buffer", "file_tail"])
def test_memory_hints_drop_a_retracted_finding(store, path):
    inv = _start("hints-case")
    gone = _store(inv, "the retracted claim about the staging cluster")
    kept = _store(inv, "the surviving claim about the staging cluster")
    if path == "file_tail":
        server._session_hints.clear()                # force the findings.jsonl path
    ids = lambda: {h["finding_id"] for h in _j(server.memory_hints(inv, limit=10))["hints"]}  # noqa: E731
    assert {gone, kept} <= ids()
    _retract(inv, gone)
    if path == "file_tail":
        server._session_hints.clear()
    got = ids()
    assert gone not in got and kept in got


# ----------------------------------------------------------------------------------------------- entities

def test_entity_lookup_drops_a_retracted_finding(store):
    inv = _start("ent-case")
    gone = _store(inv, "host 10.9.8.7 was compromised according to a hallucinated report")
    def lookup():
        return _j(server.investigation_entity_lookup("10.9.8.7", "ip", investigation_id=inv))
    assert lookup()["total_findings"] == 1
    _retract(inv, gone)            # also retracts any finding that shares the entity (its lineage rule)
    out = lookup()
    assert out["total_findings"] == 0 and "hallucinated" not in json.dumps(out)
    _store(inv, "host 10.9.8.7 is the bastion and is healthy")      # stored after the retraction: must appear
    out = lookup()
    assert out["total_findings"] == 1 and "bastion" in json.dumps(out) and "hallucinated" not in json.dumps(out)


def test_entity_list_and_timeline_ignore_retracted_findings(store):
    inv = _start("ent-list-case")
    gone = _store(inv, 'Zebulon Quark reported the "Frobnicator" outage')
    kept = _store(inv, 'Zebulon Quark later closed the "Frobnicator" ticket')
    listed = _j(server.entity_list(inv))["entities"]
    ent = next(e for e in listed if "Zebulon" in (e["name"] or ""))
    assert ent["finding_count"] == 2
    _retract(inv, gone)
    ent = next(e for e in _j(server.entity_list(inv))["entities"] if "Zebulon" in (e["name"] or ""))
    assert ent["finding_count"] == 1
    tl = _j(server.entity_timeline(inv, ent["entity_id"]))["timeline"]
    assert [t["finding_id"] for t in tl] == [kept]


def test_an_entity_whose_every_finding_was_retracted_disappears_from_the_list(store):
    inv = _start("ent-only-case")
    only = _store(inv, 'Hallucinated "Ghostwriter" actor appears here')
    assert any("Ghostwriter" in (e["name"] or "") for e in _j(server.entity_list(inv))["entities"])
    _retract(inv, only)
    assert not any("Ghostwriter" in (e["name"] or "") for e in _j(server.entity_list(inv))["entities"])


# ------------------------------------------------------------------------------------------------ confidence

def test_memory_confidence_does_not_count_a_retracted_neighbour(store, monkeypatch):
    inv = _start("conf-case")
    gone = _store(inv, "claim that will be retracted")
    _retract(inv, gone)
    live = "live-finding-id"
    rows = [
        {"id": gone, "investigation_id": inv, "text": "claim that will be retracted", "score": 0.9, "confidence": "high", "source": "s"},
        {"id": live, "investigation_id": "other-case", "text": "a live claim", "score": 0.8, "confidence": "high", "source": "s"},
    ]
    monkeypatch.setattr(server, "_confidence_retrieve", lambda q, k: (list(rows), None))
    out = _j(server.memory_confidence("anything"))
    refs = [r["finding_id"] for r in out["confidence_aggregation"]["evidence_refs"]]
    assert refs == [live], refs        # the retracted row is not evidence; the live one still is

"""Memory-use signal: which surfaced memories were actually used.

memory_surface exposures, the evidence ids a pre-answer check tied to each claim,
and the derived_from parents of stored findings go to one append-only log, with
ids and scores only. Nothing any tool returns changes.
"""
import json

import pytest

import server

CONTEXT = "tracing the flaky login redirect in the auth gateway"
FINDING_TEXT = "the auth gateway drops the session cookie on redirect"


def _json(raw):
    return json.loads(raw)


@pytest.fixture
def mem(tmp_path, monkeypatch):
    root = tmp_path / "memory-sessions"
    root.mkdir()
    original = server.MEMORY_DIR
    server.MEMORY_DIR = root
    server._session_hints.clear()
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(root))
    monkeypatch.delenv("LOCI_MEMORY_USE_LOG", raising=False)
    yield root
    server.MEMORY_DIR = original
    server._session_hints.clear()


def _events(mem):
    path = mem.parent / "instrumentation" / server.MEMORY_USE_LOG_NAME
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        assert row.pop("ts")
        assert row.pop("schema") == 1
    return rows


def _surface_stubs(monkeypatch):
    monkeypatch.setattr(server, "_get_qdrant", lambda: (object(), "loci_memory"))
    monkeypatch.setattr(server, "_qdrant_search_collection", lambda *a, **k: [
        {"id": "f-a", "investigation_id": "inv-x", "source": "q", "text": FINDING_TEXT, "score": 0.81234},
        {"id": "f-b", "investigation_id": "inv-y", "source": "q", "text": "second finding", "score": 0.5},
        {"id": "f-low", "investigation_id": "inv-y", "source": "q", "text": "below threshold", "score": 0.2},
    ])
    monkeypatch.setattr(server, "docs_search", lambda *a, **k: json.dumps({"results": [
        {"title": "runbook title", "summary": "docs summary text", "score": None},
    ]}))


def test_memory_surface_records_the_findings_it_returned(mem, monkeypatch):
    _surface_stubs(monkeypatch)
    out = _json(server.memory_surface(CONTEXT, top_k=5))
    assert [r["finding_id"] for r in out["surfaced"]] == ["f-a", "f-b", "runbook title"]
    assert _events(mem) == [{
        "event": "surfaced",
        "tool": "memory_surface",
        "scope_investigation_id": None,
        "items": [
            {"rank": 0, "finding_id": "f-a", "investigation_id": "inv-x", "score": 0.8123},
            {"rank": 1, "finding_id": "f-b", "investigation_id": "inv-y", "score": 0.5},
        ],
        "docs_rows": 1,
    }]


def test_surfaced_log_holds_no_context_or_text(mem, monkeypatch):
    _surface_stubs(monkeypatch)
    server.memory_surface(CONTEXT, top_k=5)
    raw = (mem.parent / "instrumentation" / server.MEMORY_USE_LOG_NAME).read_text()
    for forbidden in (CONTEXT, "auth gateway", "second finding", "runbook title", "docs summary"):
        assert forbidden not in raw


def test_nothing_surfaced_writes_nothing(mem, monkeypatch):
    monkeypatch.setattr(server, "_get_qdrant", lambda: (object(), "loci_memory"))
    monkeypatch.setattr(server, "_qdrant_search_collection", lambda *a, **k: [])
    monkeypatch.setattr(server, "docs_search", lambda *a, **k: json.dumps({"results": []}))
    server.memory_surface(CONTEXT, top_k=5)
    assert _events(mem) == []


def _seed(inv, text=FINDING_TEXT, derived_from=None):
    if not server._load_manifest(inv):
        server.investigation_start(investigation_id=inv, title=f"test {inv}")
    stored = _json(server.investigation_store(
        investigation_id=inv, finding_type="observed", text=text,
        source="unit-test", confidence="high", derived_from=derived_from,
        evidence_provenance_tier="tool_verified",
    ))
    return stored["finding_id"]


@pytest.fixture
def no_vector_lane(monkeypatch):
    # Lexical lane only: no embedding model load, no network.
    monkeypatch.setattr(server, "_search_qdrant_claim_evidence",
                        lambda *a, **k: ([], {"available": False}))
    monkeypatch.setattr(server, "_search_benign_context_qdrant", lambda *a, **k: [])


def test_pre_answer_check_records_evidence_ids_per_claim(mem, no_vector_lane):
    fid = _seed("use-check")
    out = _json(server.investigation_pre_answer_check(
        investigation_id="use-check",
        claims=[FINDING_TEXT, "an unrelated statement about billing exports"],
        record=False,
    ))
    assert [c["supported"] for c in out["claim_results"]] == [True, False]
    events = _events(mem)
    assert [e["event"] for e in events] == ["answer_check"]
    assert events[0] == {
        "event": "answer_check",
        "tool": "investigation_pre_answer_check",
        "investigation_id": "use-check",
        "claims": [
            {"claim_index": 0, "supported": True, "contradicted": False,
             "support_basis": "lexical",
             "support": [{"id": fid, "origin": "findings_jsonl", "score": 1.0}],
             "contradiction": [], "semantic_candidates": []},
            {"claim_index": 1, "supported": False, "contradicted": False,
             "support_basis": "none", "support": [], "contradiction": [],
             "semantic_candidates": []},
        ],
    }
    raw = (mem.parent / "instrumentation" / server.MEMORY_USE_LOG_NAME).read_text()
    assert FINDING_TEXT not in raw and "billing" not in raw



def test_every_support_and_contradiction_ref_is_logged(mem):
    """A memory cited as the 9th or later support ref must still count as used offline."""
    def refs(prefix, n):
        return [{"evidence_id": f"{prefix}-{i}", "origin": "findings_jsonl", "score": 0.5} for i in range(n)]

    assert server._record_memory_answer_check("use-many", [{
        "supported": True, "contradicted": True, "support_basis": "lexical",
        "support_refs": refs("s", 12), "contradiction_refs": refs("c", 10),
        "semantic_candidates": refs("v", 12),
    }]) is True
    [event] = _events(mem)
    [claim] = event["claims"]
    assert [r["id"] for r in claim["support"]] == [f"s-{i}" for i in range(12)]
    assert [r["id"] for r in claim["contradiction"]] == [f"c-{i}" for i in range(10)]
    # only the semantic candidates stay capped
    assert [r["id"] for r in claim["semantic_candidates"]] == [f"v-{i}" for i in range(8)]

def test_derived_from_parents_are_recorded_as_cited(mem):
    parent = _seed("use-cite")
    child = _seed("use-cite", text="the cookie drop comes from a SameSite change",
                  derived_from=[parent])
    assert _events(mem) == [{
        "event": "cited",
        "tool": "investigation_store",
        "investigation_id": "use-cite",
        "finding_id": child,
        "cited_ids": [parent],
        "cited_count": 1,
    }]


def test_rejected_derived_from_records_nothing(mem):
    _seed("use-cite-bad")
    out = _json(server.investigation_store(
        investigation_id="use-cite-bad", finding_type="observed", text="x y z",
        source="unit-test", derived_from=["no-such-parent"],
    ))
    assert "error" in out
    assert _events(mem) == []


def test_flag_off_writes_nothing_and_output_is_identical(mem, monkeypatch, no_vector_lane):
    _surface_stubs(monkeypatch)
    on = _json(server.memory_surface(CONTEXT, top_k=5))
    monkeypatch.setenv("LOCI_MEMORY_USE_LOG", "0")
    path = mem.parent / "instrumentation" / server.MEMORY_USE_LOG_NAME
    before = path.read_text()
    off = _json(server.memory_surface(CONTEXT, top_k=5))
    assert off == on
    parent = _seed("use-off")
    _seed("use-off", text="child finding", derived_from=[parent])
    server.investigation_pre_answer_check("use-off", FINDING_TEXT, record=False)
    assert path.read_text() == before


def test_log_failure_never_breaks_the_tool(mem, monkeypatch):
    _surface_stubs(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr("instrumentation_log.append_rows", _boom)
    out = _json(server.memory_surface(CONTEXT, top_k=5))
    assert out["count"] == 3
    parent = _seed("use-fail")
    child = _seed("use-fail", text="child finding", derived_from=[parent])
    assert child

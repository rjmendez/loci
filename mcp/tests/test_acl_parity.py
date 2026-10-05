"""The private-investigation ACL must hold on every tool that reads or changes an investigation.

2026-10-05 audit (S2-03, S3-03, S4-07): investigation_load, search and export checked the ACL, but memory_hints,
investigation_reason, memory_promote/demote, conflict_resolve, causal_infer, the entity tools, the queue tools,
investigation_note/reflect/finding_provenance, investigation_start (resuming returns the whole manifest),
investigation_list, docs_search/recall, procedure_search and memory_confidence did not, so a non-member could read a private
case and write into it. An investigation with neither an owner nor an ACL stays open to everyone (no behaviour change there).
"""
import json
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import inv_store  # noqa: E402
import investigation_tools as it  # noqa: E402
import mnemo_ops  # noqa: E402
import server  # noqa: E402

PRIVATE = "private-case"
OPEN = "open-case"


def _j(s):
    return json.loads(s)


@pytest.fixture
def store(tmp_path, monkeypatch):
    mem = tmp_path / "mem"
    mem.mkdir()
    monkeypatch.setattr(server, "MEMORY_DIR", mem)
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(mem))
    monkeypatch.setenv("LOCI_CODE_ROOT", str(tmp_path / "code"))
    monkeypatch.setenv("LOCI_RAG_EXPAND", "0")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("HERMES_AGENT_ID", "alice")
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


def _make(mem, inv, *, private):
    assert _j(server.investigation_start(investigation_id=inv, title="case " + inv)).get("status") == "created"
    fid = _j(server.investigation_store(inv, "observed", "SECRET host 10.4.4.4 was seen beaconing", "unit-test"))["finding_id"]
    if private:
        mpath = Path(mem) / inv / "manifest.json"
        m = json.loads(mpath.read_text())
        m["owner"], m["acl"] = "alice", ["alice", "bob"]
        mpath.write_text(json.dumps(m))
        inv_store._manifest_cache.clear()
    return fid


def _tools(inv, fid):
    """(label, callable(requesting_agent_id)) for every tool that takes an investigation id."""
    S, T = server, it
    return [
        ("memory_hints", lambda w: S.memory_hints(inv, requesting_agent_id=w)),
        ("investigation_reason", lambda w: S.investigation_reason(inv, "what happened", requesting_agent_id=w)),
        ("memory_promote", lambda w: S.memory_promote(inv, fid, "hot", requesting_agent_id=w)),
        ("memory_demote", lambda w: S.memory_demote(inv, fid, "cold", requesting_agent_id=w)),
        ("conflict_resolve", lambda w: S.conflict_resolve(inv, "cid", "a_wins", requesting_agent_id=w)),
        ("causal_infer", lambda w: S.causal_infer(inv, requesting_agent_id=w)),
        ("entity_list", lambda w: S.entity_list(inv, requesting_agent_id=w)),
        ("entity_timeline", lambda w: S.entity_timeline(inv, "eid", requesting_agent_id=w)),
        ("investigation_entity_lookup", lambda w: S.investigation_entity_lookup("10.4.4.4", "ip", investigation_id=inv, requesting_agent_id=w)),
        ("investigation_pre_answer_check", lambda w: S.investigation_pre_answer_check(inv, ["host 10.4.4.4 beaconed"], requesting_agent_id=w)),
        ("investigation_evidence_precheck", lambda w: S.investigation_evidence_precheck(inv, "beaconing", requesting_agent_id=w)),
        ("docs_search", lambda w: S.docs_search("beaconing", investigation_id=inv, requesting_agent_id=w)),
        ("docs_recall", lambda w: S.docs_recall("beaconing", investigation_id=inv, requesting_agent_id=w)),
        ("investigation_queue_enqueue", lambda w: T.investigation_queue_enqueue(inv, "i1", "{}", requesting_agent_id=w)),
        ("investigation_queue_claim", lambda w: T.investigation_queue_claim(inv, "i1", "s1", requesting_agent_id=w)),
        ("investigation_queue_complete", lambda w: T.investigation_queue_complete(inv, "i1", "s1", requesting_agent_id=w)),
        ("investigation_queue_release", lambda w: T.investigation_queue_release(inv, "i1", "s1", requesting_agent_id=w)),
        ("investigation_queue_status", lambda w: T.investigation_queue_status(inv, requesting_agent_id=w)),
        ("investigation_queue_list", lambda w: T.investigation_queue_list(inv, requesting_agent_id=w)),
        ("investigation_note", lambda w: T.investigation_note(inv, "hypothesis", "x", requesting_agent_id=w)),
        ("investigation_reflect", lambda w: T.investigation_reflect(inv, requesting_agent_id=w)),
        ("investigation_finding_provenance", lambda w: T.investigation_finding_provenance(fid, inv, requesting_agent_id=w)),
        ("investigation_start (resume)", lambda w: T.investigation_start(inv, "again", requesting_agent_id=w)),
    ]


def _denied(out):
    try:
        return json.loads(out).get("error") == "permission_denied"
    except Exception:
        return False


def test_every_listed_tool_denies_a_non_member_of_a_private_investigation(store):
    fid = _make(store, PRIVATE, private=True)
    leaks = []
    for label, call in _tools(PRIVATE, fid):
        out = call("mallory")
        if not _denied(out):
            leaks.append((label, str(out)[:120]))
        assert "SECRET" not in str(out), label
    assert not leaks, leaks


def test_a_member_is_not_denied_by_any_of_them(store):
    fid = _make(store, PRIVATE, private=True)
    refused = [label for label, call in _tools(PRIVATE, fid) if _denied(call("alice"))]
    assert not refused, refused


def test_an_investigation_with_no_owner_and_no_acl_stays_open(store):
    fid = _make(store, OPEN, private=False)
    refused = [label for label, call in _tools(OPEN, fid) if _denied(call("mallory"))]
    assert not refused, refused


def test_the_local_identity_is_held_to_the_owner_rule_without_a_requesting_agent(store, monkeypatch):
    fid = _make(store, PRIVATE, private=True)
    monkeypatch.setenv("HERMES_AGENT_ID", "mallory")          # the process itself is not a member
    assert _denied(server.memory_hints(PRIVATE))
    assert _denied(it.investigation_note(PRIVATE, "hypothesis", "x"))
    assert _denied(it.investigation_start(PRIVATE, "again"))


def test_a_denied_write_changes_nothing(store):
    fid = _make(store, PRIVATE, private=True)
    before = (Path(store) / PRIVATE / "manifest.json").read_text()
    assert _denied(it.investigation_note(PRIVATE, "hypothesis", "intruder was here", requesting_agent_id="mallory"))
    assert _denied(it.investigation_queue_enqueue(PRIVATE, "x1", "{}", requesting_agent_id="mallory"))
    assert (Path(store) / PRIVATE / "manifest.json").read_text() == before
    assert "intruder was here" not in (Path(store) / PRIVATE / "manifest.json").read_text()


# ------------------------------------------------------------------------------ cross-investigation tools

def test_cross_investigation_entity_lookup_hides_private_findings_from_a_non_member(store):
    _make(store, PRIVATE, private=True)
    _make(store, OPEN, private=False)
    def total(who):
        return _j(server.investigation_entity_lookup("10.4.4.4", "ip", requesting_agent_id=who))["total_findings"]
    assert total("alice") == 2          # member sees both
    assert total("mallory") == 1        # non-member sees only the open case
    out = server.investigation_entity_lookup("10.4.4.4", "ip", requesting_agent_id="mallory")
    assert PRIVATE not in out


def test_related_cases_does_not_name_a_private_investigation_to_a_non_member(store):
    _make(store, PRIVATE, private=True)
    out = server.investigation_related_cases("10.4.4.4", requesting_agent_id="mallory")
    assert PRIVATE not in out and "SECRET" not in out
    assert PRIVATE in server.investigation_related_cases("10.4.4.4", requesting_agent_id="alice")


def test_procedure_search_hides_private_procedures_from_a_non_member(store):
    inv = PRIVATE
    assert _j(server.investigation_start(investigation_id=inv, title="p")).get("status") == "created"
    kw = dict(procedure_preconditions="pre", procedure_steps="1. go", procedure_postconditions="post")
    _j(server.investigation_store(inv, "procedure", "rotate the private widget key", "unit", **kw))
    mpath = Path(store) / inv / "manifest.json"
    m = json.loads(mpath.read_text())
    m["owner"], m["acl"] = "alice", ["alice"]
    mpath.write_text(json.dumps(m))
    inv_store._manifest_cache.clear()
    assert _j(server.procedure_search("rotate the private widget", requesting_agent_id="alice"))["count"] == 1
    assert _j(server.procedure_search("rotate the private widget", requesting_agent_id="mallory"))["count"] == 0


def test_investigation_list_does_not_list_a_private_investigation_to_a_non_member(store):
    _make(store, PRIVATE, private=True)
    _make(store, OPEN, private=False)
    ids = lambda who: {r["id"] for r in _j(it.investigation_list(requesting_agent_id=who))["investigations"]}  # noqa: E731
    assert ids("alice") == {PRIVATE, OPEN}
    assert ids("mallory") == {OPEN}


def test_memory_confidence_does_not_count_a_private_neighbour_for_a_non_member(store, monkeypatch):
    _make(store, PRIVATE, private=True)
    rows = [{"id": "f-private", "investigation_id": PRIVATE, "text": "SECRET", "score": 0.9, "confidence": "high", "source": "s"},
            {"id": "f-open", "investigation_id": "elsewhere", "text": "ok", "score": 0.8, "confidence": "high", "source": "s"}]
    monkeypatch.setattr(server, "_confidence_retrieve", lambda q, k: (list(rows), None))
    refs = lambda who: [r["finding_id"] for r in _j(server.memory_confidence("q", requesting_agent_id=who))["confidence_aggregation"]["evidence_refs"]]  # noqa: E731
    assert refs("mallory") == ["f-open"]
    assert refs("alice") == ["f-private", "f-open"]

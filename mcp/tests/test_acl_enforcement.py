"""A non-empty ACL must keep non-members out of load, export, search and share.

Audit finding acl-filter-leaks-to-non-members: investigation_load kept findings
authored by ACL members without ever checking that the *requester* was one, so
a stranger read every member's finding; export and share did no check at all.
"""
from __future__ import annotations

import json

import pytest

import investigation_tools
import server

SECRET = "SECRET owned by alice: db password rotation plan"


@pytest.fixture
def acl_case(tmp_path, monkeypatch):
    original = server.MEMORY_DIR
    server.MEMORY_DIR = tmp_path
    # The manifest cache is keyed by id alone; a stale entry from another tmpdir
    # makes investigation_start "resume" a case that has no manifest on disk here.
    server._manifest_cache.clear()
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("HERMES_AGENT_ID", raising=False)
    monkeypatch.setattr(server, "_qdrant_upsert", lambda *a, **k: False)
    server.investigation_start("acl-audit-case", title="private case", context="x")
    server.investigation_store(investigation_id="acl-audit-case", finding_type="observed", text=SECRET,
                               source="t", confidence="high", authored_by="alice")
    server.investigation_share("acl-audit-case", ["alice", "bob"])
    yield "acl-audit-case"
    server.MEMORY_DIR = original
    server._manifest_cache.clear()


def test_non_member_cannot_load(acl_case):
    out = json.loads(server.investigation_load(acl_case, requesting_agent_id="mallory"))
    assert out.get("error") == "permission_denied", out
    assert SECRET not in json.dumps(out)
    for fidelity in ("summary", "brief"):
        out = json.loads(server.investigation_load(acl_case, requesting_agent_id="mallory", fidelity=fidelity))
        assert out.get("error") == "permission_denied", out


def test_members_still_load(acl_case):
    out = json.loads(server.investigation_load(acl_case, requesting_agent_id="bob"))
    assert out["total_findings"] == 1
    assert SECRET in out["recent_findings"][0]["text"]


def test_non_member_cannot_export(acl_case):
    out = json.loads(server.investigation_export(acl_case, requesting_agent_id="mallory"))
    assert out.get("error") == "permission_denied", out
    assert SECRET not in json.dumps(out)
    assert json.loads(server.investigation_export(acl_case, requesting_agent_id="alice"))["exported"] is True


def test_non_member_cannot_add_itself_to_the_acl(acl_case):
    out = json.loads(server.investigation_share(acl_case, ["mallory"], requesting_agent_id="mallory"))
    assert out.get("error") == "permission_denied", out
    out = json.loads(server.investigation_unshare(acl_case, ["alice"], requesting_agent_id="mallory"))
    assert out.get("error") == "permission_denied", out
    assert server._load_manifest(acl_case)["acl"] == ["alice", "bob"]
    # A member may still change it.
    assert json.loads(server.investigation_share(acl_case, ["carol"], requesting_agent_id="bob"))["shared_with"] == ["carol"]


def test_owned_investigation_refuses_a_foreign_local_agent(acl_case, monkeypatch):
    manifest = server._load_manifest(acl_case)
    manifest["owner"] = "alice"
    server._save_manifest(manifest)
    monkeypatch.setenv("HERMES_AGENT_ID", "mallory")
    out = json.loads(server.investigation_load(acl_case))
    assert out.get("error") == "permission_denied", out
    monkeypatch.setenv("HERMES_AGENT_ID", "alice")
    assert json.loads(server.investigation_load(acl_case))["total_findings"] == 1


def test_search_drops_rows_from_investigations_the_caller_cannot_read(acl_case, monkeypatch):
    row = {"id": "f-1", "finding_id": "f-1", "investigation_id": acl_case, "record_type": "observed",
           "source": "mnemo", "text": SECRET, "score": 0.9}
    monkeypatch.setattr(server, "_get_mnemo_funcs", lambda: (None, object()))
    monkeypatch.setattr(server, "_mnemo_recall", lambda *a, **k: [dict(row)])
    denied = json.loads(server.investigation_search("password rotation", investigation_id=acl_case,
                                                    requesting_agent_id="mallory"))
    assert denied.get("error") == "permission_denied", denied
    everywhere = json.loads(server.investigation_search("password rotation", requesting_agent_id="mallory"))
    assert SECRET not in json.dumps(everywhere)
    assert everywhere.get("excluded_acl") == 1, everywhere
    member = json.loads(server.investigation_search("password rotation", requesting_agent_id="bob"))
    assert SECRET in json.dumps(member)


def test_open_investigation_is_unaffected(tmp_path, monkeypatch):
    original = server.MEMORY_DIR
    server.MEMORY_DIR = tmp_path
    server._manifest_cache.clear()
    try:
        monkeypatch.setattr(server, "_qdrant_upsert", lambda *a, **k: False)
        server.investigation_start("open-case", title="t")
        server.investigation_store(investigation_id="open-case", finding_type="observed", text="shared note",
                                   source="t", confidence="high")
        assert json.loads(server.investigation_load("open-case", requesting_agent_id="anyone"))["total_findings"] == 1
        assert json.loads(investigation_tools.investigation_export("open-case"))["exported"] is True
    finally:
        server.MEMORY_DIR = original
        server._manifest_cache.clear()

"""rag_search honours investigation ACLs and the soft-retraction flag (#383 follow-ups b, c).

rag_search reads loci_memory, which holds investigation findings. Before, any
caller with a token got every hit, including findings from investigations whose
ACL excludes it and findings memory_retract had flagged retracted.

A /bootstrap session token is bound to its agent_id, so that is the identity the
ACL checks; the sender field cannot widen it.
"""

import asyncio
import datetime
import importlib.util
import json
import os
import pathlib

import pytest

os.environ.setdefault("LOCI_A2A_TOKEN", "test-token-abc123")
os.environ.setdefault("LOCI_A2A_URL", "http://localhost:8201")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MNEMOSYNE_EMBEDDING_API_URL", "http://localhost:11434/v1")

_server_path = pathlib.Path(__file__).parent.parent / "server.py"
_spec = importlib.util.spec_from_file_location("a2a_acl_impl", _server_path)
a2a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a)

SECRET = "SECRET owned by alice: db password rotation plan"


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("HERMES_AGENT_ID", raising=False)
    case = tmp_path / "private-case"
    case.mkdir()
    (case / "manifest.json").write_text(json.dumps(
        {"id": "private-case", "owner": "alice", "acl": ["alice", "bob"]}))
    hits = [
        {"id": "f-secret", "score": 0.9, "payload": {"text": SECRET, "investigation_id": "private-case"}},
        {"id": "f-retracted", "score": 0.8, "payload": {"text": "retracted claim", "retracted": True}},
        {"id": "c-1", "score": 0.7, "payload": {"text": "public code chunk"}},
    ]

    async def fake_search(col, vec, top_k=5):
        return [dict(h, payload=dict(h["payload"])) for h in hits] if col == "loci_memory" else []

    async def fake_embed(q):
        return [0.1] * 4

    monkeypatch.setattr(a2a, "_qdrant_search", fake_search)
    monkeypatch.setattr(a2a, "_embed", fake_embed)
    return tmp_path


def _run(sender, bound=None):
    task = {"input": {"query": "password rotation", "collections": ["loci_memory"]}, "sender": sender}

    async def go():
        auth = {"token_type": "session", "sender": bound} if bound else {"token_type": "primary"}
        with a2a._bind_authenticated_caller(auth):
            return await a2a.skill_rag_search(task)

    # Not asyncio.run(): it unsets the thread's event loop, and sibling test
    # modules still call asyncio.get_event_loop().
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(go())
    finally:
        loop.close()


def _contents(out):
    return [r["content"] for r in out["results"]]


def test_retracted_hits_are_withheld(mem):
    out = _run("bob", bound="bob")
    assert "retracted claim" not in _contents(out)
    assert out["excluded_retracted"] == 1


def test_session_bound_non_member_does_not_see_acl_hits(mem):
    out = _run("mallory", bound="mallory")
    assert SECRET not in json.dumps(out)
    assert out["excluded_acl"] == 1
    assert "public code chunk" in _contents(out)


def test_session_bound_member_sees_acl_hits(mem):
    out = _run("bob", bound="bob")
    assert SECRET in _contents(out) and out["excluded_acl"] == 0


def test_primary_token_sender_can_only_narrow(mem):
    # Primary token: no bound agent; this node (no HERMES_AGENT_ID) is not the owner.
    # The search itself must succeed: withholding is per hit, not an error.
    out = _run("bob")
    assert "error" not in out
    assert SECRET not in json.dumps(out)
    assert _contents(out) == ["public code chunk"]
    assert out["excluded_acl"] == 1 and out["excluded_retracted"] == 1
    assert out["total_hits"] == 1


def test_primary_token_member_node_sees_hits_for_a_member_sender(mem, monkeypatch):
    # Positive twin: this node IS a member, so a primary-token caller that names
    # another member sees the finding...
    monkeypatch.setenv("HERMES_AGENT_ID", "alice")
    out = _run("bob")
    assert "error" not in out
    assert SECRET in _contents(out) and out["excluded_acl"] == 0


def test_primary_token_member_node_narrowed_by_a_non_member_sender(mem, monkeypatch):
    # ...but naming a non-member narrows: the node may read it, the named sender may not.
    monkeypatch.setenv("HERMES_AGENT_ID", "alice")
    out = _run("mallory")
    assert SECRET not in json.dumps(out)
    assert _contents(out) == ["public code chunk"] and out["excluded_acl"] == 1


def test_acl_check_error_withholds_the_hit(mem, monkeypatch, caplog):
    # A member would normally see it; when the ACL check itself raises, the
    # investigation's hits must be withheld (fail closed), not served.
    def boom(manifest, requesting_agent_id=None, **kw):
        raise RuntimeError("acl backend exploded")

    monkeypatch.setattr(a2a._inv_store_acl, "_acl_access_denied", boom)
    with caplog.at_level("WARNING"):
        out = _run("bob", bound="bob")
    assert "error" not in out
    assert SECRET not in json.dumps(out)
    assert _contents(out) == ["public code chunk"]
    assert out["excluded_acl"] == 1
    assert "failed, withholding its hits" in caplog.text


def test_unreadable_manifest_withholds_the_hit(mem):
    (mem / "private-case" / "manifest.json").write_text("{not json")
    out = _run("bob", bound="bob")
    assert SECRET not in json.dumps(out)
    assert out["excluded_acl"] == 1


def test_acl_helpers_unavailable_withholds_investigation_hits(mem, monkeypatch):
    monkeypatch.setattr(a2a, "_inv_store_acl", None)
    out = _run("bob", bound="bob")
    assert SECRET not in json.dumps(out)
    assert _contents(out) == ["public code chunk"] and out["excluded_acl"] == 1


def test_endpoint_binds_the_session_token_agent(mem):
    from fastapi.testclient import TestClient

    tok = "session-tok-mallory"
    a2a._session_tokens[tok] = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
    a2a._session_token_agents[tok] = "mallory"
    try:
        client = TestClient(a2a.app)
        body = {"jsonrpc": "2.0", "id": "1", "method": "tasks/send",
                "params": {"skill_id": "rag_search",
                           "input": {"query": "password rotation", "collections": ["loci_memory"]}}}
        r = client.post("/a2a", json=body, headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200, r.text
        out = r.json()["result"]["output"]
        assert SECRET not in json.dumps(out)
        assert out["excluded_acl"] == 1
    finally:
        a2a._session_tokens.pop(tok, None)
        a2a._session_token_agents.pop(tok, None)

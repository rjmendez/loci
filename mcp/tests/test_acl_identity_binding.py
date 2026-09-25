"""#383 follow-up (b): ACL identity comes from the transport, and a caller-supplied
requesting_agent_id can only narrow it.

Before: the ACL trusted ``requesting_agent_id`` outright, so any caller could
name an ACL member and read the investigation, and grounding, memory_route and
investigation_as_of were not gated at all.
"""
from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest

import caller_identity
import grounding
import inv_store
import mnemo_ops
import server

SECRET = "SECRET owned by alice: db password rotation plan"
CASE = "acl-bind-case"


@pytest.fixture
def acl_case(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_store._manifest_cache.clear()
    monkeypatch.delenv("HERMES_AGENT_ID", raising=False)
    monkeypatch.setenv("LOCI_RAG_EXPAND", "0")
    no = lambda *a, **k: None  # noqa: E731
    for name, value in {
        "_qdrant_upsert": lambda *a, **k: False,
        "_mnemo_remember": lambda *a, **k: False,
        "_event_log_append": no,
        "_mirror_finding_to_ladybug": no,
        "_autolink_finding_to_ladybug": no,
        "_get_cross_encoder": lambda *a, **k: None,
        "docs_search": lambda *a, **k: json.dumps({"results": []}),
        "_retract_quarantine_verdict": lambda *a, **k: False,
        "_forget_finding_verdicts": lambda *a, **k: 0,
        "_semantic_neighbor_ids": lambda *a, **k: [],
    }.items():
        monkeypatch.setattr(server, name, value, raising=False)
    no_mnemo = lambda: (None, None)  # noqa: E731
    monkeypatch.setattr(server, "_get_mnemo_funcs", no_mnemo)
    monkeypatch.setattr(mnemo_ops, "_get_mnemo_funcs", no_mnemo)
    server.investigation_start(CASE, title="private case", context="x")
    stored = json.loads(server.investigation_store(
        investigation_id=CASE, finding_type="observed", text=SECRET,
        source="t", confidence="high", authored_by="alice"))
    server.investigation_share(CASE, ["alice", "bob"])
    manifest = server._load_manifest(CASE)
    manifest["owner"] = "alice"
    server._save_manifest(manifest)
    yield stored["finding_id"]
    inv_store._manifest_cache.clear()


def _load(**kw):
    return json.loads(server.investigation_load(CASE, **kw))


# --- the core rule: requesting_agent_id narrows, never widens -------------------

def test_naming_a_member_does_not_admit_a_non_member_process(acl_case, monkeypatch):
    monkeypatch.setenv("HERMES_AGENT_ID", "mallory")
    out = _load(requesting_agent_id="bob")
    assert out.get("error") == "permission_denied", out
    assert SECRET not in json.dumps(out)


def test_member_process_can_still_narrow_to_a_non_member(acl_case, monkeypatch):
    monkeypatch.setenv("HERMES_AGENT_ID", "alice")
    assert _load()["total_findings"] == 1
    assert _load(requesting_agent_id="bob")["total_findings"] == 1
    assert _load(requesting_agent_id="mallory").get("error") == "permission_denied"


def test_transport_bound_identity_is_what_the_acl_checks(acl_case):
    with caller_identity.bound("mallory"):
        assert _load().get("error") == "permission_denied"
        # Naming a member does not widen a bound non-member.
        assert _load(requesting_agent_id="alice").get("error") == "permission_denied"
    with caller_identity.bound("bob"):
        assert _load()["total_findings"] == 1
        assert _load(requesting_agent_id="bob")["total_findings"] == 1
        assert _load(requesting_agent_id="mallory").get("error") == "permission_denied"


def test_bound_identity_is_read_from_the_mcp_request_scope(acl_case):
    from mcp.server.lowlevel.server import request_ctx

    ctx = SimpleNamespace(request=SimpleNamespace(scope={caller_identity.SCOPE_KEY: "mallory"}))
    token = request_ctx.set(ctx)
    try:
        assert caller_identity.bound_agent_id() == "mallory"
        assert _load(requesting_agent_id="bob").get("error") == "permission_denied"
    finally:
        request_ctx.reset(token)
    assert caller_identity.bound_agent_id() is None


def test_share_cannot_be_used_to_widen(acl_case):
    with caller_identity.bound("mallory"):
        out = json.loads(server.investigation_share(CASE, ["mallory"], requesting_agent_id="alice"))
    assert out.get("error") == "permission_denied", out
    assert "mallory" not in server._load_manifest(CASE)["acl"]


def test_share_by_the_bound_owner_widens(acl_case):
    # Positive twin: the same call from the transport-bound owner succeeds.
    with caller_identity.bound("alice"):
        out = json.loads(server.investigation_share(CASE, ["carol"], requesting_agent_id="alice"))
    assert "error" not in out, out
    assert "carol" in server._load_manifest(CASE)["acl"]


def test_retract_owner_check_uses_the_bound_identity(acl_case, monkeypatch):
    monkeypatch.setattr(server, "AGENT_ID", "alice")
    with caller_identity.bound("mallory"):
        out = json.loads(server.memory_retract(CASE, acl_case, dry_run=True))
    assert out.get("error") == "permission_denied", out


def test_retract_by_the_bound_owner_is_allowed(acl_case, monkeypatch):
    # Positive twin: server AGENT_ID is someone else, the bound caller is the owner.
    monkeypatch.setattr(server, "AGENT_ID", "mallory")
    with caller_identity.bound("alice"):
        out = json.loads(server.memory_retract(CASE, acl_case, dry_run=True))
    assert "error" not in out, out
    assert (out["seed_ids"], out["count"], out["applied"]) == ([acl_case], 1, False)


# --- previously ungated paths --------------------------------------------------

def test_as_of_is_acl_gated(acl_case, monkeypatch):
    monkeypatch.setenv("HERMES_AGENT_ID", "mallory")
    out = json.loads(server.investigation_as_of(CASE, "2100-01-01T00:00:00+00:00", requesting_agent_id="bob"))
    assert out.get("error") == "permission_denied", out
    monkeypatch.setenv("HERMES_AGENT_ID", "alice")
    assert json.loads(server.investigation_as_of(CASE, "2100-01-01T00:00:00+00:00"))["count"] == 1


def _hits(acl_case):
    return [
        {"id": acl_case, "investigation_id": CASE, "text": SECRET, "score": 0.9,
         "origin": server.QDRANT_COLLECTION_PREFIX, "source": "t", "record_type": "observed",
         "authored_by": "alice"},
        {"id": "code-1", "text": "def rotate(): pass", "score": 0.8, "origin": "code"},
    ]


def _fake_qdrant(monkeypatch, hits):
    monkeypatch.setattr(server, "_get_qdrant", lambda *a, **k: (object(), server.QDRANT_COLLECTION_PREFIX))
    monkeypatch.setattr(server, "_qdrant_search_collection", lambda *a, **k: [dict(h) for h in hits])


def test_rag_context_search_drops_hits_the_caller_cannot_read(acl_case, monkeypatch):
    _fake_qdrant(monkeypatch, _hits(acl_case))
    with caller_identity.bound("mallory"):
        r = json.loads(server.rag_context_search("password rotation", expand_query=False))
    assert SECRET not in json.dumps(r)
    assert r["excluded_acl"] == 1
    with caller_identity.bound("bob"):
        r = json.loads(server.rag_context_search("password rotation", expand_query=False))
    assert "db password rotation" in r["context"] and r["excluded_acl"] == 0


def test_ground_does_not_inject_findings_the_caller_cannot_read(acl_case, monkeypatch):
    _fake_qdrant(monkeypatch, _hits(acl_case))
    monkeypatch.setenv("HERMES_AGENT_ID", "mallory")
    g = grounding.ground({"title": "password rotation", "caseIds": [CASE]}, {"memoryDir": ""})
    assert "db password rotation" not in g["block"], g["block"]
    # The requesting id is threaded through to the lanes (narrowing a member process).
    monkeypatch.setenv("HERMES_AGENT_ID", "alice")
    g = json.loads(server.ground("password rotation", case_ids=[CASE], requesting_agent_id="mallory"))
    assert "db password rotation" not in g["block"], g["block"]
    g = json.loads(server.ground("password rotation", case_ids=[CASE], requesting_agent_id="bob"))
    assert "db password rotation" in g["block"]


def test_memory_route_drops_hits_the_caller_cannot_read(acl_case, monkeypatch):
    _fake_qdrant(monkeypatch, _hits(acl_case)[:1])
    with caller_identity.bound("mallory"):
        r = json.loads(server.memory_route("password rotation", include_trace=True))
    assert SECRET not in json.dumps(r), r
    assert r["excluded_acl"] == 1
    # agent_id is caller-supplied: naming a member does not widen a bound stranger.
    with caller_identity.bound("mallory"):
        r = json.loads(server.memory_route("password rotation", agent_id="alice"))
    assert SECRET not in json.dumps(r)
    with caller_identity.bound("bob"):
        r = json.loads(server.memory_route("password rotation"))
    assert r["count"] == 1


# --- transport binding --------------------------------------------------------

def _mw_call(mw, headers, scope_extra=None):
    seen = {}

    async def inner(scope, receive, send):
        seen.update(scope)

    mw._app = inner
    scope = {"type": "http", "path": "/mcp", "headers": headers, **(scope_extra or {})}
    sent = []

    async def send(m):
        sent.append(m)

    loop = asyncio.new_event_loop()  # not asyncio.run(): it unsets the thread's loop
    try:
        loop.run_until_complete(mw(scope, None, send))
    finally:
        loop.close()
    return seen, sent


def test_per_agent_token_binds_the_agent_in_the_asgi_scope():
    mw = server._BearerAuthMiddleware(None, "shared", agent_tokens={"tok-bob": "bob"})
    seen, sent = _mw_call(mw, [(b"authorization", b"Bearer tok-bob")])
    assert sent == [] and seen[caller_identity.SCOPE_KEY] == "bob"
    # The shared token authenticates but binds nobody, even if a scope key was smuggled in.
    seen, sent = _mw_call(mw, [(b"authorization", b"Bearer shared")], {caller_identity.SCOPE_KEY: "alice"})
    assert sent == [] and caller_identity.SCOPE_KEY not in seen
    seen, sent = _mw_call(mw, [(b"authorization", b"Bearer nope")])
    assert sent[0]["status"] == 401 and not seen


def test_agent_tokens_alone_satisfy_the_wide_bind_rule_and_wrap_the_app():
    env = {"LOCI_MCP_TRANSPORT": "streamable-http", "LOCI_MCP_HOST": "0.0.0.0",
           "LOCI_MCP_AGENT_TOKENS": json.dumps({"bob": "tok-bob"})}
    with mock.patch.dict(os.environ, env, clear=True), \
         mock.patch.object(server.mcp, "streamable_http_app", return_value=mock.sentinel.app), \
         mock.patch.object(server, "logger"), mock.patch("uvicorn.run") as urun:
        server.main()
    app = urun.call_args.args[0]
    assert isinstance(app, server._BearerAuthMiddleware)
    assert app._agent_tokens == {"tok-bob": "bob"}


def test_shared_token_reused_as_an_agent_token_refuses_to_start():
    env = {"LOCI_MCP_TRANSPORT": "streamable-http", "LOCI_MCP_TOKEN": "same",
           "LOCI_MCP_AGENT_TOKENS": "bob=same"}
    with mock.patch.dict(os.environ, env, clear=True), \
         mock.patch.object(server, "logger"), mock.patch("uvicorn.run"), \
         pytest.raises(SystemExit):
        server.main()


def test_agent_token_parsing():
    assert caller_identity.load_agent_tokens({"LOCI_MCP_AGENT_TOKENS": "a=1, b=2"}) == {"1": "a", "2": "b"}
    with pytest.raises(ValueError):
        caller_identity.load_agent_tokens({"LOCI_MCP_AGENT_TOKENS": "a=1,b=1"})
    with pytest.raises(ValueError):
        caller_identity.load_agent_tokens({"LOCI_MCP_AGENT_TOKENS": "justaname"})

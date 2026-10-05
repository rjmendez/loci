"""Destructive-skill privilege is bound to the credential, not to the declared sender.

Before, `tasks/send` checked only `params.sender in LOCI_A2A_PRIVILEGED_SENDERS`.
The shared primary token does not identify an agent, and `_bound_sender` passed a
primary-token caller's `sender` through unchanged, so anyone holding the primary
token could name any privileged agent and run memory_remember / memory_sleep /
context_broadcast / mnemosyne_triple_add as it.

Now the privileged identity comes from the credential:
  * a per-agent token (LOCI_A2A_AGENT_TOKENS) or a /bootstrap session token is
    bound to its agent_id;
  * the primary token authenticates this node's operator, i.e. the local agent
    (HERMES_AGENT_ID), and nobody else.
The declared sender must equal that identity and be in the privileged set.
"""

import asyncio
import datetime
import importlib.util
import json
import os
import pathlib
import unittest
from unittest import mock

os.environ.setdefault("LOCI_A2A_TOKEN", "test-token-abc123")
os.environ.setdefault("LOCI_A2A_URL", "http://localhost:8201")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MNEMOSYNE_EMBEDDING_API_URL", "http://localhost:11434/v1")

_server_path = pathlib.Path(__file__).parent.parent / "server.py"
_spec = importlib.util.spec_from_file_location("a2a_privilege_binding_impl", _server_path)
a2a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a)

from fastapi.testclient import TestClient  # noqa: E402

PRIMARY = a2a.A2A_TOKEN
LOCAL = "local-node"
TRUSTED = "trusted-agent"
TRUSTED_TOKEN = "per-agent-token-for-trusted-0123456789"
UNPRIV_TOKEN = "per-agent-token-for-nobody-0123456789"


def _rpc(skill_id, sender=None, **inp):
    params = {"skill_id": skill_id, "message": "m", "input": inp or {"content": "x"}}
    if sender is not None:
        params["sender"] = sender
    return {"jsonrpc": "2.0", "id": "r1", "method": "tasks/send", "params": params}


def _bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


class _Base(unittest.TestCase):
    def setUp(self):
        self.dispatched = []

        async def recording_dispatch(skill_id, task):
            self.dispatched.append((skill_id, task["sender"]))
            return {"ok": True, "skill": skill_id}

        for target, value in (
            ("AGENT_ID", LOCAL),
            ("TOTP_SEED", ""),
            ("_PRIVILEGED_SENDERS", frozenset({LOCAL, TRUSTED})),
            ("_AGENT_TOKENS", {TRUSTED_TOKEN: TRUSTED, UNPRIV_TOKEN: "nobody"}),
            ("_dispatch", recording_dispatch),
        ):
            p = mock.patch.object(a2a, target, value, create=(target == "_AGENT_TOKENS"))
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(a2a.app, raise_server_exceptions=False)

    def post(self, body, tok):
        return self.client.post("/a2a", json=body, headers=_bearer(tok))


class TestPrimaryTokenCannotSpoofPrivilege(_Base):
    def test_primary_token_claiming_a_privileged_peer_is_refused(self):
        for skill in sorted(a2a.DESTRUCTIVE_SKILLS):
            with self.subTest(skill=skill):
                r = self.post(_rpc(skill, sender=TRUSTED), PRIMARY)
                self.assertEqual(r.status_code, 403, r.text)
                err = r.json()["error"]
                self.assertEqual(err["code"], -32600)
                self.assertIn("requires elevated privilege", err["message"])
        self.assertEqual(self.dispatched, [])

    def test_primary_token_acts_as_the_local_agent(self):
        # The node's own bridge posts context_broadcast with the primary token
        # and sender=HERMES_AGENT_ID: that must keep working.
        r = self.post(_rpc("memory_remember", sender=LOCAL), PRIMARY)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"]["output"], {"ok": True, "skill": "memory_remember"})
        self.assertEqual(self.dispatched, [("memory_remember", LOCAL)])

    def test_primary_token_local_agent_still_needs_the_allowlist(self):
        with mock.patch.object(a2a, "_PRIVILEGED_SENDERS", frozenset({TRUSTED})):
            r = self.post(_rpc("memory_remember", sender=LOCAL), PRIMARY)
        self.assertEqual(r.status_code, 403, r.text)
        self.assertEqual(self.dispatched, [])

    def test_primary_token_non_destructive_skill_keeps_declared_sender(self):
        r = self.post(_rpc("memory_stats", sender=TRUSTED), PRIMARY)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.dispatched, [("memory_stats", TRUSTED)])


class TestPerAgentTokens(_Base):
    def test_agent_token_runs_destructive_skill_as_its_agent(self):
        r = self.post(_rpc("memory_remember"), TRUSTED_TOKEN)   # no sender declared
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"]["output"], {"ok": True, "skill": "memory_remember"})
        self.assertEqual(self.dispatched, [("memory_remember", TRUSTED)])

    def test_agent_token_is_bound_to_its_agent(self):
        r = self.post(_rpc("memory_remember", sender=LOCAL), TRUSTED_TOKEN)
        self.assertEqual(r.status_code, 403, r.text)
        self.assertEqual(self.dispatched, [])

    def test_agent_token_for_unprivileged_agent_is_refused(self):
        r = self.post(_rpc("memory_remember"), UNPRIV_TOKEN)
        self.assertEqual(r.status_code, 403, r.text)
        self.assertEqual(r.json()["error"]["code"], -32600)
        self.assertEqual(self.dispatched, [])

    def test_agent_token_non_destructive_skill_dispatched(self):
        r = self.post(_rpc("memory_stats"), UNPRIV_TOKEN)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.dispatched, [("memory_stats", "nobody")])


class TestSessionTokenPrivilege(_Base):
    def _session(self, agent):
        tok = f"session-{agent}"
        a2a._session_tokens[tok] = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
        a2a._session_token_agents[tok] = agent
        self.addCleanup(a2a._session_tokens.pop, tok, None)
        self.addCleanup(a2a._session_token_agents.pop, tok, None)
        return tok

    def test_privileged_session_dispatched(self):
        r = self.post(_rpc("memory_remember"), self._session(TRUSTED))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.dispatched, [("memory_remember", TRUSTED)])

    def test_unprivileged_session_refused(self):
        r = self.post(_rpc("memory_remember"), self._session("mallory"))
        self.assertEqual(r.status_code, 403, r.text)
        self.assertEqual(self.dispatched, [])


class TestPrimaryTokenCompareIsExact(_Base):
    def test_exact_token_accepted(self):
        r = self.post({"jsonrpc": "2.0", "id": "1", "method": "tasks/list"}, PRIMARY)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"], {"tasks": []})

    def test_prefix_of_the_token_rejected(self):
        for tok in (PRIMARY[:-1], PRIMARY[:1]):
            with self.subTest(tok=tok):
                r = self.post({"jsonrpc": "2.0", "id": "1", "method": "tasks/list"}, tok)
                self.assertEqual(r.status_code, 401, r.text)
                self.assertIn("invalid or expired token", r.json()["detail"])

    def test_token_with_extra_suffix_rejected(self):
        r = self.post({"jsonrpc": "2.0", "id": "1", "method": "tasks/list"}, PRIMARY + "x")
        self.assertEqual(r.status_code, 401, r.text)

    def test_prefix_of_an_agent_token_rejected(self):
        r = self.post(_rpc("memory_stats"), TRUSTED_TOKEN[:-1])
        self.assertEqual(r.status_code, 401, r.text)
        self.assertEqual(self.dispatched, [])


class TestDirectCallFailsClosed(unittest.TestCase):
    """_handle_task_send without an authenticated identity must not run destructive skills."""

    def test_no_authenticated_agent_means_no_privilege(self):
        calls = []

        async def recording_dispatch(skill_id, task):
            calls.append(skill_id)
            return {}

        with mock.patch.object(a2a, "_PRIVILEGED_SENDERS", frozenset({TRUSTED})), \
             mock.patch.object(a2a, "_dispatch", recording_dispatch):
            loop = asyncio.new_event_loop()
            try:
                r = loop.run_until_complete(a2a._handle_task_send(
                    "r", {"skill_id": "memory_remember", "sender": TRUSTED,
                          "message": "m", "input": {"content": "x"}}))
            finally:
                loop.close()
        self.assertEqual(r.status_code, 403)
        self.assertEqual(json.loads(r.body)["error"]["code"], -32600)
        self.assertEqual(calls, [])


class TestLoadAgentTokens(unittest.TestCase):
    def test_json_object(self):
        self.assertEqual(
            a2a._load_agent_tokens({"LOCI_A2A_AGENT_TOKENS": '{"a": "tok-a", "b": "tok-b"}'}),
            {"tok-a": "a", "tok-b": "b"})

    def test_pairs(self):
        self.assertEqual(
            a2a._load_agent_tokens({"LOCI_A2A_AGENT_TOKENS": "a=tok-a, b=tok-b"}),
            {"tok-a": "a", "tok-b": "b"})

    def test_unset_is_empty(self):
        self.assertEqual(a2a._load_agent_tokens({}), {})

    def test_malformed_fails_closed_to_no_agent_tokens(self):
        for raw in ('{"a": "s3cr3t"', "s3cr3t-without-equals", '["s3cr3t"]',
                    '{"a": "", "b": "s3cr3t"}', "a=s3cr3t,b=s3cr3t"):
            with self.subTest(raw=raw):
                with self.assertLogs(a2a.log, level="ERROR") as logs:
                    self.assertEqual(a2a._load_agent_tokens({"LOCI_A2A_AGENT_TOKENS": raw}), {})
                self.assertIn("LOCI_A2A_AGENT_TOKENS", "\n".join(logs.output))
                self.assertNotIn("s3cr3t", "\n".join(logs.output))

    def test_primary_token_cannot_be_an_agent_token(self):
        with self.assertLogs(a2a.log, level="ERROR"):
            self.assertEqual(
                a2a._load_agent_tokens({"LOCI_A2A_AGENT_TOKENS": f"a={PRIMARY}"}, primary=PRIMARY),
                {})


if __name__ == "__main__":
    unittest.main()

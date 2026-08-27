"""Tests for per-skill privilege tier enforcement (issue #90).

Verifies:
  - Destructive skill called by unprivileged sender → 403
  - Destructive skill called by privileged sender (via LOCI_A2A_PRIVILEGED_SENDERS) → dispatched
  - Non-destructive skill callable by any authenticated sender
"""

import asyncio
import importlib.util
import os
import pathlib
import unittest
from unittest import mock

os.environ.setdefault("LOCI_A2A_TOKEN", "test-token-abc123")
os.environ.setdefault("LOCI_A2A_URL", "http://localhost:8201")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MNEMOSYNE_EMBEDDING_API_URL", "http://localhost:11434/v1")

_server_path = pathlib.Path(__file__).parent.parent / "server.py"
_spec = importlib.util.spec_from_file_location("a2a_server_impl", _server_path)
a2a_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a_server)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestDestructiveSkillBlocked(unittest.TestCase):
    """Unprivileged senders must be refused for every destructive skill."""

    def setUp(self):
        # Ensure no senders are privileged
        mock.patch.object(a2a_server, '_PRIVILEGED_SENDERS', frozenset()).start()
        self.addCleanup(mock.patch.stopall)

    def _call(self, skill_id: str, sender: str = "unprivileged-agent"):
        rpc_id = "test-rpc-1"
        params = {"skill_id": skill_id, "message": "test", "sender": sender, "input": {}}
        return _run(a2a_server._handle_task_send(rpc_id, params))

    def test_memory_remember_blocked(self):
        resp = self._call("memory_remember")
        self.assertEqual(resp.status_code, 403)
        body = resp.body.decode()
        self.assertIn("requires elevated privilege", body)

    def test_memory_sleep_blocked(self):
        resp = self._call("memory_sleep")
        self.assertEqual(resp.status_code, 403)

    def test_context_broadcast_blocked(self):
        resp = self._call("context_broadcast")
        self.assertEqual(resp.status_code, 403)

    def test_mnemosyne_triple_add_blocked(self):
        resp = self._call("mnemosyne_triple_add")
        self.assertEqual(resp.status_code, 403)

    def test_error_code_is_minus_32600(self):
        import json
        resp = self._call("memory_remember")
        body = json.loads(resp.body)
        self.assertEqual(body["error"]["code"], -32600)


class TestDestructiveSkillAllowed(unittest.TestCase):
    """Privileged senders must be able to call destructive skills (skill dispatched)."""

    PRIVILEGED = "trusted-agent"

    def setUp(self):
        mock.patch.object(
            a2a_server, '_PRIVILEGED_SENDERS', frozenset({self.PRIVILEGED})
        ).start()
        self.addCleanup(mock.patch.stopall)

    def _call_with_mock_dispatch(self, skill_id: str, dispatch_result: dict):
        rpc_id = "test-rpc-2"
        params = {
            "skill_id": skill_id,
            "message": "test",
            "sender": self.PRIVILEGED,
            "input": {},
        }
        with mock.patch.object(a2a_server, '_dispatch', return_value=dispatch_result) as m:
            resp = _run(a2a_server._handle_task_send(rpc_id, params))
            m.assert_called_once()
        return resp

    def test_memory_remember_dispatched(self):
        import json
        resp = self._call_with_mock_dispatch("memory_remember", {"stored": True})
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.body)
        self.assertEqual(body["result"]["output"], {"stored": True})

    def test_mnemosyne_triple_add_dispatched(self):
        resp = self._call_with_mock_dispatch("mnemosyne_triple_add", {"ok": True})
        self.assertEqual(resp.status_code, 200)


class TestNonDestructiveSkillOpen(unittest.TestCase):
    """Non-destructive skills must reach dispatch regardless of privileged senders config."""

    def setUp(self):
        # No privileged senders configured at all
        mock.patch.object(a2a_server, '_PRIVILEGED_SENDERS', frozenset()).start()
        self.addCleanup(mock.patch.stopall)

    def test_memory_recall_dispatched_for_any_sender(self):
        import json
        rpc_id = "test-rpc-3"
        params = {
            "skill_id": "memory_recall",
            "message": "hello",
            "sender": "random-unprivileged-agent",
            "input": {"query": "hello"},
        }
        with mock.patch.object(a2a_server, '_dispatch', return_value={"hits": []}) as m:
            resp = _run(a2a_server._handle_task_send(rpc_id, params))
            m.assert_called_once()
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.body)
        self.assertEqual(body["result"]["output"], {"hits": []})

    def test_memory_stats_dispatched_for_any_sender(self):
        params = {
            "skill_id": "memory_stats",
            "message": "",
            "sender": "any-agent",
            "input": {},
        }
        with mock.patch.object(a2a_server, '_dispatch', return_value={}) as m:
            resp = _run(a2a_server._handle_task_send("id", params))
            m.assert_called_once()
        self.assertEqual(resp.status_code, 200)


if __name__ == '__main__':
    unittest.main()

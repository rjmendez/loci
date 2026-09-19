"""Trust-boundary hardening tests for A2A b2b endpoint lanes."""

import asyncio
import hashlib
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
_spec = importlib.util.spec_from_file_location("a2a_server_trust_impl", _server_path)
a2a_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a_server)


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _boundary_for_memory_remember(content: str, source: str = "broadcast:peer", bank: str = "default"):
    payload = {"content": content, "source": source, "bank": bank}
    digest = hashlib.sha256(a2a_server._json_dumps_stable(payload).encode("utf-8")).hexdigest()
    return {
        "lane": "context_broadcast",
        "idempotency_key": "idem-key-12345",
        "artifact_sha256": digest,
        "origin_agent_id": "peer-a",
        "ts": "2026-09-18T00:00:00+00:00",
    }


class TestPeerPayloadAllowlist(unittest.TestCase):
    def test_peer_payload_rejects_non_allowlisted_skill(self):
        with self.assertRaises(ValueError):
            a2a_server._peer_task_payload("docker_status", "x", {})

    def test_peer_payload_includes_boundary_metadata(self):
        payload = a2a_server._peer_task_payload(
            "memory_remember",
            "hello",
            {"content": "hello", "source": "broadcast:peer", "bank": "default"},
        )
        boundary = payload["params"]["input"]["_boundary"]
        self.assertEqual(boundary["lane"], "context_broadcast")
        self.assertIn("idempotency_key", boundary)
        self.assertEqual(len(boundary["artifact_sha256"]), 64)


class TestBoundaryValidation(unittest.TestCase):
    def setUp(self):
        mock.patch.object(a2a_server, "_PRIVILEGED_SENDERS", frozenset({"peer-a"})).start()
        mock.patch.object(a2a_server, "_idempotency_seen", {}).start()
        self.addCleanup(mock.patch.stopall)

    def _call(self, params):
        return _run(a2a_server._handle_task_send("rpc-1", params))

    def test_rejects_non_object_input(self):
        resp = self._call({
            "skill_id": "memory_stats",
            "message": "",
            "sender": "peer-a",
            "input": "not-a-dict",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("input must be an object", resp.body.decode())

    def test_rejects_bad_boundary_hash(self):
        boundary = _boundary_for_memory_remember("hello")
        boundary["artifact_sha256"] = "0" * 64
        resp = self._call({
            "skill_id": "memory_remember",
            "message": "hello",
            "sender": "peer-a",
            "input": {
                "content": "hello",
                "source": "broadcast:peer",
                "bank": "default",
                "_boundary": boundary,
            },
        })
        self.assertEqual(resp.status_code, 403)
        body = resp.body.decode()
        self.assertIn("boundary validation failed", body)
        self.assertIn("artifact_sha256 mismatch", body)

    def test_rejects_idempotency_replay(self):
        boundary = _boundary_for_memory_remember("hello")
        params = {
            "skill_id": "memory_remember",
            "message": "hello",
            "sender": "peer-a",
            "input": {
                "content": "hello",
                "source": "broadcast:peer",
                "bank": "default",
                "_boundary": boundary,
            },
        }
        with mock.patch.object(a2a_server, "_dispatch", return_value={"stored": True}):
            first = self._call(params)
            second = self._call(params)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 403)
        self.assertIn("idempotency key replay", second.body.decode())

    def test_accepts_valid_boundary_and_dispatches(self):
        boundary = _boundary_for_memory_remember("hello")
        params = {
            "skill_id": "memory_remember",
            "message": "hello",
            "sender": "peer-a",
            "input": {
                "content": "hello",
                "source": "broadcast:peer",
                "bank": "default",
                "_boundary": boundary,
            },
        }
        with mock.patch.object(a2a_server, "_dispatch", return_value={"stored": True}) as m:
            resp = self._call(params)
            m.assert_called_once()
        self.assertEqual(resp.status_code, 200)
        self.assertIn("completed", resp.body.decode())


if __name__ == "__main__":
    unittest.main()

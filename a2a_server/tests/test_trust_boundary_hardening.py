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
        self.assertRegex(boundary["idempotency_key"], a2a_server._IDEMPOTENCY_KEY_RE)
        self.assertEqual(boundary["origin_agent_id"], a2a_server.AGENT_ID)
        self.assertEqual(payload["params"]["sender"], a2a_server.AGENT_ID)
        # The digest must be of the artifact the RECEIVER recomputes, not of {}.
        expected = hashlib.sha256(a2a_server._json_dumps_stable(
            {"content": "hello", "source": "broadcast:peer", "bank": "default"}
        ).encode("utf-8")).hexdigest()
        self.assertEqual(boundary["artifact_sha256"], expected)


class TestPeerPayloadRoundTrip(unittest.TestCase):
    """What one node sends, the receiving node's validator must accept.

    The old test only checked the digest was 64 hex chars, so a sender hashing
    the wrong artifact (every real broadcast 403s at the receiver) stayed green.
    """

    CASES = [
        ("memory_remember", "hello mesh",
         {"content": "hello mesh", "source": "broadcast:node-a", "importance": 0.7, "bank": "ops"}),
        ("memory_prime", "k3s outage",
         {"topic": "k3s outage", "skepticism_delta": 0.3, "ttl_seconds": 600, "broadcast": False}),
    ]

    def setUp(self):
        mock.patch.object(a2a_server, "_idempotency_seen", {}).start()
        self.addCleanup(mock.patch.stopall)

    def _received(self, payload):
        p = payload["params"]
        return p["skill_id"], p["sender"], {"message": p["message"], "input": p["input"]}

    def test_validator_accepts_what_the_sender_builds(self):
        for skill, message, inp in self.CASES:
            with self.subTest(skill=skill):
                payload = a2a_server._peer_task_payload(skill, message, inp)
                receipt = a2a_server._validate_boundary_metadata(*self._received(payload))
                self.assertIsNotNone(receipt)
                self.assertTrue(receipt["accepted"], receipt)
                self.assertEqual(receipt["reason"], "accepted")
                self.assertEqual(receipt["lane"], payload["params"]["input"]["_boundary"]["lane"])

    def test_tampered_content_is_rejected_after_the_round_trip(self):
        payload = a2a_server._peer_task_payload("memory_remember", "hello", {"content": "hello"})
        payload["params"]["input"]["content"] = "hello, but altered in transit"
        receipt = a2a_server._validate_boundary_metadata(*self._received(payload))
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "artifact_sha256 mismatch")

    def test_receiving_endpoint_dispatches_an_authenticated_broadcast(self):
        payload = a2a_server._peer_task_payload(*self.CASES[0])
        calls = []

        async def recording_dispatch(skill_id, task):
            calls.append((skill_id, task["sender"], task["input"]["content"]))
            return {"stored": True}

        with mock.patch.object(a2a_server, "_PRIVILEGED_SENDERS",
                               frozenset({a2a_server.AGENT_ID})), \
             mock.patch.object(a2a_server, "_dispatch", recording_dispatch):
            resp = _run(a2a_server._handle_task_send(
                payload["id"], dict(payload["params"]),
                authenticated_agent=a2a_server.AGENT_ID))
        self.assertEqual(resp.status_code, 200, resp.body)
        self.assertEqual(calls, [("memory_remember", a2a_server.AGENT_ID, "hello mesh")])


class TestBoundaryValidation(unittest.TestCase):
    def setUp(self):
        mock.patch.object(a2a_server, "_PRIVILEGED_SENDERS", frozenset({"peer-a"})).start()
        mock.patch.object(a2a_server, "_idempotency_seen", {}).start()
        self.addCleanup(mock.patch.stopall)

    def _call(self, params):
        # peer-a presented its own per-agent credential (see test_privilege_binding.py).
        return _run(a2a_server._handle_task_send("rpc-1", params, authenticated_agent="peer-a"))

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

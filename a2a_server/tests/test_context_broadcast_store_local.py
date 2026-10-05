"""context_broadcast's store_local flag.

The context bridge relays memories by POSTing them to its OWN server's context_broadcast.
That skill stored before fanning out, so a relayed memory was re-inserted into the database
it had just been read from - new id, new created_at - which made the copy look "new" to the
next bridge run and got it relayed again. Unbounded local growth, no peer required.

store_local=false lets a caller relay something the node already holds.
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
_spec = importlib.util.spec_from_file_location("a2a_storelocal_impl", _server_path)
a2a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a)


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def call(**inp):
    """Invoke context_broadcast with no peers configured, capturing local stores."""
    calls = []

    async def fake_remember(task):
        calls.append(task)
        return {"id": "stored-id", "status": "stored"}

    with mock.patch.dict(os.environ, {"PEER_A2A_URLS": ""}, clear=False), \
         mock.patch.object(a2a, "skill_memory_remember", fake_remember):
        out = run(a2a.skill_context_broadcast({"input": inp, "sender": "test"}))
    return out, calls


class TestStoreLocal(unittest.TestCase):
    def test_defaults_to_storing(self):
        out, calls = call(content="hello")
        self.assertEqual(len(calls), 1)
        self.assertTrue(out["stored_locally"])

    def test_explicit_true_stores(self):
        _, calls = call(content="hello", store_local=True)
        self.assertEqual(len(calls), 1)

    def test_false_does_not_store(self):
        out, calls = call(content="hello", store_local=False)
        self.assertEqual(calls, [])
        self.assertFalse(out["stored_locally"])

    def test_false_still_attempts_the_fan_out(self):
        # Suppressing the local write must not suppress the actual point of the skill.
        # With a peer configured the POST must actually go out (with no peer the
        # 'broadcast' key is present either way, which is what hid a skipped fan-out).
        peer = "http://peer-a:8201/a2a"
        posts = []

        class _Resp:
            status = 200

            async def json(self):
                return {"result": {"output": {"id": "peer-mem-1"}}}

            async def text(self):
                return ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _Sess:
            closed = False

            def post(self, url, json=None, headers=None):
                posts.append((url, json, headers))
                return _Resp()

        remembered = []

        async def fake_remember(task):
            remembered.append(task)
            return {"id": "stored-id"}

        saved = a2a._http_session
        a2a._http_session = _Sess()
        try:
            with mock.patch.dict(os.environ, {"PEER_A2A_URLS": peer,
                                              "PEER_A2A_TOKEN": "peer-tok",
                                              "PEER_A2A_TOKENS_JSON": "{}",
                                              "PEER_A2A_TOTP_SEED": "",
                                              "PEER_A2A_TOTP_SEEDS_JSON": "{}"}), \
                 mock.patch.object(a2a, "skill_memory_remember", fake_remember):
                out = run(a2a.skill_context_broadcast(
                    {"input": {"content": "hello", "store_local": False, "importance": 0.8},
                     "sender": "test"}))
        finally:
            a2a._http_session = saved

        self.assertEqual(remembered, [])
        self.assertFalse(out["stored_locally"])
        self.assertEqual(out["peers_count"], 1)
        self.assertEqual(out["broadcast"],
                         [{"peer": peer, "status": "ok", "output": {"id": "peer-mem-1"}}])
        self.assertEqual(len(posts), 1)
        url, body, headers = posts[0]
        self.assertEqual(url, peer)
        self.assertEqual(headers["Authorization"], "Bearer peer-tok")
        self.assertEqual(body["method"], "tasks/send")
        self.assertEqual(body["params"]["skill_id"], "memory_remember")
        self.assertEqual(body["params"]["input"]["content"], "hello")
        self.assertEqual(body["params"]["input"]["importance"], 0.8)
        self.assertEqual(body["params"]["input"]["_boundary"]["lane"], "context_broadcast")

    def test_empty_content_still_rejected(self):
        out, calls = call(content="   ", store_local=False)
        self.assertIn("error", out)
        self.assertEqual(calls, [])

    def test_source_and_importance_reach_the_local_store(self):
        _, calls = call(content="hello", source="bridge:loci", importance=0.9)
        self.assertEqual(calls[0]["input"]["source"], "bridge:loci")
        self.assertEqual(calls[0]["input"]["importance"], 0.9)


if __name__ == "__main__":
    unittest.main()

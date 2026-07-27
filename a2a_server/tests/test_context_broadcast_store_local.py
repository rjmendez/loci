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

os.environ.setdefault("HERMES_A2A_TOKEN", "test-token-abc123")
os.environ.setdefault("HERMES_A2A_URL", "http://localhost:8201")
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
        out, _ = call(content="hello", store_local=False)
        self.assertIn("broadcast", out)

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

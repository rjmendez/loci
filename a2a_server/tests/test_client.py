"""Tests for a2a_server/client.py — LociClient HTTP wire protocol.

Uses aiohttp.test_utils to spin up a fake A2A endpoint so no live server is needed.
Covers: _headers() Authorization + optional X-TOTP, _call() success and HTTP 4xx,
memory_recall() and memory_remember() payload shape.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import pathlib
import sys
import unittest
from unittest import mock

# Ensure a2a_server is importable
_A2A_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_A2A_DIR) not in sys.path:
    sys.path.insert(0, str(_A2A_DIR))

# Set required env vars before importing the client module
os.environ.setdefault("LOCI_A2A_TOKEN", "test-token-abc")
os.environ.setdefault("LOCI_A2A_URL", "http://localhost:8201")

_client_path = pathlib.Path(__file__).parent.parent / "client.py"
_spec = importlib.util.spec_from_file_location("loci_client_module", _client_path)
client_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client_mod)
LociClient = client_mod.LociClient


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestHeadersConstruction(unittest.TestCase):
    def test_authorization_header_present(self):
        c = LociClient(endpoint="http://test", token="mytoken")
        h = c._headers()
        self.assertIn("Authorization", h)
        # The client uses the literal string "Bearer" with the token
        self.assertIn("mytoken", h["Authorization"])

    def test_content_type_json(self):
        c = LociClient(endpoint="http://test", token="tok")
        h = c._headers()
        self.assertEqual(h["Content-Type"], "application/json")

    def test_no_totp_when_no_seed(self):
        c = LociClient(endpoint="http://test", token="tok", totp_seed="")
        h = c._headers()
        self.assertNotIn("X-TOTP", h)

    def test_totp_header_when_seed_and_pyotp(self):
        try:
            import pyotp
        except ImportError:
            self.skipTest("pyotp not installed")
        seed = pyotp.random_base32()
        c = LociClient(endpoint="http://test", token="tok", totp_seed=seed)
        h = c._headers()
        self.assertIn("X-TOTP", h)
        # Verify the TOTP is a 6-digit numeric string
        self.assertTrue(h["X-TOTP"].isdigit())
        self.assertEqual(len(h["X-TOTP"]), 6)


class TestCallSuccess(unittest.TestCase):
    def setUp(self):
        try:
            import aiohttp
        except ImportError:
            self.skipTest("aiohttp not installed")

    def test_call_success_returns_output(self):
        """_call() on HTTP 200 returns result.output."""
        import aiohttp

        expected_output = {"memory": "test result"}
        mock_response_data = {"result": {"output": expected_output}}

        mock_resp = mock.AsyncMock()
        mock_resp.status = 200
        mock_resp.json = mock.AsyncMock(return_value=mock_response_data)
        mock_resp.__aenter__ = mock.AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = mock.AsyncMock(return_value=False)

        mock_session = mock.AsyncMock()
        mock_session.post = mock.MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mock.AsyncMock(return_value=False)

        c = LociClient(endpoint="http://test", token="tok")
        with mock.patch("aiohttp.ClientSession", return_value=mock_session):
            result = _run(c._call("memory_recall", "query", {"query": "q"}))

        self.assertEqual(result, expected_output)

    def test_call_http_4xx_returns_error(self):
        """_call() on HTTP 4xx returns an error dict."""
        import aiohttp

        mock_resp = mock.AsyncMock()
        mock_resp.status = 401
        mock_resp.json = mock.AsyncMock(return_value={"error": "unauthorized"})
        mock_resp.__aenter__ = mock.AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = mock.AsyncMock(return_value=False)

        mock_session = mock.AsyncMock()
        mock_session.post = mock.MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = mock.AsyncMock(return_value=False)

        c = LociClient(endpoint="http://test", token="tok")
        with mock.patch("aiohttp.ClientSession", return_value=mock_session):
            result = _run(c._call("memory_recall", "q"))

        self.assertIn("error", result)
        self.assertIn("401", result["error"])


class TestMemoryRecallPayload(unittest.TestCase):
    def setUp(self):
        try:
            import aiohttp
        except ImportError:
            self.skipTest("aiohttp not installed")

    def _mock_call(self, return_value=None):
        """Return a mock that replaces _call and records the call."""
        async def _fake_call(skill_id, message="", input_data=None):
            return return_value or {}
        return _fake_call

    def test_memory_recall_passes_query(self):
        called_with = {}

        async def _fake_call(skill_id, message="", input_data=None):
            called_with.update({"skill_id": skill_id, "message": message, "input_data": input_data})
            return {}

        c = LociClient(endpoint="http://test", token="tok")
        c._call = _fake_call
        _run(c.memory_recall("DAMA telemetry", top_k=3))

        self.assertEqual(called_with["skill_id"], "memory_recall")
        self.assertEqual(called_with["message"], "DAMA telemetry")
        self.assertEqual(called_with["input_data"]["query"], "DAMA telemetry")
        self.assertEqual(called_with["input_data"]["top_k"], 3)

    def test_memory_remember_passes_content(self):
        called_with = {}

        async def _fake_call(skill_id, message="", input_data=None):
            called_with.update({"skill_id": skill_id, "message": message, "input_data": input_data})
            return {}

        c = LociClient(endpoint="http://test", token="tok")
        c._call = _fake_call
        _run(c.memory_remember("Resolved k3s at 03:00 UTC", sender="hermes-agent"))

        self.assertEqual(called_with["skill_id"], "memory_remember")
        self.assertIn("content", called_with["input_data"])
        self.assertEqual(called_with["input_data"]["content"], "Resolved k3s at 03:00 UTC")


if __name__ == "__main__":
    unittest.main(verbosity=2)

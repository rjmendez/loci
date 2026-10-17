"""Tests for TOTP authentication paths in a2a_server (issues #98).

Covers: 401 on bad TOTP code, 429 after rate-limit exceeded, and pass-through
when TOTP_SEED is not set. No live Qdrant / pyotp TOTP generation required.
"""

import collections
import time
import unittest
from unittest.mock import patch

# a2a_server module is loaded by test_routing.py; reuse it.
import importlib.util
import os
import pathlib

os.environ.setdefault("LOCI_A2A_TOKEN", "test-token-abc123")
os.environ.setdefault("LOCI_A2A_URL", "http://localhost:8201")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MNEMOSYNE_EMBEDDING_API_URL", "http://localhost:11434/v1")

_server_path = pathlib.Path(__file__).parent.parent / "server.py"
_spec = importlib.util.spec_from_file_location("a2a_server_impl", _server_path)
a2a_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a_server)

from fastapi.testclient import TestClient  # noqa: E402

_AUTH = {"Authorization": "Bearer test-token-abc123"}
_JSON = {"Content-Type": "application/json"}
_HEADERS = {**_AUTH, **_JSON}

_TEST_SEED = "JBSWY3DPEHPK3PXP"  # valid base32 seed for testing


def _rpc(method: str, params: dict = None) -> dict:
    body = {"jsonrpc": "2.0", "id": "t-1", "method": method}
    if params is not None:
        body["params"] = params
    return body


class TestTOTPAuth(unittest.TestCase):
    """TOTP 401 / 429 / disabled paths."""

    def setUp(self):
        # Reset rate-limit counters before every test.
        a2a_server._totp_attempts.clear()

    def _client_with_totp(self) -> TestClient:
        return TestClient(a2a_server.app, raise_server_exceptions=False)

    def test_401_on_missing_totp_header_when_seed_set(self):
        """When TOTP_SEED is set, requests without X-TOTP return 401."""
        with patch.object(a2a_server, "TOTP_SEED", _TEST_SEED):
            client = self._client_with_totp()
            resp = client.post("/a2a", json=_rpc("tasks/list"), headers=_HEADERS)
        self.assertEqual(resp.status_code, 401)
        self.assertIn("X-TOTP", resp.json().get("detail", ""))

    def test_401_on_bad_totp_code(self):
        """A wrong TOTP code returns 401 even if the auth token is valid."""
        mock_totp = unittest.mock.MagicMock()
        mock_totp.return_value.verify.return_value = False  # always reject

        with patch.object(a2a_server, "TOTP_SEED", _TEST_SEED), \
             patch.object(a2a_server, "pyotp") as mock_pyotp_mod:
            mock_pyotp_mod.TOTP.return_value.verify.return_value = False
            client = self._client_with_totp()
            resp = client.post(
                "/a2a",
                json=_rpc("tasks/list"),
                headers={**_HEADERS, "X-TOTP": "000000"},
            )
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Invalid TOTP", resp.json().get("detail", ""))

    def test_429_after_max_attempts_from_same_ip(self):
        """After _TOTP_MAX_ATTEMPTS failed attempts the next call returns 429."""
        max_attempts = a2a_server._TOTP_MAX_ATTEMPTS
        now = time.monotonic()

        with patch.object(a2a_server, "TOTP_SEED", _TEST_SEED), \
             patch.object(a2a_server, "pyotp") as mock_pyotp_mod:
            mock_pyotp_mod.TOTP.return_value.verify.return_value = False

            client = self._client_with_totp()

            # Prefill the attempts counter to simulate previous failures.
            # TestClient uses "testclient" as the host by default.
            client_ip = "testclient"
            a2a_server._totp_attempts[client_ip] = [
                now - i for i in range(max_attempts)
            ]

            resp = client.post(
                "/a2a",
                json=_rpc("tasks/list"),
                headers={**_HEADERS, "X-TOTP": "000000"},
            )
        self.assertEqual(resp.status_code, 429)
        self.assertIn("Too many", resp.json().get("detail", ""))

    def test_pass_through_when_totp_seed_not_set(self):
        """When TOTP_SEED is empty, X-TOTP is not required and requests succeed."""
        with patch.object(a2a_server, "TOTP_SEED", ""):
            client = self._client_with_totp()
            resp = client.post("/a2a", json=_rpc("tasks/list"), headers=_HEADERS)
        # 200 OK or a JSON-RPC result (not 401/429)
        self.assertNotIn(resp.status_code, (401, 429))

    def test_valid_totp_code_passes(self):
        """A TOTP code accepted by pyotp.TOTP.verify allows the request through."""
        with patch.object(a2a_server, "TOTP_SEED", _TEST_SEED), \
             patch.object(a2a_server, "pyotp") as mock_pyotp_mod:
            mock_pyotp_mod.TOTP.return_value.verify.return_value = True
            client = self._client_with_totp()
            resp = client.post(
                "/a2a",
                json=_rpc("tasks/list"),
                headers={**_HEADERS, "X-TOTP": "123456"},
            )
        self.assertNotIn(resp.status_code, (401, 429))


if __name__ == "__main__":
    unittest.main()


class TestTOTPAttemptKeyLeak(unittest.TestCase):
    """The rate-limit dict pruned timestamps but never its keys.

    Each distinct client IP left an entry behind forever — an unauthenticated
    caller could grow it by rotating source addresses, and a busy server grew it
    just by being used. Only the per-IP lists were bounded.
    """

    # Read via getattr with a fallback ON PURPOSE. Referencing the new constant
    # directly made this test fail on the old code with AttributeError — proving
    # the constant was absent, not that keys leaked. With a fallback the test
    # populates the dict, makes a request, and asserts it shrank, which is the
    # behaviour under test.
    SWEEP_AFTER = getattr(a2a_server, "_TOTP_SWEEP_AFTER", 1024)

    def setUp(self):
        a2a_server._totp_attempts.clear()

    def tearDown(self):
        a2a_server._totp_attempts.clear()

    def _post_one_valid_request(self):
        """One authenticated request, using this file's mock-pyotp convention."""
        with patch.object(a2a_server, "TOTP_SEED", _TEST_SEED), \
             patch.object(a2a_server, "pyotp") as mock_pyotp_mod:
            mock_pyotp_mod.TOTP.return_value.verify.return_value = True
            client = TestClient(a2a_server.app, raise_server_exceptions=False)
            client.post("/a2a", json=_rpc("tasks/list"),
                        headers={**_HEADERS, "X-TOTP": "000000"})

    def test_expired_keys_are_swept_not_just_emptied(self):
        stale = time.monotonic() - (a2a_server._TOTP_WINDOW + 60)
        for i in range(self.SWEEP_AFTER + 5):
            a2a_server._totp_attempts[f"10.0.{i // 256}.{i % 256}"] = [stale]
        before = len(a2a_server._totp_attempts)
        self.assertGreater(before, self.SWEEP_AFTER)

        self._post_one_valid_request()

        after = len(a2a_server._totp_attempts)
        self.assertLess(after, before,
                        "stale client-IP keys must be removed, not merely emptied")
        self.assertLessEqual(after, 2, f"expected a swept dict, got {after} keys")

    def test_a_live_client_is_not_swept(self):
        """The sweep must only remove keys whose window has fully expired."""
        now = time.monotonic()
        a2a_server._totp_attempts["10.1.1.1"] = [now]
        for i in range(self.SWEEP_AFTER + 5):
            a2a_server._totp_attempts[f"10.9.{i // 256}.{i % 256}"] = [
                now - (a2a_server._TOTP_WINDOW + 60)]

        self._post_one_valid_request()

        self.assertIn("10.1.1.1", a2a_server._totp_attempts,
                      "a client inside its window must survive the sweep")

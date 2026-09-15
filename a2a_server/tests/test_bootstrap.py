"""Tests for POST /bootstrap and the session tokens it issues.

The point of a session token is that it authenticates on its own — a caller
that presented the pre-shared bootstrap key should not also need the TOTP
seed. These tests run with TOTP ENABLED, because that is the only
configuration in which the feature does anything.
"""

import datetime
import importlib.util
import os
import pathlib
import time
import unittest

import pyotp

TOTP_SEED = pyotp.random_base32()

os.environ["LOCI_ENV_FILE"] = "/nonexistent-env-file-for-tests"
os.environ["LOCI_A2A_TOKEN"] = "test-token-abc123"
os.environ["LOCI_A2A_TOTP_SEED"] = TOTP_SEED
os.environ["LOCI_A2A_BOOTSTRAP_KEY"] = "test-bootstrap-key"
os.environ.setdefault("LOCI_A2A_URL", "http://localhost:8201")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MNEMOSYNE_EMBEDDING_API_URL", "http://localhost:11434/v1")

_server_path = pathlib.Path(__file__).parent.parent / "server.py"
_spec = importlib.util.spec_from_file_location("a2a_bootstrap_impl", _server_path)
a2a_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a_server)

# server.py latches TOTP_SEED / BOOTSTRAP_KEY into module globals at import, so
# drop them from the shared environment now. Otherwise the next test module to
# load its own copy of server.py in the same pytest session inherits TOTP and
# every one of its unauthenticated-by-design requests starts 401ing.
os.environ.pop("LOCI_A2A_TOTP_SEED", None)
os.environ.pop("LOCI_A2A_BOOTSTRAP_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(a2a_server.app, raise_server_exceptions=False)

RPC = {
    "jsonrpc": "2.0",
    "id": "t",
    "method": "tasks/send",
    "params": {"skill_id": "memory_stats", "message": "", "input": {}, "sender": "test"},
}


def _bootstrap(key="test-bootstrap-key", **extra):
    return client.post("/bootstrap", json={"bootstrap_key": key, **extra})


def _session_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


class TestBootstrapIssuance(unittest.TestCase):
    def setUp(self):
        a2a_server._bootstrap_attempts.clear()

    def test_correct_key_issues_token(self):
        r = _bootstrap()
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["session_token"])

    def test_wrong_key_401s(self):
        self.assertEqual(_bootstrap(key="nope").status_code, 401)

    def test_missing_key_401s(self):
        self.assertEqual(client.post("/bootstrap", json={}).status_code, 401)

    def test_invalid_json_400s(self):
        r = client.post(
            "/bootstrap", content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(r.status_code, 400)

    def test_ttl_capped_at_7_days(self):
        exp = datetime.datetime.fromisoformat(_bootstrap(ttl_hours=9999).json()["expires_at"])
        ahead = exp - datetime.datetime.now(datetime.timezone.utc)
        self.assertLessEqual(ahead, datetime.timedelta(hours=168))

    def test_response_advertises_no_totp_required(self):
        self.assertFalse(_bootstrap().json()["totp_required"])

    def test_rate_limits_after_too_many_failed_attempts(self):
        now = time.monotonic()
        a2a_server._bootstrap_attempts["testclient"] = [
            now - i for i in range(a2a_server._BOOTSTRAP_MAX_ATTEMPTS)
        ]
        r = _bootstrap(key="still-wrong")
        self.assertEqual(r.status_code, 429)
        self.assertIn("Too many bootstrap attempts", r.json()["detail"])

    def test_successful_bootstrap_clears_failed_attempts(self):
        a2a_server._bootstrap_attempts["testclient"] = [time.monotonic()]
        r = _bootstrap()
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("testclient", a2a_server._bootstrap_attempts)


class TestSessionTokenAuth(unittest.TestCase):
    """TOTP is enabled in this module, so these assert the actual feature."""

    def setUp(self):
        a2a_server._tasks.clear()

    def test_session_token_works_without_totp(self):
        tok = _bootstrap(agent_id="test").json()["session_token"]
        r = client.post(
            "/a2a", json=RPC,
            headers=_session_headers(tok),
        )
        self.assertEqual(r.status_code, 200, r.text)

    def test_primary_token_still_requires_totp(self):
        r = client.post(
            "/a2a", json=RPC,
            headers={"Authorization": "Bearer test-token-abc123"},
        )
        self.assertEqual(r.status_code, 401)

    def test_primary_token_with_valid_totp_works(self):
        r = client.post(
            "/a2a", json=RPC,
            headers={
                "Authorization": "Bearer test-token-abc123",
                "X-TOTP": pyotp.TOTP(TOTP_SEED).now(),
            },
        )
        self.assertEqual(r.status_code, 200, r.text)

    def test_unknown_bearer_401s(self):
        r = client.post("/a2a", json=RPC, headers={"Authorization": "Bearer bogus"})
        self.assertEqual(r.status_code, 401)

    def test_expired_session_token_401s(self):
        tok = _bootstrap(ttl_hours=0).json()["session_token"]
        r = client.post("/a2a", json=RPC, headers=_session_headers(tok))
        self.assertEqual(r.status_code, 401)

    def test_expired_session_token_does_not_bypass_totp(self):
        # An expired token must not sneak past the TOTP gate either.
        tok = _bootstrap(ttl_hours=0).json()["session_token"]
        a2a_server._session_tokens[tok] = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
        )
        r = client.post("/a2a", json=RPC, headers=_session_headers(tok))
        self.assertEqual(r.status_code, 401)

    def test_session_token_works_on_tasks_get_route(self):
        tok = _bootstrap(agent_id="bootstrap-agent").json()["session_token"]
        r = client.get(
            "/a2a/tasks/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": f"Bearer {tok}"},
        )
        # 404 (unknown task) proves auth passed; 401 would mean it did not.
        self.assertEqual(r.status_code, 404)

    def test_session_token_cannot_impersonate_another_sender(self):
        tok = _bootstrap(agent_id="bootstrap-agent").json()["session_token"]
        r = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "x",
                "method": "tasks/list",
                "params": {"sender": "other-agent"},
            },
            headers=_session_headers(tok),
        )
        self.assertEqual(r.status_code, 403)
        self.assertIn("bound to their issuing agent_id", r.json()["detail"])



class TestHealthReporting(unittest.TestCase):
    def test_health_reports_bootstrap_configured(self):
        self.assertTrue(client.get("/health").json()["bootstrap_configured"])

    def test_health_counts_active_sessions(self):
        before = client.get("/health").json()["active_sessions"]
        _bootstrap()
        self.assertGreater(client.get("/health").json()["active_sessions"], before)


if __name__ == "__main__":
    unittest.main()

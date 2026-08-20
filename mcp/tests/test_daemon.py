"""Tests for mcp/memcheck/daemon.py — Unix socket protocol, routing, and resilience.

Uses a real Unix socket with an InMemoryBackend-backed engine so no live Qdrant
is required. A background thread runs serve(); tests connect with a raw socket.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

from memcheck.backend import InMemoryBackend
from memcheck.engine import EmlConfig, VerdictEngine
from memcheck import daemon as daemon_mod


def _make_engine() -> VerdictEngine:
    return VerdictEngine(InMemoryBackend(), EmlConfig(promote_after=3))


def _connect_and_send(sock_path: str, payload: dict) -> dict:
    """Send a JSON payload over a raw Unix socket; return the parsed response."""
    raw = (json.dumps(payload) + "\n").encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(sock_path)
        s.sendall(raw)
        s.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    return json.loads(b"".join(chunks).strip())


class TestMemcheckDaemonProtocol(unittest.TestCase):
    """Daemon golden-path and error-path tests over a real Unix socket."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._sock_path = os.path.join(self._tmp.name, "memcheck.sock")
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=daemon_mod.serve,
            kwargs={
                "socket_path": self._sock_path,
                "engine_factory": _make_engine,
                "ready_event": self._ready,
            },
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5)
        # Find the server object to shut it down later
        self._server = None  # shutdown via socket closure is sufficient for tests

    def tearDown(self):
        # Best-effort cleanup: the daemon thread is daemonic so it dies with the process.
        self._tmp.cleanup()

    def _send(self, payload: dict) -> dict:
        return _connect_and_send(self._sock_path, payload)

    # --- golden path ---------------------------------------------------------

    def test_pretooluse_returns_expected_keys(self):
        resp = self._send({
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "ls /tmp"},
            "session_id": "test-session-1",
        })
        self.assertIn("would_flag", resp)
        self.assertIn("occurrences", resp)
        self.assertIn("qdrant", resp)

    def test_pretooluse_occurrences_increment(self):
        payload = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"path": "/etc/hosts"},
            "session_id": "test-session-2",
        }
        r1 = self._send(payload)
        r2 = self._send(payload)
        self.assertGreater(r2["occurrences"], r1["occurrences"])

    def test_posttooluse_routing(self):
        resp = self._send({
            "hook_event_name": "PostToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": "/nonexistent/foo.py", "new_file_contents": "x=1"},
            "session_id": "test-session-3",
        })
        # PostToolUse is advisory — would_flag is always False
        self.assertFalse(resp["would_flag"])
        self.assertIn("occurrences", resp)

    # --- error paths ---------------------------------------------------------

    def test_malformed_json_returns_fallback(self):
        """A non-JSON payload must not crash the daemon; it returns a fail-open response."""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(self._sock_path)
            s.sendall(b"THIS IS NOT JSON\n")
            s.shutdown(socket.SHUT_WR)
            chunks = []
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        resp = json.loads(b"".join(chunks).strip())
        self.assertFalse(resp["would_flag"])
        # Malformed JSON degrades to empty payload → process_action({}, engine) → occurrences=1
        self.assertIn("occurrences", resp)

    def test_empty_payload_returns_fallback(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(self._sock_path)
            s.sendall(b"")
            s.shutdown(socket.SHUT_WR)
            chunks = []
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        resp = json.loads(b"".join(chunks).strip())
        self.assertIn("would_flag", resp)

    def test_daemon_survives_bad_request_and_continues_serving(self):
        """After a malformed request the daemon must still handle a valid one."""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(self._sock_path)
            s.sendall(b"bad json\n")
            s.shutdown(socket.SHUT_WR)
            while s.recv(4096):
                pass

        # Second request should succeed normally
        resp = self._send({
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {},
            "session_id": "session-after-error",
        })
        self.assertIn("would_flag", resp)


class TestSocketPathResolution(unittest.TestCase):
    def test_socket_path_env_override(self, monkeypatch=None):
        os.environ["MEMCHECK_SOCKET"] = "/tmp/custom.sock"
        p = daemon_mod.socket_path()
        self.assertEqual(str(p), "/tmp/custom.sock")
        del os.environ["MEMCHECK_SOCKET"]

    def test_socket_path_default(self):
        os.environ.pop("MEMCHECK_SOCKET", None)
        p = daemon_mod.socket_path()
        self.assertIn("memcheck.sock", str(p))

    def test_socket_path_arg_wins(self):
        p = daemon_mod.socket_path("/override/path.sock")
        self.assertEqual(str(p), "/override/path.sock")


if __name__ == "__main__":
    unittest.main(verbosity=2)

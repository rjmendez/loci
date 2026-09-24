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
from unittest import mock

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

    # What the absolute-boundary fallback writes when the handler raises. A
    # golden-path assertion must pin values this dict cannot produce, or a
    # handler that always raises would still pass.
    FALLBACK = {"would_flag": False, "occurrences": 0, "qdrant": "unavailable"}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._audit_log = os.path.join(self._tmp.name, "audit.jsonl")
        self._prev_audit = os.environ.get("MEMCHECK_AUDIT_LOG")
        os.environ["MEMCHECK_AUDIT_LOG"] = self._audit_log
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
        if self._prev_audit is None:
            os.environ.pop("MEMCHECK_AUDIT_LOG", None)
        else:
            os.environ["MEMCHECK_AUDIT_LOG"] = self._prev_audit
        self._tmp.cleanup()

    def _send(self, payload: dict) -> dict:
        return _connect_and_send(self._sock_path, payload)

    def _audit_lines(self) -> list:
        with open(self._audit_log, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    # --- golden path ---------------------------------------------------------

    def test_pretooluse_returns_expected_keys(self):
        resp = self._send({
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "ls /tmp"},
            "session_id": "test-session-1",
        })
        # First observation against a working engine: 1 occurrence, backend ok.
        self.assertEqual(resp, {"would_flag": False, "occurrences": 1, "qdrant": "ok"})
        (line,) = self._audit_lines()
        self.assertEqual(line["tool_name"], "Bash")
        self.assertEqual(line["session_id"], "test-session-1")

    def test_pretooluse_occurrences_increment(self):
        payload = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"path": "/etc/hosts"},
            "session_id": "test-session-2",
        }
        responses = [self._send(payload) for _ in range(3)]
        self.assertEqual([r["occurrences"] for r in responses], [1, 2, 3])
        # PROMOTE_AFTER=3: only the third observation is a promoted block.
        self.assertEqual([r["would_flag"] for r in responses], [False, False, True])
        self.assertEqual({r["qdrant"] for r in responses}, {"ok"})

    def test_posttooluse_routing(self):
        # Two LH009 smells: process_code reports occurrences = n_issues = 2,
        # which neither process_action (1) nor the fallback (0) can produce.
        py_file = os.path.join(self._tmp.name, "smelly.py")
        with open(py_file, "w", encoding="utf-8") as fh:
            fh.write(
                "import asyncio\n\n\n"
                "async def g():\n    return 1\n\n\n"
                "async def f():\n    return asyncio.run(g())\n\n\n"
                "async def h():\n    return asyncio.run(g())\n"
            )
        resp = self._send({
            "hook_event_name": "PostToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": py_file},
            "session_id": "test-session-3",
        })
        # PostToolUse is advisory — would_flag is always False
        self.assertEqual(resp, {"would_flag": False, "occurrences": 2, "qdrant": "ok"})
        (line,) = self._audit_lines()
        self.assertEqual(line["event"], "code")
        self.assertEqual(line["codes"], ["LH009", "LH009"])

    def test_posttooluse_skip_is_routed_to_code_path(self):
        resp = self._send({
            "hook_event_name": "PostToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": "/nonexistent/foo.py"},
            "session_id": "test-session-3b",
        })
        self.assertEqual(resp, {"would_flag": False, "occurrences": 0, "qdrant": "ok"})
        (line,) = self._audit_lines()
        self.assertEqual((line["event"], line["skipped"]), ("code", True))

    def test_handler_exception_returns_the_fallback_and_keeps_serving(self):
        # Stub the dependency (process_action), not the handler under test.
        calls = []

        def exploding(payload, engine):
            calls.append(payload.get("session_id"))
            raise RuntimeError("backend blew up")

        payload = {"hook_event_name": "PreToolUse", "tool_name": "Bash",
                   "tool_input": {}, "session_id": "boom"}
        with mock.patch.object(daemon_mod.cli, "process_action", exploding):
            resp = self._send(payload)
        self.assertEqual(calls, ["boom"])
        self.assertEqual(resp, self.FALLBACK)
        # Worker survived: the next request gets a real answer.
        payload["session_id"] = "after"
        self.assertEqual(self._send(payload),
                         {"would_flag": False, "occurrences": 1, "qdrant": "ok"})

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
        # Malformed JSON degrades to empty payload → process_action({}, engine)
        # → occurrences=1 with a working backend (not the raise-fallback's 0).
        self.assertEqual(resp, {"would_flag": False, "occurrences": 1, "qdrant": "ok"})
        (line,) = self._audit_lines()
        self.assertEqual(line["tool_name"], "")

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
        self.assertEqual(resp, {"would_flag": False, "occurrences": 1, "qdrant": "ok"})

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
        self.assertEqual(resp, {"would_flag": False, "occurrences": 1, "qdrant": "ok"})


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

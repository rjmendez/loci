"""Tests for memcheck/cli.py — process_action, process_code, hash_embed, etc.

Exercises the shared testable core without a live Qdrant or model load.
InMemoryBackend is used so backends exercise the same recall/record paths as
QdrantBackend but without any network dependency.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

from memcheck import cli
from memcheck.backend import InMemoryBackend
from memcheck.engine import EmlConfig, VerdictEngine


def _engine() -> VerdictEngine:
    return VerdictEngine(InMemoryBackend(), EmlConfig(promote_after=3))


class TestHashEmbed(unittest.TestCase):
    def test_deterministic(self):
        v1 = cli.hash_embed("hello")
        v2 = cli.hash_embed("hello")
        self.assertEqual(v1, v2)

    def test_length(self):
        v = cli.hash_embed("some text")
        self.assertEqual(len(v), cli.EMBED_DIM)

    def test_values_in_range(self):
        v = cli.hash_embed("range test")
        self.assertTrue(all(-1.0 <= x <= 1.0 for x in v))

    def test_different_inputs_differ(self):
        self.assertNotEqual(cli.hash_embed("abc"), cli.hash_embed("xyz"))

    def test_none_input(self):
        v = cli.hash_embed(None)
        self.assertEqual(len(v), cli.EMBED_DIM)


class TestBuildDescriptor(unittest.TestCase):
    def test_format(self):
        d = cli.build_descriptor("Bash", {"command": "ls"})
        self.assertTrue(d.startswith("Bash "))
        self.assertIn("command", d)

    def test_secret_redacted(self):
        d = cli.build_descriptor("Tool", {"token": "supersecret"})
        self.assertNotIn("supersecret", d)
        self.assertIn("[REDACTED]", d)

    def test_none_input(self):
        d = cli.build_descriptor("Tool", None)
        self.assertTrue(d.startswith("Tool "))


class TestProcessAction(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["MEMCHECK_AUDIT_LOG"] = os.path.join(self._tmp.name, "audit.jsonl")

    def tearDown(self):
        os.environ.pop("MEMCHECK_AUDIT_LOG", None)
        self._tmp.cleanup()

    def test_returns_expected_keys(self):
        payload = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "echo hello"},
            "session_id": "s1",
        }
        record = cli.process_action(payload, _engine())
        for key in ("ts", "tool_name", "signature", "occurrences", "would_flag", "qdrant"):
            self.assertIn(key, record, f"missing key: {key}")

    def test_occurrences_increment_on_repeated_call(self):
        payload = {
            "tool_name": "Write",
            "tool_input": {"path": "/etc/passwd", "content": "x"},
            "session_id": "s2",
        }
        engine = _engine()
        r1 = cli.process_action(payload, engine)
        r2 = cli.process_action(payload, engine)
        self.assertEqual(r1["occurrences"], 1)
        self.assertEqual(r2["occurrences"], 2)

    def test_would_flag_false_below_promote_threshold(self):
        payload = {"tool_name": "Read", "tool_input": {}, "session_id": "s3"}
        r = cli.process_action(payload, _engine())
        # First occurrence — below PROMOTE_AFTER (3), must be False
        self.assertFalse(r["would_flag"])

    def test_would_flag_true_at_promote_threshold(self):
        payload = {
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
            "session_id": "s4",
        }
        engine = _engine()
        # PROMOTE_AFTER=3: would_flag should be True when occurrences >= 3
        for _ in range(cli.PROMOTE_AFTER):
            r = cli.process_action(payload, engine)
        self.assertTrue(r["would_flag"])

    def test_no_engine_still_returns_record(self):
        payload = {"tool_name": "Read", "tool_input": {}, "session_id": "s5"}
        record = cli.process_action(payload, None)
        self.assertIn("qdrant", record)
        self.assertEqual(record["qdrant"], "unavailable")

    def test_tool_name_in_record(self):
        payload = {"tool_name": "MyTool", "tool_input": {}, "session_id": "s6"}
        record = cli.process_action(payload, None)
        self.assertEqual(record["tool_name"], "MyTool")

    def test_session_id_in_record(self):
        payload = {"tool_name": "T", "tool_input": {}, "session_id": "my-session-id"}
        record = cli.process_action(payload, None)
        self.assertEqual(record["session_id"], "my-session-id")

    def test_audit_log_written(self):
        log_path = os.environ["MEMCHECK_AUDIT_LOG"]
        payload = {"tool_name": "Audit", "tool_input": {}, "session_id": "s7"}
        cli.process_action(payload, None)
        self.assertTrue(os.path.exists(log_path))
        with open(log_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        self.assertGreater(len(lines), 0)
        import json
        data = json.loads(lines[-1])
        self.assertEqual(data["tool_name"], "Audit")


class TestProcessCode(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["MEMCHECK_AUDIT_LOG"] = os.path.join(self._tmp.name, "audit.jsonl")
        # Write a trivial Python file for the PostToolUse path to check
        self._py_file = os.path.join(self._tmp.name, "example.py")
        with open(self._py_file, "w") as f:
            f.write("def add(a, b):\n    return a + b\n")

    def tearDown(self):
        os.environ.pop("MEMCHECK_AUDIT_LOG", None)
        self._tmp.cleanup()

    def test_returns_dict_for_py_file(self):
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": self._py_file},
        }
        record = cli.process_code(payload, _engine(), repo_root=self._tmp.name)
        self.assertIsInstance(record, dict)
        self.assertIn("event", record)
        self.assertEqual(record["event"], "code")

    def test_skips_non_py_file(self):
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": "/some/file.txt"},
        }
        record = cli.process_code(payload, None)
        self.assertIn("event", record)
        self.assertEqual(record["event"], "code")

    def test_no_engine_still_returns_record(self):
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": self._py_file},
        }
        record = cli.process_code(payload, None, repo_root=self._tmp.name)
        self.assertIsInstance(record, dict)


class TestStatsDegradation(unittest.TestCase):
    """A zeroed stats page is what a never-firing memcheck AND an unreachable store
    both produce. The degraded one has to say so."""

    class _DeadBackend(InMemoryBackend):
        async def stats(self) -> dict:
            raise RuntimeError("qdrant unreachable")

    def test_healthy_stats_pass_through_without_an_ok_flag(self):
        import asyncio

        stats = asyncio.run(_engine().stats())
        self.assertEqual(stats.get("total_verdicts"), 0)
        self.assertNotIn("error", stats)
        self.assertTrue(stats.get("ok", True))

    def test_backend_error_is_marked_not_measured(self):
        import asyncio

        engine = VerdictEngine(self._DeadBackend(), EmlConfig())
        stats = asyncio.run(engine.stats())
        self.assertEqual(stats["total_verdicts"], 0)
        self.assertIs(stats["ok"], False)
        self.assertIn("qdrant unreachable", stats["error"])

    def test_cmd_stats_exit_code_follows_the_ok_flag(self):
        import io
        import contextlib
        from unittest import mock

        with mock.patch.object(cli, "_build_qdrant_backend", lambda: self._DeadBackend()), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = cli._cmd_stats([])
        self.assertEqual(rc, 1)
        self.assertIn('"ok": false', out.getvalue())

        with mock.patch.object(cli, "_build_qdrant_backend", lambda: InMemoryBackend()), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli._cmd_stats([]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

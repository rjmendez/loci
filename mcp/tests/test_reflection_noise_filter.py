"""The reflection loop reads agent transcripts for failures, not for the harness telling an agent no.

2026-10-05: of 1,115 stored findings, 1,079 came from subagent transcripts and the top "errors" were the harness's
own refusals (a sleep-then-poll block, a worktree-isolation notice, a structured-output schema mismatch) and
user-role prompt text (workflow task prompts, teammate messages, system notifications) that merely contained
the word "error". A learner trained on that learns what the harness refuses.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import server  # noqa: E402


def _result_event(text, is_error=True):
    return {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "is_error": is_error, "content": text}]}}


def _text_event(role, text):
    return {"type": role, "message": {"role": role, "content": text}}


class _Scan(unittest.TestCase):
    def scan(self, *events):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.jsonl"
            path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
            return server._process_reflection_item("claude_code_event", str(path), max_lines=500)


class TestHarnessGuardResults(_Scan):
    GUARDS = (
        "<tool_use_error>blocked: sleep 30 followed by: cat out.txt. to wait for a condition use Monitor",
        "<tool_use_error>Blocked: start-sleep 20 followed by: wsl -e bash",
        "<tool_use_error>This session is isolated in the worktree C:\\x\\wt. Edit the worktree copy",
        "<tool_use_error>Subagents should return findings as text, not write report files.",
        "Output does not match required schema: root: must have required property 'findings'",
        "blocked: sleep 5 followed by: ls",
        "<tool_use_error>This subagent's parent bg session hasn't isolated yet, so writes to the shared checkout are blocked. Re-spawn",
        "<tool_use_error>This subagent’s parent bg session hasn’t isolated yet, so writes are blocked.",
    )

    def test_a_harness_refusal_is_counted_as_a_guard_not_as_an_error(self):
        for text in self.GUARDS:
            with self.subTest(text=text[:50]):
                out = self.scan(_result_event(text))
                self.assertEqual(out["errors"], {})
                self.assertEqual(out["warnings"], {})
                self.assertEqual(out["events"].get("harness_guard"), 1)
                self.assertNotIn("tool_result_error", out["events"])

    def test_a_real_tool_failure_is_still_an_error(self):
        """Positive twin: same block shape, a genuine failure."""
        for text in ("Exit code 1\nTraceback: boom", "File does not exist. Note: your current working directory is x",
                     "<tool_use_error>InputValidationError: bad field</tool_use_error>"):
            with self.subTest(text=text[:40]):
                out = self.scan(_result_event(text))
                self.assertEqual(out["events"].get("tool_result_error"), 1)
                self.assertNotIn("harness_guard", out["events"])
                self.assertEqual(len(out["errors"]), 1)
                self.assertTrue(next(iter(out["errors"])).startswith("claude tool_result error: "))

    def test_a_guard_phrase_that_is_not_the_first_line_is_not_a_guard(self):
        out = self.scan(_result_event("Exit code 2\nblocked: the build step was blocked by a lock"))
        self.assertEqual(out["events"].get("tool_result_error"), 1)
        self.assertNotIn("harness_guard", out["events"])

    def test_a_guard_phrase_in_the_middle_of_the_first_line_is_not_a_guard(self):
        out = self.scan(_result_event("Exit code 1: the deploy was blocked: sleep did not help"))
        self.assertEqual(out["events"].get("tool_result_error"), 1)
        self.assertNotIn("harness_guard", out["events"])

    def test_a_non_error_result_is_ignored_whatever_it_says(self):
        out = self.scan(_result_event("blocked: sleep 30 followed by: cat x", is_error=False))
        self.assertEqual((out["errors"], out["warnings"], out["events"].get("harness_guard")), ({}, {}, None))

    def test_guards_and_failures_in_one_file_are_separated(self):
        out = self.scan(_result_event(self.GUARDS[0]), _result_event("Exit code 1\nboom"), _result_event(self.GUARDS[2]))
        self.assertEqual(out["events"].get("harness_guard"), 2)
        self.assertEqual(out["events"].get("tool_result_error"), 1)
        self.assertEqual(list(out["errors"].values()), [1])


class TestUserPromptsAreNotScanned(_Scan):
    TEXT = "ERROR: disk full while writing the report"

    def test_error_words_in_a_user_role_message_are_not_a_finding(self):
        out = self.scan(_text_event("user", self.TEXT))
        self.assertEqual((out["errors"], out["warnings"]), ({}, {}))
        self.assertEqual(out["events"].get("user"), 1)

    def test_the_same_words_in_an_assistant_message_still_are(self):
        """Positive twin: only the role changed."""
        out = self.scan(_text_event("assistant", self.TEXT))
        self.assertEqual(len(out["errors"]), 1)
        self.assertIn("disk full", next(iter(out["errors"])))

    def test_harness_notices_arriving_as_user_messages_are_not_findings(self):
        for text in ("[SYSTEM NOTIFICATION - NOT USER INPUT] Error: task failed",
                     "[workflow harness - computed task] Warning: retry limit exceeded",
                     "<teammate-message teammate_id=\"x\">failed: tests broke</teammate-message>"):
            with self.subTest(text=text[:40]):
                out = self.scan(_text_event("user", text))
                self.assertEqual((out["errors"], out["warnings"]), ({}, {}))

    def test_an_interrupted_turn_is_still_seen_in_a_user_message(self):
        out = self.scan(_text_event("user", "[Request interrupted by user for tool use]"))
        self.assertEqual(out["events"].get("interrupted_turn"), 1)
        self.assertEqual(len(out["warnings"]), 1)

    def test_a_failed_tool_result_inside_a_user_event_is_still_read(self):
        out = self.scan(_result_event("Exit code 1\nboom"))
        self.assertEqual(out["events"].get("user"), 1)
        self.assertEqual(out["events"].get("tool_result_error"), 1)

    def test_a_permission_denial_inside_a_user_event_is_still_read(self):
        out = self.scan(_result_event("Permission to use Bash has been denied."))
        self.assertEqual(out["events"].get("permission_denied"), 1)


if __name__ == "__main__":
    unittest.main()

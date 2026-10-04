"""Per-source coverage of the reflection loop: Claude Code, Copilot and Hermes (synthetic fixtures)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_MCP_DIR = str(Path(__file__).resolve().parent.parent)
if _MCP_DIR not in sys.path:
    sys.path.insert(0, _MCP_DIR)

import server  # noqa: E402

# Obviously fake credentials: none may ever reach a stored finding or a tool result.
FAKE_BEARER = "FAKEBEARERVALUE1234567890"
FAKE_APIKEY = "sk-FAKEKEY000000000000"
FAKE_TOKEN = "FAKETOKENVALUE987654"
FAKE_HEX = "deadbeef" * 8
FAKE_B64 = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5"
FAKE_PEM = "FAKEPRIVATEKEYBODYLINE"
SECRETS = [FAKE_BEARER, FAKE_APIKEY, FAKE_TOKEN, FAKE_HEX, FAKE_B64, FAKE_PEM]


def _w(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


HERMES_ERRORS = "\n".join([
    "2026-01-01 00:00:01,100 INFO agent: started ok retry later",
    f"2026-01-01 00:00:02,200 ERROR agent: upstream call failed Authorization: Bearer {FAKE_BEARER}",
    f'2026-01-01 00:00:03,300 ERROR gateway: bad config {{"api_key": "{FAKE_APIKEY}", "x": 1}}',
    f"2026-01-01 00:00:04,400 WARNING gateway: token={FAKE_TOKEN} expiring soon",
    f"2026-01-01 00:00:05,500 ERROR agent: blob {FAKE_HEX} and {FAKE_B64}",
    "Traceback (most recent call last):",
    f"-----BEGIN PRIVATE KEY-----\n{FAKE_PEM}\n-----END PRIVATE KEY-----",
    "ValueError: boom",
]) + "\n"

CLAUDE_LINES = [
    {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
         "content": f"Exit code 1\nfailed to run step token={FAKE_TOKEN}"}]}},
    {"type": "user", "message": {"role": "user", "content": [
        {"type": "text", "text": "[Request interrupted by user for tool use]"}]}},
    {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t2", "is_error": True,
         "content": "Permission for this action has been denied."}]}},
    {"type": "assistant", "isApiErrorMessage": True,
     "message": {"role": "assistant", "content": [{"type": "text", "text": "API Error: 529 overloaded"}]}},
    {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t3", "is_error": False, "content": "all good"}]}},
]


class ReflectionSourcesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.claude = tmp / "claude" / "projects"
        self.copilot = tmp / "copilot"
        self.hermes = tmp / "hermes"
        _w(self.claude / "proj" / "s.jsonl", "\n".join(json.dumps(x) for x in CLAUDE_LINES) + "\n")
        _w(self.copilot / "logs" / "process-1.log",
           f"2026-01-01T00:00:00.000Z [ERROR] copilot failed password={FAKE_TOKEN}\n")
        _w(self.copilot / "session-state" / "a" / "events.jsonl",
           json.dumps({"type": "x", "assistant_content": "Error: copilot session failed"}) + "\n")
        _w(self.hermes / "logs" / "errors.log", HERMES_ERRORS)
        _w(self.hermes / "logs" / "agent.log", "2026-01-01 00:00:09,000 ERROR agent: other failure\n")
        self.state_file = tmp / "state" / "state.json"
        self.state = None
        env = {
            "LOCI_REFLECT_CLAUDE_ROOTS": str(self.claude),
            "LOCI_REFLECT_COPILOT_ROOTS": str(self.copilot),
            "LOCI_REFLECT_HERMES_ROOTS": str(self.hermes),
        }
        self.stored: list[dict] = []

        def fake_store(**kwargs):
            self.stored.append(kwargs)
            return json.dumps({"stored": True})

        for ctx in (
            patch.dict(os.environ, env),
            patch.object(server, "REFLECTION_STATE_FILE", self.state_file),
            patch.object(server, "_ensure_investigation_exists", side_effect=lambda *a, **k: None),
            patch.object(server, "investigation_store", side_effect=fake_store),
        ):
            ctx.start()
            self.addCleanup(ctx.stop)

    def _seed(self, **kw):
        return json.loads(server.reflection_loop_seed(investigation_id="t-inv", **kw))

    def _tick_all(self):
        outs = []
        for _ in range(10):
            raw = server.reflection_loop_tick(max_items=20, max_lines_per_file=500)
            outs.append(raw)
            if json.loads(raw).get("remaining_queue", 0) == 0:
                break
        return outs

    def test_every_source_seeds_ticks_and_stores_a_source_tagged_finding(self):
        res = self._seed()
        for src in ("claude", "copilot", "hermes"):
            self.assertGreaterEqual(res["per_source"][src]["enqueued"], 1, src)
        kinds = {i["kind"]: i["source"] for i in json.loads(self.state_file.read_text())["queue"]}
        self.assertEqual(kinds["hermes_log"], "hermes")
        self.assertEqual(kinds["claude_code_event"], "claude")
        self.assertEqual(kinds["process_log"], "copilot")
        self._tick_all()
        observed = [c for c in self.stored if c["finding_type"] == "observed"]
        for src in ("claude", "copilot", "hermes"):
            self.assertTrue(any(f"source-{src}" in c["tags"] for c in observed), src)
        status = json.loads(server.reflection_loop_status())
        by_source = status["stats"]["by_source"]
        for src in ("claude", "copilot", "hermes"):
            self.assertGreaterEqual(by_source[src]["files_processed"], 1, src)
        self.assertGreaterEqual(by_source["hermes"]["errors_seen"], 3)

    def test_missing_and_unreadable_roots_are_reported_per_source(self):
        gone = Path(self._tmp.name) / "nope"
        with patch.dict(os.environ, {"LOCI_REFLECT_HERMES_ROOTS": str(gone)}):
            res = self._seed()
        h = res["per_source"]["hermes"]
        self.assertEqual(h["roots"], [{"root": str(gone), "status": "missing"}])
        self.assertEqual(h["skipped_missing_root"], 1)
        self.assertEqual(h["enqueued"], 0)
        self.assertGreaterEqual(res["per_source"]["claude"]["enqueued"], 1)
        with patch.object(server.Path, "iterdir", side_effect=PermissionError("denied")):
            res = self._seed(dry_run=True)
        self.assertEqual(res["per_source"]["claude"]["roots"][0]["status"], "unreadable")

    def test_paths_list_uses_pathsep_and_duplicates_are_counted(self):
        extra = Path(self._tmp.name) / "hermes2"
        _w(extra / "logs" / "gateway.log", "2026-01-01 00:00:09,000 ERROR gateway: x\n")
        joined = os.pathsep.join([str(self.hermes), str(extra)])
        with patch.dict(os.environ, {"LOCI_REFLECT_HERMES_ROOTS": joined}):
            first = self._seed()
            second = self._seed()
        self.assertEqual(first["per_source"]["hermes"]["enqueued"], 3)
        self.assertEqual(second["per_source"]["hermes"]["enqueued"], 0)
        self.assertEqual(second["per_source"]["hermes"]["skipped_duplicate"], 3)

    def test_dry_run_enqueues_and_persists_nothing(self):
        res = self._seed(dry_run=True)
        self.assertTrue(res["dry_run"])
        for src in ("claude", "copilot", "hermes"):
            self.assertGreaterEqual(res["per_source"][src]["candidates"], 1, src)
        self.assertFalse(self.state_file.exists())
        status = json.loads(server.reflection_loop_status())
        self.assertEqual(status["queue_size"], 0)
        self.assertEqual(self.stored, [])

    def test_secrets_never_reach_findings_or_tool_output(self):
        outs = [json.dumps(self._seed())]
        outs += self._tick_all()
        outs.append(server.reflection_loop_status(queue_preview=50))
        blob = "\n".join(outs) + json.dumps(self.stored) + self.state_file.read_text()
        for secret in SECRETS:
            self.assertNotIn(secret, blob)
        self.assertTrue(self.stored)

    def test_scrub_forms(self):
        cases = [
            f"Authorization: Bearer {FAKE_BEARER}",
            f'curl -H "x-api-key: {FAKE_APIKEY}"',
            f"api_key={FAKE_APIKEY}&other=1",
            f"{{'token': '{FAKE_TOKEN}'}}",
            f"hash {FAKE_HEX}",
            f"blob {FAKE_B64}",
            f"-----BEGIN RSA PRIVATE KEY-----\n{FAKE_PEM}\n-----END RSA PRIVATE KEY-----",
        ]
        for text in cases:
            out = server._reflection_scrub(text)
            for secret in SECRETS:
                self.assertNotIn(secret, out, text)
        self.assertEqual(server._reflection_scrub("plain error: disk full"), "plain error: disk full")

    def test_hermes_scan_extracts_levelled_lines_and_dedupes(self):
        summary = server._process_reflection_item(
            "hermes_log", str(self.hermes / "logs" / "errors.log"), max_lines=500)
        self.assertEqual(summary["status"], "processed")
        self.assertEqual(summary["sampling_mode"], "tail")
        self.assertGreaterEqual(len(summary["errors"]), 3)
        self.assertEqual(len(summary["warnings"]), 1)
        # INFO line containing "retry" is not a warning.
        self.assertFalse(any("started ok" in k for k in summary["warnings"]))
        text = _w(Path(self._tmp.name) / "dup.log", "\n".join(
            f"2026-01-01 00:00:{i:02d},000 ERROR agent: failed id {i}" for i in range(20)))
        dup = server._process_reflection_item("hermes_log", str(text), max_lines=500)
        self.assertEqual(list(dup["errors"].values()), [20])

    def test_hermes_tail_is_bounded(self):
        big = _w(Path(self._tmp.name) / "big.log", "\n".join(
            f"2026-01-01 00:00:00,000 ERROR agent: line {i}" for i in range(5000)))
        with patch.object(server, "REFLECTION_HERMES_TAIL_LINES", 100):
            summary = server._process_reflection_item("hermes_log", str(big), max_lines=4000)
        self.assertEqual(summary["lines_scanned"], 100)

    def test_claude_scan_finds_tool_errors_interrupts_permission_and_api_errors(self):
        path = str(self.claude / "proj" / "s.jsonl")
        summary = server._process_reflection_item("claude_code_event", path, max_lines=500)
        self.assertEqual(summary["events"].get("tool_result_error"), 1)
        self.assertEqual(summary["events"].get("interrupted_turn"), 1)
        self.assertEqual(summary["events"].get("permission_denied"), 1)
        self.assertEqual(summary["events"].get("api_error"), 1)
        self.assertTrue(any("tool_result error" in k for k in summary["errors"]))
        self.assertTrue(any("api error" in k for k in summary["errors"]))
        self.assertTrue(any("interrupted" in k for k in summary["warnings"]))
        self.assertTrue(any("permission denied" in k for k in summary["warnings"]))
        self.assertFalse(any("all good" in k for k in list(summary["errors"]) + list(summary["warnings"])))

    def test_llm_triage_stays_opt_in(self):
        self._seed()
        with patch.object(server, "_reflection_llm_triage_metadata") as triage:
            server.reflection_loop_tick(max_items=20)
        triage.assert_not_called()


if __name__ == "__main__":
    unittest.main()

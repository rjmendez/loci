import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Also handled by conftest.py; this covers direct `python test_reflection_loop.py` runs.
_MCP_DIR = str(Path(__file__).resolve().parent.parent)
if _MCP_DIR not in sys.path:
    sys.path.insert(0, _MCP_DIR)

import server  # noqa: E402


class ReflectionLoopTests(unittest.TestCase):
    def test_process_log_uses_tail_sampling_for_large_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "process.log"
            head = "".join(f"info line {i}\n" for i in range(200))
            tail = "final error tail marker\n"
            log_path.write_text(head + tail, encoding="utf-8")

            with patch.object(server, "REFLECTION_LOG_TAIL_MIN_FILE_BYTES", 100), patch.object(
                server, "REFLECTION_LOG_TAIL_READ_BYTES", 160
            ):
                result = server._process_reflection_item("process_log", str(log_path), max_lines=5)

        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["sampling_mode"], "tail")
        self.assertLessEqual(result["lines_scanned"], 5)
        self.assertGreaterEqual(sum(result["errors"].values()), 1)

    def test_tick_saturates_signatures_and_marks_unreceipted_observed(self):
        state = server._reflection_default_state()
        state["investigation_id"] = "test-inv"
        state["queue"] = [
            {"kind": "process_log", "path": "/tmp/a.log"},
            {"kind": "process_log", "path": "/tmp/b.log"},
        ]

        stored_calls: list[dict] = []
        summary = {
            "status": "processed",
            "kind": "process_log",
            "path": "/tmp/x.log",
            "lines_scanned": 10,
            "bytes_scanned": 100,
            "sampling_mode": "full",
            "events": {},
            "tools": {},
            "errors": {"repeat-signature": 5},
            "warnings": {},
        }

        def fake_store(**kwargs):
            stored_calls.append(kwargs)
            return json.dumps({"stored": True})

        with patch.object(server, "_load_reflection_state", side_effect=lambda: state), patch.object(
            server, "_save_reflection_state", side_effect=lambda new_state: state.update(new_state)
        ), patch.object(
            server, "_ensure_investigation_exists", side_effect=lambda *args, **kwargs: None
        ), patch.object(
            server, "_process_reflection_item", side_effect=[summary, summary]
        ), patch.object(
            server, "investigation_store", side_effect=fake_store
        ), patch.object(
            server, "REFLECTION_SIGNATURE_OBSERVE_LIMIT", 1
        ):
            result = json.loads(
                server.reflection_loop_tick(max_items=2, max_lines_per_file=200, store_item_findings=True)
            )

        self.assertEqual(result["processed_items"], 2)
        self.assertGreaterEqual(result["stats"]["error_signatures_suppressed"], 5)

        observed = [c for c in stored_calls if c["finding_type"] == "observed"]
        self.assertEqual(len(observed), 2)
        for call in observed:
            self.assertEqual(call["confidence"], "low")
            self.assertIn("unreceipted-observed", call["tags"])

        self.assertIn("saturated=1 signatures (5 hits)", observed[1]["text"])

    def test_tick_prioritizes_process_logs_before_session_events(self):
        state = server._reflection_default_state()
        state["investigation_id"] = "test-inv"
        state["queue"] = [
            {"kind": "session_event", "path": "/tmp/session.log"},
            {"kind": "process_log", "path": "/tmp/process.log"},
        ]
        processed_kinds: list[str] = []

        def fake_process(kind, path, max_lines):
            processed_kinds.append(kind)
            return {
                "status": "processed",
                "kind": kind,
                "path": path,
                "lines_scanned": 1,
                "bytes_scanned": 1,
                "sampling_mode": "full",
                "events": {},
                "tools": {},
                "errors": {},
                "warnings": {},
            }

        with patch.object(server, "_load_reflection_state", side_effect=lambda: state), patch.object(
            server, "_save_reflection_state", side_effect=lambda new_state: state.update(new_state)
        ), patch.object(
            server, "_ensure_investigation_exists", side_effect=lambda *args, **kwargs: None
        ), patch.object(
            server, "_process_reflection_item", side_effect=fake_process
        ), patch.object(
            server, "investigation_store", return_value=json.dumps({"stored": True})
        ):
            server.reflection_loop_tick(max_items=2, max_lines_per_file=100, store_item_findings=False)

        self.assertEqual(processed_kinds[0], "process_log")
        self.assertEqual(processed_kinds[1], "session_event")

    def test_tick_batches_low_signal_session_events_into_one_observed(self):
        state = server._reflection_default_state()
        state["investigation_id"] = "test-inv"
        state["queue"] = [
            {"kind": "session_event", "path": "/tmp/s1.log"},
            {"kind": "session_event", "path": "/tmp/s2.log"},
        ]
        stored_calls: list[dict] = []
        summary = {
            "status": "processed",
            "kind": "session_event",
            "path": "/tmp/s.log",
            "lines_scanned": 10,
            "bytes_scanned": 100,
            "sampling_mode": "full",
            "events": {"hook.start": 1},
            "tools": {},
            "errors": {},
            "warnings": {},
        }

        def fake_store(**kwargs):
            stored_calls.append(kwargs)
            return json.dumps({"stored": True})

        with patch.object(server, "_load_reflection_state", side_effect=lambda: state), patch.object(
            server, "_save_reflection_state", side_effect=lambda new_state: state.update(new_state)
        ), patch.object(
            server, "_ensure_investigation_exists", side_effect=lambda *args, **kwargs: None
        ), patch.object(
            server, "_process_reflection_item", side_effect=[summary, summary]
        ), patch.object(
            server, "investigation_store", side_effect=fake_store
        ):
            server.reflection_loop_tick(max_items=2, max_lines_per_file=100, store_item_findings=True)

        observed = [c for c in stored_calls if c["finding_type"] == "observed"]
        self.assertEqual(len(observed), 1)
        self.assertIn("batched low-signal session_event files count=2", observed[0]["text"])
        self.assertIn("batched-low-signal", observed[0]["tags"])


class ReflectionQueueBoundsTests(unittest.TestCase):
    """The queue, the dedupe map and the re-queue path must all be bounded.

    One seed offers ~620 candidates while a tick drains at most 20, and the whole
    state file is rewritten on every call, so anything that only ever grows here
    is paid for on every seed and every tick.
    """

    def _seed_into(self, home: Path, state: dict, queue_max: int) -> dict:
        with patch.object(server.Path, "home", staticmethod(lambda: home)), patch.object(
            server, "REFLECTION_QUEUE_MAX", queue_max
        ), patch.object(
            server, "_load_reflection_state", side_effect=lambda: state
        ), patch.object(
            server, "_save_reflection_state", side_effect=lambda new_state: state.update(new_state)
        ), patch.object(
            server, "_ensure_investigation_exists", side_effect=lambda *args, **kwargs: None
        ):
            return json.loads(server.reflection_loop_seed(investigation_id="test-inv"))

    def test_seed_caps_the_queue_and_keeps_the_high_priority_kinds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            logs = home / ".copilot" / "logs"
            logs.mkdir(parents=True)
            for i in range(3):
                (logs / f"process-{i}.log").write_text("info\n", encoding="utf-8")
            projects = home / ".claude" / "projects" / "proj"
            projects.mkdir(parents=True)
            for i in range(5):
                (projects / f"s{i}.jsonl").write_text("{}\n", encoding="utf-8")

            state = server._reflection_default_state()
            result = self._seed_into(home, state, queue_max=4)

        # 8 candidates offered, 4 kept.
        self.assertEqual(result["queued_added"], 8)
        self.assertEqual(result["queue_size"], 4)
        self.assertEqual(result["queued_dropped_over_cap"], 4)
        self.assertEqual(len(state["queue"]), 4)
        # process_log is priority 0, so every one of them survives the cap.
        kinds = [item["kind"] for item in state["queue"]]
        self.assertEqual(kinds.count("process_log"), 3)
        self.assertEqual(kinds.count("claude_code_event"), 1)

    def test_seed_under_the_cap_drops_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            logs = home / ".copilot" / "logs"
            logs.mkdir(parents=True)
            (logs / "process-0.log").write_text("info\n", encoding="utf-8")

            state = server._reflection_default_state()
            result = self._seed_into(home, state, queue_max=4)

        self.assertEqual(result["queue_size"], 1)
        self.assertEqual(result["queued_dropped_over_cap"], 0)

    def test_save_prunes_the_processed_dedupe_map_to_the_newest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state = server._reflection_default_state()
            state["processed"] = {
                f"process_log|/tmp/{i}.log": {"processed_at": f"2026-08-{i + 1:02d}T00:00:00Z"}
                for i in range(6)
            }
            with patch.object(server, "REFLECTION_STATE_FILE", state_file), patch.object(
                server, "REFLECTION_PROCESSED_MAX", 3
            ):
                server._save_reflection_state(state)
            written = json.loads(state_file.read_text(encoding="utf-8"))

        self.assertEqual(len(written["processed"]), 3)
        # Newest processed_at survives; the oldest three are gone.
        self.assertEqual(
            sorted(written["processed"]),
            ["process_log|/tmp/3.log", "process_log|/tmp/4.log", "process_log|/tmp/5.log"],
        )

    def _tick_with_status(self, status: str) -> tuple[dict, dict, list]:
        state = server._reflection_default_state()
        state["investigation_id"] = "test-inv"
        state["queue"] = [{"kind": "session_event", "path": "/tmp/gone.jsonl"}]
        summary = {
            "status": status,
            "kind": "session_event",
            "path": "/tmp/gone.jsonl",
            "lines_scanned": 0,
            "bytes_scanned": 0,
            "events": {},
            "tools": {},
            "errors": {},
            "warnings": {},
        }
        stored_calls: list = []

        def fake_store(**kwargs):
            stored_calls.append(kwargs)
            return json.dumps({"stored": True})

        with patch.object(server, "_load_reflection_state", side_effect=lambda: state), patch.object(
            server, "_save_reflection_state", side_effect=lambda new_state: state.update(new_state)
        ), patch.object(
            server, "_ensure_investigation_exists", side_effect=lambda *args, **kwargs: None
        ), patch.object(
            server, "_process_reflection_item", side_effect=[summary]
        ), patch.object(
            server, "investigation_store", side_effect=fake_store
        ):
            result = json.loads(
                server.reflection_loop_tick(max_items=1, max_lines_per_file=100, store_item_findings=True)
            )
        return result, state, stored_calls

    def test_terminal_status_item_is_not_requeued(self):
        result, state, stored_calls = self._tick_with_status("missing")

        # A file that no longer exists will still be missing next tick; re-queueing
        # it replays the item and its "gap" finding forever.
        self.assertEqual(result["remaining_queue"], 0)
        self.assertEqual(state["queue"], [])
        gaps = [c for c in stored_calls if c["finding_type"] == "gap"]
        self.assertEqual(len(gaps), 1)
        self.assertIn("status=missing", gaps[0]["text"])
        self.assertIn("terminal", gaps[0]["tags"])

    def test_retryable_status_item_is_still_requeued(self):
        result, state, stored_calls = self._tick_with_status("transient_failure")

        self.assertEqual(result["remaining_queue"], 1)
        self.assertEqual(state["queue"], [{"kind": "session_event", "path": "/tmp/gone.jsonl"}])
        gaps = [c for c in stored_calls if c["finding_type"] == "gap"]
        self.assertEqual(len(gaps), 1)
        self.assertIn("re-queued", gaps[0]["tags"])


if __name__ == "__main__":
    unittest.main()

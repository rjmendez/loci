import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server


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


if __name__ == "__main__":
    unittest.main()

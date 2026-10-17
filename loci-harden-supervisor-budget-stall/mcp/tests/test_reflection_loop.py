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

    def _tick_with(self, store_item_findings):
        state = server._reflection_default_state()
        state["investigation_id"] = "test-inv"
        state["queue"] = [{"kind": "process_log", "path": "/tmp/a.log"}]
        summary = {
            "status": "processed",
            "kind": "process_log",
            "path": "/tmp/a.log",
            "lines_scanned": 10,
            "bytes_scanned": 100,
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
            server, "_process_reflection_item", side_effect=[summary]
        ), patch.object(
            server, "investigation_store", return_value=json.dumps({"stored": True})
        ):
            server.reflection_loop_tick(max_items=1, max_lines_per_file=100,
                                        store_item_findings=store_item_findings)
        return state

    def test_preview_tick_does_not_retire_the_item_it_stored_nothing_for(self):
        """reflection_loop_seed drops every key in processed and only reset_queue=True
        clears it, so an item marked processed by a tick that wrote no finding is out of
        the reflection corpus for good."""
        state = self._tick_with(store_item_findings=False)
        self.assertEqual(state["processed"], {})

    def test_storing_tick_still_marks_the_item_processed(self):
        state = self._tick_with(store_item_findings=True)
        self.assertEqual(list(state["processed"]), ["process_log|/tmp/a.log"])

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

    def test_tick_attaches_llm_triage_metadata_when_enabled(self):
        state = server._reflection_default_state()
        state["investigation_id"] = "test-inv"
        state["queue"] = [{"kind": "process_log", "path": "/tmp/a.log"}]
        summary = {
            "status": "processed",
            "kind": "process_log",
            "path": "/tmp/a.log",
            "lines_scanned": 7,
            "bytes_scanned": 70,
            "sampling_mode": "full",
            "events": {},
            "tools": {"pytest": 1},
            "errors": {"assertion failed": 1},
            "warnings": {},
        }
        stored_calls: list[dict] = []

        def fake_store(**kwargs):
            stored_calls.append(kwargs)
            return json.dumps({"stored": True})

        with patch.object(server, "_load_reflection_state", side_effect=lambda: state), patch.object(
            server, "_save_reflection_state", side_effect=lambda new_state: state.update(new_state)
        ), patch.object(
            server, "_ensure_investigation_exists", side_effect=lambda *args, **kwargs: None
        ), patch.object(
            server, "_process_reflection_item", return_value=summary
        ), patch.object(
            server, "_reflection_llm_triage_metadata",
            return_value={"llm_triage": {"category": "real_regression", "novelty": "novel_signal"}},
        ) as triage_mock, patch.object(
            server, "investigation_store", side_effect=fake_store
        ):
            server.reflection_loop_tick(
                max_items=1,
                max_lines_per_file=100,
                store_item_findings=True,
                enable_llm_triage=True,
                max_llm_items=1,
            )

        observed = [c for c in stored_calls if c["finding_type"] == "observed"]
        self.assertEqual(len(observed), 1)
        self.assertEqual(
            observed[0]["metadata"],
            {"llm_triage": {"category": "real_regression", "novelty": "novel_signal"}},
        )
        triage_mock.assert_called_once()

    def test_tick_llm_triage_fail_open_marks_degraded(self):
        state = server._reflection_default_state()
        state["investigation_id"] = "test-inv"
        state["queue"] = [{"kind": "process_log", "path": "/tmp/a.log"}]
        summary = {
            "status": "processed",
            "kind": "process_log",
            "path": "/tmp/a.log",
            "lines_scanned": 7,
            "bytes_scanned": 70,
            "sampling_mode": "full",
            "events": {},
            "tools": {},
            "errors": {"assertion failed": 1},
            "warnings": {},
        }
        stored_calls: list[dict] = []

        def fake_store(**kwargs):
            stored_calls.append(kwargs)
            return json.dumps({"stored": True})

        with patch.object(server, "_load_reflection_state", side_effect=lambda: state), patch.object(
            server, "_save_reflection_state", side_effect=lambda new_state: state.update(new_state)
        ), patch.object(
            server, "_ensure_investigation_exists", side_effect=lambda *args, **kwargs: None
        ), patch.object(
            server, "_process_reflection_item", return_value=summary
        ), patch.object(
            server, "_reflection_llm_triage_metadata",
            return_value={"llm_triage": {"degraded": True}},
        ), patch.object(
            server, "investigation_store", side_effect=fake_store
        ):
            server.reflection_loop_tick(
                max_items=1,
                max_lines_per_file=100,
                store_item_findings=True,
                enable_llm_triage=True,
                max_llm_items=1,
            )

        observed = [c for c in stored_calls if c["finding_type"] == "observed"]
        self.assertEqual(observed[0]["metadata"], {"llm_triage": {"degraded": True}})

    def test_tick_respects_max_llm_items_budget(self):
        state = server._reflection_default_state()
        state["investigation_id"] = "test-inv"
        state["queue"] = [
            {"kind": "process_log", "path": "/tmp/a.log"},
            {"kind": "process_log", "path": "/tmp/b.log"},
            {"kind": "process_log", "path": "/tmp/c.log"},
        ]
        summary = {
            "status": "processed",
            "kind": "process_log",
            "path": "/tmp/a.log",
            "lines_scanned": 1,
            "bytes_scanned": 10,
            "sampling_mode": "full",
            "events": {},
            "tools": {},
            "errors": {"repeatable failure": 1},
            "warnings": {},
        }
        stored_calls: list[dict] = []

        def fake_store(**kwargs):
            stored_calls.append(kwargs)
            return json.dumps({"stored": True})

        with patch.object(server, "_load_reflection_state", side_effect=lambda: state), patch.object(
            server, "_save_reflection_state", side_effect=lambda new_state: state.update(new_state)
        ), patch.object(
            server, "_ensure_investigation_exists", side_effect=lambda *args, **kwargs: None
        ), patch.object(
            server, "_process_reflection_item", side_effect=[summary, summary, summary]
        ), patch.object(
            server, "_reflection_llm_triage_metadata",
            return_value={"llm_triage": {"category": "unknown", "novelty": "unclear"}},
        ) as triage_mock, patch.object(
            server, "investigation_store", side_effect=fake_store
        ):
            server.reflection_loop_tick(
                max_items=3,
                max_lines_per_file=100,
                store_item_findings=True,
                enable_llm_triage=True,
                max_llm_items=2,
            )

        observed = [c for c in stored_calls if c["finding_type"] == "observed"]
        self.assertEqual(triage_mock.call_count, 2)
        self.assertEqual(sum("metadata" in call for call in observed), 2)

    def test_tick_off_keeps_store_payload_and_stats_unchanged(self):
        state = server._reflection_default_state()
        state["investigation_id"] = "test-inv"
        state["queue"] = [{"kind": "process_log", "path": "/tmp/a.log"}]
        summary = {
            "status": "processed",
            "kind": "process_log",
            "path": "/tmp/a.log",
            "lines_scanned": 10,
            "bytes_scanned": 100,
            "sampling_mode": "full",
            "events": {},
            "tools": {},
            "errors": {"repeat-signature": 2},
            "warnings": {"warn-signature": 1},
        }
        stored_calls: list[dict] = []

        def fake_store(**kwargs):
            stored_calls.append(kwargs)
            return json.dumps({"stored": True})

        expected_stats = {
            "files_processed": 1,
            "lines_scanned": 10,
            "errors_seen": 2,
            "warnings_seen": 1,
            "bytes_scanned": 100,
            "error_signatures_suppressed": 0,
            "warning_signatures_suppressed": 0,
            "error_signature_observations": {"repeat-signature": 1},
            "warning_signature_observations": {"warn-signature": 1},
            "last_error_signatures": [{"signature": "repeat-signature", "count": 2}],
            "last_warning_signatures": [{"signature": "warn-signature", "count": 1}],
        }

        with patch.object(server, "_load_reflection_state", side_effect=lambda: state), patch.object(
            server, "_save_reflection_state", side_effect=lambda new_state: state.update(new_state)
        ), patch.object(
            server, "_ensure_investigation_exists", side_effect=lambda *args, **kwargs: None
        ), patch.object(
            server, "_process_reflection_item", return_value=summary
        ), patch.object(
            server, "_reflection_llm_triage_metadata"
        ) as triage_mock, patch.object(
            server, "investigation_store", side_effect=fake_store
        ):
            result = json.loads(
                server.reflection_loop_tick(
                    max_items=1,
                    max_lines_per_file=100,
                    store_item_findings=True,
                )
            )

        triage_mock.assert_not_called()
        observed = [c for c in stored_calls if c["finding_type"] == "observed"]
        self.assertEqual(len(observed), 1)
        self.assertNotIn("metadata", observed[0])
        self.assertEqual(
            observed[0],
            {
                "investigation_id": "test-inv",
                "finding_type": "observed",
                "text": (
                    "reflection_loop_tick processed process_log target=/tmp/a.log; "
                    "lines=10 bytes=100; sampling=full; top_events={}; top_tools={}; "
                    "errors=repeat-signature (2); warnings=warn-signature (1)."
                ),
                "source": "reflection_loop_tick",
                "confidence": "low",
                "tags": "self-reflection,loop-tick,artifact-mining,unreceipted-observed",
            },
        )
        self.assertEqual(result["stats"], expected_stats)


if __name__ == "__main__":
    unittest.main()

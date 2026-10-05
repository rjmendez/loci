"""The reflection loop stores what is news, counts what is not, and reports status in a bounded size.

2026-10-05: 716 of 776 findings in the loop's investigation were one sentence ("reflection_loop_tick processed
<kind> target=...; errors={}; warnings={}"), and they surfaced as top retrieval hits for unrelated queries.
reflection_loop_status returned 166 KB because it carried every error and warning line ever seen.
"""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import server  # noqa: E402


def _summary(kind, errors=None, warnings=None, lines=10, size=100):
    return {"status": "processed", "kind": kind, "path": f"/tmp/{kind}.log", "lines_scanned": lines,
            "bytes_scanned": size, "sampling_mode": "full", "events": {"e": 1}, "tools": {},
            "errors": errors or {}, "warnings": warnings or {}}


class _TickBase(unittest.TestCase):
    def tick(self, summaries, env=None, observe_limit=None):
        """Run one tick over `summaries`, one queue item each; return (stored findings, resulting state)."""
        state = server._reflection_default_state()
        state["investigation_id"] = "test-inv"
        state["queue"] = [{"kind": s["kind"], "path": s["path"]} for s in summaries]
        stored = []

        def fake_store(**kwargs):
            stored.append(kwargs)
            return json.dumps({"stored": True})
        patches = [
            patch.object(server, "_load_reflection_state", side_effect=lambda: state),
            patch.object(server, "_save_reflection_state", side_effect=lambda new: state.update(new)),
            patch.object(server, "_ensure_investigation_exists", side_effect=lambda *a, **k: None),
            patch.object(server, "_process_reflection_item", side_effect=list(summaries)),
            patch.object(server, "investigation_store", side_effect=fake_store),
            patch.dict(os.environ, env or {"LOCI_REFLECTION_STORE_LOW_SIGNAL": ""}),
        ]
        if observe_limit is not None:
            patches.append(patch.object(server, "REFLECTION_SIGNATURE_OBSERVE_LIMIT", observe_limit))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        server.reflection_loop_tick(max_items=len(summaries), max_lines_per_file=100, store_item_findings=True)
        return [c for c in stored if c["finding_type"] == "observed"], state


class TestOnlyNewsBecomesAFinding(_TickBase):
    KINDS = ("claude_code_event", "process_log", "copilot_log", "session_event")

    def test_an_item_with_no_errors_or_warnings_is_not_stored_whatever_its_kind(self):
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                observed, state = self.tick([_summary(kind)])
                self.assertEqual(observed, [])
                self.assertEqual(state["stats"]["low_signal"]["by_kind"], {kind: 1})

    def test_the_same_item_with_an_error_or_a_warning_is_stored(self):
        """Positive twin: only the absence of errors and warnings changed."""
        for kind in self.KINDS:
            for field in ("errors", "warnings"):
                with self.subTest(kind=kind, field=field):
                    observed, state = self.tick([_summary(kind, **{field: {"disk full": 3}})])
                    self.assertEqual(len(observed), 1)
                    self.assertIn("disk full", observed[0]["text"])
                    self.assertNotIn("low_signal", state["stats"])

    def test_an_item_whose_every_signature_is_already_known_is_counted_not_stored(self):
        # Limit 1: the first sighting is visible, the second (5 hits of the same line) is wholly suppressed.
        observed, state = self.tick([_summary("process_log", errors={"repeat": 5}),
                                     _summary("process_log", errors={"repeat": 5})], observe_limit=1)
        self.assertEqual(len(observed), 1)
        self.assertEqual(state["stats"]["error_signatures_suppressed"], 5)
        self.assertEqual(state["stats"]["low_signal"]["files"], 1)

    def test_the_skipped_items_totals_are_kept_as_counters_and_add_up_across_ticks(self):
        _, state = self.tick([_summary("claude_code_event", lines=7, size=70), _summary("process_log", lines=3, size=30)])
        self.assertEqual(state["stats"]["low_signal"],
                         {"files": 2, "lines": 10, "bytes": 100, "by_kind": {"claude_code_event": 1, "process_log": 1}})
        server._reflection_note_low_signal(state["stats"], [{"kind": "process_log", "lines_scanned": 5, "bytes_scanned": 50}])
        self.assertEqual(state["stats"]["low_signal"],
                         {"files": 3, "lines": 15, "bytes": 150, "by_kind": {"claude_code_event": 1, "process_log": 2}})

    def test_the_item_is_still_marked_processed_so_it_is_not_requeued(self):
        _, state = self.tick([_summary("claude_code_event")])
        self.assertEqual(len(state["processed"]), 1)

    def test_the_rollup_finding_is_off_by_default_and_on_for_each_truthy_value(self):
        for value in ("", "0", "no", "off", "false"):
            with self.subTest(value=value), patch.dict(os.environ, {"LOCI_REFLECTION_STORE_LOW_SIGNAL": value}):
                self.assertFalse(server._reflection_store_low_signal())
        for value in ("1", "true", "YES", "On"):
            with self.subTest(value=value), patch.dict(os.environ, {"LOCI_REFLECTION_STORE_LOW_SIGNAL": value}):
                self.assertTrue(server._reflection_store_low_signal())
        observed, _ = self.tick([_summary("claude_code_event"), _summary("process_log")],
                                env={"LOCI_REFLECTION_STORE_LOW_SIGNAL": "1"})
        self.assertEqual(len(observed), 1)
        self.assertIn("batched low-signal files count=2", observed[0]["text"])


class TestStatusFitsInAContextWindow(unittest.TestCase):
    def _state(self, n):
        state = server._reflection_default_state()
        state["investigation_id"] = "test-inv"
        state["stats"]["error_signature_observations"] = {f"error line number {i} " + "x" * 120: 1 for i in range(n)}
        state["stats"]["warning_signature_observations"] = {f"warning line number {i} " + "y" * 120: 2 for i in range(n)}
        state["stats"]["last_error_signatures"] = [{"signature": "disk full", "count": 9}]
        state["stats"]["files_processed"] = 1033
        return state

    def _status(self, state, **kw):
        with patch.object(server, "_load_reflection_state", return_value=state):
            return server.reflection_loop_status(**kw)

    def test_the_default_status_carries_counts_not_the_observation_maps(self):
        out = self._status(self._state(3000))
        data = json.loads(out)
        self.assertLess(len(out), 6000)
        self.assertEqual(data["stats"]["error_signature_observation_count"], 3000)
        self.assertEqual(data["stats"]["warning_signature_observation_count"], 3000)
        self.assertNotIn("error_signature_observations", data["stats"])
        self.assertNotIn("warning_signature_observations", data["stats"])
        self.assertEqual(data["stats"]["files_processed"], 1033)
        self.assertEqual(data["stats"]["last_error_signatures"], [{"signature": "disk full", "count": 9}])

    def test_verbose_returns_the_maps_whole(self):
        """Positive twin: same state, same tool, the maps are there when asked for."""
        data = json.loads(self._status(self._state(3000), verbose=True))
        self.assertEqual(len(data["stats"]["error_signature_observations"]), 3000)
        self.assertEqual(len(data["stats"]["warning_signature_observations"]), 3000)
        self.assertNotIn("error_signature_observation_count", data["stats"])

    def test_clip_bounds_lists_dicts_and_strings_and_says_what_it_dropped(self):
        self.assertEqual(server._reflection_clip(list(range(20)), limit=3), [0, 1, 2, "... 17 more"])
        self.assertEqual(server._reflection_clip({str(i): i for i in range(5)}, limit=2), {"0": 0, "1": 1, "...": "3 more"})
        self.assertEqual(server._reflection_clip("a" * 500, text=10), "a" * 10)
        self.assertEqual(server._reflection_clip([1, 2], limit=12), [1, 2])

    def test_an_empty_loop_still_reports(self):
        data = json.loads(self._status(server._reflection_default_state()))
        self.assertEqual((data["queue_size"], data["processed_count"]), (0, 0))


class TestTickResultFitsInAContextWindow(unittest.TestCase):
    """2026-10-05: a tick returned 171 KB (158 KB of it the observation maps) and overflowed the tool-result limit."""

    N = 3000

    def _tick(self, **kw):
        state = server._reflection_default_state()
        state["investigation_id"] = "test-inv"
        state["stats"]["error_signature_observations"] = {f"error line {i} " + "x" * 120: 1 for i in range(self.N)}
        state["stats"]["warning_signature_observations"] = {f"warning line {i} " + "y" * 120: 2 for i in range(self.N)}
        state["queue"] = [{"kind": "process_log", "path": "/tmp/process_log.log"}]
        patches = [
            patch.object(server, "_load_reflection_state", side_effect=lambda: state),
            patch.object(server, "_save_reflection_state", side_effect=lambda new: state.update(new)),
            patch.object(server, "_ensure_investigation_exists", side_effect=lambda *a, **k: None),
            patch.object(server, "_process_reflection_item", side_effect=[_summary("process_log", errors={"disk full": 3})]),
            patch.object(server, "investigation_store", side_effect=lambda **k: json.dumps({"stored": True})),
            patch.object(server, "REFLECTION_SIGNATURE_OBSERVE_LIMIT", 10 ** 6),
            patch.object(server, "_prune_signature_observations", side_effect=lambda m: m),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        out = server.reflection_loop_tick(max_items=1, max_lines_per_file=100, **kw)
        return out, state

    def test_the_default_tick_carries_counts_not_the_observation_maps(self):
        out, _ = self._tick()
        data = json.loads(out)
        self.assertLess(len(out), 8000)
        self.assertEqual(data["stats"]["error_signature_observation_count"], self.N + 1)
        self.assertEqual(data["stats"]["warning_signature_observation_count"], self.N)
        self.assertNotIn("error_signature_observations", data["stats"])
        self.assertNotIn("warning_signature_observations", data["stats"])
        self.assertEqual(data["stats"]["errors_seen"], 3)
        self.assertEqual(data["stats"]["last_error_signatures"], [{"signature": "disk full", "count": 3}])
        self.assertEqual((data["processed_items"], data["remaining_queue"]), (1, 0))

    def test_verbose_returns_the_maps_whole(self):
        """Positive twin: same tick, the maps are there when asked for."""
        data = json.loads(self._tick(verbose=True)[0])
        self.assertEqual(len(data["stats"]["error_signature_observations"]), self.N + 1)
        self.assertEqual(len(data["stats"]["warning_signature_observations"]), self.N)
        self.assertNotIn("error_signature_observation_count", data["stats"])

    def test_the_saved_state_keeps_the_maps_whatever_the_view(self):
        _, state = self._tick()
        self.assertEqual(len(state["stats"]["error_signature_observations"]), self.N + 1)
        self.assertEqual(len(state["stats"]["warning_signature_observations"]), self.N)


if __name__ == "__main__":
    unittest.main()

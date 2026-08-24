"""cheap_batch — resolving workflow tasks off the Claude tiers.

The premise, measured on wf_6e13c6c2-8e0: agent *returns* were 3.3% of subagent
token spend; the other 96.7% was input the agents read. So offloading is about
answering a task once, cheaply, and injecting it — not about which model an
agent() runs on, which is not selectable.
"""
import importlib.util
import pathlib
import unittest
from unittest import mock

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("cheap_batch", _SCRIPTS / "cheap_batch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load()

TASKS = [{"key": "a", "prompt": "p1"}, {"key": "b", "prompt": "p2"}]


def _ok(text, model="local-model"):
    return {"text": text, "ok": True, "model": model}


def _fail(why=""):
    return {"text": "", "ok": False, "why": why}


class TestResolve(unittest.TestCase):
    def test_the_local_tier_answers_and_is_credited(self):
        out = cb.resolve(TASKS, local_fn=lambda p, f: [_ok("A"), _ok("B")])
        self.assertEqual(out["a"]["text"], "A")
        self.assertEqual(out["a"]["tier"], "local")
        self.assertTrue(out["a"]["ok"])

    def test_remote_is_not_consulted_unless_asked(self):
        called = []
        cb.resolve(TASKS, local_fn=lambda p, f: [_fail(), _fail()],
                   remote_fn=lambda p, f: called.append(p) or [_ok("X"), _ok("Y")])
        self.assertEqual(called, [], "leaving the box must be an explicit choice")

    def test_remote_picks_up_only_what_local_could_not_serve(self):
        seen = {}

        def remote(prompts, fmt):
            seen["prompts"] = list(prompts)
            return [_ok("from-remote", model="or-model")]

        out = cb.resolve(TASKS, remote=True,
                         local_fn=lambda p, f: [_ok("A"), _fail("429")],
                         remote_fn=remote)
        self.assertEqual(seen["prompts"], ["p2"], "p1 was already served locally")
        self.assertEqual(out["a"]["tier"], "local")
        self.assertEqual(out["b"]["tier"], "remote")
        self.assertEqual(out["b"]["text"], "from-remote")

    def test_an_unserved_task_is_returned_not_dropped(self):
        # loci-native degrades a factless task to a mechanical agent — it can only
        # do that if it can see the gap.
        out = cb.resolve(TASKS, remote=True,
                         local_fn=lambda p, f: [_fail("no tier"), _fail("no tier")],
                         remote_fn=lambda p, f: [_fail("429"), _fail("429")])
        self.assertEqual(set(out), {"a", "b"})
        self.assertFalse(out["a"]["ok"])
        self.assertIsNone(out["a"]["tier"])
        self.assertTrue(out["a"]["why"], "an unserved task must say why")

    def test_no_tasks_costs_nothing(self):
        called = []
        self.assertEqual(cb.resolve([], local_fn=lambda p, f: called.append(1)), {})
        self.assertEqual(called, [])


class TestTaskParsing(unittest.TestCase):
    def test_a_list_of_key_prompt_objects(self):
        with mock.patch("builtins.open", mock.mock_open(read_data='[{"key":"k","prompt":"p"}]')):
            self.assertEqual(cb._load_tasks("f.json"), [{"key": "k", "prompt": "p"}])

    def test_a_plain_mapping_is_accepted(self):
        with mock.patch("builtins.open", mock.mock_open(read_data='{"k": "p"}')):
            self.assertEqual(cb._load_tasks("f.json"), [{"key": "k", "prompt": "p"}])

    def test_id_and_focus_are_accepted_as_aliases(self):
        with mock.patch("builtins.open", mock.mock_open(read_data='[{"id":"k","focus":"p"}]')):
            self.assertEqual(cb._load_tasks("f.json"), [{"key": "k", "prompt": "p"}])

    def test_entries_missing_a_key_or_prompt_are_skipped(self):
        data = '[{"key":"k","prompt":"p"},{"key":"","prompt":"p"},{"key":"z","prompt":"  "}]'
        with mock.patch("builtins.open", mock.mock_open(read_data=data)):
            self.assertEqual(cb._load_tasks("f.json"), [{"key": "k", "prompt": "p"}])


class TestCli(unittest.TestCase):
    def test_unreadable_tasks_exit_two_rather_than_traceback(self):
        with mock.patch("builtins.open", side_effect=OSError("nope")):
            self.assertEqual(cb.main(["missing.json"]), 2)

    def test_an_empty_task_list_emits_an_empty_object(self):
        with mock.patch("builtins.open", mock.mock_open(read_data="[]")):
            self.assertEqual(cb.main(["f.json"]), 0)


if __name__ == "__main__":
    unittest.main()

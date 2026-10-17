"""The PreToolUse gate that keeps a workflow from running on one model tier.

verdict() is pure so the rule is testable without the harness, which is the
point of its shape. main() is covered too, because the interesting bug was
there: it read only tool_input["script"] and returned 0 for a scriptPath, so a
workflow invoked the way the Workflow tool recommends for iteration escaped the
rule entirely. A uniformly-Opus workflow shipped that way before this was fixed.
"""
from __future__ import annotations

import importlib.util
import io
import json
import pathlib
from unittest import mock

import pytest

HOOK = pathlib.Path(__file__).resolve().parents[1] / "workflow_balanced_models.py"


def _load():
    spec = importlib.util.spec_from_file_location("_wbm_uut", HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BALANCED = """
const a = await agent('x', {model: 'haiku', effort: 'low'})
const b = await agent('y', {model: 'opus', effort: 'xhigh'})
"""
UNIFORM = """
const a = await agent('x', {model: 'opus', effort: 'high'})
const b = await agent('y', {model: 'opus', effort: 'high'})
"""
NO_MODEL = "const a = await agent('x', {effort: 'high'})"
NO_EFFORT = """
const a = await agent('x', {model: 'haiku'})
const b = await agent('y', {model: 'opus'})
"""


def test_two_tiers_and_an_effort_pass():
    assert _load().verdict(BALANCED)[0] is True


def test_one_tier_everywhere_fails():
    ok, reason = _load().verdict(UNIFORM)
    assert ok is False and "opus" in reason


def test_no_model_key_at_all_fails():
    ok, reason = _load().verdict(NO_MODEL)
    assert ok is False and "model" in reason


def test_models_without_an_effort_fails():
    assert _load().verdict(NO_EFFORT)[0] is False


def test_an_empty_script_passes():
    assert _load().verdict("")[0] is True


def _run(mod, payload):
    out = io.StringIO()
    with mock.patch.object(mod.sys, "stdin", io.StringIO(json.dumps(payload))), \
            mock.patch.object(mod.sys, "stdout", out):
        rc = mod.main()
    return rc, out.getvalue()


def test_an_inline_unbalanced_script_is_denied():
    mod = _load()
    rc, out = _run(mod, {"tool_name": "Workflow", "tool_input": {"script": UNIFORM}})
    assert rc == 0
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_a_scriptpath_is_read_rather_than_waved_through(tmp_path):
    """The Workflow tool recommends scriptPath for iteration, so a gate blind to
    it is a gate that does not apply to the common case."""
    p = tmp_path / "wf.js"
    p.write_text(UNIFORM)
    mod = _load()
    rc, out = _run(mod, {"tool_name": "Workflow", "tool_input": {"scriptPath": str(p)}})
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_a_balanced_scriptpath_is_allowed(tmp_path):
    p = tmp_path / "wf.js"
    p.write_text(BALANCED)
    mod = _load()
    assert _run(mod, {"tool_name": "Workflow", "tool_input": {"scriptPath": str(p)}}) == (0, "")


@pytest.mark.parametrize("tool_input", [
    {"name": "some-saved-workflow"},
    {"scriptPath": "/nonexistent/never.js"},
    {},
])
def test_what_cannot_be_read_fails_open(tool_input):
    """A gate that blocked work it could not inspect would make the tool unusable
    the first time it saw an input shape nobody anticipated."""
    mod = _load()
    assert _run(mod, {"tool_name": "Workflow", "tool_input": tool_input}) == (0, "")


def test_other_tools_are_untouched():
    mod = _load()
    assert _run(mod, {"tool_name": "Bash", "tool_input": {"command": "ls"}}) == (0, "")


def test_unreadable_stdin_fails_open():
    mod = _load()
    out = io.StringIO()
    with mock.patch.object(mod.sys, "stdin", io.StringIO("not json")), \
            mock.patch.object(mod.sys, "stdout", out):
        assert mod.main() == 0
    assert out.getvalue() == ""

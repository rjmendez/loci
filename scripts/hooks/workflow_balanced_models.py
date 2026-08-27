#!/usr/bin/env python3
"""PreToolUse gate: a workflow script must spread work across models and efforts.

Operator standing rule: every workflow, always, uses balanced models and efforts. Memory cannot
enforce that -- it is guidance to the assistant, which forgets. This runs in the harness, so it
holds whether or not the model remembers.

Passes when the script sets `model:` on agent() calls with at least TWO DISTINCT values and sets
`effort:` at least once. Uniform model+effort across a fan-out wastes the cheap tiers on trivial
stages and the expensive ones on stages that do not need them.

Fails OPEN by design: anything it cannot inspect (a saved workflow by `name`, a `scriptPath` on
disk, malformed stdin) is allowed through. A gate that blocks work it cannot read would make the
Workflow tool unusable the first time it saw an input shape nobody anticipated.
"""
import json
import re
import sys

MIN_DISTINCT_MODELS = 2

# model: 'opus' | model: "sonnet" | model: l.model  -- only literals are counted as distinct
_MODEL_LIT = re.compile(r"""\bmodel\s*:\s*['"]([A-Za-z0-9._\-\[\]]+)['"]""")
_MODEL_ANY = re.compile(r"\bmodel\s*:")
_EFFORT_ANY = re.compile(r"\beffort\s*:")


def verdict(script):
    """-> (ok, reason). Pure, so the rule is testable without the harness."""
    if not script:
        return True, "nothing to inspect"

    literals = set(_MODEL_LIT.findall(script))
    has_model_key = bool(_MODEL_ANY.search(script))
    has_effort = bool(_EFFORT_ANY.search(script))

    problems = []
    if len(literals) < MIN_DISTINCT_MODELS:
        if not has_model_key:
            problems.append("no `model:` on any agent() call -- every agent inherits one tier")
        else:
            # model: is present but indirect (model: l.model) or all one value. A table of
            # per-lens models is the normal shape, so say what is missing rather than guess.
            seen = ", ".join(sorted(literals)) if literals else "none as literals"
            problems.append(
                "fewer than %d distinct model literals (found: %s). If models come from a lens "
                "table, put the literals in that table so the spread is visible here."
                % (MIN_DISTINCT_MODELS, seen))
    if not has_effort:
        problems.append("no `effort:` anywhere -- set it per stage (low for mechanical, "
                        "xhigh for the hardest verify/judge)")

    if problems:
        return False, ("This workflow is not balanced across models/efforts:\n  - "
                       + "\n  - ".join(problems))
    return True, "ok"


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0                                   # unreadable stdin: fail open

    if payload.get("tool_name") != "Workflow":
        return 0

    tool_input = payload.get("tool_input") or {}
    script = tool_input.get("script")
    if not script:
        # scriptPath IS readable, and the Workflow tool recommends it for
        # iteration — so the rule was unenforceable through its most common
        # invocation path. A saved workflow by `name` still fails open.
        path = tool_input.get("scriptPath")
        if path:
            try:
                script = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                return 0
    if not script:
        return 0                                   # name only: nothing to read, fail open

    ok, reason = verdict(script)
    if ok:
        return 0

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""PreToolUse gate: a Workflow script must use balanced models and efforts.

Passes when the script sets `model:` with at least TWO DISTINCT literal values and sets
`effort:` at least once. Enforced in the harness because memory/guidance is forgotten.

Fails OPEN: anything it cannot inspect (saved workflow by `name`, unreadable scriptPath,
malformed stdin) is allowed through.
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
    """-> (ok, reason). Pure, testable without the harness."""
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
            # model: present but indirect (model: l.model) or a single value.
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
        # scriptPath is readable and the common invocation path; `name` alone fails open.
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

"""Copilot CLI → memcheck bridge.

Translates the Copilot CLI hook payload (camelCase keys, ``toolResult`` for
``postToolUse``) into the Claude Code format the memcheck daemon expects, then
delegates to :func:`hook_client.run`.

The daemon dispatches on ``hook_event_name``:
  - ``PostToolUse`` → code-hallucination check (PostToolUse path)
  - ``PreToolUse``  → action-verdict audit (PreToolUse path)

Wire this as a Copilot hook in ``~/.copilot/hooks/hooks.json``:

.. code-block:: json

    {
      "hooks": {
        "preToolUse": [
          {
            "type": "command",
            "command": "PYTHONPATH=/path/to/loci-mcp /path/to/.venv/bin/python -m memcheck.copilot_adapter"
          }
        ],
        "postToolUse": [
          {
            "type": "command",
            "command": "PYTHONPATH=/path/to/loci-mcp /path/to/.venv/bin/python -m memcheck.copilot_adapter"
          }
        ]
      }
    }

Like ``hook_client``, this module never writes to stdout and always exits 0 so
it cannot break a Copilot CLI session.

Copilot preToolUse payload shape::

    {"sessionId": "...", "cwd": "...", "toolName": "bash", "toolArgs": {...}}

Copilot postToolUse payload shape::

    {"sessionId": "...", "cwd": "...", "toolName": "bash", "toolArgs": {...},
     "toolResult": {"resultType": "success", "error": "", "output": "..."}}
"""

from __future__ import annotations

import json
import sys

from .hook_client import run


def translate(payload: dict) -> dict:
    """Normalise a Copilot hook payload to the Claude Code Pre/PostToolUse shape.

    Detection: ``toolResult`` (or ``tool_result``) present → PostToolUse,
    otherwise PreToolUse.
    """
    has_result = "toolResult" in payload or "tool_result" in payload
    event = "PostToolUse" if has_result else "PreToolUse"

    tool_name = str(
        payload.get("toolName") or payload.get("tool_name") or "unknown"
    )
    tool_input = payload.get("toolArgs") or payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {"value": str(tool_input)}

    translated: dict = {
        "hook_event_name": event,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "session_id": str(
            payload.get("sessionId") or payload.get("session_id") or ""
        ),
        "cwd": str(payload.get("cwd") or ""),
    }

    if has_result:
        tr = payload.get("toolResult") or payload.get("tool_result") or {}
        translated["tool_response"] = {
            "content": str(tr.get("output") or tr.get("content") or ""),
            "error": str(tr.get("error") or ""),
        }

    return translated


def main() -> int:
    """Entry point: read Copilot payload from stdin, translate, forward to daemon."""
    try:
        raw = sys.stdin.buffer.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
    except Exception:  # noqa: BLE001 — never raise from a hook
        return 0

    translated = translate(payload)
    return run(json.dumps(translated, separators=(",", ":")).encode("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())

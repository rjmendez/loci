"""Hermes pre_tool_call guard (defense in depth, layer 2 of 3).

Blocks a tool call with exit code 2 when:
  - the tool input is not a JSON object (malformed call), or
  - any string in it names a path that does not exist on this Windows host (/mnt/..., /Users/..., /home/...,
    /Volumes/...). These are the guessed Linux/macOS paths seen in the Hermes logs.

Fails closed in config (fail_closed: true): if this script crashes, the call is blocked, not allowed.
Every block is appended to hooks/guard_tool.log so refusals are auditable.

Wire (from hermes shell_hooks): stdin JSON {hook_event_name, tool_name, tool_input, ...}.
"""
import json
import os
import sys
import time

FORBIDDEN_PREFIXES = ("/mnt/", "/users/", "/home/", "/volumes/")
# Linux-only commands that do not exist on this Windows host (seen in Hermes logs: ps aux, systemctl).
FORBIDDEN_COMMANDS = ("ps aux", "systemctl", "apt-get", "apt install", "/bin/", "/usr/bin/", "sudo ")
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guard_tool.log")


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _log(reason: str, tool: str) -> None:
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": round(time.time(), 3), "tool": tool, "reason": reason}) + "\n")
    except OSError:
        pass  # the block still happens; only the audit line is lost


def main() -> int:
    payload = json.load(sys.stdin)
    tool = str(payload.get("tool_name") or "")
    args = payload.get("tool_input")
    if not isinstance(args, dict):
        _log("malformed_tool_input", tool)
        print("guard: tool input is not a JSON object; fix the call and retry", file=sys.stderr)
        return 2
    for text in _strings(args):
        low = text.lower()
        for cmd in FORBIDDEN_COMMANDS:
            if cmd in low:
                _log(f"linux_only_command:{cmd.strip()}", tool)
                print(f"guard: {cmd.strip()!r} is not available on this Windows host; use a PowerShell or Windows equivalent",
                      file=sys.stderr)
                return 2
        for prefix in FORBIDDEN_PREFIXES:
            if prefix in low:
                _log(f"non_host_path:{prefix}", tool)
                print(f"guard: path {prefix!r} does not exist on this Windows host; use C:\\, D:\\ or F:\\ paths",
                      file=sys.stderr)
                return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # fail closed: an unparseable or crashing guard blocks the call
        print(f"guard: internal error, blocking the call: {exc!r}", file=sys.stderr)
        sys.exit(2)

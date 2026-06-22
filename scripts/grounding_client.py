#!/usr/bin/env python3
"""
Thin client for grounding_daemon.py.

Connects to the UNIX socket, sends stdin as payload, prints response.
Falls back to running the hook script directly if the daemon is down.

Used as the hook command in config.yaml:
  command: python3 /path/to/grounding_client.py
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys

SOCK_PATH  = os.environ.get("GROUNDING_SOCK", "/tmp/hermes-grounding-{}.sock".format(os.environ.get("HERMES_AGENT_ID", "hermes")))
HOOK_PATH  = os.path.join(os.path.dirname(__file__), "hooks", "pre_llm_grounding.py")
TIMEOUT    = 4.5   # Must be under hook timeout (5s)
CONNECT_TO = 0.5   # Max wait for socket connect


def _via_daemon(payload: bytes) -> str | None:
    """Try the daemon socket. Returns response string or None on failure."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(CONNECT_TO)
    try:
        s.connect(SOCK_PATH)
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        s.close()
        return None
    s.settimeout(TIMEOUT)
    try:
        s.sendall(payload)
        s.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = s.recv(8192)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")
    except Exception:
        return None
    finally:
        s.close()


def _via_subprocess(payload: bytes) -> str:
    """Fallback: spawn the hook script directly."""
    py = sys.executable
    r = subprocess.run(
        [py, HOOK_PATH],
        input=payload,
        capture_output=True,
        timeout=TIMEOUT,
    )
    return r.stdout.decode("utf-8", errors="replace")


def main():
    payload = sys.stdin.buffer.read()
    if not payload.strip():
        sys.exit(0)

    result = _via_daemon(payload)
    if result is None:
        # Daemon not running — fall back silently
        result = _via_subprocess(payload)

    if result:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()

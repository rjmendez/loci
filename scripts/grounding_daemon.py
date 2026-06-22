#!/usr/bin/env python3
"""
Grounding hook daemon — eliminates per-turn Python startup cost.

Runs as a persistent UNIX socket server. Hermes fires the hook by
calling the thin client (grounding_client.py) which connects, sends
the JSON payload, and returns stdout. Total round-trip: ~50ms vs ~140ms
for a fresh subprocess spawn.

Protocol:
  client -> server: raw JSON payload bytes, then EOF (SHUT_WR)
  server -> client: hook stdout bytes (JSON or empty), then close

Socket: /tmp/hermes-grounding-{HERMES_AGENT_ID}.sock
PID:    /tmp/hermes-grounding-{HERMES_AGENT_ID}.pid

Managed by systemd user unit: hermes-grounding-{HERMES_AGENT_ID}.service
Run: systemctl --user start hermes-grounding-{HERMES_AGENT_ID}
"""
from __future__ import annotations

import importlib.util
import io
import contextlib
import os
import signal
import socket
import sys
import time
import logging
from pathlib import Path

# Socket/pid names are agent-scoped; set HERMES_AGENT_ID in your environment.
_agent_id  = os.environ.get("HERMES_AGENT_ID", "hermes")
SOCK_PATH  = os.environ.get("GROUNDING_SOCK", f"/tmp/hermes-grounding-{_agent_id}.sock")
PID_PATH   = os.environ.get("GROUNDING_PID",  f"/tmp/hermes-grounding-{_agent_id}.pid")
HOOK_PATH  = os.environ.get(
    "GROUNDING_HOOK",
    str(Path(__file__).parent / "hooks" / "pre_llm_grounding.py")
)

# Log dir: $HERMES_HOME/.hermes/<profile>/logs or ~/.hermes/logs as fallback
_hermes_home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
_profile     = os.environ.get("HERMES_PROFILE", "")
_log_dir     = (
    os.path.join(_hermes_home, "profiles", _profile, "logs")
    if _profile
    else os.path.join(_hermes_home, "logs")
)
LOG_PATH = os.environ.get("GROUNDING_LOG", os.path.join(_log_dir, "grounding_daemon.log"))

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("grounding_daemon")


def _load_hook():
    """Import the hook module in-process (avoids re-importing every request)."""
    spec = importlib.util.spec_from_file_location("pre_llm_grounding", HOOK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _handle(conn: socket.socket, mod) -> None:
    """Read payload from conn, run hook, write result back."""
    data = b""
    try:
        while True:
            chunk = conn.recv(8192)
            if not chunk:
                break
            data += chunk
    except Exception as e:
        log.warning("recv error: %s", e)
        conn.close()
        return

    buf = io.StringIO()
    old_stdin = sys.stdin
    sys.stdin = io.TextIOWrapper(io.BytesIO(data))

    try:
        with contextlib.redirect_stdout(buf):
            try:
                mod.main()
            except SystemExit:
                pass
    except Exception as e:
        log.error("hook error: %s", e, exc_info=True)
    finally:
        sys.stdin = old_stdin

    result = buf.getvalue()
    try:
        conn.sendall(result.encode("utf-8"))
    except Exception as e:
        log.warning("send error: %s", e)
    finally:
        conn.close()


def _cleanup(sock_path, pid_path):
    for p in (sock_path, pid_path):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


def main():
    # Ensure log dir
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    # Write PID
    Path(PID_PATH).write_text(str(os.getpid()))

    # Cleanup socket on exit
    def _sigterm(signum, frame):
        log.info("SIGTERM — shutting down")
        _cleanup(SOCK_PATH, PID_PATH)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    # Remove stale socket
    try:
        os.unlink(SOCK_PATH)
    except FileNotFoundError:
        pass

    # Load hook module once
    log.info("loading hook from %s", HOOK_PATH)
    try:
        mod = _load_hook()
        log.info("hook loaded OK")
    except Exception as e:
        log.error("failed to load hook: %s", e)
        sys.exit(1)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(SOCK_PATH)
    srv.listen(16)
    os.chmod(SOCK_PATH, 0o600)

    log.info("listening on %s  pid=%d", SOCK_PATH, os.getpid())
    print(f"grounding_daemon ready  sock={SOCK_PATH}  pid={os.getpid()}", flush=True)

    while True:
        try:
            conn, _ = srv.accept()
            _handle(conn, mod)
        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error("accept error: %s", e)
            time.sleep(0.1)

    _cleanup(SOCK_PATH, PID_PATH)


if __name__ == "__main__":
    main()

"""memcheck hook client — the ultra-thin PreToolUse hook entrypoint.

This replaces ``python -m memcheck.cli check-action`` as the wired hook. It
imports ONLY the standard library (``socket``, ``sys``, ``os``, ``subprocess``)
— never ``memcheck`` heavy modules and never ``qdrant_client`` — so its startup
is a few milliseconds, not the ~0.5–0.9s the cold CLI path costs. All the real
work (warm qdrant client + engine) lives in the long-running daemon
(:mod:`memcheck.daemon`); this client just ferries the PreToolUse payload to it
over a Unix socket.

Contract (identical to the CLI hook, audit-only):
  * Read all of stdin (the PreToolUse JSON), send it to the daemon, half-close,
    optionally read+discard the reply, close.
  * ALWAYS ``sys.exit(0)`` with EMPTY stdout — on success OR any exception
    (connection refused, timeout, anything). The hook can neither block a tool
    nor auto-approve one.
  * Self-healing: if the daemon is down (connection refused / no socket), best-
    effort fire-and-forget spawn it detached, then still exit 0 immediately. The
    current call is skipped; the daemon will be warm for the next one.

Importing this module must not connect to anything — only ``main()`` /
``run()`` do I/O, so it stays ``python -m memcheck.hook_client`` friendly.
"""

import os
import socket
import sys

# Keep stdlib-only and lazy: ``subprocess`` is imported inside the spawn path so
# the common (daemon-up) case never pays for it.

DEFAULT_SOCKET = "~/.hermes/memcheck.sock"
# Total budget for connect + send + (optional) read. The hook is on the hot path
# of every tool call, so this is deliberately tight; on timeout we just exit 0.
_TIMEOUT_S = 0.7


def _socket_path():
    raw = os.environ.get("MEMCHECK_SOCKET") or DEFAULT_SOCKET
    return os.path.expanduser(raw)


def _spawn_daemon():
    """Best-effort detached daemon launch. Never raises, never waits.

    Fire-and-forget: we do NOT wait for readiness — this call is sacrificed so
    the daemon is warm for the next hook invocation. v1 has no spawn-storm guard;
    a future hardening could touch a rate-limit file (e.g.
    ``~/.hermes/memcheck.spawn``) and skip if spawned within the last few seconds.
    """
    try:
        import subprocess

        subprocess.Popen(
            [sys.executable, "-m", "memcheck.cli", "daemon"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        # Spawning is opportunistic; failure just means the next call retries.
        pass


def run(stdin_bytes):
    """Ferry ``stdin_bytes`` to the daemon. Always returns 0; never raises.

    Split out from :func:`main` so tests can assert the exit code and the
    (empty) stdout without the process actually exiting.
    """
    path = _socket_path()
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_TIMEOUT_S)
        try:
            sock.connect(path)
        except (FileNotFoundError, ConnectionRefusedError):
            # Daemon down — best-effort start it for next time, then bail (0).
            _spawn_daemon()
            sock.close()
            return 0
        try:
            sock.sendall(stdin_bytes)
            sock.shutdown(socket.SHUT_WR)
            # Read+discard the response so the daemon's send doesn't block; we
            # don't parse it (the hook is audit-only — the body is debug-only).
            try:
                while sock.recv(65536):
                    pass
            except (OSError, socket.timeout):
                pass
        finally:
            sock.close()
    except Exception:
        # Absolute fail-open boundary: timeout, reset, anything — exit 0.
        pass
    return 0


def main():
    """Hook entrypoint: read stdin, ferry to daemon, exit 0 with empty stdout."""
    try:
        stdin_bytes = sys.stdin.buffer.read()
    except Exception:
        stdin_bytes = b""
    return run(stdin_bytes)


if __name__ == "__main__":
    sys.exit(main())

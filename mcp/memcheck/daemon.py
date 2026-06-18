"""memcheck warm daemon — the long-running half of the PreToolUse hook.

The ``check-action`` CLI path is correct but slow: every tool call spawns a
fresh Python, imports ``qdrant_client``, opens a connection, and ensures the
collection — ~0.5–0.9s of cold start on the hot path. This daemon pays that
cost ONCE at startup and then holds a warm :class:`QdrantBackend` (wrapped in a
:class:`VerdictEngine`) for the life of the process. The hook becomes a tiny
stdlib-only client (:mod:`memcheck.hook_client`) that ferries the PreToolUse
payload over a Unix socket and exits 0.

Design rules (inherited from the CLI):

* **Audit-only** — the daemon never blocks and never auto-approves. It runs the
  SAME :func:`memcheck.cli.process_action` the in-process path uses, so the
  audit-log line and ``would_flag`` accounting are byte-for-byte identical.
* **Fail-open / resilient** — qdrant being down at startup does NOT crash the
  daemon. It builds the backend lazily and, if qdrant is unreachable on a given
  request, that request records ``qdrant: "unavailable"`` (exactly like the CLI)
  and the daemon keeps serving. The whole per-connection handler is wrapped so a
  single malformed request can never kill a worker or the daemon.

Launch: ``python -m memcheck.cli daemon [--socket PATH]``.
Socket: ``MEMCHECK_SOCKET`` env, default ``~/.hermes/memcheck.sock`` (perms 0600).
"""

from __future__ import annotations

import json
import os
import signal
import socketserver
import stat
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from . import cli

__all__ = [
    "DEFAULT_SOCKET",
    "socket_path",
    "build_engine",
    "MemcheckDaemon",
    "serve",
]

DEFAULT_SOCKET = "~/.hermes/memcheck.sock"

# A request is at most a single PreToolUse payload; bound how much we will read
# so a runaway/hostile client cannot exhaust memory.
_MAX_REQUEST_BYTES = 4 * 1024 * 1024


def _resolve(override: Optional[str]) -> Path:
    """Resolve the daemon socket path: arg > ``MEMCHECK_SOCKET`` > default."""
    raw = override or os.environ.get("MEMCHECK_SOCKET") or DEFAULT_SOCKET
    return Path(raw).expanduser()


def socket_path(override: Optional[str] = None) -> Path:
    """Public alias of :func:`_resolve` — the configured daemon socket path."""
    return _resolve(override)


# --------------------------------------------------------------------------- #
# Warm engine construction (qdrant-down tolerant; lazily (re)ensures collection)
# --------------------------------------------------------------------------- #
def build_engine() -> Optional[object]:
    """Build ONE warm ``VerdictEngine`` over a connected ``QdrantBackend``.

    Returns the engine, or ``None`` if qdrant cannot be reached / the collection
    cannot be ensured right now. ``None`` is not fatal: the daemon retries on the
    next request, and meanwhile requests degrade to ``qdrant: "unavailable"``.
    """
    from .engine import EmlConfig, VerdictEngine

    try:
        backend = cli._build_qdrant_backend()
    except Exception:  # noqa: BLE001 — qdrant down / import error: stay alive
        return None
    if backend is None:
        return None
    return VerdictEngine(backend, EmlConfig(promote_after=cli.PROMOTE_AFTER))


class _EngineHolder:
    """Holds the warm engine and lazily (re)builds it when qdrant comes back.

    Thread-safe: the build is guarded so concurrent worker threads do not race
    to construct duplicate clients. Once an engine exists it is shared read-
    mostly (the qdrant client is thread-safe for our retrieve+upsert usage), so
    the hot path takes no lock.
    """

    def __init__(self, factory: Callable[[], Optional[object]]) -> None:
        self._factory = factory
        self._engine: Optional[object] = None
        self._lock = threading.Lock()

    def get(self) -> Optional[object]:
        engine = self._engine
        if engine is not None:
            return engine
        with self._lock:
            if self._engine is None:
                try:
                    self._engine = self._factory()
                except Exception:  # noqa: BLE001 — never let a build kill a request
                    self._engine = None
            return self._engine


# --------------------------------------------------------------------------- #
# Server + per-connection handler
# --------------------------------------------------------------------------- #
class _Handler(socketserver.BaseRequestHandler):
    """One hook request per connection (PreToolUse action OR PostToolUse code).

    Protocol: the client writes the whole JSON payload then half-closes its
    write side (``shutdown(SHUT_WR)``); we read to EOF and dispatch on the
    payload's ``hook_event_name``:

    * ``PostToolUse`` → :func:`memcheck.cli.process_code` (the new code path).
    * otherwise (``PreToolUse`` / absent / default) →
      :func:`memcheck.cli.process_action` (the existing action path, unchanged).

    We write back one compact JSON line ``{"would_flag", "occurrences",
    "qdrant"}`` (for debugging/``stats`` — the hook client ignores the body).
    The ENTIRE body is wrapped fail-open so one bad request can never take down
    a worker thread or the daemon.
    """

    def handle(self) -> None:  # noqa: C901 — single try/except wrapper is intentional
        try:
            raw = self._read_all()
            try:
                payload = json.loads(raw) if raw.strip() else {}
            except Exception:  # noqa: BLE001 — malformed request: degrade, stay up
                payload = {}
            if not isinstance(payload, dict):
                payload = {}

            engine = self.server.engine_holder.get()  # type: ignore[attr-defined]

            event = str(payload.get("hook_event_name", "") or "")
            if event == "PostToolUse":
                record = cli.process_code(payload, engine)
                response = {
                    "would_flag": False,  # code path is advisory; never blocks
                    "occurrences": int(record.get("n_issues", 0)),
                    "qdrant": record.get("qdrant", "unavailable"),
                }
            else:
                record = cli.process_action(payload, engine)
                response = {
                    "would_flag": bool(record.get("would_flag", False)),
                    "occurrences": int(record.get("occurrences", 0)),
                    "qdrant": record.get("qdrant", "unavailable"),
                }
            self._write_response(response)
        except Exception:  # noqa: BLE001 — absolute boundary: never crash a worker
            try:
                self._write_response(
                    {"would_flag": False, "occurrences": 0, "qdrant": "unavailable"}
                )
            except Exception:  # noqa: BLE001
                pass

    def _read_all(self) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = self.request.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_REQUEST_BYTES:
                break
        return b"".join(chunks)

    def _write_response(self, obj: dict) -> None:
        line = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            self.request.sendall(line)
        except Exception:  # noqa: BLE001 — client may have already gone away
            pass


class MemcheckDaemon(socketserver.ThreadingUnixStreamServer):
    """Threaded Unix-socket server holding the warm engine.

    ``ThreadingUnixStreamServer`` gives one worker thread per connection; the
    warm qdrant client is shared (read-mostly). ``engine_holder`` lazily builds
    the engine so the daemon survives qdrant being down at startup.
    """

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64

    def __init__(
        self,
        path: Path,
        engine_factory: Callable[[], Optional[object]] = build_engine,
    ) -> None:
        self.engine_holder = _EngineHolder(engine_factory)
        super().__init__(str(path), _Handler)

    def handle_error(self, request, client_address) -> None:  # noqa: D401
        # Never propagate a per-request error out of a worker; the handler
        # already wraps everything fail-open, this is just belt-and-suspenders.
        pass


def _prepare_socket_path(path: Path) -> None:
    """Create the parent dir and remove any stale socket file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        try:
            path.unlink()
        except OSError:
            pass


def serve(
    socket_path: Optional[str] = None,
    *,
    engine_factory: Callable[[], Optional[object]] = build_engine,
    ready_event: Optional[threading.Event] = None,
) -> int:
    """Construct and run the daemon until SIGTERM/SIGINT, then clean up.

    ``engine_factory`` is injectable so tests can supply an in-memory-backed
    engine (no live qdrant). ``ready_event`` (if given) is set once the daemon
    is bound and serving — tests use it to avoid racing the bind. Returns 0 on a
    clean shutdown.
    """
    path = _resolve(socket_path)
    _prepare_socket_path(path)

    server = MemcheckDaemon(path, engine_factory=engine_factory)
    # User-only access to the socket (0600).
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass

    def _shutdown(_signum, _frame) -> None:
        # shutdown() must run off the serve thread; spawn a tiny stopper.
        threading.Thread(target=server.shutdown, daemon=True).start()

    installed_signals = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _shutdown)
            installed_signals.append(sig)
        except (ValueError, OSError):
            # Not on the main thread (e.g. under a test) — skip signal install.
            pass

    if ready_event is not None:
        ready_event.set()

    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        try:
            if path.exists() or path.is_symlink():
                path.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":  # pragma: no cover — convenience entrypoint
    sys.exit(serve())

"""Structured route-decision events for the generation tiers.

Each event records which tier/route was chosen and why. It never records prompt
text: only a truncated SHA-256 and the length. Events go to the shared
append-only event log (``LOCI_EVENT_LOG``, default ``~/.hermes/event_log.jsonl``).

Set ``LOCI_ROUTE_AUDIT=0`` to turn route events off. The log has no automatic
rotation; ``scripts/event_log.py compact`` archives old events.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("loci.route_audit")
_warned_paths: set[str] = set()


def _enabled() -> bool:
    return os.environ.get("LOCI_ROUTE_AUDIT", "1").strip().lower() not in {"0", "false", "no", "off"}


def _default_log_path() -> str:
    # Read at call time so tests and operators can redirect it after import.
    return os.path.expanduser(os.environ.get("LOCI_EVENT_LOG", "~/.hermes/event_log.jsonl"))


def _scripts_dir() -> str:
    return str(Path(__file__).resolve().parent.parent / "scripts")


def _hash_prompt(prompt: str | None) -> str:
    value = (prompt or "").encode("utf-8", errors="replace")
    return hashlib.sha256(value).hexdigest()[:16]


def _warn_once(path: str, msg: str, *args: Any) -> None:
    # The log path is fixed per process, so one warning is enough; a failing
    # disk must not turn every generate() call into a log line.
    if path in _warned_paths:
        return
    _warned_paths.add(path)
    _LOG.warning(msg, *args)


def record_route_event(
    *,
    tier: str,
    route: str,
    reason: str,
    prompt: str | None = None,
    model: str | None = None,
    degraded: bool = False,
    ok: bool | None = None,
    fallback: str | None = None,
    status_code: int | None = None,
    source: str = "route_audit",
    log_path: str | None = None,
) -> bool:
    """Append a structured route decision event to the main event log.

    Returns True only when the event was actually written.
    """
    if not _enabled():
        return False
    record: dict[str, Any] = {
        "op": "route_event",
        "event_type": "route_decision",
        "source": source,
        "tier": tier,
        "route": route,
        "reason": reason,
        "model": model or "",
        "prompt_hash": _hash_prompt(prompt),
        "prompt_len": len(prompt or ""),
        "degraded": bool(degraded),
        "ok": bool(ok) if ok is not None else None,
        "fallback": fallback or "",
        "status_code": status_code,
    }
    path = log_path or _default_log_path()
    try:
        # event_log.append() only creates the parent dir for its own default
        # path; with an explicit path a missing dir made every write fail
        # silently while this function still reported success.
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        scripts_dir = _scripts_dir()
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from event_log import append as _append_event
        written = bool(_append_event(record, log_path=path))
    except Exception as exc:
        _warn_once(path, "route audit write failed for %s/%s at %s: %r", tier, route, path, exc)
        return False
    if not written:
        _warn_once(path, "route audit write failed for %s/%s at %s", tier, route, path)
    return written

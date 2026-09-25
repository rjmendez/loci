from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("loci.route_audit")


def _default_log_path() -> str:
    return os.path.expanduser(os.environ.get("LOCI_EVENT_LOG", "~/.hermes/event_log.jsonl"))


def _scripts_dir() -> str:
    return str(Path(__file__).resolve().parent.parent / "scripts")


def _hash_prompt(prompt: str | None) -> str:
    value = (prompt or "").encode("utf-8", errors="replace")
    return hashlib.sha256(value).hexdigest()[:16]


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
    """Append a structured route decision event to the main event log."""
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
    try:
        scripts_dir = _scripts_dir()
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from event_log import append as _append_event
        _append_event(record, log_path=log_path or _default_log_path())
        return True
    except Exception as exc:
        _LOG.warning("route audit write failed for %s/%s: %r", tier, route, exc)
        return False

#!/usr/bin/env python3
"""scripts/event_log.py — Immutable append-only event log for memory operations.

Implements the AgentCore Memory three-tier pattern (AWS, 2025) and the
Stability and Safety Governed Memory (SSGM, arXiv:2603.11768) recommendation:
all memory mutations should be written to an immutable raw log BEFORE being
applied, enabling replay, re-extraction, and audit trails independent of the
live store.

This module provides:
  - append(event): write one event to the log (fail-open)
  - replay(start_ts, end_ts): read back events in time range
  - compact(before_ts): archive old events to a compressed file

Callers in mcp/server.py should call append() from:
  - investigation_store (finding written)
  - memory_retract (retraction)
  - investigation_note (note added)
  - memory_consolidate (consolidation triggered)

The log is an append-only JSONL file; compaction creates dated .jsonl.gz archives.

Usage:
    from scripts.event_log import append, replay, compact

    # Record an event
    append({"op": "store", "investigation_id": "...", "finding_id": "..."})

    # Audit recent events
    for ev in replay(start_ts=None, limit=100):
        print(ev)
"""

import gzip
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_LOG = os.path.expanduser(
    os.environ.get("HERMES_EVENT_LOG", "~/.hermes/event_log.jsonl")
)
_DEFAULT_ARCHIVE_DIR = os.path.expanduser(
    os.environ.get("HERMES_EVENT_ARCHIVE", "~/.hermes/event_archive/")
)


def _log_path() -> Path:
    p = Path(_DEFAULT_LOG)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append(event: dict, log_path: str | None = None) -> bool:
    """Append an event to the immutable log. Fail-open: returns False on error.

    The event dict is augmented with:
      - ts: unix timestamp (float)
      - iso: ISO-8601 UTC timestamp string
    """
    path = Path(log_path) if log_path else _log_path()
    record = {
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        return True
    except Exception:
        return False


def replay(
    start_ts: float | None = None,
    end_ts: float | None = None,
    limit: int | None = None,
    op_filter: str | None = None,
    log_path: str | None = None,
) -> list[dict]:
    """Read events from the log within a time range.

    Args:
        start_ts: Earliest unix timestamp to include (inclusive). None = no lower bound.
        end_ts:   Latest unix timestamp to include (inclusive). None = no upper bound.
        limit:    Maximum number of events to return (most recent first if set).
        op_filter: If set, only return events where event["op"] == op_filter.
        log_path: Override the log path.

    Returns list of event dicts in chronological order.
    """
    path = Path(log_path) if log_path else _log_path()
    if not path.exists():
        return []

    events = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = ev.get("ts", 0.0)
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            if op_filter is not None and ev.get("op") != op_filter:
                continue
            events.append(ev)

    if limit is not None:
        events = events[-limit:]  # most recent N
    return events


def compact(
    before_ts: float | None = None,
    archive_dir: str | None = None,
    log_path: str | None = None,
) -> dict:
    """Archive events older than before_ts to a dated .jsonl.gz file.

    The live log is rewritten with only the events AFTER before_ts.
    Returns {"archived": int, "remaining": int, "archive_path": str}.

    before_ts defaults to 90 days ago.
    """
    if before_ts is None:
        before_ts = time.time() - 90 * 86400

    path = Path(log_path) if log_path else _log_path()
    if not path.exists():
        return {"archived": 0, "remaining": 0, "archive_path": ""}

    arch_dir = Path(archive_dir) if archive_dir else Path(_DEFAULT_ARCHIVE_DIR)
    arch_dir.mkdir(parents=True, exist_ok=True)

    old_events, new_events = [], []
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("ts", 0.0) < before_ts:
                old_events.append(ev)
            else:
                new_events.append(ev)

    if not old_events:
        return {"archived": 0, "remaining": len(new_events), "archive_path": ""}

    dt_str = datetime.fromtimestamp(before_ts, tz=timezone.utc).strftime("%Y%m%d")
    arch_path = arch_dir / f"event_log_{dt_str}.jsonl.gz"
    with gzip.open(str(arch_path), "wt") as gz:
        for ev in old_events:
            gz.write(json.dumps(ev) + "\n")

    # Rewrite live log with only recent events
    with open(path, "w") as fh:
        for ev in new_events:
            fh.write(json.dumps(ev) + "\n")

    return {
        "archived": len(old_events),
        "remaining": len(new_events),
        "archive_path": str(arch_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Hermes immutable event log tool")
    sub = ap.add_subparsers(dest="cmd")

    rp = sub.add_parser("replay", help="Replay events from the log")
    rp.add_argument("--limit", type=int, default=50)
    rp.add_argument("--op", default=None)
    rp.add_argument("--since-hours", type=float, default=None)

    cp = sub.add_parser("compact", help="Archive old events")
    cp.add_argument("--before-days", type=float, default=90)

    sp = sub.add_parser("stats", help="Print log statistics")

    a = ap.parse_args()

    if a.cmd == "replay":
        start = (time.time() - a.since_hours * 3600) if a.since_hours else None
        events = replay(start_ts=start, limit=a.limit, op_filter=a.op)
        for ev in events:
            print(json.dumps(ev))
        print(f"# {len(events)} events")

    elif a.cmd == "compact":
        before_ts = time.time() - a.before_days * 86400
        result = compact(before_ts=before_ts)
        print(f"[event_log] archived={result['archived']} remaining={result['remaining']} "
              f"archive={result['archive_path']}")

    elif a.cmd == "stats":
        events = replay()
        ops: dict[str, int] = {}
        for ev in events:
            op = ev.get("op", "unknown")
            ops[op] = ops.get(op, 0) + 1
        print(f"[event_log] total={len(events)} ops={ops}")
        path = _log_path()
        size = path.stat().st_size if path.exists() else 0
        print(f"[event_log] path={path} size={size:,} bytes")

    else:
        ap.print_help()


if __name__ == "__main__":
    main()

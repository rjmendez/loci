#!/usr/bin/env python3
"""mlops/memory/live_evo.py — Live-Evo memory adaptation from explicit failures.

The old hook-state reader expected fixture-only guard logs that no production
path wrote. This module now adapts memory only from an explicit failure stream,
passed either as Python records or as a JSONL file on the CLI.

Usage:
    python3 mlops/memory/live_evo.py \
        --db ~/.hermes/mnemosyne/data/mnemosyne.db \
        [--failures-jsonl failures.jsonl] \
        --dry-run
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_DEFAULT_DB = os.path.expanduser(
    os.environ.get("MNEMOSYNE_DB", "~/.hermes/mnemosyne/data/mnemosyne.db")
)
_ADAPTATION_LOG = Path(__file__).parent / "live_evo_log.jsonl"

DEFAULT_PENALTY = 0.15
DEFAULT_CONFIDENCE_FLOOR = 0.05
SIMILARITY_WORDS = 6
FAILURE_EVENTS = frozenset({
    "hallucination_detected",
    "grounding_fail",
    "bleed_detected",
    "retraction",
})


def _normalize_failures(records: Iterable[object] | None) -> list[dict]:
    failures = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        event = rec.get("event")
        if event not in FAILURE_EVENTS:
            continue
        failures.append({
            "session_id": rec.get("session_id", ""),
            "content": rec.get("content", "") or rec.get("text", ""),
            "event": event,
        })
    return failures


def load_failures_jsonl(path: str | None) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, errors="ignore") as fh:
            records = []
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return _normalize_failures(records)


def _word_overlap(a: str, b: str, n: int = SIMILARITY_WORDS) -> bool:
    return len(set(a.lower().split()) & set(b.lower().split())) >= n


def _find_correlated_entries(conn, failures: list[dict]) -> list[tuple]:
    if not failures:
        return []
    fail_sessions = {f["session_id"] for f in failures if f["session_id"]}
    correlated = []
    try:
        rows = conn.execute(
            "SELECT id, content, importance, session_id FROM working_memory WHERE importance IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    for row in rows:
        matched_event = None
        if row[3] and row[3] in fail_sessions:
            for f in failures:
                if f["session_id"] == row[3]:
                    matched_event = f["event"]
                    break
        if matched_event is None:
            for f in failures:
                if f.get("content") and _word_overlap(row[1] or "", f["content"]):
                    matched_event = f["event"]
                    break
        if matched_event is not None:
            correlated.append((row[0], float(row[2] or 0.0), matched_event))
    return correlated


def adapt(
    db_path: str = _DEFAULT_DB,
    failures: Iterable[object] | None = None,
    penalty: float = DEFAULT_PENALTY,
    importance_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    dry_run: bool = False,
) -> dict:
    if not os.path.exists(db_path):
        return {"error": f"db not found: {db_path}", "n_failures": 0, "n_correlated": 0}
    normalized_failures = _normalize_failures(failures)
    if not normalized_failures:
        return {"n_failures": 0, "n_correlated": 0, "n_penalized": 0, "dry_run": dry_run}
    conn = sqlite3.connect(db_path)
    try:
        correlated = _find_correlated_entries(conn, normalized_failures)
        updates = []
        for mem_id, current_importance, event in correlated:
            penalized = max(importance_floor, current_importance * (1.0 - penalty))
            if abs(penalized - current_importance) > 1e-6:
                updates.append((penalized, mem_id))
        if not dry_run and updates:
            conn.executemany("UPDATE working_memory SET importance = ? WHERE id = ?", updates)
            conn.commit()
    finally:
        conn.close()
    record = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_failures_loaded": len(normalized_failures),
        "n_correlated": len(correlated),
        "n_penalized": len(updates),
        "dry_run": dry_run,
    }
    if not dry_run:
        with open(_ADAPTATION_LOG, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    return {"n_failures": len(normalized_failures), "n_correlated": len(correlated),
            "n_penalized": len(updates), "dry_run": dry_run}


def main() -> None:
    ap = argparse.ArgumentParser(description="Live-Evo memory adaptation from explicit failures")
    ap.add_argument("--db", default=_DEFAULT_DB)
    ap.add_argument("--failures-jsonl", default=None,
                    help="Optional JSONL file containing explicit failure events")
    ap.add_argument("--penalty", type=float, default=DEFAULT_PENALTY)
    ap.add_argument("--floor", type=float, default=DEFAULT_CONFIDENCE_FLOOR)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    stats = adapt(db_path=a.db, failures=load_failures_jsonl(a.failures_jsonl),
                  penalty=a.penalty, importance_floor=a.floor, dry_run=a.dry_run)
    print(f"[live_evo] failures={stats.get('n_failures')} correlated={stats.get('n_correlated')} "
          f"penalized={stats.get('n_penalized')} dry_run={a.dry_run}")


if __name__ == "__main__":
    main()

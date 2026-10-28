#!/usr/bin/env python3
"""Learned-flag registry check (Tripwire-Governor S15).

Every default-off learned feature must be registered in ``scripts/policy_flags.toml`` with a reason it is
off and a ``review_by`` date. CI fails when an entry is incomplete, malformed, duplicated, or when its
``review_by`` is in the past, so a shipped-OFF experiment cannot be forgotten: it is re-reviewed or
removed. (Design: "A failed gate ships OFF with review_by +30 days and is not re-tuned.")

    [[flag]]
    name = "learned_router"
    reason = "Stage-0 gate failed on coverage; not re-tuned, retry needs a new registration"
    review_by = 2026-11-17

``review_by`` equal to today still passes; the day after it fails. An empty registry (no ``[[flag]]``) is
valid; a MISSING registry file is an error, because a check that silently finds nothing is not a check.

    python3 scripts/check_policy_flags.py [--registry PATH] [--today YYYY-MM-DD]

Exit status: 0 clean, 1 at least one problem, 2 the registry cannot be read. Stdlib only (Python 3.11+).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import tomllib
from pathlib import Path
from typing import Any, Optional

DEFAULT_REGISTRY = Path(__file__).resolve().parent / "policy_flags.toml"

__all__ = ["DEFAULT_REGISTRY", "check", "load_registry", "main"]


def load_registry(path: Path) -> list[dict[str, Any]]:
    """The ``[[flag]]`` entries of ``path``. Raises OSError / tomllib.TOMLDecodeError if unreadable."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    flags = data.get("flag", [])
    if not isinstance(flags, list):
        raise tomllib.TOMLDecodeError("'flag' must be an array of tables ([[flag]])")
    return flags


def _as_date(value: Any) -> Optional[dt.date]:
    if isinstance(value, dt.datetime):
        return None  # a timestamp is not a review date
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value)
        except ValueError:
            return None
    return None


def check(flags: list[dict[str, Any]], today: dt.date) -> list[str]:
    """One message per problem, each naming the flag; an empty list means the registry is clean."""
    problems: list[str] = []
    seen: set[str] = set()
    for i, entry in enumerate(flags, 1):
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name.strip():
            problems.append(f"flag #{i}: missing 'name'")
            continue
        if name in seen:
            problems.append(f"{name}: registered twice")
        seen.add(name)
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            problems.append(f"{name}: missing 'reason' (why is this learned feature default-off?)")
        if "review_by" not in entry:
            problems.append(f"{name}: missing 'review_by'")
            continue
        review_by = _as_date(entry["review_by"])
        if review_by is None:
            problems.append(f"{name}: 'review_by' must be a date (YYYY-MM-DD), got {entry['review_by']!r}")
        elif review_by < today:
            problems.append(f"{name}: review_by {review_by.isoformat()} has expired (today {today.isoformat()}); "
                            f"re-review the flag and move the date, or remove it")
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--today", type=dt.date.fromisoformat, default=None, help="override today's date (tests)")
    args = ap.parse_args(argv)
    try:
        flags = load_registry(args.registry)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"check_policy_flags: cannot read registry {args.registry}: {exc}", file=sys.stderr)
        return 2
    problems = check(flags, args.today or dt.date.today())
    for p in problems:
        print(f"check_policy_flags: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"check_policy_flags: {len(flags)} learned flag(s) registered, none expired")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""mlops/memory/decay.py — Weibull temporal decay for Mnemosyne working_memory.

Implements the decay function from Human-Inspired Memory Architecture (arXiv:2605.08538,
Microsoft, May 2026): importance × exp(-((age_days / λ)^k))

With λ=30, k=0.8:
  7 days  → 80% retention
  30 days → 37% retention
  90 days → 10% retention

Decay is computed from ``base_importance`` -- a snapshot of the authored value
taken the first time this runs -- and NOT from the live ``importance``. Reading
and writing the same column compounds: retention is a function of absolute age,
so a second run multiplies an already-decayed value by (almost) the same factor
again, and the corpus lands on the floor. It also destroys the authored value,
which nothing else stores.

Run as a step in the MLOps loop or standalone:
    python3 mlops/memory/decay.py --db ~/.hermes/mnemosyne/data/mnemosyne.db --dry-run
"""

import argparse
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = os.path.expanduser(
    os.environ.get("MNEMOSYNE_DB", "~/.hermes/mnemosyne/data/mnemosyne.db")
)
DEFAULT_LAMBDA = 30.0
DEFAULT_K = 0.8
DEFAULT_MIN_IMPORTANCE = 0.05

# scripts/hooks/pre_llm_grounding.py filters recall hits at this importance in
# three places. Rows below it are still in the table and invisible to grounding,
# so it is the number that says whether a decay run costs anything.
GROUNDING_MIN_IMPORTANCE = float(os.environ.get("HOOK_RECALL_MIN_IMPORTANCE", "0.2"))


def weibull_retention(age_days: float, lambda_days: float = DEFAULT_LAMBDA, k: float = DEFAULT_K) -> float:
    if age_days <= 0:
        return 1.0
    return math.exp(-((age_days / lambda_days) ** k))


def _has_baseline(conn: sqlite3.Connection) -> bool:
    return "base_importance" in {r[1] for r in conn.execute("PRAGMA table_info(working_memory)")}


def _ensure_baseline(conn: sqlite3.Connection) -> None:
    """Snapshot the authored importance once, so decay has a fixed input.

    Idempotent: the column is added only if absent, and only rows with no
    baseline yet are seeded. Existing baselines are never overwritten -- that is
    what makes a decay run reversible (UPDATE importance = base_importance).

    Never called under dry_run: a schema change is a write.
    """
    if not _has_baseline(conn):
        conn.execute("ALTER TABLE working_memory ADD COLUMN base_importance REAL")
    conn.execute(
        "UPDATE working_memory SET base_importance = importance "
        "WHERE base_importance IS NULL AND importance IS NOT NULL"
    )
    conn.commit()


def apply_decay(
    db_path: str = DEFAULT_DB,
    lambda_days: float = DEFAULT_LAMBDA,
    k: float = DEFAULT_K,
    min_importance: float = DEFAULT_MIN_IMPORTANCE,
    dry_run: bool = False,
) -> dict:
    if not os.path.exists(db_path):
        return {"error": f"db not found: {db_path}", "n_rows": 0, "n_decayed": 0}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    now = datetime.now(timezone.utc)

    try:
        try:
            if not dry_run:
                _ensure_baseline(conn)
            base_col = "base_importance" if _has_baseline(conn) else "importance AS base_importance"
            rows = conn.execute(
                f"SELECT id, importance, {base_col}, created_at FROM working_memory "
                "WHERE importance IS NOT NULL"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            return {"error": str(exc), "n_rows": 0, "n_decayed": 0}

        updates = []
        retentions = []
        n_visible = n_visible_before = 0
        for row in rows:
            created_raw = row["created_at"] or ""
            try:
                created_dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                # SQLite CURRENT_TIMESTAMP writes naive UTC; stamp it so the
                # subtraction against an aware `now` does not raise.
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                age_days = (now - created_dt).total_seconds() / 86400.0
            except (ValueError, TypeError) as exc:
                print(
                    f"[decay] skipping row {row['id']}: bad created_at {created_raw!r} ({exc})",
                    file=sys.stderr,
                )
                continue

            retention = weibull_retention(age_days, lambda_days, k)
            retentions.append(retention)
            current = float(row["importance"] or 0.0)
            # From the baseline, never from `current` -- see the module docstring.
            base = row["base_importance"]
            base = float(base) if base is not None else current
            decayed = max(min_importance, base * retention)
            if decayed >= GROUNDING_MIN_IMPORTANCE:
                n_visible += 1
            if current >= GROUNDING_MIN_IMPORTANCE:
                n_visible_before += 1
            if abs(decayed - current) > 1e-6:
                updates.append((decayed, row["id"]))

        if not dry_run and updates:
            conn.executemany("UPDATE working_memory SET importance = ? WHERE id = ?", updates)
            conn.commit()
    finally:
        conn.close()

    return {
        "n_rows": len(rows),
        "n_decayed": len(updates),
        "mean_retention": sum(retentions) / len(retentions) if retentions else 1.0,
        "min_retention": min(retentions) if retentions else 1.0,
        "n_grounding_visible_before": n_visible_before,
        "n_grounding_visible_after": n_visible,
        "grounding_min_importance": GROUNDING_MIN_IMPORTANCE,
        "lambda_days": lambda_days,
        "k": k,
        "dry_run": dry_run,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply Weibull temporal decay to Mnemosyne working_memory")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--lambda-days", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--min-importance", type=float, default=DEFAULT_MIN_IMPORTANCE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    stats = apply_decay(db_path=a.db, lambda_days=a.lambda_days, k=a.k,
                        min_importance=a.min_importance, dry_run=a.dry_run)

    print(f"[decay] n_rows={stats.get('n_rows')} n_decayed={stats.get('n_decayed')} "
          f"mean_retention={stats.get('mean_retention', 0):.3f} "
          f"min_retention={stats.get('min_retention', 0):.3f} "
          f"lambda={a.lambda_days}d k={a.k} dry_run={a.dry_run}")
    print(f"[decay] rows visible to grounding (importance >= "
          f"{stats.get('grounding_min_importance')}): "
          f"{stats.get('n_grounding_visible_before')} -> {stats.get('n_grounding_visible_after')}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

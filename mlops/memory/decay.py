#!/usr/bin/env python3
"""mlops/memory/decay.py — Weibull temporal decay for Mnemosyne working_memory.

Implements the decay function from Human-Inspired Memory Architecture (arXiv:2605.08538,
Microsoft, May 2026): importance × exp(-((age_days / λ)^k))

With λ=30, k=0.8:
  7 days  → 80% retention
  30 days → 37% retention
  90 days → 10% retention

Run as a step in the MLOps loop or standalone:
    python3 mlops/memory/decay.py --db ~/.hermes/mnemosyne/data/mnemosyne.db --dry-run
"""

import argparse
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = os.path.expanduser(
    os.environ.get("MNEMOSYNE_DB", "~/.hermes/mnemosyne/data/mnemosyne.db")
)
DEFAULT_LAMBDA = 30.0
DEFAULT_K = 0.8
DEFAULT_MIN_IMPORTANCE = 0.05


def weibull_retention(age_days: float, lambda_days: float = DEFAULT_LAMBDA, k: float = DEFAULT_K) -> float:
    if age_days <= 0:
        return 1.0
    return math.exp(-((age_days / lambda_days) ** k))


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
        rows = conn.execute(
            "SELECT id, importance, created_at FROM working_memory WHERE importance IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        conn.close()
        return {"error": str(exc), "n_rows": 0, "n_decayed": 0}

    updates = []
    retentions = []
    for row in rows:
        created_raw = row["created_at"] or ""
        try:
            created_dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            age_days = (now - created_dt).total_seconds() / 86400.0
        except Exception:
            continue

        retention = weibull_retention(age_days, lambda_days, k)
        retentions.append(retention)
        current = float(row["importance"] or 0.0)
        decayed = max(min_importance, current * retention)
        if abs(decayed - current) > 1e-6:
            updates.append((decayed, row["id"]))

    if not dry_run and updates:
        conn.executemany("UPDATE working_memory SET importance = ? WHERE id = ?", updates)
        conn.commit()

    conn.close()

    return {
        "n_rows": len(rows),
        "n_decayed": len(updates),
        "mean_retention": sum(retentions) / len(retentions) if retentions else 1.0,
        "min_retention": min(retentions) if retentions else 1.0,
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

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

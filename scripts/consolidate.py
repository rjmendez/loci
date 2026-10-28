#!/usr/bin/env python3
"""Nightly consolidation (Tripwire-Governor S18).  NOT SCHEDULED, NOT WIRED: nothing runs this yet.

"Sleep replay and downscaling": read the trajectory rows, write per-policy summaries with a forgetting weight
(0.995 per call, newest first) and a compacted replay log (rows older than the TTL dropped, newest 500 kept),
then advance the heartbeat ``last_sleep.json``.  Dry-run is the default: nothing is written without ``--apply``.

Rules it enforces (design.md sections 4 and 3.3):

* L0 only: any other vitals level (or a missing/stale vitals heartbeat) refuses with exit 4.
* ``rows_read == 0`` is a failure (exit 3): a consolidation that read nothing proves nothing, and the
  heartbeat does NOT advance.
* The learn dir is capped at 256 MiB (exit 6, nothing written) and free space must stay >= 2 GB (exit 5,
  nothing written; this is checked again before the first byte).
* Every file is written to a temp file and renamed over its target; the outputs are renamed first and
  ``last_sleep.json`` LAST, and only after every output succeeded.  A failure anywhere leaves the heartbeat
  at its previous value (exit 7).
* ``--check-heartbeat`` raises the alarm (exit 2) when ``last_sleep.json`` is missing, unreadable, from the
  future, or older than 36 h.
* One job at a time (lock file, exit 8), 10 min wall clock (exit 9), BELOW_NORMAL priority and a 512 MiB Job
  Object on this process; a cap that cannot be applied refuses the run (exit 10).

    python scripts/consolidate.py --trajectory-dir DIR --learn-dir DIR --vitals-state FILE [--apply]
    python scripts/consolidate.py --learn-dir DIR --check-heartbeat

Stdlib only (Python 3.11+).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loci.drpg import (  # noqa: E402
    CONSOLIDATE_WALL_S, MAX_POLICIES_PER_CALL, MIN_FREE_BYTES, Deadline, LearnerBusy, LearnWriter,
    LimitsNotApplied, LowDiskSpace, OsFs, OsLimits, PolicyLimitError, QuotaExceeded, RunLock, SandboxViolation,
    TimeLimitExceeded, TrajectoryScorer, TrajectoryStore, parse_ts, apply_os_limits, learner_blocked,
)
from loci.vitals import check_on_c  # noqa: E402

EXIT_OK = 0
EXIT_ALARM = 2
EXIT_NO_ROWS = 3
EXIT_NOT_L0 = 4
EXIT_LOW_DISK = 5
EXIT_QUOTA = 6
EXIT_FAILED = 7
EXIT_BUSY = 8
EXIT_TIMEOUT = 9
EXIT_LIMITS = 10

HEARTBEAT_MAX_S = 36 * 3600     # heartbeat older than 36 h is an alarm
HEARTBEAT_FUTURE_SKEW_S = 300
TTL_S = 14 * 86400              # replay rows older than 14 days are compacted away
KEEP_ROWS = 500                 # newest rows kept per policy
DECAY = 0.995                   # forgetting weight per call, newest first
LAST_SLEEP = "last_sleep.json"
SCHEMA = 1

__all__ = ["EXIT_OK", "EXIT_ALARM", "EXIT_NO_ROWS", "EXIT_NOT_L0", "EXIT_LOW_DISK", "EXIT_QUOTA", "EXIT_FAILED",
           "EXIT_BUSY", "EXIT_TIMEOUT", "EXIT_LIMITS", "HEARTBEAT_MAX_S", "heartbeat_alarm", "plan_outputs", "run",
           "main"]


def heartbeat_alarm(learn_dir: Path, now: float, fs: Optional[OsFs] = None) -> Optional[str]:
    """None when ``last_sleep.json`` is fresh (age <= 36 h); otherwise the alarm reason."""
    fs = fs or OsFs()
    path = Path(learn_dir) / LAST_SLEEP
    if not fs.exists(path):
        return "heartbeat_missing: no last_sleep.json"
    try:
        ts = json.loads(fs.read_bytes(path).decode("utf-8"))["ts"]
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
            raise ValueError(f"ts {ts!r}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"heartbeat_unreadable: {type(exc).__name__}: {exc}"
    age = now - ts
    if age < -HEARTBEAT_FUTURE_SKEW_S:
        return f"heartbeat_in_future: last_sleep.json is {-age:.0f}s ahead of the clock"
    if age > HEARTBEAT_MAX_S:
        return f"heartbeat_stale: last_sleep.json is {age:.0f}s old (limit {HEARTBEAT_MAX_S}s)"
    return None


def _dump(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def plan_outputs(store: TrajectoryStore, policy_ids: List[str], learn_dir: Path, now: float,
                 deadline: Optional[Deadline] = None) -> tuple:
    """``(outputs, rows_read, stats)``: ``outputs`` is a list of ``(path, bytes)`` for every summary and replay
    file; nothing is written here."""
    scorer = TrajectoryScorer()
    outputs: List[tuple] = []
    rows_read = kept_total = expired_total = undated_total = 0
    for pid in policy_ids:
        if deadline is not None:
            deadline.check()
        rows = store.read(pid)
        rows_read += len(rows)
        fresh: List[Dict[str, Any]] = []
        expired = undated = 0
        for row in rows:
            ts = parse_ts(row.get("_ts", row.get("ts")))
            if ts is None:
                undated += 1
                fresh.append(row)
            elif now - ts > TTL_S:
                expired += 1
            else:
                fresh.append(row)
        kept = fresh[-KEEP_ROWS:]
        num = den = 0.0
        for i, row in enumerate(reversed(rows)):
            weight = DECAY ** i
            num += weight * scorer.score([row])
            den += weight
        summary = {"policy_id": pid, "rows": len(rows), "kept": len(kept), "expired": expired, "undated": undated,
                   "decayed_mean_score": round(num / den, 9) if den else 0.0, "as_of": now, "schema": SCHEMA}
        outputs.append((Path(learn_dir) / "consolidated" / f"{pid}.json", _dump(summary)))
        outputs.append((Path(learn_dir) / "replay" / f"{pid}.jsonl", b"".join(_dump(r) for r in kept)))
        kept_total += len(kept)
        expired_total += expired
        undated_total += undated
    return outputs, rows_read, {"policies": len(policy_ids), "rows_kept": kept_total, "rows_expired": expired_total,
                                "rows_undated": undated_total}


def run(*, trajectory_dir: Path, learn_dir: Path, vitals_state: Path, apply: bool = False,
        clock: Callable[[], float] = time.time, fs: Optional[OsFs] = None, limits: Optional[OsLimits] = None,
        apply_limits: bool = False, protected: tuple = (), min_free_bytes: int = MIN_FREE_BYTES) -> tuple:
    """``(exit_code, report)``.  See the module docstring for what each exit code means."""
    fs = fs or OsFs()
    learn_dir = Path(learn_dir)
    report: Dict[str, Any] = {"mode": "apply" if apply else "dry-run", "rows_read": 0}

    reason = learner_blocked(vitals_state, clock)
    if reason:
        report["refused"] = reason
        return EXIT_NOT_L0, report
    now = clock()
    writer = LearnWriter(learn_dir, fs=fs, protected=protected, min_free_bytes=min_free_bytes)
    deadline = Deadline(CONSOLIDATE_WALL_S, clock)
    try:
        if apply_limits:
            report["limits"] = apply_os_limits(limits)
        if not Path(trajectory_dir).is_dir():
            report["refused"] = f"rows_read == 0: trajectory dir {trajectory_dir} does not exist"
            return EXIT_NO_ROWS, report
        store = TrajectoryStore(Path(trajectory_dir))
        policy_ids = sorted(store.list_policies())
        if len(policy_ids) > MAX_POLICIES_PER_CALL:
            raise PolicyLimitError(f"{len(policy_ids)} policies; at most {MAX_POLICIES_PER_CALL}")
        outputs, rows_read, stats = plan_outputs(store, policy_ids, learn_dir, now, deadline)
        report.update(stats)
        report["rows_read"] = rows_read
        if rows_read == 0:
            report["refused"] = "rows_read == 0: nothing to consolidate; the heartbeat is NOT advanced"
            return EXIT_NO_ROWS, report
        report["files_planned"] = len(outputs) + 1
        if not apply:
            return EXIT_OK, report
        if fs.free_bytes(learn_dir) < min_free_bytes:
            raise LowDiskSpace(f"free space is below {min_free_bytes} bytes; aborting before any write")
        with RunLock(writer, learn_dir / "consolidate.lock", clock):
            writer.write_many(outputs)
            deadline.check()
            heartbeat = {"ts": now, "rows_read": rows_read, "policies": len(policy_ids), "schema": SCHEMA}
            writer.write(learn_dir / LAST_SLEEP, _dump(heartbeat))
        report["last_sleep"] = now
        return EXIT_OK, report
    except LimitsNotApplied as exc:
        report["refused"] = str(exc)
        return EXIT_LIMITS, report
    except LearnerBusy as exc:
        report["refused"] = str(exc)
        return EXIT_BUSY, report
    except LowDiskSpace as exc:
        report["refused"] = str(exc)
        return EXIT_LOW_DISK, report
    except QuotaExceeded as exc:
        report["refused"] = str(exc)
        return EXIT_QUOTA, report
    except TimeLimitExceeded as exc:
        report["failed"] = str(exc)
        return EXIT_TIMEOUT, report
    except (SandboxViolation, PolicyLimitError, OSError, ValueError) as exc:
        report["failed"] = f"{type(exc).__name__}: {exc}"
        return EXIT_FAILED, report


def main(argv: Optional[List[str]] = None, *, clock: Callable[[], float] = time.time,
         fs: Optional[OsFs] = None, limits: Optional[OsLimits] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--learn-dir", type=Path, required=True, help="learn dir (on C:, <= 256 MiB)")
    ap.add_argument("--trajectory-dir", type=Path, default=None)
    ap.add_argument("--vitals-state", type=Path, default=None)
    ap.add_argument("--apply", action="store_true", help="write the outputs and advance last_sleep.json")
    ap.add_argument("--check-heartbeat", action="store_true", help="exit 2 if last_sleep.json is missing or older than 36 h")
    args = ap.parse_args(argv)
    try:
        check_on_c(args.learn_dir)
    except ValueError as exc:
        print(f"consolidate: {exc}", file=sys.stderr)
        return EXIT_FAILED
    if args.check_heartbeat:
        alarm = heartbeat_alarm(args.learn_dir, clock(), fs)
        if alarm:
            print(json.dumps({"alarm": alarm}), file=sys.stderr)
            return EXIT_ALARM
        print(json.dumps({"alarm": None}))
        return EXIT_OK
    if args.trajectory_dir is None or args.vitals_state is None:
        ap.error("--trajectory-dir and --vitals-state are required unless --check-heartbeat")
    code, report = run(trajectory_dir=args.trajectory_dir, learn_dir=args.learn_dir, vitals_state=args.vitals_state,
                       apply=args.apply, clock=clock, fs=fs, limits=limits, apply_limits=True)
    print(json.dumps(report, sort_keys=True), file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())

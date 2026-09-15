#!/usr/bin/env python3
"""Run the Hermes cron job list without dropping overdue work on the floor.

The live gateway scheduler on the reference host could keep "fast-forwarding"
an overdue job's ``next_run_at`` in memory without ever persisting that update.
On the next tick it re-read the same stale timestamp, declared the same run
missed again, and never executed the job. This runner owns the backlog collapse
and state persistence itself:

* a job whose ``next_run_at`` is in the past executes once now
* missed intervals collapse to one catch-up run, never a replay storm
* the next scheduled run is written back to ``jobs.json`` on the same tick
* execution failures are recorded in ``last_status`` / ``last_error``
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

try:  # pragma: no cover - exercised on Linux; harmless elsewhere.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


DEFAULT_JOBS_FILE = Path(__file__).resolve().parents[1] / "cron" / "jobs.json"


@dataclass
class TickResult:
    executed: int = 0
    failed: int = 0
    skipped_locked: bool = False


def _now() -> datetime:
    return datetime.now().astimezone()


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_dt(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    return _ensure_aware(datetime.fromisoformat(raw))


def _interval_for(job: dict[str, Any]) -> timedelta:
    schedule = job.get("schedule") or {}
    if schedule.get("kind") != "interval":
        raise ValueError(f"unsupported schedule kind: {schedule.get('kind')!r}")
    minutes = int(schedule.get("minutes") or 0)
    if minutes <= 0:
        raise ValueError(f"invalid interval minutes: {schedule.get('minutes')!r}")
    return timedelta(minutes=minutes)


def _scheduled_due_at(job: dict[str, Any], now: datetime) -> datetime:
    next_run = _parse_dt(job.get("next_run_at"))
    if next_run is not None:
        return next_run
    last_run = _parse_dt(job.get("last_run_at"))
    if last_run is not None:
        return last_run + _interval_for(job)
    return now


def _next_after(now: datetime, due_at: datetime, interval: timedelta) -> datetime:
    if due_at > now:
        return due_at
    steps = int((now - due_at).total_seconds() // interval.total_seconds()) + 1
    return due_at + (interval * steps)


def _resolve_script(jobs_file: Path, job: dict[str, Any]) -> Path:
    script = str(job.get("script") or "").strip()
    if not script:
        raise ValueError("job has no script")
    candidate = Path(script).expanduser()
    if candidate.is_absolute():
        return candidate
    if job.get("workdir"):
        return Path(job["workdir"]).expanduser() / script
    repo_or_profile = jobs_file.resolve().parents[1]
    preferred = repo_or_profile / "scripts" / script
    if preferred.exists():
        return preferred
    return repo_or_profile / script


def _command_for(script: Path, python_executable: str) -> list[str]:
    if script.suffix == ".py":
        return [python_executable, str(script)]
    if script.suffix == ".sh":
        return ["bash", str(script)]
    return [str(script)]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


@contextmanager
def _tick_lock(lock_path: Path) -> Iterator[bool]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+", encoding="utf-8")
    if fcntl is None:
        try:
            yield True
        finally:
            fh.close()
        return

    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        yield False
        return

    try:
        yield True
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def tick(
    jobs_file: Path = DEFAULT_JOBS_FILE,
    *,
    now: Optional[datetime] = None,
    python_executable: Optional[str] = None,
) -> TickResult:
    jobs_file = Path(jobs_file).expanduser()
    python_executable = python_executable or os.environ.get("LOCI_PY") or sys.executable
    now = _ensure_aware(now or _now())
    result = TickResult()

    with _tick_lock(jobs_file.with_suffix(".lock")) as locked:
        if not locked:
            result.skipped_locked = True
            return result

        payload = json.loads(jobs_file.read_text(encoding="utf-8"))
        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError(f"{jobs_file} does not contain a jobs list")

        mutated = False
        for job in jobs:
            if not job.get("enabled", True):
                continue

            interval = _interval_for(job)
            due_at = _scheduled_due_at(job, now)
            if due_at > now:
                continue

            next_run_at = _next_after(now + interval, due_at, interval).isoformat()
            script = _resolve_script(jobs_file, job)
            cmd = _command_for(script, python_executable)
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(script.parent),
            )

            job["last_run_at"] = now.isoformat()
            job["next_run_at"] = next_run_at
            if completed.returncode == 0:
                job["last_status"] = "ok"
                job["last_error"] = None
                result.executed += 1
            else:
                stderr = (completed.stderr or "").strip()
                stdout = (completed.stdout or "").strip()
                detail = stderr or stdout or f"exit {completed.returncode}"
                job["last_status"] = "error"
                job["last_error"] = detail
                result.executed += 1
                result.failed += 1
            mutated = True

        if mutated:
            _atomic_write_json(jobs_file, payload)

    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-file",
        default=os.environ.get("HERMES_CRON_JOBS_FILE", str(DEFAULT_JOBS_FILE)),
        help="Path to the jobs.json file to tick",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("LOCI_PY") or sys.executable,
        help="Python interpreter for .py job scripts",
    )
    args = parser.parse_args(argv)

    result = tick(Path(args.jobs_file), python_executable=args.python)
    if result.skipped_locked:
        return 0
    if result.executed:
        print(f"hermes_cron_runner: executed={result.executed} failed={result.failed}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

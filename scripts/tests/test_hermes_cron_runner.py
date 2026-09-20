"""Regression proof for the Hermes cron catch-up loop from issue #205."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from datetime import datetime, timedelta

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "hermes_cron_runner.py"


def _load():
    spec = importlib.util.spec_from_file_location("hermes_cron_runner_uut", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    sys.modules.pop("hermes_cron_runner_uut", None)
    return mod


def _write_jobs(path: pathlib.Path, job: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"jobs": [job]}, indent=2) + "\n", encoding="utf-8")


def test_overdue_job_executes_once_after_a_long_gap(tmp_path):
    root = tmp_path / "profile"
    jobs_file = root / "cron" / "jobs.json"
    ran_file = root / "ran.txt"
    job_script = root / "scripts" / "count_runs.py"
    job_script.parent.mkdir(parents=True, exist_ok=True)
    job_script.write_text(
        "from pathlib import Path\n"
        "p = Path(__file__).resolve().parents[1] / 'ran.txt'\n"
        "n = int(p.read_text()) if p.exists() else 0\n"
        "p.write_text(str(n + 1))\n",
        encoding="utf-8",
    )

    due_at = datetime.fromisoformat("2026-06-18T13:12:34-04:00")
    now = datetime.fromisoformat("2026-08-25T10:52:00-04:00")
    _write_jobs(
        jobs_file,
        {
            "id": "job-1",
            "name": "test-job",
            "script": "count_runs.py",
            "enabled": True,
            "schedule": {"kind": "interval", "minutes": 20},
            "next_run_at": due_at.isoformat(),
            "last_run_at": (due_at - timedelta(minutes=20)).isoformat(),
            "last_status": "ok",
            "last_error": None,
        },
    )

    uut = _load()
    first = uut.tick(jobs_file, now=now, python_executable=sys.executable)

    assert first.executed == 1
    assert first.failed == 0
    assert ran_file.read_text(encoding="utf-8") == "1"

    saved = json.loads(jobs_file.read_text(encoding="utf-8"))["jobs"][0]
    assert saved["last_status"] == "ok"
    assert saved["last_error"] is None
    assert saved["last_run_at"] == now.isoformat()
    assert datetime.fromisoformat(saved["next_run_at"]) > now

    second = uut.tick(
        jobs_file,
        now=now + timedelta(minutes=1),
        python_executable=sys.executable,
    )
    assert second.executed == 0
    assert ran_file.read_text(encoding="utf-8") == "1"


def test_job_failures_are_persisted_instead_of_silently_swallowed(tmp_path):
    root = tmp_path / "profile"
    jobs_file = root / "cron" / "jobs.json"
    job_script = root / "scripts" / "fail.py"
    job_script.parent.mkdir(parents=True, exist_ok=True)
    job_script.write_text(
        "import sys\n"
        "sys.stderr.write('boom from cron job\\n')\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )

    now = datetime.fromisoformat("2026-08-25T11:00:00-04:00")
    _write_jobs(
        jobs_file,
        {
            "id": "job-2",
            "name": "failing-job",
            "script": "fail.py",
            "enabled": True,
            "schedule": {"kind": "interval", "minutes": 5},
            "next_run_at": "2026-08-25T10:55:00-04:00",
            "last_run_at": "2026-08-25T10:50:00-04:00",
            "last_status": "ok",
            "last_error": None,
        },
    )

    uut = _load()
    result = uut.tick(jobs_file, now=now, python_executable=sys.executable)

    assert result.executed == 1
    assert result.failed == 1
    saved = json.loads(jobs_file.read_text(encoding="utf-8"))["jobs"][0]
    assert saved["last_status"] == "error"
    assert "boom from cron job" in saved["last_error"]


def test_absolute_script_path_is_rejected_and_recorded(tmp_path):
    root = tmp_path / "profile"
    jobs_file = root / "cron" / "jobs.json"
    _write_jobs(
        jobs_file,
        {
            "id": "job-abs",
            "name": "absolute-script",
            "script": str((root / "scripts" / "noop.py").resolve()),
            "enabled": True,
            "schedule": {"kind": "interval", "minutes": 5},
            "next_run_at": "2026-08-25T10:55:00-04:00",
            "last_run_at": "2026-08-25T10:50:00-04:00",
            "last_status": "ok",
            "last_error": None,
        },
    )
    uut = _load()
    result = uut.tick(jobs_file, now=datetime.fromisoformat("2026-08-25T11:00:00-04:00"),
                      python_executable=sys.executable)
    assert result.executed == 1
    assert result.failed == 1
    saved = json.loads(jobs_file.read_text(encoding="utf-8"))["jobs"][0]
    assert saved["last_status"] == "error"
    assert "absolute script paths are not allowed" in saved["last_error"]


def test_parent_traversal_script_path_is_rejected_and_recorded(tmp_path):
    root = tmp_path / "profile"
    jobs_file = root / "cron" / "jobs.json"
    _write_jobs(
        jobs_file,
        {
            "id": "job-traversal",
            "name": "traversal-script",
            "script": "../outside.py",
            "enabled": True,
            "schedule": {"kind": "interval", "minutes": 5},
            "next_run_at": "2026-08-25T10:55:00-04:00",
            "last_run_at": "2026-08-25T10:50:00-04:00",
            "last_status": "ok",
            "last_error": None,
        },
    )
    uut = _load()
    result = uut.tick(jobs_file, now=datetime.fromisoformat("2026-08-25T11:00:00-04:00"),
                      python_executable=sys.executable)
    assert result.executed == 1
    assert result.failed == 1
    saved = json.loads(jobs_file.read_text(encoding="utf-8"))["jobs"][0]
    assert saved["last_status"] == "error"
    assert "path traversal" in saved["last_error"]

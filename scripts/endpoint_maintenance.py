#!/usr/bin/env python3
"""Deterministic maintenance automation for a remote endpoint over SSH."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timeout: bool = False


def _run_ssh(endpoint: str, script: str, timeout: int) -> CommandResult:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, timeout)}",
        endpoint,
        "bash",
        "-lc",
        script,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1, timeout) + 10,
            check=False,
        )
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timeout=True,
        )


def _parse_key_values(text: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _build_sync_script(repo_path: str, repo_url: str, branch: str) -> str:
    repo_path_q = shlex.quote(repo_path)
    repo_url_q = shlex.quote(repo_url)
    branch_q = shlex.quote(branch)
    return f"""
set -euo pipefail
REPO_PATH={repo_path_q}
REPO_URL={repo_url_q}
BRANCH={branch_q}

if [ -e "$REPO_PATH" ] && [ ! -d "$REPO_PATH/.git" ]; then
  echo "ERROR=repo_path_exists_but_not_git_repository"
  exit 41
fi

CLONED=0
if [ ! -d "$REPO_PATH/.git" ]; then
  mkdir -p "$(dirname "$REPO_PATH")"
  git clone --origin origin "$REPO_URL" "$REPO_PATH"
  CLONED=1
fi

git -C "$REPO_PATH" remote set-url origin "$REPO_URL"
git -C "$REPO_PATH" fetch --prune origin
git -C "$REPO_PATH" checkout -B "$BRANCH" "origin/$BRANCH"
git -C "$REPO_PATH" reset --hard "origin/$BRANCH"
git -C "$REPO_PATH" clean -fd

SHA="$(git -C "$REPO_PATH" rev-parse HEAD)"
PORCELAIN="$(git -C "$REPO_PATH" status --porcelain=v1)"
if [ -z "$PORCELAIN" ]; then
  CLEAN=1
  DIRTY_COUNT=0
else
  CLEAN=0
  DIRTY_COUNT="$(printf '%s\\n' "$PORCELAIN" | sed '/^$/d' | wc -l | tr -d ' ')"
fi

printf 'CLONED=%s\\n' "$CLONED"
printf 'SHA=%s\\n' "$SHA"
printf 'CLEAN=%s\\n' "$CLEAN"
printf 'DIRTY_COUNT=%s\\n' "$DIRTY_COUNT"
"""


def _build_health_script(health_url: str, health_timeout: int) -> str:
    health_url_q = shlex.quote(health_url)
    return f"""
set -eu
URL={health_url_q}
TIMEOUT={max(1, health_timeout)}

if [ -z "$URL" ]; then
  echo "PRESENT=0"
  echo "REASON=disabled"
  exit 0
fi

if command -v curl >/dev/null 2>&1; then
  CODE="$(curl -sS -o /dev/null -m "$TIMEOUT" -w '%{{http_code}}' "$URL" || true)"
  echo "PRESENT=1"
  echo "TOOL=curl"
  echo "HTTP_STATUS=${{CODE:-0}}"
  exit 0
fi

if command -v wget >/dev/null 2>&1; then
  CODE="$(wget --spider --server-response -T "$TIMEOUT" "$URL" 2>&1 | awk '/^  HTTP\\/|^HTTP\\// {{c=$2}} END {{print c}}')"
  echo "PRESENT=1"
  echo "TOOL=wget"
  echo "HTTP_STATUS=${{CODE:-0}}"
  exit 0
fi

echo "PRESENT=0"
echo "REASON=http_client_missing"
"""


def _build_service_script(unit_name: str) -> str:
    unit_q = shlex.quote(unit_name)
    return f"""
set -eu
UNIT={unit_q}

if ! command -v systemctl >/dev/null 2>&1; then
  echo "PRESENT=0"
  echo "REASON=systemctl_unavailable"
  exit 0
fi

LOAD_STATE="$(systemctl --user show "$UNIT" -p LoadState --value 2>/dev/null || true)"
if [ -z "$LOAD_STATE" ] || [ "$LOAD_STATE" = "not-found" ]; then
  echo "PRESENT=0"
  echo "REASON=unit_not_found"
  exit 0
fi

ACTIVE_STATE="$(systemctl --user show "$UNIT" -p ActiveState --value 2>/dev/null || true)"
SUB_STATE="$(systemctl --user show "$UNIT" -p SubState --value 2>/dev/null || true)"
UNIT_FILE_STATE="$(systemctl --user show "$UNIT" -p UnitFileState --value 2>/dev/null || true)"

echo "PRESENT=1"
echo "LOAD_STATE=${{LOAD_STATE:-unknown}}"
echo "ACTIVE_STATE=${{ACTIVE_STATE:-unknown}}"
echo "SUB_STATE=${{SUB_STATE:-unknown}}"
echo "UNIT_FILE_STATE=${{UNIT_FILE_STATE:-unknown}}"
"""


def _add_error(errors: List[Dict[str, Any]], step: str, message: str, critical: bool) -> None:
    errors.append({"step": step, "message": message, "critical": critical})


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Maintain and inspect a remote Loci endpoint over SSH."
    )
    parser.add_argument("--endpoint", default="edge-endpoint", help="SSH host alias.")
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to sync to origin/<branch>.",
    )
    parser.add_argument(
        "--repo-path",
        default="~/development/loci",
        help="Remote absolute or home-relative repository path.",
    )
    parser.add_argument(
        "--repo-url",
        default=os.getenv("LOCI_REPO_URL", "https://github.com/rjmendez/loci.git"),
        help="Repository URL used for clone and origin reset.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="SSH command timeout (seconds) for each remote step.",
    )
    parser.add_argument(
        "--health-url",
        default="http://127.0.0.1:8201/health",
        help="Remote health URL to probe.",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=5,
        help="Remote health probe timeout (seconds).",
    )
    parser.add_argument(
        "--service",
        action="append",
        dest="services",
        default=[],
        help="systemd --user unit to inspect (repeatable).",
    )
    parser.add_argument(
        "--default-services",
        action="store_true",
        help="Also inspect common service names.",
    )

    args = parser.parse_args(argv)

    service_names = list(args.services)
    if args.default_services:
        service_names.extend(
            [
                "a2a.service",
                "context-bridge.service",
                "loci-context-bridge.service",
            ]
        )
    service_names = sorted(set(service_names))

    errors: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {
        "endpoint": args.endpoint,
        "ok": False,
        "sync": {
            "repo_path": args.repo_path,
            "repo_url": args.repo_url,
            "branch": args.branch,
            "attempted": True,
            "cloned": None,
            "remote_sha": None,
            "clean": None,
            "dirty_count": None,
        },
        "health": {
            "checked": bool(args.health_url),
            "url": args.health_url,
            "present": None,
            "http_status": None,
            "ok": None,
        },
        "services": [],
        "errors": errors,
    }

    sync_cmd = _build_sync_script(args.repo_path, args.repo_url, args.branch)
    sync_run = _run_ssh(args.endpoint, sync_cmd, args.timeout)
    if sync_run.timeout:
        _add_error(errors, "sync", "ssh sync command timed out", True)
    elif sync_run.returncode != 0:
        detail = (sync_run.stderr or sync_run.stdout).strip() or f"exit {sync_run.returncode}"
        _add_error(errors, "sync", detail, True)
    else:
        sync_data = _parse_key_values(sync_run.stdout)
        result["sync"]["cloned"] = sync_data.get("CLONED") == "1"
        result["sync"]["remote_sha"] = sync_data.get("SHA")
        result["sync"]["clean"] = sync_data.get("CLEAN") == "1"
        try:
            result["sync"]["dirty_count"] = int(sync_data.get("DIRTY_COUNT", "0"))
        except ValueError:
            result["sync"]["dirty_count"] = None
        if not result["sync"]["remote_sha"]:
            _add_error(errors, "sync", "missing remote commit SHA in sync output", True)
        if result["sync"]["clean"] is False:
            _add_error(errors, "sync", "repository remains dirty after sync", True)

    if args.health_url:
        health_cmd = _build_health_script(args.health_url, args.health_timeout)
        health_run = _run_ssh(args.endpoint, health_cmd, args.timeout)
        if health_run.timeout:
            _add_error(errors, "health", "health check command timed out", False)
        elif health_run.returncode != 0:
            detail = (health_run.stderr or health_run.stdout).strip() or f"exit {health_run.returncode}"
            _add_error(errors, "health", detail, False)
        else:
            health_data = _parse_key_values(health_run.stdout)
            present = health_data.get("PRESENT") == "1"
            result["health"]["present"] = present
            http_status = health_data.get("HTTP_STATUS")
            if http_status and http_status.isdigit():
                result["health"]["http_status"] = int(http_status)
            else:
                result["health"]["http_status"] = None
            if present and result["health"]["http_status"] is not None:
                result["health"]["ok"] = 200 <= result["health"]["http_status"] < 300
            else:
                result["health"]["ok"] = None
            if "REASON" in health_data:
                result["health"]["reason"] = health_data["REASON"]
            if "TOOL" in health_data:
                result["health"]["tool"] = health_data["TOOL"]

    for unit_name in service_names:
        unit_cmd = _build_service_script(unit_name)
        unit_run = _run_ssh(args.endpoint, unit_cmd, args.timeout)
        unit_result: Dict[str, Any] = {"name": unit_name, "present": None}
        if unit_run.timeout:
            unit_result["error"] = "timeout"
            _add_error(errors, "services", f"{unit_name}: status command timed out", False)
        elif unit_run.returncode != 0:
            detail = (unit_run.stderr or unit_run.stdout).strip() or f"exit {unit_run.returncode}"
            unit_result["error"] = detail
            _add_error(errors, "services", f"{unit_name}: {detail}", False)
        else:
            unit_data = _parse_key_values(unit_run.stdout)
            present = unit_data.get("PRESENT") == "1"
            unit_result["present"] = present
            if present:
                unit_result["load_state"] = unit_data.get("LOAD_STATE")
                unit_result["active_state"] = unit_data.get("ACTIVE_STATE")
                unit_result["sub_state"] = unit_data.get("SUB_STATE")
                unit_result["unit_file_state"] = unit_data.get("UNIT_FILE_STATE")
                unit_result["ok"] = unit_data.get("ACTIVE_STATE") == "active"
            elif "REASON" in unit_data:
                unit_result["reason"] = unit_data["REASON"]
        result["services"].append(unit_result)

    critical_errors = any(bool(error.get("critical")) for error in errors)
    result["ok"] = not critical_errors

    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

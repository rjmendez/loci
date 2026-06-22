#!/usr/bin/env python3
"""
ua-watch.py — Scans project directories for completed understand-anything graphs
and ingests any that haven't been pushed to Qdrant yet.

Run manually:   python3 ua-watch.py [<dir1> <dir2> ...]
Run via cron:   Checks all known project roots from ~/.ua-projects

State file:     ~/.ua-watch-state.json  — maps project_root -> last_ingested_git_hash

If no directories provided, scans:
  - ~/development/
  - ~/projects/
  - /mnt/c/Users/*/development/   (Windows drive)
"""

import sys
import os
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path.home() / ".ua-watch-state.json"
SCAN_ROOTS = [
    Path.home() / "development",
    Path.home() / "projects",
]


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_git_hash(project_root):
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def find_ua_projects(roots):
    """Find directories with a completed understand-anything pipeline."""
    found = []
    for root in roots:
        if not root.exists():
            continue
        # Look for .understand-anything/intermediate/final-graph.json
        for fg in root.glob("**/.understand-anything/intermediate/final-graph.json"):
            project = fg.parent.parent.parent
            found.append(project)
    return found


def run_ingest(project_root):
    """Run ua-ingest.py for a project. Returns True on success."""
    _default = Path(__file__).parent / "ua-ingest.py"
    script = Path(os.environ.get("UA_INGEST_SCRIPT", str(_default)))
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".local" / "bin") + ":" + env.get("PATH", "")

    result = subprocess.run(
        [sys.executable, str(script), str(project_root)],
        capture_output=True, text=True, timeout=300, env=env
    )
    if result.returncode == 0:
        print(result.stdout)
        return True
    print(f"ERROR ingesting {project_root}:")
    print(result.stdout[-2000:])
    print(result.stderr[-500:])
    return False


def main():
    state = load_state()

    # Determine which roots to scan
    if len(sys.argv) > 1:
        roots = [Path(p) for p in sys.argv[1:]]
        # If they passed a project root directly (has .understand-anything), use as-is
        direct = [r for r in roots if (r / ".understand-anything").exists()]
        scan   = [r for r in roots if r not in direct]
    else:
        direct = []
        scan   = SCAN_ROOTS

    projects = direct + find_ua_projects(scan)

    if not projects:
        print("No completed understand-anything projects found.")
        return

    ingested = 0
    skipped  = 0

    for project in projects:
        key      = str(project.resolve())
        git_hash = get_git_hash(project)
        last     = state.get(key, {}).get("git_hash", "")

        # Check if final-graph.json is newer than last ingest
        fg_path  = project / ".understand-anything" / "intermediate" / "final-graph.json"
        if not fg_path.exists():
            fg_path = project / ".understand-anything" / "intermediate" / "assembled-graph.json"
        if not fg_path.exists():
            continue

        fg_mtime = fg_path.stat().st_mtime
        last_mtime = state.get(key, {}).get("fg_mtime", 0)

        if git_hash and git_hash == last:
            skipped += 1
            print(f"  SKIP {project.name} (git unchanged: {git_hash[:8]})")
            continue

        if fg_mtime <= last_mtime and last_mtime > 0:
            skipped += 1
            print(f"  SKIP {project.name} (graph unchanged)")
            continue

        print(f"\n  INGEST {project.name} (git: {git_hash[:8] or 'unknown'})")
        ok = run_ingest(project)
        if ok:
            state[key] = {
                "git_hash":  git_hash,
                "fg_mtime":  fg_mtime,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "project":   project.name,
            }
            save_state(state)
            ingested += 1

    print(f"\nDone. Ingested: {ingested}  Skipped: {skipped}")


if __name__ == "__main__":
    main()

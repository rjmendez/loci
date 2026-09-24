#!/usr/bin/env python3
"""Move legacy access rows out of findings.jsonl and into access.jsonl.

Older builds of rag_context_search appended an ``{id, record_type: "access",
last_accessed, query}`` row to findings.jsonl for every hit. The row reused the
finding's own id. Readers where the last row wins then saw a text-less access
row in place of the finding. The server now writes these rows to a separate
per-investigation access.jsonl, and every findings.jsonl reader ignores legacy
access rows. This script finishes the job by moving the legacy rows out.

Safety:
  * A dry run is the default. It only reports what would move.
  * --apply copies findings.jsonl to findings.jsonl.bak-<UTC timestamp> before
    touching anything.
  * Both files are rewritten through a same-directory temp file and os.replace.
  * All other lines, including ones that do not parse, are kept byte-for-byte.
  * Rerunning is safe. A file with no access rows is left alone, and a line
    already copied into access.jsonl by an interrupted run is not copied twice.
  * Stop loci-mcp before --apply. A writer that is appending while the file is
    replaced could lose its line.

Usage:
    migrate_access_rows.py                       # dry run over $LOCI_MEMORY_DIR
    migrate_access_rows.py --memory-dir DIR --apply
    migrate_access_rows.py --investigation INV --apply
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

FINDINGS = "findings.jsonl"
ACCESS = "access.jsonl"
ACCESS_TYPES = frozenset({"access"})


def default_memory_dir() -> Path:
    return Path(os.environ.get("LOCI_MEMORY_DIR", Path.home() / ".loci" / "memory-sessions"))


def is_access_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    try:
        rec = json.loads(stripped)
    except ValueError:
        return False
    return isinstance(rec, dict) and (rec.get("record_type") or rec.get("type") or "") in ACCESS_TYPES


def split_lines(text: str) -> tuple[list[str], list[str]]:
    """Return (kept_lines, access_lines). Both keep the original line text."""
    kept, access = [], []
    for line in text.splitlines():
        (access if is_access_line(line) else kept).append(line)
    return kept, access


def _atomic_write(path: Path, lines: list[str]) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".migrate.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def migrate_investigation(inv_dir: Path, apply: bool, stamp: str) -> dict:
    findings = inv_dir / FINDINGS
    report = {"investigation_id": inv_dir.name, "access_rows": 0, "kept_lines": 0}
    if not apply:
        # A dry run writes nothing, not even a lock file.
        kept, access = split_lines(findings.read_text())
        report.update(access_rows=len(access), kept_lines=len(kept))
        return report
    # The server's rewrite paths serialise on <inv>/.lock, so take it as well.
    with open(inv_dir / ".lock", "a+") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        kept, access = split_lines(findings.read_text())
        report.update(access_rows=len(access), kept_lines=len(kept))
        if not access:
            return report

        backup = inv_dir / f"{FINDINGS}.bak-{stamp}"
        n = 1
        while backup.exists():  # never overwrite an earlier backup
            backup = inv_dir / f"{FINDINGS}.bak-{stamp}.{n}"
            n += 1
        shutil.copy2(findings, backup)
        report["backup"] = str(backup)

        access_path = inv_dir / ACCESS
        existing = access_path.read_text().splitlines() if access_path.exists() else []
        # An interrupted earlier run may already have copied some of these lines.
        already = collections.Counter(line.strip() for line in existing)
        to_add = []
        for line in access:
            key = line.strip()
            if already[key] > 0:
                already[key] -= 1
                continue
            to_add.append(key)
        # access.jsonl first: a crash between the two replaces leaves a
        # duplicate that the next run skips, never a lost row.
        if to_add:
            _atomic_write(access_path, existing + to_add)
        _atomic_write(findings, kept)
        report["moved"] = len(to_add)
        report["already_in_access_log"] = len(access) - len(to_add)
    return report


def run(memory_dir: Path, apply: bool, investigation: str | None = None) -> dict:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if investigation:
        paths = [memory_dir / investigation / FINDINGS]
    else:
        paths = sorted(memory_dir.glob(f"*/{FINDINGS}"))
    per_inv = []
    for path in paths:
        if not path.is_file():
            continue
        r = migrate_investigation(path.parent, apply, stamp)
        if r["access_rows"]:
            per_inv.append(r)
    return {
        "memory_dir": str(memory_dir),
        "mode": "apply" if apply else "dry-run",
        "investigations_scanned": len(paths),
        "investigations_with_access_rows": len(per_inv),
        "access_rows": sum(r["access_rows"] for r in per_inv),
        "per_investigation": per_inv,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--memory-dir", type=Path, default=None,
                    help="store root (default: $LOCI_MEMORY_DIR or ~/.loci/memory-sessions)")
    ap.add_argument("--investigation", default=None, help="migrate a single investigation")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default is a dry run)")
    args = ap.parse_args(argv)
    if args.investigation is not None and (
            not args.investigation or "/" in args.investigation or args.investigation in {".", ".."}):
        print(f"invalid investigation id: {args.investigation!r}", file=sys.stderr)
        return 2
    memory_dir = args.memory_dir or default_memory_dir()
    if not memory_dir.is_dir():
        print(f"memory dir not found: {memory_dir}", file=sys.stderr)
        return 2
    print(json.dumps(run(memory_dir, args.apply, args.investigation), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fail when .vulture_whitelist.py carries an entry that no longer suppresses anything.

A suppression entry is a claim: "this name is reported unused, and that report is
wrong." Claims expire. The caller gets deleted, the framework registration goes
away, the threshold changes -- and the entry stays, because nothing re-checks it.
An append-only list eventually suppresses genuinely dead code, and the check goes
on reporting clean.

The test here is exact rather than heuristic: run vulture with no whitelist,
collect every name it reports, and flag any entry whose name is absent. If
removing the entry would not change the output, the entry is inert.

That is not a style nit. Before this existed, the vulture job ran at
--min-confidence 80, which excludes the entire "unused function" class of
findings -- so 75 of the list's 78 entries suppressed nothing, and the three that
did were unused-variable entries. The list looked like a maintained root model and
was mostly decoration.

Usage:
    python3 scripts/check_vulture_whitelist.py

Exit status is 1 if any entry is inert, or if vulture is not installed. Keep the
argument list in sync with the vulture step in .github/workflows/ci.yml -- the
two must scan the same tree for the comparison to mean anything.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
WHITELIST = REPO / ".vulture_whitelist.py"

# Must match the vulture step in ci.yml.
SCAN = ["mcp/server.py", "mcp/memcheck/", "scripts/", "a2a_server/server.py"]
MIN_CONFIDENCE = "60"
EXCLUDE = "*/callgraph/fixtures/*,*/callgraph/docs/legacy/*"

ENTRY = re.compile(r"^_ = ([A-Za-z_][A-Za-z0-9_]*)", re.M)
# "path/to/f.py:12: unused function 'name' (60% confidence)"
FINDING = re.compile(r"unused \w+ '([^']+)'")


def whitelist_entries() -> list[str]:
    return ENTRY.findall(WHITELIST.read_text(encoding="utf-8"))


def reported_names() -> set[str]:
    """Every name vulture reports with the whitelist NOT applied."""
    proc = subprocess.run(
        [sys.executable, "-m", "vulture", *SCAN,
         "--min-confidence", MIN_CONFIDENCE, "--exclude", EXCLUDE],
        cwd=REPO, capture_output=True, text=True,
    )
    # vulture exits 3 when it has findings, 0 when clean; anything else is a
    # real failure (not installed, bad arguments) and must not read as "clean".
    if proc.returncode not in (0, 3):
        print("vulture could not run:\n" + (proc.stderr or proc.stdout), file=sys.stderr)
        sys.exit(1)
    return set(FINDING.findall(proc.stdout))


def main() -> int:
    entries = whitelist_entries()
    if not entries:
        print(f"{WHITELIST.name} has no entries to check", file=sys.stderr)
        return 1

    reported = reported_names()
    inert = [name for name in entries if name not in reported]

    if inert:
        print(f"{WHITELIST.name}: {len(inert)} of {len(entries)} entries suppress nothing.\n")
        print("Each of these names is no longer reported by vulture, so the entry has no")
        print("effect. Either the code it covered is gone -- delete the entry -- or the")
        print("name became genuinely dead and the entry is now hiding it. Check which:\n")
        for name in inert:
            print(f"  {name}")
        return 1

    print(f"{WHITELIST.name}: all {len(entries)} entries still suppress a live finding")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Flag commits that change source code and weaken its tests in the same commit.

The 2026-09 test audit traced several hollow tests to a single commit that changed
the code *and* bent the test to match: assertions emptied out while the probe
timeouts changed, exact counts turned into ``>=`` floors, expected values rewritten
to whatever HEAD produced. Rule 9 of AGENTS.md forbids that unless the commit says
why. This script looks for it, per commit, in a revision range.

A commit is flagged when it modifies a non-test ``.py`` file AND, in some test
file it also touches:

  * the number of assertions (``assert`` statements and ``assert*`` calls) drops,
  * ``==`` comparisons inside asserts are replaced by ordering / membership
    comparisons (``>=``, ``<=``, ``<``, ``>``, ``in``),
  * skip / xfail markers are added, or
  * a new check_test_honesty.py violation appears;

or it deletes a test file. A ``Test-weakened-because: <why>`` trailer in the commit
message acknowledges the change and silences the flag for that commit.

This is a heuristic, so CI runs it as a warning: legitimate refactors (splitting a
test file, deleting a feature with its tests) trip it too. The trailer is the fix.

Usage:
    python3 scripts/check_test_weakening.py                       # origin/main..HEAD
    python3 scripts/check_test_weakening.py --range A..B [--strict]

Exit status is 0 unless --strict is given and a commit is flagged (or git fails).
"""
from __future__ import annotations

import argparse
import ast
import collections
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_test_honesty as honesty  # noqa: E402

REPO = honesty.REPO
TRAILER = "test-weakened-because:"
LOOSE_OPS = (ast.GtE, ast.LtE, ast.Gt, ast.Lt, ast.In)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def is_test_path(rel: str) -> bool:
    parts = rel.split("/")
    name = parts[-1]
    return "tests" in parts[:-1] or name.startswith("test_") or name.endswith("_test.py") \
        or name == "conftest.py"


def profile(source: str) -> collections.Counter:
    """Counts that a weakening moves: assertions, == vs loose compares, skip/xfail marks."""
    counts: collections.Counter = collections.Counter()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return counts
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            counts["asserts"] += 1
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Compare):
                    counts["eq"] += sum(isinstance(op, ast.Eq) for op in sub.ops)
                    counts["loose"] += sum(isinstance(op, LOOSE_OPS) for op in sub.ops)
        elif isinstance(node, ast.Call):
            name = honesty._call_name(node)
            if honesty._is_assert_call(node):
                counts["asserts"] += 1
                if name in {"assertEqual", "assertEquals", "assertListEqual", "assertDictEqual"}:
                    counts["eq"] += 1
                elif name in {"assertGreaterEqual", "assertLessEqual", "assertGreater",
                              "assertLess", "assertIn"}:
                    counts["loose"] += 1
            if name in {"skip", "skipif", "skipIf", "skipUnless", "xfail", "skipTest"}:
                counts["skips"] += 1
        elif isinstance(node, ast.Attribute) and node.attr in {"skip", "xfail"} \
                and honesty._dotted(node).startswith("pytest.mark"):
            counts["skips"] += 1
    return counts


def _show(rev: str, path: str) -> str | None:
    try:
        return _git("show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return None


def weakening_signals(commit: str, path: str, cfg: honesty.Config) -> list[str]:
    old, new = _show(f"{commit}^", path), _show(commit, path)
    if old is None or new is None:
        return []
    before, after = profile(old), profile(new)
    out = []
    if after["asserts"] < before["asserts"]:
        out.append(f"assertions {before['asserts']} -> {after['asserts']}")
    if after["eq"] < before["eq"] and after["loose"] > before["loose"]:
        out.append(f"exact compares {before['eq']} -> {after['eq']} while loose compares "
                   f"{before['loose']} -> {after['loose']}")
    if after["skips"] > before["skips"]:
        out.append(f"skip/xfail markers {before['skips']} -> {after['skips']}")
    fake = REPO / path
    old_v = {v.key[1:] for v in honesty.check_file(fake, cfg, old)}
    new_v = [v for v in honesty.check_file(fake, cfg, new) if v.key[1:] not in old_v]
    out.extend(f"new {v.rule} in {v.test}" for v in new_v)
    return out


def check_commit(commit: str, cfg: honesty.Config) -> list[str]:
    message = _git("log", "-1", "--format=%B", commit)
    if any(line.strip().lower().startswith(TRAILER) for line in message.splitlines()):
        return []
    status = _git("diff-tree", "--no-commit-id", "-r", "--name-status", "--no-renames", commit)
    changes = [line.split("\t", 1) for line in status.splitlines() if "\t" in line]
    py = [(s, p) for s, p in changes if p.endswith(".py")]
    if not any(not is_test_path(p) for _, p in py):
        return []
    signals = []
    for state, path in py:
        if not is_test_path(path):
            continue
        if state == "D" and honesty._is_candidate(path):
            signals.append(f"{path}: test file deleted")
        elif state == "M":
            signals.extend(f"{path}: {s}" for s in weakening_signals(commit, path, cfg))
    return signals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--range", default="origin/main..HEAD", help="git revision range")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a commit is flagged")
    args = ap.parse_args(argv)
    try:
        commits = _git("rev-list", "--reverse", "--no-merges", args.range).split()
    except subprocess.CalledProcessError as exc:
        print(f"check_test_weakening: cannot list {args.range}: {exc.stderr.strip()}")
        return 1
    cfg = honesty.load_config()
    gha = os.environ.get("GITHUB_ACTIONS") == "true"
    flagged = 0
    for commit in commits:
        signals = check_commit(commit, cfg)
        if not signals:
            continue
        flagged += 1
        subject = _git("log", "-1", "--format=%h %s", commit).strip()
        print(f"{subject}: changes source and weakens tests without a "
              f"'Test-weakened-because:' trailer")
        for s in signals:
            print(f"    {s}")
        if gha:
            print(f"::warning title=test weakened with source change::{subject}: "
                  + "; ".join(signals))
    print(f"test-weakening: {len(commits)} commits checked, {flagged} flagged")
    return 1 if flagged and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())

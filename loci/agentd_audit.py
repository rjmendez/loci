"""agentd containment audit (Tripwire-Governor slice S11): DETECTION, not prevention.

Path rules are DETECTION, not prevention, until the restricted-token containment test passes (see
``loci/agentd_token_win.py`` and ``docs/agentd.md``). This module can tell that a run wrote somewhere it
should not have; it cannot stop the write.

What it audits, per run:

* ``sandbox_violation`` / ``protected_path``: :func:`check_write` raises :class:`SandboxViolation` for a
  write outside the worktree (``protected_path`` when the target is on the protected list: ``~/.loci``,
  ``~/.ssh``, ``~/.claude``, ``~/.hermes``, the LiteLLM directory under ``F:/services``, ``F:/services``).
  :func:`snapshot_tree` / :func:`diff_snapshots` find writes that nobody reported.
* ``persistence_change``: the scheduled-task, service, Run-key and startup-folder listings are taken before
  and after the run and diffed. A listing that cannot be taken is ``persistence_unknown`` (fail closed).
* ``diff_budget``: more than 2,000 changed lines, 40 files, or a binary over 1 MiB.

Everything that touches the OS (git, listings, file sizes) is injected. Nothing runs at import time.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

PATH_RULES_MODE = "detection"  # flips to "prevention" only in the change that lands the passing token test

CODE_PROTECTED = "protected_path"
CODE_PERSISTENCE = "persistence_change"
CODE_PERSISTENCE_UNKNOWN = "persistence_unknown"
CODE_SANDBOX = "sandbox_violation"
CODE_DIFF = "diff_budget"
# Highest severity first: a run that touched a protected path is reported as that, whatever else it did.
PRECEDENCE = (CODE_PROTECTED, CODE_PERSISTENCE, CODE_PERSISTENCE_UNKNOWN, CODE_SANDBOX, CODE_DIFF)

DIFF_MAX_LINES = 2000
DIFF_MAX_FILES = 40
DIFF_MAX_BINARY_BYTES = 1024 * 1024
AUDIT_ENTRY_MAX_BYTES = 1024
SNAPSHOT_MAX_ENTRIES = 50_000
HASH_MAX_BYTES = 64 * 1024
LINE_SCAN_CAP_BYTES = 2 * 1024 * 1024
PERSISTENCE_KINDS = ("scheduled_tasks", "services", "run_keys", "startup_folder")


class SandboxViolation(Exception):
    """A write landed outside the run's worktree. ``code`` is the stop code for the run."""

    def __init__(self, message: str, code: str = CODE_SANDBOX):
        super().__init__(message)
        self.code = code


class PersistenceListingError(RuntimeError):
    """A persistence listing could not be taken."""


class AuditError(RuntimeError):
    """The diff audit could not be completed (git failed)."""


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str = ""


# ----------------------------------------------------------------------------- paths

def _norm(path: Any) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def is_within(path: Any, root: Any) -> bool:
    a, b = _norm(path), _norm(root)
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:  # different drives
        return False


def default_protected_paths(home: Any, extra: Iterable[Any] = ()) -> tuple:
    """The detection list (decision 3). ``~/.hermes`` is where Hermes keeps its config on this machine."""
    home = Path(home)
    base = [home / ".loci", home / ".ssh", home / ".claude", home / ".hermes", Path("F:/services")]
    return tuple(base) + tuple(Path(p) for p in extra)


def check_write(path: Any, worktree: Any, protected: Sequence[Any] = ()) -> None:
    """Return if ``path`` is inside ``worktree``; raise :class:`SandboxViolation` otherwise."""
    if is_within(path, worktree):
        return
    for root in protected:
        if is_within(path, root):
            raise SandboxViolation(f"write outside worktree to protected path: {path}", CODE_PROTECTED)
    raise SandboxViolation(f"write outside worktree: {path}", CODE_SANDBOX)


# ----------------------------------------------------------------------------- protected-path snapshots

@dataclass(frozen=True)
class Snapshot:
    files: Mapping[str, tuple]       # normalised path -> (size, mtime_ns, sha256 hex or None)
    truncated: bool = False          # more than max_entries files: the comparison is partial, REPORTED
    errors: tuple = ()               # walk errors, REPORTED


def fingerprint(path: Any) -> Optional[tuple]:
    """(size, mtime_ns, sha256 for files up to 64 KiB). None if the file is absent."""
    try:
        st = os.stat(path)
        digest = None
        if st.st_size <= HASH_MAX_BYTES:
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
        return (st.st_size, st.st_mtime_ns, digest)
    except FileNotFoundError:
        return None


def snapshot_tree(roots: Iterable[Any], max_entries: int = SNAPSHOT_MAX_ENTRIES) -> Snapshot:
    files: dict = {}
    errors: list = []
    truncated = False
    for root in roots:
        root = os.fspath(root)
        if os.path.isfile(root):
            candidates: Iterator = iter([(os.path.dirname(root), [], [os.path.basename(root)])])
        elif os.path.isdir(root):
            candidates = os.walk(root, onerror=lambda e: errors.append(f"{type(e).__name__}: {e}"),
                                 followlinks=False)
        else:
            continue
        for dirpath, _dirs, names in candidates:
            for name in names:
                if len(files) >= max_entries:
                    truncated = True
                    break
                full = os.path.join(dirpath, name)
                try:
                    fp = fingerprint(full)
                except OSError as exc:
                    errors.append(f"{full}: {type(exc).__name__}: {exc}")
                    continue
                if fp is not None:
                    files[_norm(full)] = fp
            if truncated:
                break
        if truncated:
            break
    return Snapshot(files, truncated, tuple(errors))


def diff_snapshots(before: Snapshot, after: Snapshot) -> list:
    """Paths added, removed or modified. A truncated snapshot compares only what both sides hold."""
    partial = before.truncated or after.truncated
    changed = set()
    for path, fp in after.files.items():
        old = before.files.get(path)
        if old is None:
            if not partial:
                changed.add(path)
        elif old != fp:
            changed.add(path)
    if not partial:
        changed.update(p for p in before.files if p not in after.files)
    return sorted(changed)


# ----------------------------------------------------------------------------- persistence listings

Lister = Callable[[], Iterable[str]]


def _run_text(argv: Sequence[str]) -> str:
    done = subprocess.run(list(argv), capture_output=True, text=True, timeout=60, check=False)
    if done.returncode != 0:
        raise PersistenceListingError(f"{argv[0]} exited {done.returncode}: {done.stderr.strip()[:200]}")
    return done.stdout


def parse_schtasks_csv(text: str) -> frozenset:
    """Task NAMES only (first column), so a changed next-run time is not a persistence change."""
    names = set()
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0] in ("TaskName", "") or row[0].startswith("INFO:"):
            continue
        names.add(row[0])
    return frozenset(names)


def list_scheduled_tasks(run: Callable[[Sequence[str]], str] = _run_text) -> frozenset:
    return parse_schtasks_csv(run(["schtasks", "/query", "/fo", "csv", "/nh"]))


def list_services() -> frozenset:
    if sys.platform != "win32":
        raise PersistenceListingError("service listing needs Windows")
    import winreg
    names = set()
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Services") as key:
        index = 0
        while True:
            try:
                names.add(winreg.EnumKey(key, index))
            except OSError:
                break
            index += 1
    return frozenset(names)


def list_run_keys() -> frozenset:
    if sys.platform != "win32":
        raise PersistenceListingError("Run key listing needs Windows")
    import winreg
    entries = set()
    hives = (("HKCU", winreg.HKEY_CURRENT_USER), ("HKLM", winreg.HKEY_LOCAL_MACHINE))
    subkeys = (r"Software\Microsoft\Windows\CurrentVersion\Run",
               r"Software\Microsoft\Windows\CurrentVersion\RunOnce",
               r"Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Run")
    for hive_name, hive in hives:
        for sub in subkeys:
            try:
                key = winreg.OpenKey(hive, sub)
            except FileNotFoundError:
                continue
            with key:
                index = 0
                while True:
                    try:
                        name, value, _ = winreg.EnumValue(key, index)
                    except OSError:
                        break
                    entries.add(f"{hive_name}\\{sub}\\{name}={value}")
                    index += 1
    return frozenset(entries)


def list_startup_folder(environ: Mapping[str, str] = os.environ) -> frozenset:
    entries = set()
    for var, tail in (("APPDATA", r"Microsoft\Windows\Start Menu\Programs\Startup"),
                      ("PROGRAMDATA", r"Microsoft\Windows\Start Menu\Programs\StartUp")):
        base = environ.get(var)
        if not base:
            continue
        folder = os.path.join(base, tail)
        try:
            for name in os.listdir(folder):
                entries.add(os.path.join(folder, name))
        except FileNotFoundError:
            continue
    return frozenset(entries)


def default_listers() -> dict:
    """The four real listings on Windows; an empty mapping elsewhere (agentd REPORTS that as degraded)."""
    if sys.platform != "win32":
        return {}
    return {"scheduled_tasks": list_scheduled_tasks, "services": list_services,
            "run_keys": list_run_keys, "startup_folder": list_startup_folder}


def snapshot_persistence(listers: Mapping[str, Lister]) -> dict:
    out = {}
    for kind, lister in listers.items():
        try:
            out[kind] = frozenset(str(x) for x in lister())
        except Exception as exc:
            raise PersistenceListingError(f"{kind} listing failed: {type(exc).__name__}: {exc}") from exc
    return out


def diff_persistence(before: Mapping[str, frozenset], after: Mapping[str, frozenset]) -> dict:
    """{kind: {'added': [...], 'removed': [...]}} for the kinds that changed."""
    changed = {}
    for kind in sorted(set(before) | set(after)):
        old, new = before.get(kind, frozenset()), after.get(kind, frozenset())
        if old != new:
            changed[kind] = {"added": sorted(new - old), "removed": sorted(old - new)}
    return changed


# ----------------------------------------------------------------------------- diff budget

@dataclass(frozen=True)
class Change:
    path: str
    added: int
    removed: int
    binary: bool
    size: int


def parse_numstat(text: str) -> list:
    """``git diff --numstat`` rows as (added, removed, path); added/removed are None for a binary file."""
    rows = []
    for line in text.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        a, r, path = parts
        rows.append((None, None, path) if a == "-" and r == "-" else (int(a), int(r), path))
    return rows


def audit_diff(changes: Sequence[Change]) -> Optional[Violation]:
    """None if the change set is within budget (2,000 lines, 40 files, 1 MiB per binary), else a Violation."""
    lines = sum(c.added + c.removed for c in changes if not c.binary)
    problems = []
    if lines > DIFF_MAX_LINES:
        problems.append(f"{lines} changed lines (cap {DIFF_MAX_LINES})")
    if len(changes) > DIFF_MAX_FILES:
        problems.append(f"{len(changes)} files (cap {DIFF_MAX_FILES})")
    for c in changes:
        if c.binary and c.size > DIFF_MAX_BINARY_BYTES:
            problems.append(f"binary {c.path} is {c.size} bytes (cap {DIFF_MAX_BINARY_BYTES})")
    return Violation(CODE_DIFF, "; ".join(problems)[:300]) if problems else None


def _scan_untracked(full: str) -> tuple:
    """(lines, binary, size) of an untracked file, reading at most 2 MiB."""
    size = os.path.getsize(full)
    with open(full, "rb") as fh:
        head = fh.read(LINE_SCAN_CAP_BYTES)
    if b"\0" in head[:8000]:
        return 0, True, size
    lines = head.count(b"\n") + (1 if head and not head.endswith(b"\n") else 0)
    return lines, False, size


def collect_changes(git: Callable[..., Any], worktree: Any, base_sha: str) -> list:
    """Everything the run changed against ``base_sha``: tracked (numstat) plus untracked files."""
    wt = os.fspath(worktree)
    res = git(["-c", "core.quotepath=off", "diff", "--numstat", base_sha], wt)
    if res.rc != 0:
        raise AuditError(f"git diff --numstat failed: {res.err.strip()[:200]}")
    changes = []
    for added, removed, path in parse_numstat(res.out):
        full = os.path.join(wt, path)
        size = os.path.getsize(full) if os.path.exists(full) else 0
        changes.append(Change(path, added or 0, removed or 0, added is None, size))
    res = git(["ls-files", "--others", "--exclude-standard", "-z"], wt)
    if res.rc != 0:
        raise AuditError(f"git ls-files failed: {res.err.strip()[:200]}")
    for path in (p for p in res.out.split("\0") if p):
        lines, binary, size = _scan_untracked(os.path.join(wt, path))
        changes.append(Change(path, lines, 0, binary, size))
    return changes


# ----------------------------------------------------------------------------- audit entries and the per-run auditor

def audit_entry(run_id: str, violation: Violation, ts: float) -> str:
    """One JSON line, at most 1 KiB including the newline. The detail is cut (by characters, found by bisection,
    so JSON escaping cannot push the line over) until the encoded line fits."""
    def render(detail: str) -> str:
        entry = {"run_id": run_id, "code": violation.code, "detail": detail, "ts": ts}
        return json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n"

    def fits(detail: str) -> bool:
        return len(render(detail).encode("utf-8")) <= AUDIT_ENTRY_MAX_BYTES

    if fits(violation.detail):
        return render(violation.detail)
    lo, hi = 0, len(violation.detail)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(violation.detail[:mid]):
            lo = mid
        else:
            hi = mid - 1
    return render(violation.detail[:lo])


@dataclass(frozen=True)
class AuditBaseline:
    protected: Snapshot
    persistence: Mapping[str, frozenset]
    critical: Mapping[str, Optional[tuple]]
    base_sha: str
    worktree: str


@dataclass
class AuditReport:
    violations: list = field(default_factory=list)
    degraded: list = field(default_factory=list)  # things the audit could not check; never silent

    @property
    def top_code(self) -> Optional[str]:
        codes = {v.code for v in self.violations}
        return next((c for c in PRECEDENCE if c in codes), None)


class RunAuditor:
    """Baseline at start, cheap critical-file check every monitor tick, full audit at the end."""

    def __init__(self, *, protected: Sequence[Any], critical_files: Sequence[Any] = (),
                 listers: Mapping[str, Lister], git: Callable[..., Any],
                 max_entries: int = SNAPSHOT_MAX_ENTRIES):
        self.protected = tuple(protected)
        self.critical_files = tuple(critical_files)
        self.listers = dict(listers)
        self._git = git
        self._max_entries = max_entries

    def begin(self, worktree: Any, base_sha: str) -> AuditBaseline:
        return AuditBaseline(
            protected=snapshot_tree(self.protected, self._max_entries),
            persistence=snapshot_persistence(self.listers),
            critical={_norm(p): fingerprint(p) for p in self.critical_files},
            base_sha=base_sha, worktree=os.fspath(worktree))

    def quick_check(self, baseline: AuditBaseline) -> Optional[Violation]:
        for path in self.critical_files:
            if fingerprint(path) != baseline.critical.get(_norm(path)):
                return Violation(CODE_PROTECTED, f"critical file changed: {path}")
        return None

    def final_check(self, baseline: AuditBaseline) -> AuditReport:
        report = AuditReport()
        quick = self.quick_check(baseline)
        if quick is not None:
            report.violations.append(quick)
        after = snapshot_tree(self.protected, self._max_entries)
        changed = diff_snapshots(baseline.protected, after)
        if changed:
            report.violations.append(Violation(CODE_PROTECTED, ("; ".join(changed[:5]))[:300]))
        if baseline.protected.truncated or after.truncated:
            report.degraded.append("protected_snapshot_truncated")
        for err in (*baseline.protected.errors, *after.errors):
            report.degraded.append(f"protected_snapshot_error: {err}"[:200])
        if not self.listers:
            report.degraded.append("persistence_listers_absent")
        try:
            moved = diff_persistence(baseline.persistence, snapshot_persistence(self.listers))
        except PersistenceListingError as exc:
            report.violations.append(Violation(CODE_PERSISTENCE_UNKNOWN, str(exc)[:300]))
        else:
            if moved:
                report.violations.append(Violation(CODE_PERSISTENCE, json.dumps(moved, sort_keys=True)[:300]))
        try:
            diff = audit_diff(collect_changes(self._git, baseline.worktree, baseline.base_sha))
        except (AuditError, OSError) as exc:
            diff = Violation(CODE_DIFF, f"diff audit failed: {type(exc).__name__}: {exc}"[:300])
        if diff is not None:
            report.violations.append(diff)
        return report

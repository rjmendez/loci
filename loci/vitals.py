"""Host vitals sampler and static degradation ladder (Tripwire-Governor slice S9).

One thread samples commit charge, host ``bash.exe`` / ``git.exe`` counts and free disk every tick (5 s),
maps the readings to a degradation level L0..L4 with STATIC thresholds, applies hysteresis so the level
does not flap, appends every transition to an audit sink and writes a tiny (<4 KiB) state file
atomically (temp + rename) as the heartbeat. Other processes read it with :func:`read_state`.

Ladder (design section 4)::

    commit charge, percent of CommitLimit (>=):   L1 70   L2 80   L3 88   L4 94
    host process counts (>):                      bash.exe L2 150, L3 400    git.exe L2 20, L3 50
    free disk, GB (<):                            L1 15     L2 8     L3 4

    L0 normal   L1 pause learning   L2 one delegate at a time
    L3 no new delegates or learning, pause lanes   L4 kill agentd jobs, ZERO tier only

Hysteresis: escalate after ``ESCALATE_TICKS`` (2) consecutive ticks above the current level, to the lowest
level those ticks agree on; de-escalate only after the readings have stayed below the current level's
thresholds minus a margin (commit -5 points, process counts -10 %, disk +2 GB) for ``DEESCALATE_S``
(300 s), to the highest level those relaxed readings justified. The very first sample sets the level
directly (there is nothing to be hysteretic against).

A reading that cannot be taken (sampler raised, wrong type, NaN, negative) is DEGRADED: it is reported
by name in the state file and the audit, and it floors the level at L1. It is never read as "fine".

Everything that touches the OS is injected (commit reader, process counter, disk reader, wall clock,
monotonic clock, audit sink, single-instance lock); the defaults use ctypes / ``/proc`` /
``shutil.disk_usage`` and never spawn a subprocess. This module is not wired into any server or
scheduler; ``main()`` is the entry point a scheduled task would run later.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

SCHEMA_VERSION = 1
MAX_LEVEL = 4
TICK_S = 5.0
ESCALATE_TICKS = 2
DEESCALATE_S = 300.0
STALE_AFTER_S = 30.0
MAX_FUTURE_SKEW_S = 5.0
STATE_MAX_BYTES = 4096
GIB = 1024 ** 3
MUTEX_NAME = "Global\\loci_vitals_sampler"
DEGRADED_FLOOR = 1
UNSAMPLED_LEVEL = 3  # what current_level() reports before the first sample, and for a stale / missing state file

DEFAULT_STATE_PATH = Path.home() / ".loci" / "vitals_state.json"
DEFAULT_AUDIT_PATH = Path.home() / ".loci" / "vitals_audit.jsonl"


# ----------------------------------------------------------------------------- thresholds and the pure level function

@dataclass(frozen=True)
class Thresholds:
    """Static ladder. Tuples are ordered by level, starting at the level named in the comment."""
    commit_pct: tuple = (70.0, 80.0, 88.0, 94.0)   # L1..L4, reading >= value
    bash: tuple = (150, 400)                        # L2, L3, count > value
    git: tuple = (20, 50)                           # L2, L3, count > value
    disk_free_gb: tuple = (15.0, 8.0, 4.0)          # L1..L3, free < value
    commit_hold_margin: float = 5.0                 # points subtracted when deciding whether to HOLD a level
    process_hold_pct: int = 10                      # percent subtracted from the count thresholds when holding
    disk_hold_margin_gb: float = 2.0                # GB added to the disk thresholds when holding


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class Readings:
    """One sample. ``None`` means the reading failed; ``errors`` says which and why (name, reason)."""
    commit_pct: Optional[float]
    bash: Optional[int]
    git: Optional[int]
    disk_free_gb: Optional[float]
    errors: tuple = ()

    @property
    def degraded(self) -> tuple:
        return tuple(sorted({name for name, _ in self.errors}))


def _count_level(value: int, limits: tuple, first_level: int, margin_pct: int) -> int:
    level = 0
    for i, limit in enumerate(limits):
        if value > limit - (limit * margin_pct) // 100:
            level = first_level + i
    return level


def level_for(readings: Readings, t: Thresholds = DEFAULT_THRESHOLDS, *, relaxed: bool = False) -> int:
    """Pure: readings -> level 0..4. ``relaxed`` applies the hold margins (used to decide a de-escalation).

    Commit and process thresholds are inclusive/exclusive exactly as the design states: commit charge
    ``>=``, process counts ``>``, free disk ``<``. Any failed reading floors the result at L1.
    """
    level = 0
    if readings.commit_pct is not None:
        margin = t.commit_hold_margin if relaxed else 0.0
        for i, limit in enumerate(t.commit_pct):
            if readings.commit_pct >= limit - margin:
                level = max(level, 1 + i)
    pct = t.process_hold_pct if relaxed else 0
    if readings.bash is not None:
        level = max(level, _count_level(readings.bash, t.bash, 2, pct))
    if readings.git is not None:
        level = max(level, _count_level(readings.git, t.git, 2, pct))
    if readings.disk_free_gb is not None:
        margin = t.disk_hold_margin_gb if relaxed else 0.0
        for i, limit in enumerate(t.disk_free_gb):
            if readings.disk_free_gb < limit + margin:
                level = max(level, 1 + i)
    if readings.errors:
        level = max(level, DEGRADED_FLOOR)
    return level


def drivers_for(readings: Readings, t: Thresholds = DEFAULT_THRESHOLDS) -> tuple:
    """Names of the signals that sit at the raw level (for the audit trail)."""
    top = level_for(readings, t)
    if top == 0:
        return ()
    names = []
    for name, sub in (
        ("commit_pct", Readings(readings.commit_pct, None, None, None)),
        ("bash", Readings(None, readings.bash, None, None)),
        ("git", Readings(None, None, readings.git, None)),
        ("disk_free_gb", Readings(None, None, None, readings.disk_free_gb)),
    ):
        if level_for(sub, t) == top:
            names.append(name)
    for name in readings.degraded:
        if DEGRADED_FLOOR == top and name not in names:
            names.append(name)
    return tuple(names)


# ----------------------------------------------------------------------------- what each level allows

# action -> (category, highest level at which it is allowed, level at which only one may run)
_ACTIONS = {
    "start_learning": ("spawn", 0, None),
    "spawn_delegate": ("spawn", 2, 2),
    "new_worktree": ("spawn", 2, None),
    "keep_agentd_jobs": ("spawn", 3, None),
    "serve_nonzero_tier": ("serve", 3, None),
}
ACTIONS = tuple(_ACTIONS)


def allows_at(level: int, action: str, active: int = 0) -> bool:
    """Is ``action`` allowed at ``level``? ``active`` = how many of that thing already run (L2: one delegate).

    An unknown action is DENIED (False), not raised: a caller that catches broad exceptions around the gate
    must not turn a typo into permission.
    """
    spec = _ACTIONS.get(action)
    if spec is None:
        return False
    _, highest, single_at = spec
    if level > highest:
        return False
    if single_at is not None and level == single_at and active >= 1:
        return False
    return True


# ----------------------------------------------------------------------------- volume guard, atomic write, audit sink

_POSIX_MOUNT = re.compile(r"^/mnt/([A-Za-z])(?:/|$)")
_WIN_DRIVE = re.compile(r"^([A-Za-z]):")


def check_on_c(path: Any) -> Path:
    """Refuse a state or audit path that lives on another volume (F: / D:, also as /mnt/f).

    Multi-write appends to the 9p-mounted drives have lost bytes before; the heartbeat must live on C:.
    """
    text = str(path).replace("\\", "/")
    m = _POSIX_MOUNT.match(text)
    drive = m.group(1) if m else None
    if drive is None:
        w = _WIN_DRIVE.match(text)
        drive = w.group(1) if w else None
    if drive is not None and drive.lower() != "c":
        raise ValueError(f"vitals path {str(path)!r} is not on C: (drive {drive.upper()}:); refusing to write there")
    return Path(path)


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temp file in the same directory, then ``os.replace``.

    A reader sees the old file or the new file, never a partial one. On failure the temp file is removed
    and the previous content is untouched.
    """
    check_on_c(path)
    data = text.encode("utf-8")
    if len(data) >= STATE_MAX_BYTES:
        raise ValueError(f"vitals state is {len(data)} bytes, over the {STATE_MAX_BYTES - 1} byte cap")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class JsonlAuditSink:
    """Append-only audit file: one JSON object per line. Callable with the event dict."""

    def __init__(self, path: Any):
        self.path = check_on_c(path)

    def __call__(self, event: Mapping) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")


# ----------------------------------------------------------------------------- default OS samplers (no subprocess)

def _windows_commit_pct() -> float:
    import ctypes
    from ctypes import wintypes

    class PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
            ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD),
        ]

    info = PERFORMANCE_INFORMATION()
    info.cb = ctypes.sizeof(info)
    fn = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
    fn.argtypes = [ctypes.POINTER(PERFORMANCE_INFORMATION), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    if not fn(ctypes.byref(info), info.cb):
        raise OSError(f"GetPerformanceInfo failed (error {ctypes.get_last_error()})")
    if info.CommitLimit == 0:
        raise OSError("GetPerformanceInfo reported CommitLimit 0")
    return 100.0 * info.CommitTotal / info.CommitLimit


def _posix_commit_pct(meminfo: Any = "/proc/meminfo") -> float:
    fields = {}
    with open(meminfo, encoding="ascii") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            if key in ("Committed_AS", "CommitLimit"):
                fields[key] = int(rest.split()[0])
    if fields.get("CommitLimit", 0) <= 0 or "Committed_AS" not in fields:
        raise OSError("/proc/meminfo has no usable CommitLimit / Committed_AS")
    return 100.0 * fields["Committed_AS"] / fields["CommitLimit"]


def default_commit_reader() -> float:
    """Commit charge as percent of CommitLimit (GetPerformanceInfo on Windows, /proc/meminfo elsewhere)."""
    return _windows_commit_pct() if sys.platform == "win32" else _posix_commit_pct()


WATCHED_PROCESSES = ("bash", "git")


def _windows_process_counts(watched: Sequence[str]) -> dict:
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.Process32FirstW.restype = wintypes.BOOL
    k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.Process32NextW.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if snap in (None, ctypes.c_void_p(-1).value, 0xFFFFFFFF):
        raise OSError(f"CreateToolhelp32Snapshot failed (error {ctypes.get_last_error()})")
    counts = {name: 0 for name in watched}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            name = entry.szExeFile.lower()
            if name.endswith(".exe"):
                name = name[:-4]
            if name in counts:
                counts[name] += 1
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return counts


def _posix_process_counts(watched: Sequence[str], proc: Any = "/proc") -> dict:
    counts = {name: 0 for name in watched}
    for entry in os.scandir(proc):
        if not entry.name.isdigit():
            continue
        try:
            with open(os.path.join(proc, entry.name, "comm"), encoding="utf-8", errors="replace") as fh:
                name = fh.read().strip().lower()
        except OSError:
            continue  # the process exited between scandir and open
        if name in counts:
            counts[name] += 1
    return counts


def default_process_counter() -> dict:
    """{'bash': n, 'git': m}: image-name census via Toolhelp snapshot (Windows) or /proc. No subprocess."""
    if sys.platform == "win32":
        return _windows_process_counts(WATCHED_PROCESSES)
    return _posix_process_counts(WATCHED_PROCESSES)


def make_disk_reader(paths: Sequence[Any]) -> Callable[[], float]:
    """Reader of the smallest free space, in GB (GiB), over ``paths``. Any path that fails to read raises."""
    paths = tuple(str(p) for p in paths)
    if not paths:
        raise ValueError("make_disk_reader needs at least one path")

    def read() -> float:
        return min(shutil.disk_usage(p).free for p in paths) / GIB

    return read


def default_disk_reader() -> Callable[[], float]:
    if sys.platform == "win32":
        roots = [os.environ.get("SystemDrive", "C:") + "\\"]
        if os.path.exists("F:\\"):
            roots.append("F:\\")
        return make_disk_reader(roots)
    return make_disk_reader(["/"])


# ----------------------------------------------------------------------------- single instance

class _WindowsMutex:
    ERROR_ALREADY_EXISTS = 183
    ERROR_ACCESS_DENIED = 5

    def __init__(self, name: str):
        self.name = name
        self._handle = None

    def try_acquire(self) -> bool:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        k32.CreateMutexW.restype = wintypes.HANDLE
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        ctypes.set_last_error(0)
        handle = k32.CreateMutexW(None, False, self.name)
        err = ctypes.get_last_error()
        if not handle:
            if err == self.ERROR_ACCESS_DENIED:
                return False  # the mutex exists and belongs to someone we cannot open
            raise OSError(f"CreateMutexW({self.name!r}) failed (error {err})")
        if err == self.ERROR_ALREADY_EXISTS:
            k32.CloseHandle(handle)
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle:
            import ctypes
            ctypes.WinDLL("kernel32").CloseHandle(self._handle)
            self._handle = None


class _FileLock:
    def __init__(self, path: Any):
        self.path = Path(path)
        self._fh = None

    def try_acquire(self) -> bool:
        import fcntl
        fh = open(self.path, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is not None:
            self._fh.close()  # closing drops the flock
            self._fh = None


def default_single_instance(name: str = MUTEX_NAME):
    """Named kernel mutex on Windows, an flock'ed file in the temp dir elsewhere. ``try_acquire() -> bool``."""
    if sys.platform == "win32":
        return _WindowsMutex(name)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return _FileLock(Path(tempfile.gettempdir()) / f"{safe}.lock")


# ----------------------------------------------------------------------------- the sampler

def _clean_number(value: Any, *, integer: bool) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"reading is not finite: {value!r}")
    if value < 0:
        raise ValueError(f"reading is negative: {value!r}")
    if integer:
        if int(value) != value:
            raise ValueError(f"count is not a whole number: {value!r}")
        return int(value)
    return float(value)


class Vitals:
    """The sampler. ``tick()`` is one sample + level update + heartbeat; ``start()`` runs it on one thread.

    Memory is bounded: counters plus the last ``recent`` transitions.
    """

    def __init__(
        self,
        *,
        commit_reader: Callable[[], float] = default_commit_reader,
        process_counter: Callable[[], Mapping] = default_process_counter,
        disk_reader: Optional[Callable[[], float]] = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        state_path: Any = DEFAULT_STATE_PATH,
        audit_sink: Optional[Callable[[Mapping], None]] = None,
        tick_s: float = TICK_S,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        recent: int = 64,
    ):
        if tick_s <= 0:
            raise ValueError("tick_s must be positive")
        self.state_path = check_on_c(state_path)
        self._commit_reader = commit_reader
        self._process_counter = process_counter
        self._disk_reader = disk_reader if disk_reader is not None else default_disk_reader()
        self._clock = clock
        self._mono = monotonic
        self._audit = audit_sink if audit_sink is not None else JsonlAuditSink(DEFAULT_AUDIT_PATH)
        self.tick_s = float(tick_s)
        self.thresholds = thresholds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._level: Optional[int] = None
        self._degraded: tuple = ()
        self._readings: Optional[Readings] = None
        self._up_n = 0
        self._up_target = 0
        self._down_since: Optional[float] = None
        self._down_target = 0
        self._seq = 0
        self.transitions: deque = deque(maxlen=recent)
        self.state_write_errors = 0
        self.audit_errors = 0
        self.last_error: Optional[str] = None

    # -- sampling
    def sample(self) -> Readings:
        errors = []

        def one(name: str, getter: Callable[[], Any], integer: bool = False):
            try:
                return _clean_number(getter(), integer=integer)
            except Exception as exc:  # a failed reading is data: degraded, reported, never "fine"
                errors.append((name, f"{type(exc).__name__}: {exc}"))
                return None

        commit = one("commit_pct", self._commit_reader)
        disk = one("disk_free_gb", self._disk_reader)
        counts: Optional[Mapping] = None
        try:
            counts = self._process_counter()
            if not isinstance(counts, Mapping):
                raise TypeError(f"process counter returned {type(counts).__name__}, not a mapping")
        except Exception as exc:
            for name in WATCHED_PROCESSES:
                errors.append((name, f"{type(exc).__name__}: {exc}"))
        bash = git = None
        if counts is not None:
            def from_counts(name: str):
                if name not in counts:
                    errors.append((name, f"KeyError: process counter reported no {name!r}"))
                    return None
                return one(name, lambda: counts[name], integer=True)
            bash = from_counts("bash")
            git = from_counts("git")
        return Readings(commit, bash, git, disk, tuple(errors))

    # -- one tick
    def tick(self) -> int:
        readings = self.sample()
        now_w = self._clock()
        now_m = self._mono()
        t = self.thresholds
        raw = level_for(readings, t)
        hold = level_for(readings, t, relaxed=True)
        events = []
        with self._lock:
            prev = self._level
            new = raw if prev is None else prev
            if prev is not None:
                if raw > prev:
                    self._up_n += 1
                    self._up_target = raw if self._up_n == 1 else min(self._up_target, raw)
                    self._down_since = None
                    if self._up_n >= ESCALATE_TICKS:
                        new = self._up_target
                        self._up_n = 0
                else:
                    self._up_n = 0
                    if hold < prev:
                        if self._down_since is None:
                            self._down_since = now_m
                            self._down_target = hold
                        else:
                            self._down_target = max(self._down_target, hold)
                        if now_m - self._down_since >= DEESCALATE_S:
                            new = self._down_target
                            self._down_since = None
                    else:
                        self._down_since = None
            if new != prev:
                self._up_n = 0
                self._down_since = None
                events.append({
                    "ts": now_w, "event": "start" if prev is None else "level_change",
                    "from": prev, "to": new, "raw_level": raw,
                    "drivers": list(drivers_for(readings, t)), "degraded": list(readings.degraded),
                    "readings": self._readings_dict(readings),
                })
            if prev is not None and readings.degraded != self._degraded:
                events.append({
                    "ts": now_w, "event": "degraded_change", "from": list(self._degraded),
                    "to": list(readings.degraded), "level": new, "errors": [list(e) for e in readings.errors],
                })
            self._degraded = readings.degraded
            self._level = new
            self._readings = readings
            self._seq += 1
            state = {
                "v": SCHEMA_VERSION, "ts": now_w, "seq": self._seq, "pid": os.getpid(), "tick_s": self.tick_s,
                "level": new, "raw_level": raw, "degraded": list(readings.degraded),
                "readings": self._readings_dict(readings),
            }
            for ev in events:
                self.transitions.append(ev)
        for ev in events:
            try:
                self._audit(ev)
            except Exception as exc:
                self.audit_errors += 1
                self.last_error = f"audit: {type(exc).__name__}: {exc}"
        try:
            write_atomic(self.state_path, json.dumps(state, sort_keys=True))
        except Exception as exc:
            self.state_write_errors += 1
            self.last_error = f"state: {type(exc).__name__}: {exc}"
        return new

    @staticmethod
    def _readings_dict(r: Readings) -> dict:
        return {"commit_pct": r.commit_pct, "bash": r.bash, "git": r.git, "disk_free_gb": r.disk_free_gb}

    # -- queries
    def current_level(self) -> int:
        """The level; L3 until the first sample exists (unknown is not 'fine')."""
        with self._lock:
            return UNSAMPLED_LEVEL if self._level is None else self._level

    def allows(self, action: str, active: int = 0) -> bool:
        return allows_at(self.current_level(), action, active)

    def health(self) -> dict:
        with self._lock:
            return {
                "level": self._level, "ticks": self._seq, "degraded": list(self._degraded),
                "state_write_errors": self.state_write_errors, "audit_errors": self.audit_errors,
                "last_error": self.last_error,
            }

    # -- thread
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("vitals sampler already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="loci-vitals", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                self.tick()
            except Exception as exc:  # last resort: the loop must outlive a bug in one tick
                self.last_error = f"tick: {type(exc).__name__}: {exc}"
            if self._stop.wait(self.tick_s):
                return

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("vitals sampler thread did not stop")
            self._thread = None


# ----------------------------------------------------------------------------- reading the state from another process

@dataclass(frozen=True)
class StateView:
    """What another process learns from the state file.

    ``status`` is 'ok', 'stale', 'missing', 'unreadable' or 'corrupt'. Only 'ok' carries a trustworthy
    ``level``; anything else is treated as L3 for spawning and learning (serving is unaffected).
    """
    status: str
    level: Optional[int] = None
    age_s: Optional[float] = None
    degraded: tuple = ()
    readings: Mapping = field(default_factory=dict)
    detail: str = ""

    def level_for(self, category: str) -> int:
        trusted = self.status == "ok"
        if category == "spawn":
            return self.level if trusted else UNSAMPLED_LEVEL
        return self.level if trusted else 0

    def allows(self, action: str, active: int = 0) -> bool:
        spec = _ACTIONS.get(action)
        if spec is None:
            return False
        return allows_at(self.level_for(spec[0]), action, active)


def read_state(path: Any, *, clock: Callable[[], float] = time.time, stale_after_s: float = STALE_AFTER_S) -> StateView:
    """Read the sampler's heartbeat file. Missing, unreadable, corrupt or old (>= stale_after_s) => not 'ok'."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(STATE_MAX_BYTES + 1)
    except FileNotFoundError:
        return StateView("missing", detail="no state file")
    except OSError as exc:
        return StateView("unreadable", detail=f"{type(exc).__name__}: {exc}")
    if len(raw) >= STATE_MAX_BYTES:
        return StateView("corrupt", detail=f"state file is {len(raw)} bytes or more, cap {STATE_MAX_BYTES - 1}")
    try:
        doc = json.loads(raw.decode("utf-8"))
        ts = doc["ts"]
        level = doc["level"]
        if (isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts)
                or isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= MAX_LEVEL):
            raise ValueError(f"bad ts {ts!r} or level {level!r}")
        degraded = tuple(str(x) for x in doc.get("degraded", ()))
        readings = dict(doc.get("readings", {}))
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return StateView("corrupt", detail=f"{type(exc).__name__}: {exc}")
    age = clock() - ts
    if age >= stale_after_s or age < -MAX_FUTURE_SKEW_S:
        return StateView("stale", level=level, age_s=age, degraded=degraded, readings=readings,
                         detail=f"heartbeat is {age:.1f}s old (limit {stale_after_s:g}s)")
    return StateView("ok", level=level, age_s=age, degraded=degraded, readings=readings)


# ----------------------------------------------------------------------------- entry point (not scheduled anywhere yet)

def _sleep_forever(_vitals: Vitals) -> None:
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    instance_factory: Callable[[str], Any] = default_single_instance,
    vitals_factory: Callable[..., Vitals] = Vitals,
    wait: Callable[[Vitals], None] = _sleep_forever,
) -> int:
    """Run the sampler until interrupted. A second copy exits with the exit code 'already_running'."""
    ap = argparse.ArgumentParser(prog="loci.vitals", description=__doc__.splitlines()[0])
    ap.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    ap.add_argument("--audit", default=str(DEFAULT_AUDIT_PATH))
    ap.add_argument("--tick", type=float, default=TICK_S)
    args = ap.parse_args(argv)
    instance = instance_factory(MUTEX_NAME)
    if not instance.try_acquire():
        raise SystemExit("already_running")
    vitals = None
    try:
        vitals = vitals_factory(state_path=args.state, audit_sink=JsonlAuditSink(args.audit), tick_s=args.tick)
        vitals.start()
        wait(vitals)
    finally:
        if vitals is not None:
            vitals.stop()
        instance.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())

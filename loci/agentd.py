"""agentd: bounded delegation of coding agents (Tripwire-Governor slice S10).

Hermes asks agentd to run a seat-billed coding agent (Claude Code, Copilot CLI) on a task. agentd gives the
run a LEASE (wall clock, turns, runs per day, concurrency), a git WORKTREE of its own, and a Windows JOB
OBJECT (process, commit and CPU limits, kill-on-close), supervises it, and writes one summary row.

SAFE BY DEFAULT: agentd is OFF unless ``LOCI_AGENTD=1`` is in the environment. With the flag off,
:meth:`Agentd.start` refuses with ``flag_off`` before it touches git, the disk, the store or the OS. Nothing
in this module is wired into Hermes or any scheduler. Read-write delegation is a second gate on top: it
needs ``AgentdConfig.rw_enabled``, a per-run ``rw_flag`` AND a containment object (S11); until the
restricted-token test passes it stays off (docs/agentd.md).

Lease (decision 9, design section 4)::

    wall      20 min default, 60 min hard     turns   60 default, 150 hard
    runs      <= 12 per local day             concurrency  2 at L0/L1, 1 at L2, 0 at L3/L4

Start is refused, with these exact codes (first failing check wins, in this order)::

    flag_off  paused  invalid_request  wall_over_hard  turns_over_hard  rw_disabled  rw_requires_containment
    governor_unavailable  governor_level  concurrency_limit  run_count_limit  run_count_unknown
    cwd_is_home  disk_free_low  repo_untracked_limit  worktree_failed  job_failed  containment_failed

Stop codes of a run: ``completed`` ``exit_nonzero`` ``wall_limit`` ``turn_limit`` ``deadman`` plus the
S11 codes. Progress for the deadman is transcript GROWTH plus child CPU delta, never file mtimes: a loop
that touches files proves nothing.

Everything that touches the OS is injected: clocks, the governor level, git, the untracked-file counter,
free-disk and directory-size readers, the job factory, the run store. The defaults are the real thing and
are only reached when the flag is on.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, NamedTuple, Optional, Sequence

from loci import vitals

FLAG_ENV = "LOCI_AGENTD"
GIB = 1024 ** 3

DEFAULT_WALL_S = 20 * 60.0
HARD_WALL_S = 60 * 60.0
DEFAULT_TURNS = 60
HARD_TURNS = 150
MAX_RUNS_PER_DAY = 12
MIN_FREE_GB_TO_START = 10.0
MAX_UNTRACKED_FILES = 5000
DEADMAN_S = 90.0
CPU_PROGRESS_MIN_S = 0.5
SUPERVISE_INTERVAL_S = 5.0
STORE_MAX_BYTES = 16 * 1024 * 1024
KNOWN_BACKENDS = ("claude_code", "copilot_cli")

# refusal codes
FLAG_OFF = "flag_off"
PAUSED = "paused"
INVALID_REQUEST = "invalid_request"
WALL_OVER_HARD = "wall_over_hard"
TURNS_OVER_HARD = "turns_over_hard"
RW_DISABLED = "rw_disabled"
RW_REQUIRES_CONTAINMENT = "rw_requires_containment"
GOVERNOR_UNAVAILABLE = "governor_unavailable"
GOVERNOR_LEVEL = "governor_level"
CONCURRENCY_LIMIT = "concurrency_limit"
RUN_COUNT_LIMIT = "run_count_limit"
RUN_COUNT_UNKNOWN = "run_count_unknown"
CWD_IS_HOME = "cwd_is_home"
DISK_FREE_LOW = "disk_free_low"
REPO_UNTRACKED_LIMIT = "repo_untracked_limit"
WORKTREE_FAILED = "worktree_failed"
CONTAINMENT_FAILED = "containment_failed"
JOB_FAILED = "job_failed"

# stop codes
COMPLETED = "completed"
EXIT_NONZERO = "exit_nonzero"
WALL_LIMIT = "wall_limit"
TURN_LIMIT = "turn_limit"
DEADMAN = "deadman"


class GitResult(NamedTuple):
    rc: int
    out: str
    err: str


def default_git(args: Sequence[str], cwd: Any) -> GitResult:
    """Run ``git <args>`` in ``cwd`` (UTF-8, 2-minute timeout so a wedged git cannot wedge agentd)."""
    done = subprocess.run(["git", *args], cwd=os.fspath(cwd), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120, check=False)
    return GitResult(done.returncode, done.stdout, done.stderr)


class StartRefused(Exception):
    """A run was not started. ``code`` is one of the refusal codes above."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class GovernorUnavailable(Exception):
    """The governor level could not be read (no fresh heartbeat). agentd fails closed."""


def agentd_enabled(environ: Mapping[str, str] = os.environ) -> bool:
    return environ.get(FLAG_ENV, "").strip() == "1"


def max_concurrency(level: int) -> int:
    """Concurrent runs allowed at a governor level: 2 at L0/L1, 1 at L2, 0 at L3 and L4."""
    return {0: 2, 1: 2, 2: 1}.get(level, 0)


def default_governor_level(path: Any = vitals.DEFAULT_STATE_PATH,
                           clock: Callable[[], float] = time.time) -> Callable[[], int]:
    def read() -> int:
        view = vitals.read_state(path, clock=clock)
        if view.status != "ok" or view.level is None:
            raise GovernorUnavailable(f"vitals state is {view.status}: {view.detail}")
        return view.level
    return read


# ----------------------------------------------------------------------------- default OS helpers

def count_untracked(repo: Any, limit: int) -> int:
    """Untracked, not-ignored files in ``repo``; stops reading once the count passes ``limit``."""
    proc = subprocess.Popen(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=os.fspath(repo),
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    count = 0
    stopped = False
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(65536)
        if not chunk:
            break
        count += chunk.count(b"\0")
        if count > limit:
            stopped = True
            proc.kill()
            break
    proc.stdout.close()
    rc = proc.wait()
    if rc != 0 and not stopped:
        raise OSError(f"git ls-files exited {rc} in {repo}")
    return count


def _nearest_existing(path: Path) -> Path:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def free_gb(path: Any) -> float:
    return shutil.disk_usage(_nearest_existing(Path(path))).free / GIB


def dir_size(path: Any) -> int:
    total = 0
    for root, _dirs, files in os.walk(os.fspath(path), followlinks=False):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


# ----------------------------------------------------------------------------- run store

class RunStore:
    """Append-only JSONL of run rows (``start`` and ``summary``). Past ``max_bytes`` writes are DROPPED and
    counted, never silent; a corrupt line is skipped and counted."""

    def __init__(self, path: Any, max_bytes: int = STORE_MAX_BYTES):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.dropped = 0
        self.corrupt_lines = 0

    def append(self, row: Mapping[str, Any]) -> bool:
        line = json.dumps(row, sort_keys=True) + "\n"
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            size = 0
        if size + len(line.encode("utf-8")) > self.max_bytes:
            self.dropped += 1
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
        return True

    def rows(self) -> Iterator[dict]:
        try:
            fh = open(self.path, "r", encoding="utf-8")
        except FileNotFoundError:
            return
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    self.corrupt_lines += 1


def local_day(ts: float) -> str:
    return datetime.fromtimestamp(ts).date().isoformat()


# ----------------------------------------------------------------------------- progress (deadman)

class FileProbe:
    """Progress sources: transcript size in bytes (growth) and child CPU seconds. Never an mtime."""

    def __init__(self, transcript: Optional[Any], cpu_reader: Callable[[], float]):
        self._transcript = transcript
        self._cpu = cpu_reader

    def transcript_bytes(self) -> int:
        if self._transcript is None:
            return 0
        try:
            return os.stat(self._transcript).st_size
        except OSError:
            return 0

    def cpu_seconds(self) -> float:
        return float(self._cpu())


class ProgressTracker:
    """The deadman. Progress = transcript growth, or CPU time since the last progress >= ``cpu_min_s``."""

    def __init__(self, now: float, probe: Any, deadman_s: float = DEADMAN_S, cpu_min_s: float = CPU_PROGRESS_MIN_S):
        self._probe = probe
        self.deadman_s = deadman_s
        self.cpu_min_s = cpu_min_s
        self._size = probe.transcript_bytes()
        self._cpu = probe.cpu_seconds()
        self._last = now

    def observe(self, now: float) -> bool:
        size = self._probe.transcript_bytes()
        cpu = self._probe.cpu_seconds()
        progressed = size > self._size or cpu - self._cpu >= self.cpu_min_s
        if progressed:
            self._last = now
            self._size, self._cpu = size, cpu
        elif size < self._size:
            self._size = size  # the transcript was rotated or truncated: new baseline, not progress
        return progressed

    def silent_for(self, now: float) -> float:
        return now - self._last

    def expired(self, now: float) -> bool:
        return self.silent_for(now) >= self.deadman_s


# ----------------------------------------------------------------------------- agentd

@dataclass(frozen=True)
class StartRequest:
    backend: str
    argv: Sequence[str]
    cwd: Any
    repo: Optional[Any] = None
    mode: str = "ro"
    rw_flag: bool = False
    wall_s: Optional[float] = None
    turns: Optional[int] = None
    base_ref: str = "HEAD"
    transcript_path: Optional[Any] = None
    env: Optional[Mapping[str, str]] = None


@dataclass(frozen=True)
class AgentdConfig:
    worktrees_root: Any                       # required: agentd never picks a place on disk by itself
    home: Any = field(default_factory=Path.home)
    rw_enabled: bool = False                  # stays False until the restricted-token test passes
    pause_path: Optional[Any] = None          # a file that exists = PAUSE: no new runs
    backends: Sequence[str] = KNOWN_BACKENDS
    min_free_gb: float = MIN_FREE_GB_TO_START
    max_untracked: int = MAX_UNTRACKED_FILES
    runs_per_day: int = MAX_RUNS_PER_DAY
    deadman_s: float = DEADMAN_S


@dataclass
class Run:
    run_id: str
    backend: str
    mode: str
    worktree: Path
    base_sha: str
    job: Any
    wall_s: float
    turns_limit: int
    started: float
    started_wall: float
    progress: ProgressTracker
    turns: int = 0
    stop_code: Optional[str] = None
    finalized: bool = False
    summary: Optional[dict] = None


class Agentd:
    def __init__(
        self,
        config: AgentdConfig,
        *,
        job_factory: Callable[[], Any],
        governor_level: Optional[Callable[[], int]] = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        day_of: Callable[[float], str] = local_day,
        enabled: Callable[[], bool] = agentd_enabled,
        git: Callable[..., GitResult] = default_git,
        untracked_counter: Callable[[Any, int], int] = count_untracked,
        free_gb_reader: Callable[[Any], float] = free_gb,
        dir_size_reader: Callable[[Any], int] = dir_size,
        store: Optional[RunStore] = None,
        containment: Any = None,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex[:12],
    ):
        self.config = config
        self._job_factory = job_factory
        self._governor_level = governor_level if governor_level is not None else default_governor_level()
        self._clock = clock
        self._wall = wall_clock
        self._day_of = day_of
        self._enabled = enabled
        self._git = git
        self._count_untracked = untracked_counter
        self._free_gb = free_gb_reader
        self._dir_size = dir_size_reader
        self.store = store
        self._containment = containment
        self._new_id = id_factory
        self._lock = threading.Lock()
        self._active: dict = {}
        self._starts: dict = {}          # local day -> runs started (seeded from the store, then in memory)
        self._seeded: set = set()
        self._refusals: Counter = Counter()
        self._stops: Counter = Counter()
        self.leaks = 0

    # -- reporting
    def stats(self) -> dict:
        return {
            "enabled": bool(self._enabled()),
            "active": len(self._active),
            "refusals": dict(self._refusals),
            "stops": dict(self._stops),
            "leaks": self.leaks,
            "store_dropped": self.store.dropped if self.store else 0,
            "store_corrupt_lines": self.store.corrupt_lines if self.store else 0,
        }

    # -- start
    def _level(self) -> int:
        try:
            level = self._governor_level()
        except Exception as exc:
            raise StartRefused(GOVERNOR_UNAVAILABLE, f"{type(exc).__name__}: {exc}") from exc
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= vitals.MAX_LEVEL:
            raise StartRefused(GOVERNOR_UNAVAILABLE, f"governor level {level!r} is not 0..{vitals.MAX_LEVEL}")
        return level

    def _runs_today(self, day: str) -> int:
        if day not in self._seeded:
            seen = 0
            if self.store is not None:
                try:
                    seen = sum(1 for row in self.store.rows() if row.get("event") == "start" and row.get("day") == day)
                except OSError as exc:
                    raise StartRefused(RUN_COUNT_UNKNOWN, f"{type(exc).__name__}: {exc}") from exc
            self._starts[day] = max(self._starts.get(day, 0), seen)
            self._seeded.add(day)
        return self._starts[day]

    def check_start(self, req: StartRequest) -> tuple:
        """Raise :class:`StartRefused` or return ``(wall_s, turns)`` for the lease. Touches nothing on failure."""
        cfg = self.config
        if not self._enabled():
            raise StartRefused(FLAG_OFF, f"{FLAG_ENV}=1 is not set")
        if cfg.pause_path is not None and Path(cfg.pause_path).exists():
            raise StartRefused(PAUSED, f"{cfg.pause_path} exists")
        if req.backend not in cfg.backends or not list(req.argv) or req.mode not in ("ro", "rw"):
            raise StartRefused(INVALID_REQUEST, f"backend={req.backend!r} mode={req.mode!r} argv={list(req.argv)!r}")
        wall = DEFAULT_WALL_S if req.wall_s is None else float(req.wall_s)
        turns = DEFAULT_TURNS if req.turns is None else req.turns
        if wall <= 0 or isinstance(turns, bool) or not isinstance(turns, int) or turns <= 0:
            raise StartRefused(INVALID_REQUEST, f"wall_s={req.wall_s!r} turns={req.turns!r}")
        if wall > HARD_WALL_S:
            raise StartRefused(WALL_OVER_HARD, f"{wall:g}s > hard {HARD_WALL_S:g}s")
        if turns > HARD_TURNS:
            raise StartRefused(TURNS_OVER_HARD, f"{turns} > hard {HARD_TURNS}")
        if req.mode == "rw":
            if not (cfg.rw_enabled and req.rw_flag):
                raise StartRefused(RW_DISABLED, "rw delegation needs rw_enabled and an explicit per-run rw_flag")
            if self._containment is None:
                raise StartRefused(RW_REQUIRES_CONTAINMENT, "no containment configured")
        level = self._level()
        active = len(self._active)
        if level >= 3 or (level == 2 and active >= 1):
            raise StartRefused(GOVERNOR_LEVEL, f"governor level L{level}, {active} run(s) active")
        if active >= max_concurrency(level):
            raise StartRefused(CONCURRENCY_LIMIT, f"{active} active, limit {max_concurrency(level)} at L{level}")
        day = self._day_of(self._wall())
        if self._runs_today(day) >= cfg.runs_per_day:
            raise StartRefused(RUN_COUNT_LIMIT, f"{cfg.runs_per_day} runs already started on {day}")
        repo = Path(req.repo if req.repo is not None else req.cwd)
        if _same_path(req.cwd, cfg.home):
            raise StartRefused(CWD_IS_HOME, f"cwd {req.cwd} is the home folder")
        free = self._free_gb(cfg.worktrees_root)
        if free < cfg.min_free_gb:
            raise StartRefused(DISK_FREE_LOW, f"{free:.2f} GB free, need {cfg.min_free_gb:g}")
        try:
            untracked = self._count_untracked(repo, cfg.max_untracked)
        except Exception as exc:
            raise StartRefused(REPO_UNTRACKED_LIMIT, f"cannot count untracked files: {type(exc).__name__}: {exc}") from exc
        if untracked > cfg.max_untracked:
            raise StartRefused(REPO_UNTRACKED_LIMIT, f"{untracked} untracked files, cap {cfg.max_untracked}")
        return wall, turns

    def start(self, req: StartRequest) -> Run:
        with self._lock:
            try:
                wall, turns = self.check_start(req)
                return self._start_locked(req, wall, turns)
            except StartRefused as refused:
                self._refusals[refused.code] += 1
                raise

    def _start_locked(self, req: StartRequest, wall: float, turns: int) -> Run:
        repo = Path(req.repo if req.repo is not None else req.cwd)
        run_id = self._new_id()
        worktree = Path(self.config.worktrees_root) / run_id
        head = self._git(["rev-parse", "--verify", req.base_ref + "^{commit}"], repo)
        if head.rc != 0:
            raise StartRefused(WORKTREE_FAILED, f"cannot resolve {req.base_ref}: {head.err.strip()[:200]}")
        base_sha = head.out.strip()
        made = self._git(["worktree", "add", "-b", f"agentd/{run_id}", os.fspath(worktree), base_sha], repo)
        if made.rc != 0:
            raise StartRefused(WORKTREE_FAILED, f"git worktree add failed: {made.err.strip()[:200]}")
        job = self._job_factory()
        spawner = None
        try:
            if req.mode == "rw":
                try:
                    self._containment.prepare(worktree)
                except Exception as exc:
                    raise StartRefused(CONTAINMENT_FAILED, f"{type(exc).__name__}: {exc}") from exc
                spawner = self._containment.spawn_suspended
            try:
                job.start(list(req.argv), worktree, req.env, spawner=spawner)
            except Exception as exc:
                if req.mode == "rw":
                    self._release_containment(worktree)
                raise StartRefused(JOB_FAILED, f"{type(exc).__name__}: {exc}") from exc
        except StartRefused:
            self._discard_fresh_worktree(repo, worktree, run_id)
            raise
        now = self._clock()
        wall_now = self._wall()
        day = self._day_of(wall_now)
        probe = FileProbe(req.transcript_path, job.cpu_seconds)
        run = Run(run_id=run_id, backend=req.backend, mode=req.mode, worktree=worktree, base_sha=base_sha, job=job,
                  wall_s=wall, turns_limit=turns, started=now, started_wall=wall_now,
                  progress=ProgressTracker(now, probe, self.config.deadman_s))
        self._active[run_id] = run
        self._starts[day] = self._starts.get(day, 0) + 1
        self._record({"event": "start", "run_id": run_id, "backend": req.backend, "mode": req.mode,
                      "day": day, "ts": round(wall_now, 3)})
        return run

    def _discard_fresh_worktree(self, repo: Path, worktree: Path, run_id: str) -> None:
        self._git(["worktree", "remove", "--force", os.fspath(worktree)], repo)
        self._git(["branch", "-D", f"agentd/{run_id}"], repo)

    def _release_containment(self, worktree: Path) -> None:
        try:
            self._containment.release(worktree)
        except Exception:
            self._stops["containment_release_error"] += 1

    def _record(self, row: Mapping[str, Any]) -> None:
        if self.store is None:
            return
        try:
            self.store.append(row)
        except OSError:
            self._stops["store_write_error"] += 1

    # -- supervision
    def record_turn(self, run: Run, count: int = 1) -> None:
        """Called for each agent turn the transcript reader sees. Counted by events, never by mtime."""
        run.turns += count

    def kill_run(self, run: Run, code: str) -> None:
        if run.stop_code is None:
            run.stop_code = code
            self._stops[code] += 1
            run.job.kill()

    def tick(self, run: Run) -> Optional[str]:
        """One supervision step. Returns ``'exited'``, the stop code if this step killed the run, or None."""
        if run.finalized:
            return "exited"
        if run.stop_code is not None:
            return run.stop_code
        if run.job.exit_code() is not None:
            return "exited"
        now = self._clock()
        if now - run.started >= run.wall_s:
            self.kill_run(run, WALL_LIMIT)
            return WALL_LIMIT
        if run.turns > run.turns_limit:
            self.kill_run(run, TURN_LIMIT)
            return TURN_LIMIT
        run.progress.observe(now)
        if run.progress.expired(now):
            self.kill_run(run, DEADMAN)
            return DEADMAN
        return None

    def supervise(self, run: Run, sleep: Callable[[float], None] = time.sleep,
                  interval: float = SUPERVISE_INTERVAL_S) -> dict:
        while self.tick(run) is None:
            sleep(interval)
        return self.finalize(run)

    def finalize(self, run: Run) -> dict:
        """Census, LEAK handling, summary row. Idempotent. The worktree is KEPT (a killed run's diff stays)."""
        if run.finalized and run.summary is not None:
            return run.summary
        exit_code = run.job.exit_code()
        report = run.job.finish()
        if report.leak:
            self.leaks += 1
        stop = run.stop_code
        if stop is None:
            stop = COMPLETED if exit_code == 0 else EXIT_NONZERO
            self._stops[stop] += 1
        if run.mode == "rw" and self._containment is not None:
            self._release_containment(run.worktree)
        wall = self._clock() - run.started
        summary = {"event": "summary", "run_id": run.run_id, "backend": run.backend, "mode": run.mode,
                   "wall_s": round(wall, 3), "turns": run.turns, "stop_code": stop, "leak": report.leak,
                   "survivors": report.survivors, "day": self._day_of(run.started_wall),
                   "ts": round(self._wall(), 3), "worktree_kept": True}
        run.summary = summary
        run.finalized = True
        self._active.pop(run.run_id, None)
        self._record(summary)
        return summary


def _same_path(a: Any, b: Any) -> bool:
    return os.path.normcase(os.path.realpath(os.fspath(a))) == os.path.normcase(os.path.realpath(os.fspath(b)))

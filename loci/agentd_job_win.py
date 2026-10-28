"""Windows Job Object wrapper for agentd (Tripwire-Governor slice S10).

One Job Object per delegated run. Limits (design section 4, decision 9):

* ``KILL_ON_JOB_CLOSE``: closing the last handle to the job kills every process in it.
* ``ActiveProcessLimit`` 64: CreateProcess fails past the limit (the agent sees a spawn error).
* per-job commit (``JobMemoryLimit``) 6 GiB: allocation fails past it.
* CPU hard cap 50 %.
* no ``BREAKAWAY_OK``: a child cannot leave the job.

Every OS call goes through a :class:`JobOsApi`. The production implementation is :class:`Win32JobApi`
(ctypes); unit tests use a recording fake, and exactly one real-Windows test drives :class:`Win32JobApi`
against a harmless child of its own. Nothing in this module touches the OS at import time or at
construction time: only :meth:`JobObject.start` and friends do, through the injected API.

Order of operations in :meth:`JobObject.start` matters and is asserted by the tests: create the job, set
the limits, spawn the process SUSPENDED, assign it to the job, and only then resume it. A process that is
resumed before it is in the job could fork children that escape.

Census: :meth:`JobObject.finish` counts the processes still in the job after the agent exited. A nonzero
count that was not caused by agentd's own kill is a LEAK: it is logged (event ``leak``), counted, and the
job is killed.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

GIB = 1024 ** 3

DEFAULT_ACTIVE_PROCESS_LIMIT = 64
DEFAULT_JOB_MEMORY_BYTES = 6 * GIB
DEFAULT_CPU_PERCENT = 50

JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

# After a kill, poll the census this many times (sleeping KILL_POLL_S between) before calling it a survivor.
KILL_POLL_TRIES = 20
KILL_POLL_S = 0.1
# A process whose exit code is already visible can still be counted by the job for a few milliseconds.
# Before a nonzero census is called a LEAK, poll this many times (SETTLE_S apart) for it to drain.
SETTLE_TRIES = 10
SETTLE_S = 0.05


class JobError(RuntimeError):
    """A Job Object operation failed. The partially built job and process are already torn down."""


@dataclass(frozen=True)
class JobLimits:
    active_process_limit: int = DEFAULT_ACTIVE_PROCESS_LIMIT
    job_memory_bytes: int = DEFAULT_JOB_MEMORY_BYTES
    cpu_percent: int = DEFAULT_CPU_PERCENT

    def __post_init__(self) -> None:
        for name in ("active_process_limit", "job_memory_bytes", "cpu_percent"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.cpu_percent > 100:
            raise ValueError(f"cpu_percent must be 1..100, got {self.cpu_percent}")

    @property
    def limit_flags(self) -> int:
        return (JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_JOB_MEMORY
                | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)


@dataclass(frozen=True)
class SpawnedProcess:
    """A process created suspended. ``handle`` is an OS handle (an int); ``pid`` its id."""
    pid: int
    handle: int


@dataclass(frozen=True)
class FinishReport:
    census: int          # processes still in the job when the agent was found exited
    leak: bool           # census > 0 although agentd did not kill the job itself
    survivors: int       # processes still alive after the kill and the poll window (0 is the goal)
    killed: bool         # the job was terminated by finish()


class JobOsApi:
    """What the wrapper needs from the OS. The recording fake in the tests and Win32JobApi implement it."""

    def create_job(self) -> int:
        raise NotImplementedError

    def set_extended_limits(self, job: int, flags: int, active_process_limit: int, job_memory_bytes: int) -> None:
        raise NotImplementedError

    def set_cpu_rate(self, job: int, percent: int) -> None:
        raise NotImplementedError

    def spawn_suspended(self, argv: Sequence[str], cwd: Any, env: Optional[Mapping[str, str]],
                        stdout: Any = None, stderr: Any = None) -> SpawnedProcess:
        raise NotImplementedError

    def assign(self, job: int, proc: SpawnedProcess) -> None:
        raise NotImplementedError

    def resume(self, proc: SpawnedProcess) -> None:
        raise NotImplementedError

    def terminate_process(self, proc: SpawnedProcess, exit_code: int) -> None:
        raise NotImplementedError

    def exit_code(self, proc: SpawnedProcess) -> Optional[int]:
        """None while the process is running."""
        raise NotImplementedError

    def active_process_count(self, job: int) -> int:
        raise NotImplementedError

    def cpu_seconds(self, job: int) -> float:
        """User + kernel CPU of every process the job ever held."""
        raise NotImplementedError

    def process_ids(self, job: int) -> list:
        """PIDs currently in the job (for the LEAK log)."""
        raise NotImplementedError

    def terminate_job(self, job: int, exit_code: int) -> None:
        raise NotImplementedError

    def close_handle(self, handle: int) -> None:
        raise NotImplementedError


Spawner = Callable[..., SpawnedProcess]
EventSink = Callable[[Mapping[str, Any]], None]


class JobObject:
    """One job, one agent. Not thread safe: agentd drives each run from its own supervision loop."""

    def __init__(self, api: JobOsApi, limits: JobLimits = JobLimits(), *,
                 sink: Optional[EventSink] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self._api = api
        self.limits = limits
        self._sink = sink
        self._sleep = sleep
        self._job: Optional[int] = None
        self._proc: Optional[SpawnedProcess] = None
        self._closed = False
        self._killed = False
        self.leaks = 0

    # -- lifecycle
    def start(self, argv: Sequence[str], cwd: Any, env: Optional[Mapping[str, str]] = None, *,
              spawner: Optional[Spawner] = None, stdout: Any = None, stderr: Any = None) -> SpawnedProcess:
        """create job -> limits -> spawn suspended -> assign -> resume. Any failure tears everything down."""
        if self._job is not None:
            raise JobError("job already started")
        api = self._api
        try:
            self._job = api.create_job()
            api.set_extended_limits(self._job, self.limits.limit_flags, self.limits.active_process_limit,
                                    self.limits.job_memory_bytes)
            api.set_cpu_rate(self._job, self.limits.cpu_percent)
            spawn = spawner if spawner is not None else api.spawn_suspended
            self._proc = spawn(argv, cwd, env, stdout=stdout, stderr=stderr)
            api.assign(self._job, self._proc)
            api.resume(self._proc)
        except Exception as exc:
            self._teardown_after_failed_start()
            raise JobError(f"job start failed: {type(exc).__name__}: {exc}") from exc
        return self._proc

    def _teardown_after_failed_start(self) -> None:
        for step in (self._terminate_proc, self._close_job):
            try:
                step()
            except Exception as exc:  # teardown must not mask the original failure
                self._emit({"event": "teardown_error", "error": f"{type(exc).__name__}: {exc}"})

    def _terminate_proc(self) -> None:
        if self._proc is not None:
            self._api.terminate_process(self._proc, 1)
            self._api.close_handle(self._proc.handle)
            self._proc = None

    def _close_job(self) -> None:
        if self._job is not None and not self._closed:
            self._closed = True
            self._api.close_handle(self._job)

    # -- observation
    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    def exit_code(self) -> Optional[int]:
        if self._proc is None:
            return None
        return self._api.exit_code(self._proc)

    def census(self) -> int:
        if self._job is None or self._closed:
            return 0
        return self._api.active_process_count(self._job)

    def cpu_seconds(self) -> float:
        if self._job is None or self._closed:
            return 0.0
        return self._api.cpu_seconds(self._job)

    # -- teardown
    def kill(self, exit_code: int = 1) -> None:
        if self._job is None or self._closed or self._killed:
            return
        self._killed = True
        self._api.terminate_job(self._job, exit_code)

    def finish(self) -> FinishReport:
        """Census the job, treat a nonzero count after a natural exit as a LEAK, kill, poll, close."""
        census = self.census()
        if census > 0 and not self._killed:
            for _ in range(SETTLE_TRIES):
                self._sleep(SETTLE_S)
                census = self.census()
                if census == 0:
                    break
        leak = census > 0 and not self._killed
        if leak:
            self.leaks += 1
            self._emit({"event": "leak", "pid": self.pid, "census": census, "pids": self._pids()})
        killed = self._killed
        if leak:
            self.kill()
            killed = True
        survivors = 0
        if killed and self._job is not None and not self._closed:
            survivors = self._wait_for_empty()
            if survivors:
                self._emit({"event": "survivors_after_kill", "pid": self.pid, "census": survivors})
        try:
            if self._proc is not None:
                self._api.close_handle(self._proc.handle)
                self._proc = None
        finally:
            self._close_job()
        return FinishReport(census=census, leak=leak, survivors=survivors, killed=killed)

    def _pids(self) -> list:
        try:
            return self._api.process_ids(self._job)
        except Exception as exc:  # diagnostics only: a failure here must not hide the leak itself
            return [f"unavailable: {type(exc).__name__}"]

    def close(self) -> None:
        """Close the handles WITHOUT terminating: KILL_ON_JOB_CLOSE must do the killing (what happens if
        agentd itself dies). :meth:`finish` is the normal path."""
        try:
            if self._proc is not None:
                self._api.close_handle(self._proc.handle)
                self._proc = None
        finally:
            self._close_job()

    def _wait_for_empty(self) -> int:
        count = self.census()
        for _ in range(KILL_POLL_TRIES):
            if count == 0:
                return 0
            self._sleep(KILL_POLL_S)
            count = self.census()
        return count

    def _emit(self, event: Mapping[str, Any]) -> None:
        if self._sink is not None:
            self._sink(dict(event))


# ----------------------------------------------------------------------------- the real implementation (win32 only)

class Win32JobApi(JobOsApi):
    """ctypes implementation. Construct it only on Windows; the constructor itself makes no OS call."""

    PROCESS_ALL_ACCESS = 0x1F0FFF
    CREATE_SUSPENDED = 0x00000004
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    STILL_ACTIVE = 259
    JobObjectBasicAccountingInformation = 1
    JobObjectBasicProcessIdList = 3
    JobObjectExtendedLimitInformation = 9
    JobObjectCpuRateControlInformation = 15
    JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x1
    JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x4

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise JobError("Win32JobApi needs Windows")
        import ctypes
        from ctypes import wintypes
        self._ct = ctypes
        self._wt = wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32 = k32
        H = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.CreateJobObjectW.restype = H
        k32.SetInformationJobObject.argtypes = [H, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.QueryInformationJobObject.argtypes = [H, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                                  ctypes.POINTER(wintypes.DWORD)]
        k32.QueryInformationJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [H, H]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.TerminateJobObject.argtypes = [H, wintypes.UINT]
        k32.TerminateJobObject.restype = wintypes.BOOL
        k32.TerminateProcess.argtypes = [H, wintypes.UINT]
        k32.TerminateProcess.restype = wintypes.BOOL
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = H
        k32.GetExitCodeProcess.argtypes = [H, ctypes.POINTER(wintypes.DWORD)]
        k32.GetExitCodeProcess.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [H]
        k32.CloseHandle.restype = wintypes.BOOL
        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtResumeProcess.argtypes = [H]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        self._ntdll = ntdll
        self._popens: dict = {}
        self._define_structures()

    def _define_structures(self) -> None:
        ct, wt = self._ct, self._wt

        class BASIC_LIMIT(ct.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ct.c_int64), ("PerJobUserTimeLimit", ct.c_int64),
                        ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ct.c_size_t),
                        ("MaximumWorkingSetSize", ct.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                        ("Affinity", ct.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

        class IO_COUNTERS(ct.Structure):
            _fields_ = [(n, ct.c_uint64) for n in ("ReadOps", "WriteOps", "OtherOps",
                                                   "ReadBytes", "WriteBytes", "OtherBytes")]

        class EXTENDED_LIMIT(ct.Structure):
            _fields_ = [("Basic", BASIC_LIMIT), ("Io", IO_COUNTERS), ("ProcessMemoryLimit", ct.c_size_t),
                        ("JobMemoryLimit", ct.c_size_t), ("PeakProcessMemoryUsed", ct.c_size_t),
                        ("PeakJobMemoryUsed", ct.c_size_t)]

        class CPU_RATE(ct.Structure):
            _fields_ = [("ControlFlags", wt.DWORD), ("CpuRate", wt.DWORD)]

        class ACCOUNTING(ct.Structure):
            _fields_ = [("TotalUserTime", ct.c_int64), ("TotalKernelTime", ct.c_int64),
                        ("ThisPeriodTotalUserTime", ct.c_int64), ("ThisPeriodTotalKernelTime", ct.c_int64),
                        ("TotalPageFaultCount", wt.DWORD), ("TotalProcesses", wt.DWORD),
                        ("ActiveProcesses", wt.DWORD), ("TotalTerminatedProcesses", wt.DWORD)]

        self._EXTENDED_LIMIT, self._CPU_RATE, self._ACCOUNTING = EXTENDED_LIMIT, CPU_RATE, ACCOUNTING

    def _fail(self, what: str) -> JobError:
        return JobError(f"{what} failed (winerror {self._ct.get_last_error()})")

    def create_job(self) -> int:
        handle = self._k32.CreateJobObjectW(None, None)
        if not handle:
            raise self._fail("CreateJobObjectW")
        return int(handle)

    def set_extended_limits(self, job: int, flags: int, active_process_limit: int, job_memory_bytes: int) -> None:
        info = self._EXTENDED_LIMIT()
        info.Basic.LimitFlags = flags
        info.Basic.ActiveProcessLimit = active_process_limit
        info.JobMemoryLimit = job_memory_bytes
        ok = self._k32.SetInformationJobObject(job, self.JobObjectExtendedLimitInformation,
                                               self._ct.byref(info), self._ct.sizeof(info))
        if not ok:
            raise self._fail("SetInformationJobObject(extended limits)")

    def set_cpu_rate(self, job: int, percent: int) -> None:
        info = self._CPU_RATE()
        info.ControlFlags = (self.JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | self.JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP)
        info.CpuRate = percent * 100
        ok = self._k32.SetInformationJobObject(job, self.JobObjectCpuRateControlInformation,
                                               self._ct.byref(info), self._ct.sizeof(info))
        if not ok:
            raise self._fail("SetInformationJobObject(cpu rate)")

    def spawn_suspended(self, argv: Sequence[str], cwd: Any, env: Optional[Mapping[str, str]],
                        stdout: Any = None, stderr: Any = None) -> SpawnedProcess:
        import subprocess
        popen = subprocess.Popen(
            list(argv), cwd=str(cwd) if cwd is not None else None, env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=stdout if stdout is not None else subprocess.DEVNULL,
            stderr=stderr if stderr is not None else subprocess.DEVNULL,
            creationflags=self.CREATE_SUSPENDED | self.CREATE_NEW_PROCESS_GROUP)
        handle = self._k32.OpenProcess(self.PROCESS_ALL_ACCESS, False, popen.pid)
        if not handle:
            popen.kill()
            raise self._fail("OpenProcess")
        self._popens[popen.pid] = popen  # keeps the Popen alive so its own handle is not closed early
        return SpawnedProcess(pid=popen.pid, handle=int(handle))

    def assign(self, job: int, proc: SpawnedProcess) -> None:
        if not self._k32.AssignProcessToJobObject(job, proc.handle):
            raise self._fail("AssignProcessToJobObject")

    def resume(self, proc: SpawnedProcess) -> None:
        status = self._ntdll.NtResumeProcess(proc.handle)
        if status != 0:
            raise JobError(f"NtResumeProcess failed (NTSTATUS {status & 0xFFFFFFFF:#x})")

    def terminate_process(self, proc: SpawnedProcess, exit_code: int) -> None:
        self._k32.TerminateProcess(proc.handle, exit_code)

    def exit_code(self, proc: SpawnedProcess) -> Optional[int]:
        code = self._wt.DWORD()
        if not self._k32.GetExitCodeProcess(proc.handle, self._ct.byref(code)):
            raise self._fail("GetExitCodeProcess")
        return None if code.value == self.STILL_ACTIVE else int(code.value)

    def _accounting(self, job: int) -> Any:
        info = self._ACCOUNTING()
        ok = self._k32.QueryInformationJobObject(job, self.JobObjectBasicAccountingInformation,
                                                 self._ct.byref(info), self._ct.sizeof(info), None)
        if not ok:
            raise self._fail("QueryInformationJobObject(accounting)")
        return info

    def active_process_count(self, job: int) -> int:
        return int(self._accounting(job).ActiveProcesses)

    def cpu_seconds(self, job: int) -> float:
        info = self._accounting(job)
        return (info.TotalUserTime + info.TotalKernelTime) / 1e7  # 100 ns units

    def process_ids(self, job: int) -> list:
        ct, wt = self._ct, self._wt

        class PID_LIST(ct.Structure):
            _fields_ = [("Assigned", wt.DWORD), ("InList", wt.DWORD), ("Ids", ct.c_size_t * 128)]

        info = PID_LIST()
        ok = self._k32.QueryInformationJobObject(job, self.JobObjectBasicProcessIdList, ct.byref(info),
                                                 ct.sizeof(info), None)
        if not ok:
            raise self._fail("QueryInformationJobObject(process ids)")
        return [int(info.Ids[i]) for i in range(info.InList)]

    def terminate_job(self, job: int, exit_code: int) -> None:
        if not self._k32.TerminateJobObject(job, exit_code):
            raise self._fail("TerminateJobObject")

    def close_handle(self, handle: int) -> None:
        self._k32.CloseHandle(handle)

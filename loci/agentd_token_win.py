"""Restricted-token containment for read-write delegates (Tripwire-Governor slice S11, decision 3).

STATUS: the plan below is implemented against an injected :class:`TokenOsApi` and unit-tested with a
recording fake. :class:`Win32TokenApi` (ctypes + icacls) has NOT been exercised against a real token or a
real ACL: creating either is a live change to the machine and the slice forbids doing it from a test run
here. Until a real-Windows test shows a child under the token getting access-denied on ``~/.loci`` and the
audit directory, the path rules are DETECTION, not prevention, and read-write delegation stays OFF.

The plan (Chrome-sandbox style):

* a restricted token is derived from the caller's own token: privileges stripped (``DISABLE_MAX_PRIVILEGE``),
  the user SID and Administrators made deny-only, and a restricting-SID list that includes the
  ``RESTRICTED`` SID (``S-1-5-12``);
* an explicit DENY ACE for ``S-1-5-12`` is put on every protected path. Only a restricted token carries that
  SID, so the owner's normal processes are unaffected;
* an ALLOW ACE for ``S-1-5-12`` is put on the run's worktree so the delegate can write there;
* the ACEs are reference counted (two concurrent rw runs share them) and removed when the last run
  releases. A failure half way through ``prepare`` removes whatever it had added.

Whether the toolchains (node, git, Claude Code, Copilot CLI) survive the token is the open question the
owner's smoke test answers; the design's alternatives are a dedicated account or AppContainer.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Mapping, Optional, Sequence

from loci.agentd_job_win import SpawnedProcess

RESTRICTED_SID = "S-1-5-12"
EVERYONE_SID = "S-1-1-0"
USERS_SID = "S-1-5-32-545"
ADMINISTRATORS_SID = "S-1-5-32-544"
DENY_RIGHTS = "F"      # icacls full control: a denied token can neither read nor write
WORKTREE_RIGHTS = "M"  # icacls modify

DISABLE_MAX_PRIVILEGE = 0x1


class ContainmentError(RuntimeError):
    """Containment could not be set up, so the run does not start."""


class TokenOsApi:
    """What containment needs from the OS. The recording fake in the tests and Win32TokenApi implement it."""

    def add_deny_ace(self, path: str, sid: str, rights: str) -> None:
        raise NotImplementedError

    def add_allow_ace(self, path: str, sid: str, rights: str) -> None:
        raise NotImplementedError

    def remove_ace(self, path: str, sid: str) -> None:
        raise NotImplementedError

    def current_user_sid(self) -> str:
        raise NotImplementedError

    def create_restricted_token(self, flags: int, disable_sids: Sequence[str],
                                restricting_sids: Sequence[str]) -> int:
        raise NotImplementedError

    def spawn_suspended_as(self, token: int, argv: Sequence[str], cwd: Any,
                           env: Optional[Mapping[str, str]], stdout: Any = None,
                           stderr: Any = None) -> SpawnedProcess:
        raise NotImplementedError

    def close_handle(self, handle: int) -> None:
        raise NotImplementedError


def _key(path: Any) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


class RestrictedTokenContainment:
    """``prepare(worktree)`` before the run, ``spawn_suspended`` as the job's spawner, ``release`` after."""

    def __init__(self, api: TokenOsApi, protected: Sequence[Any], *,
                 exists: Any = os.path.exists, sid: str = RESTRICTED_SID,
                 extra_restricting: Sequence[str] = (EVERYONE_SID, USERS_SID)):
        self._api = api
        self._protected = tuple(os.fspath(p) for p in protected)
        self._exists = exists
        self._sid = sid
        self._restricting = (*extra_restricting, sid)
        self._lock = threading.Lock()
        self._refs = 0
        self._denied: list = []
        self._tokens: dict = {}
        self.skipped: list = []     # protected paths that did not exist, so no ACE was possible
        self.release_errors = 0

    def prepare(self, worktree: Any) -> None:
        wt = os.fspath(worktree)
        added_deny: list = []
        added_allow = False
        token: Optional[int] = None
        with self._lock:
            first = self._refs == 0
            try:
                if first:
                    for path in self._protected:
                        if not self._exists(path):
                            self.skipped.append(path)
                            continue
                        self._api.add_deny_ace(path, self._sid, DENY_RIGHTS)
                        added_deny.append(path)
                self._api.add_allow_ace(wt, self._sid, WORKTREE_RIGHTS)
                added_allow = True
                user = self._api.current_user_sid()
                token = self._api.create_restricted_token(
                    DISABLE_MAX_PRIVILEGE, (user, ADMINISTRATORS_SID), self._restricting)
            except Exception as exc:
                self._undo(added_deny, wt if added_allow else None, token)
                raise ContainmentError(f"containment setup failed: {type(exc).__name__}: {exc}") from exc
            if first:
                self._denied = added_deny
            self._refs += 1
            self._tokens[_key(wt)] = token

    def _undo(self, deny_paths: Sequence[str], allow_path: Optional[str], token: Optional[int]) -> None:
        for path in reversed(list(deny_paths)):
            self._safe(self._api.remove_ace, path, self._sid)
        if allow_path is not None:
            self._safe(self._api.remove_ace, allow_path, self._sid)
        if token is not None:
            self._safe(self._api.close_handle, token)

    def _safe(self, fn: Any, *args: Any) -> None:
        try:
            fn(*args)
        except Exception:
            self.release_errors += 1

    def spawn_suspended(self, argv: Sequence[str], cwd: Any, env: Optional[Mapping[str, str]],
                        stdout: Any = None, stderr: Any = None) -> SpawnedProcess:
        token = self._tokens.get(_key(cwd))
        if token is None:
            raise ContainmentError(f"no restricted token prepared for {cwd}")
        return self._api.spawn_suspended_as(token, argv, cwd, env, stdout=stdout, stderr=stderr)

    def release(self, worktree: Any) -> None:
        wt = os.fspath(worktree)
        with self._lock:
            token = self._tokens.pop(_key(wt), None)
            if token is None:
                return
            last = self._refs == 1
            self._refs -= 1
            deny = self._denied if last else []
            if last:
                self._denied = []
            self._undo(deny, wt, token)


# ----------------------------------------------------------------------------- the real implementation (win32 only)

class Win32TokenApi(TokenOsApi):
    """ctypes + icacls. UNEXERCISED against a real token or ACL (see the module docstring)."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise ContainmentError("Win32TokenApi needs Windows")

    def _icacls(self, path: str, *args: str) -> None:
        import subprocess
        done = subprocess.run(["icacls", path, *args], capture_output=True, text=True, timeout=120, check=False)
        if done.returncode != 0:
            raise ContainmentError(f"icacls {' '.join(args)} on {path} exited {done.returncode}: "
                                   f"{(done.stdout + done.stderr).strip()[:200]}")

    def add_deny_ace(self, path: str, sid: str, rights: str) -> None:
        self._icacls(path, "/deny", f"*{sid}:(OI)(CI)({rights})")

    def add_allow_ace(self, path: str, sid: str, rights: str) -> None:
        self._icacls(path, "/grant", f"*{sid}:(OI)(CI)({rights})")

    def remove_ace(self, path: str, sid: str) -> None:
        self._icacls(path, "/remove:d", f"*{sid}")
        self._icacls(path, "/remove:g", f"*{sid}")

    def current_user_sid(self) -> str:
        import subprocess
        done = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True,
                              timeout=30, check=False)
        parts = [p.strip('"') for p in done.stdout.strip().split(",")]
        if done.returncode != 0 or len(parts) < 2 or not parts[1].startswith("S-1-"):
            raise ContainmentError(f"cannot resolve the current user SID: {done.stdout.strip()[:100]}")
        return parts[1]

    def create_restricted_token(self, flags: int, disable_sids: Sequence[str],
                                restricting_sids: Sequence[str]) -> int:
        import ctypes
        from ctypes import wintypes
        adv = ctypes.WinDLL("advapi32", use_last_error=True)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        adv.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
        adv.CreateRestrictedToken.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
            ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.HANDLE)]

        class SID_AND_ATTRIBUTES(ctypes.Structure):
            _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

        def sid_array(sids: Sequence[str]) -> Any:
            arr = (SID_AND_ATTRIBUTES * max(len(sids), 1))()
            for i, text in enumerate(sids):
                ptr = ctypes.c_void_p()
                if not adv.ConvertStringSidToSidW(text, ctypes.byref(ptr)):
                    raise ContainmentError(f"bad SID {text!r} (winerror {ctypes.get_last_error()})")
                arr[i].Sid = ptr
            return arr

        TOKEN_ALL = 0xF01FF
        base = wintypes.HANDLE()
        if not adv.OpenProcessToken(k32.GetCurrentProcess(), TOKEN_ALL, ctypes.byref(base)):
            raise ContainmentError(f"OpenProcessToken failed (winerror {ctypes.get_last_error()})")
        try:
            new = wintypes.HANDLE()
            disable, restrict = sid_array(disable_sids), sid_array(restricting_sids)
            ok = adv.CreateRestrictedToken(base, flags, len(disable_sids), disable, 0, None,
                                           len(restricting_sids), restrict, ctypes.byref(new))
            if not ok:
                raise ContainmentError(f"CreateRestrictedToken failed (winerror {ctypes.get_last_error()})")
            return int(new.value)
        finally:
            k32.CloseHandle(base)

    def spawn_suspended_as(self, token: int, argv: Sequence[str], cwd: Any,
                           env: Optional[Mapping[str, str]], stdout: Any = None,
                           stderr: Any = None) -> SpawnedProcess:
        import ctypes
        import subprocess
        from ctypes import wintypes
        adv = ctypes.WinDLL("advapi32", use_last_error=True)

        class STARTUPINFOW(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR), ("lpDesktop", wintypes.LPWSTR),
                        ("lpTitle", wintypes.LPWSTR), ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                        ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
                        ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
                        ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                        ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                        ("lpReserved2", ctypes.c_void_p), ("hStdInput", wintypes.HANDLE),
                        ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE)]

        class PROCESS_INFORMATION(ctypes.Structure):
            _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                        ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]

        adv.CreateProcessAsUserW.argtypes = [
            wintypes.HANDLE, wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
            wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
            ctypes.POINTER(STARTUPINFOW), ctypes.POINTER(PROCESS_INFORMATION)]
        si, pi = STARTUPINFOW(), PROCESS_INFORMATION()
        si.cb = ctypes.sizeof(si)
        cmdline = subprocess.list2cmdline(list(argv))
        env_block = None
        if env is not None:
            env_block = ctypes.create_unicode_buffer("\0".join(f"{k}={v}" for k, v in env.items()) + "\0\0")
        flags = 0x4 | 0x200 | 0x400  # CREATE_SUSPENDED | NEW_PROCESS_GROUP | UNICODE_ENVIRONMENT
        ok = adv.CreateProcessAsUserW(token, None, cmdline, None, None, False, flags, env_block,
                                      os.fspath(cwd) if cwd is not None else None,
                                      ctypes.byref(si), ctypes.byref(pi))
        if not ok:
            raise ContainmentError(f"CreateProcessAsUserW failed (winerror {ctypes.get_last_error()})")
        ctypes.WinDLL("kernel32").CloseHandle(wintypes.HANDLE(pi.hThread))
        return SpawnedProcess(pid=int(pi.dwProcessId), handle=int(pi.hProcess))

    def close_handle(self, handle: int) -> None:
        import ctypes
        from ctypes import wintypes
        ctypes.WinDLL("kernel32").CloseHandle(wintypes.HANDLE(handle))

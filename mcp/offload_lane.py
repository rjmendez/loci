"""Command offload lanes: serve a role that maps to a tmux session from a configured command.

A lane is configured in ``~/.loci/backends.toml`` as ``[offload_lane.<name>]`` and is
referenced by putting ``<name>`` in ``[tmux_offload] role_sessions``. A lane name wins
over a tmux session of the same name; with no lane configured nothing here runs and the
tmux path is unchanged.

Contract with the command (the bma-offload ``offload.py`` tool is the reference):
  * ``<command> health``                                       exit 0 = a backend is up
  * ``<command> ask --prompt - [ask_args] [--local-only]``     prompt on stdin, answer on stdout

``--local-only`` is passed unless the lane sets ``allow_cloud = true``, so nothing reaches
a cloud lane unless the config says so. The command sees an allow-listed environment only
(no tokens or API keys unless named in ``env_passthrough``) and runs in its own process
group; on timeout the whole process tree is killed. Every function fails open: no exception
escapes, failures come back as ``ok=False`` with a reason so callers fall back as before.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

_CACHE: dict = {}                             # lane name -> (probed_at, available)
_LOCKS: dict = {}                             # lane name -> Lock (one in-flight health probe)
_LOCKS_GUARD = threading.Lock()
_clock: Callable[[], float] = time.monotonic  # seam for tests

# Environment a lane command may see. Everything else (LOCI_TOKEN, *_API_KEY, ...) is dropped
# unless the lane lists it in ``env_passthrough``. Compared upper-case; OFFLOAD_* always passes.
_ENV_ALLOW = frozenset({
    "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "SYSTEMDRIVE", "TEMP", "TMP", "TMPDIR",
    "USERPROFILE", "HOME", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "PUBLIC", "USERNAME", "USERDOMAIN", "COMPUTERNAME",
    "LANG", "LC_ALL", "LC_CTYPE", "TERM",
})


def child_env(lane: dict) -> dict:
    """Allow-listed copy of os.environ for the lane command."""
    extra = {str(k).upper() for k in lane.get("env_passthrough", [])}
    return {
        k: v for k, v in os.environ.items()
        if k.upper() in _ENV_ALLOW or k.upper().startswith("OFFLOAD_") or k.upper() in extra
    }


def _kill_tree(proc: "subprocess.Popen") -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10, check=False)
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _exec(cmd: list, input: Optional[str], timeout: float, env: dict) -> tuple:
    """Run ``cmd`` in its own process group; on timeout kill the whole tree and return promptly.

    Returns ``(returncode, stdout)``; raises ``subprocess.TimeoutExpired``.
    """
    kw: dict = dict(stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
                    text=True, encoding="utf-8", errors="replace")
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kw)
    try:
        out, _ = proc.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=2)
        except Exception:
            pass
        raise subprocess.TimeoutExpired(cmd, timeout)
    return proc.returncode, out or ""


_run = _exec                                  # seam for tests: (cmd, input, timeout, env) -> (rc, stdout)


def reset_cache() -> None:
    _CACHE.clear()


def _lock_for(name: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(name, threading.Lock())


def available(name: str, lane: dict) -> bool:
    """True when ``<command> health`` exits 0 within the lane's health timeout.

    Both outcomes are cached for ``cache_ttl_s`` (stamped after the probe), and concurrent
    callers share one in-flight probe.
    """
    with _lock_for(name):
        hit = _CACHE.get(name)
        if hit is not None and _clock() - hit[0] < float(lane["cache_ttl_s"]):
            return hit[1]
        try:
            rc, _out = _run(list(lane["command"]) + ["health"], None,
                            float(lane["health_timeout_s"]), child_env(lane))
            ok = rc == 0
        except Exception:
            ok = False
        _CACHE[name] = (_clock(), ok)
        return ok


def last_known(name: str, lane: dict) -> Optional[bool]:
    """The cached health result if it is at most 5 TTLs old, else None. Never runs a command."""
    hit = _CACHE.get(name)
    if hit is not None and _clock() - hit[0] < 5 * float(lane["cache_ttl_s"]):
        return hit[1]
    return None


def ask(lane: dict, prompt: str) -> dict:
    """Run one prompt through the lane. Returns ``{'ok','text','why'}``; never raises."""
    cmd = list(lane["command"]) + ["ask", "--prompt", "-"] + list(lane.get("ask_args", []))
    if not lane["allow_cloud"]:
        cmd.append("--local-only")
    try:
        rc, stdout = _run(cmd, prompt, float(lane["timeout_s"]), child_env(lane))
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": "", "why": f"command lane timed out after {lane['timeout_s']}s"}
    except Exception as exc:
        return {"ok": False, "text": "", "why": f"command lane failed to run: {type(exc).__name__}: {exc}"[:200]}
    if rc != 0:
        return {"ok": False, "text": "", "why": f"command lane exited {rc}"}
    text, why = _answer(stdout or "")
    if why:
        return {"ok": False, "text": "", "why": why}
    if not text.strip():
        return {"ok": False, "text": "", "why": "command lane returned no text"}
    return {"ok": True, "text": text[: int(lane["max_chars"])], "why": ""}


def _answer(stdout: str) -> tuple:
    """``(text, why)``. A JSON object must carry ``answer``; any non-object stdout is the text itself."""
    try:
        obj = json.loads(stdout)
    except ValueError:
        return stdout.strip(), ""
    if isinstance(obj, dict):
        if "answer" not in obj:
            detail = str(obj.get("error", ""))[:120]
            return "", ("command lane reply had no answer" + (f": {detail}" if detail else ""))
        return str(obj["answer"] or "").strip(), ""
    return stdout.strip(), ""


def lane_for(name: Optional[str]) -> Optional[dict]:
    """The configured lane called ``name``, or None."""
    if not name:
        return None
    try:
        import backends
        return backends.offload_lane(name)
    except Exception:
        return None

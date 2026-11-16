"""Command offload lanes: serve a role that maps to a tmux session from a configured command.

A lane is configured in ``~/.loci/backends.toml`` as ``[offload_lane.<name>]`` and is
referenced by putting ``<name>`` in ``[tmux_offload] role_sessions``. A lane name wins
over a tmux session of the same name; with no lane configured nothing here runs and the
tmux path is unchanged.

Contract with the command (the bma-offload ``offload.py`` tool is the reference):
  * ``<command> health``                           exit 0 = a backend is up
  * ``<command> ask --prompt - [--local-only]``    prompt on stdin, answer on stdout

``--local-only`` is passed unless the lane sets ``allow_cloud = true``, so nothing reaches
a cloud lane unless the config says so. Every function fails open: no exception escapes,
failures come back as ``ok=False`` with a reason so callers fall back as before.
"""
from __future__ import annotations

import json
import subprocess
import time
from typing import Callable, Optional

_CACHE: dict = {}                             # lane name -> (checked_at, available)
_clock: Callable[[], float] = time.monotonic  # seam for tests
_run = subprocess.run                         # seam for tests


def reset_cache() -> None:
    _CACHE.clear()


def available(name: str, lane: dict) -> bool:
    """True when ``<command> health`` exits 0 within the lane's health timeout (cached)."""
    now = _clock()
    hit = _CACHE.get(name)
    if hit is not None and now - hit[0] < float(lane["cache_ttl_s"]):
        return hit[1]
    try:
        proc = _run(
            list(lane["command"]) + ["health"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=float(lane["health_timeout_s"]),
            check=False,
        )
        ok = proc.returncode == 0
    except Exception:
        ok = False
    _CACHE[name] = (now, ok)
    return ok


def ask(lane: dict, prompt: str) -> dict:
    """Run one prompt through the lane. Returns ``{'ok','text','why'}``; never raises."""
    cmd = list(lane["command"]) + ["ask", "--prompt", "-"]
    if not lane["allow_cloud"]:
        cmd.append("--local-only")
    try:
        proc = _run(
            cmd,
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(lane["timeout_s"]),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": "", "why": f"command lane timed out after {lane['timeout_s']}s"}
    except Exception as exc:
        return {"ok": False, "text": "", "why": f"command lane failed to run: {type(exc).__name__}: {exc}"[:200]}
    if proc.returncode != 0:
        return {"ok": False, "text": "", "why": f"command lane exited {proc.returncode}"}
    text = _answer(proc.stdout or "")
    if not text.strip():
        return {"ok": False, "text": "", "why": "command lane returned no text"}
    return {"ok": True, "text": text[: int(lane["max_chars"])], "why": ""}


def _answer(stdout: str) -> str:
    """The offload tool prints a JSON object with an ``answer``; any other stdout is the text itself."""
    try:
        obj = json.loads(stdout)
    except ValueError:
        return stdout.strip()
    if isinstance(obj, dict) and "answer" in obj:
        return str(obj["answer"] or "").strip()
    return stdout.strip()


def lane_for(name: Optional[str]) -> Optional[dict]:
    """The configured lane called ``name``, or None."""
    if not name:
        return None
    try:
        import backends
        return backends.offload_lane(name)
    except Exception:
        return None

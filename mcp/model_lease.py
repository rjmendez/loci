"""Model leases: borrow GPU headroom from Ollama for a special job, then give it back.

A job that needs a card to itself (a FlyBrain evolution run, a big batch on a larger model)
asks for ``need_gb`` of free VRAM on ONE GPU. The lease unloads resident models, worst
first, until some single GPU has that much free, hands the job a lease id, and on release
(or expiry) loads back what it evicted.

Who may be evicted, and in what order (see ``plan``):

* never a pinned or ``evictable = false`` pool entry (the embedder is pinned by ``init``),
  and never a model a Loci call is using right now (``inflight``);
* with ``priority = "normal"`` never a model that is the current primary for a pooled role
  (the one ``model_pool.pick`` would return): only spare models go;
* with ``priority = "critical"`` primaries may go too, still not pinned or in-flight;
* order: models outside the pool first, then the worst pool rank first, larger first on ties.

Progress is verified, not assumed. After each eviction the lease waits for the model to leave
``/api/ps`` and for ``nvidia-smi`` to show the memory returned, and stops as soon as one GPU
has ``need_gb`` free. If everything evictable is gone and it still does not fit, the lease
puts back what it took and reports the shortfall: a failed lease leaves the box as it found it.
Without ``nvidia-smi`` it cannot verify, so it evicts every eligible model and says so
(``verified: false``).

While a lease is active the pool treats the evicted models as unavailable so a Loci call does
not reload one into the headroom the job just reserved. Leases expire (default 30 min) and
are reaped lazily, so a crashed job cannot hold the GPU forever.

State: ``<data home>/leases/model_leases.json`` (ids, names, sizes, times; no text).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger("loci-mcp.model_lease")

DEFAULT_TTL_S = 1800
DEFAULT_WAIT_S = 60.0
_POLL_S = 0.5
_UNLOAD_TIMEOUT_S = 30.0
_LOAD_TIMEOUT_S = 180.0
_RESTORE_KEEP_ALIVE = "30m"
_HELD_OUT_TTL_S = 2.0
PRIORITIES = ("normal", "critical")

_lock = threading.RLock()


# ------------------------------------------------------------------------------ in-flight

_inflight: dict[str, int] = {}


@contextlib.contextmanager
def inflight(model: str):
    """Mark ``model`` as in use by a Loci call so a lease will not evict it mid-request."""
    with _lock:
        _inflight[model] = _inflight.get(model, 0) + 1
    try:
        yield
    finally:
        with _lock:
            n = _inflight.get(model, 1) - 1
            if n <= 0:
                _inflight.pop(model, None)
            else:
                _inflight[model] = n


def inflight_models() -> set[str]:
    with _lock:
        return set(_inflight)


# --------------------------------------------------------------------------------- types

@dataclass
class Victim:
    name: str
    size_gb: float
    reason: str


@dataclass
class Plan:
    need_gb: float
    priority: str
    victims: list[Victim] = field(default_factory=list)
    protected: list[tuple[str, str]] = field(default_factory=list)     # (name, why)


# --------------------------------------------------------------------------------- pure plan

def _best_rank(name: str, pool_entries: Iterable) -> Optional[float]:
    ranks = []
    for e in pool_entries:
        if e.name == name or f"{e.name}:latest" == name or e.name == f"{name}:latest":
            ranks.extend(e.rank_for(r) for r in e.roles)
    return min(ranks) if ranks else None


def _entry(name: str, pool_entries: Iterable):
    for e in pool_entries:
        if e.name == name or f"{e.name}:latest" == name or e.name == f"{name}:latest":
            return e
    return None


def plan(resident: list[dict], need_gb: float, priority: str, pool_entries: list,
         primaries: dict[str, str], busy: set[str]) -> Plan:
    """Who would be evicted, in order, and who is protected and why. Pure."""
    priority = priority if priority in PRIORITIES else "normal"
    primary_for: dict[str, list[str]] = {}
    for role, name in primaries.items():
        if name:
            primary_for.setdefault(name, []).append(role)
    out = Plan(need_gb=need_gb, priority=priority)
    ranked: list[tuple[tuple, Victim]] = []
    for m in resident:
        name = str(m.get("name") or "")
        if not name:
            continue
        size = float(m.get("size_gb") or 0.0)
        if size <= 0.0:
            out.protected.append((name, "holds no VRAM (unloading it would free nothing)"))
            continue
        entry = _entry(name, pool_entries)
        if entry is not None and (entry.pinned or not entry.evictable):
            out.protected.append((name, "pinned"))
            continue
        if name in busy:
            out.protected.append((name, "in use by a Loci call"))
            continue
        roles = primary_for.get(name, [])
        if roles and priority != "critical":
            out.protected.append((name, "primary for " + ", ".join(sorted(roles))))
            continue
        rank = _best_rank(name, pool_entries)
        reason = "not in the pool" if rank is None else f"pool rank {rank:g}"
        if roles:
            reason += " (primary, critical lease)"
        # outside the pool first, then worst (highest) rank, then the bigger one (frees more)
        key = (0 if rank is None else 1, -(rank if rank is not None else 0.0), -size, name)
        ranked.append((key, Victim(name=name, size_gb=size, reason=reason)))
    out.victims = [v for _, v in sorted(ranked, key=lambda t: t[0])]
    return out


# ------------------------------------------------------------------------------- transport

def _base_url() -> str:
    try:
        import model_pool
        return model_pool._gen_url().rstrip("/")
    except Exception:
        return ""


def _http_json(url: str, body: Optional[dict] = None, timeout: float = 5.0) -> Optional[dict]:
    try:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                     headers={"Content-Type": "application/json"} if data is not None else {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-configured URL
            raw = resp.read().decode("utf-8", "replace")
        return json.loads(raw) if raw.strip() else {}
    except Exception as exc:
        logger.debug("model_lease: %s failed: %r", url, exc)
        return None


def resident_detail(base_url: str) -> Optional[list[dict]]:
    """Resident models with their VRAM use in GB, or None when Ollama is unreachable."""
    data = _http_json(base_url + "/api/ps", timeout=5.0)
    if data is None:
        return None
    out = []
    for m in data.get("models") or []:
        size = m.get("size_vram") or 0            # VRAM only: a CPU-resident model frees none
        out.append({"name": str(m.get("name") or m.get("model") or ""), "size_gb": float(size) / 1e9})
    return out


def gpu_free_gb() -> Optional[list[float]]:
    """Free VRAM per GPU in GB via nvidia-smi (``LOCI_LEASE_GPU_CMD`` overrides), None if unavailable."""
    cmd = os.environ.get("LOCI_LEASE_GPU_CMD")
    argv = cmd.split() if cmd else ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=5)
        if proc.returncode != 0:
            return None
        vals = [float(x.strip().split()[0]) / 1024.0 for x in proc.stdout.splitlines() if x.strip()]
        return vals or None
    except Exception as exc:
        logger.debug("model_lease: nvidia-smi unavailable: %r", exc)
        return None


def _is_embed(name: str) -> bool:
    try:
        import model_pool
        return model_pool.classify_model(name) == ("embed",)
    except Exception:
        return "embed" in name.lower()


def unload(base_url: str, name: str) -> bool:
    """Ask Ollama to drop ``name`` now (keep_alive 0). True when the request was accepted."""
    if _is_embed(name):
        r = _http_json(base_url + "/api/embed", {"model": name, "input": "x", "keep_alive": 0}, _UNLOAD_TIMEOUT_S)
    else:
        r = _http_json(base_url + "/api/generate", {"model": name, "prompt": "", "stream": False,
                                                    "keep_alive": 0}, _UNLOAD_TIMEOUT_S)
    return r is not None


def load(base_url: str, name: str, keep_alive: str = _RESTORE_KEEP_ALIVE) -> bool:
    """Load ``name`` back into memory (a zero-token request). True when Ollama accepted it."""
    if _is_embed(name):
        r = _http_json(base_url + "/api/embed", {"model": name, "input": "warm", "keep_alive": keep_alive},
                       _LOAD_TIMEOUT_S)
    else:
        r = _http_json(base_url + "/api/generate", {"model": name, "prompt": "", "stream": False,
                                                    "keep_alive": keep_alive,
                                                    "options": {"num_predict": 0}}, _LOAD_TIMEOUT_S)
    return r is not None


# --------------------------------------------------------------------------------- ledger

def _ledger_path() -> Path:
    mem = os.environ.get("LOCI_MEMORY_DIR") or str(Path.home() / ".loci" / "memory-sessions")
    return Path(mem).parent / "leases" / "model_leases.json"


_held_cache: tuple[float, float, set] = (0.0, -1.0, set())


def _read() -> dict:
    try:
        data = json.loads(_ledger_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write(data: dict) -> None:
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _restore_keep_alive() -> str:
    try:
        import model_pool
        return str(model_pool._models_cfg().get("restore_keep_alive") or _RESTORE_KEEP_ALIVE)
    except Exception:
        return _RESTORE_KEEP_ALIVE


def active_leases(now: Optional[float] = None) -> dict:
    now = time.time() if now is None else now
    return {k: v for k, v in _read().items() if v.get("expires_at", 0) > now}


def held_out() -> set[str]:
    """Models evicted by an active lease. The pool treats these as unavailable. Fail-open: empty."""
    global _held_cache
    try:
        path = _ledger_path()
        mtime = path.stat().st_mtime if path.exists() else -1.0
        now = time.monotonic()
        if now - _held_cache[0] < _HELD_OUT_TTL_S and mtime == _held_cache[1]:
            return set(_held_cache[2])
        names: set[str] = set()
        for rec in active_leases().values():
            names.update(rec.get("evicted") or [])
        _held_cache = (now, mtime, names)
        return set(names)
    except Exception:
        return set()


# ------------------------------------------------------------------------- acquire/release

def _primaries() -> dict[str, str]:
    """Role -> the model pick() would return right now, ignoring existing leases."""
    try:
        import model_pool
        entries = model_pool.entries()
        inv, res = model_pool.inventory(), model_pool.resident_models()
        roles = sorted({r for e in entries for r in e.roles})
        return {r: model_pool.rank_role(r, pool=entries, inv=inv, resident=res, held_out=set()).chosen
                for r in roles}
    except Exception:
        return {}


def _pool_entries() -> list:
    try:
        import model_pool
        return model_pool.entries()
    except Exception:
        return []


def _wait_for(pred: Callable[[], bool], wait_s: float, sleep: Callable[[float], None]) -> bool:
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        if pred():
            return True
        if time.monotonic() >= deadline:
            return False
        sleep(_POLL_S)


def acquire(job: str, need_gb: float, priority: str = "normal", ttl_s: int = DEFAULT_TTL_S,
            wait_s: float = DEFAULT_WAIT_S, restore: bool = True, base_url: Optional[str] = None,
            _sleep: Callable[[float], None] = time.sleep) -> dict:
    """Free ``need_gb`` on one GPU for ``job``. Returns a dict; ``granted`` says whether it worked."""
    job = (job or "").strip() or "unnamed-job"
    priority = priority if priority in PRIORITIES else "normal"
    need_gb = float(need_gb)
    base = (base_url if base_url is not None else _base_url()).rstrip("/")
    with _lock:
        reap(base_url=base, _sleep=_sleep)
        resident = resident_detail(base) if base else None
        if resident is None:
            return {"granted": False, "job": job, "reason": "Ollama unreachable; nothing was changed"}
        pl = plan(resident, need_gb, priority, _pool_entries(), _primaries(), inflight_models())
        result = {"granted": False, "job": job, "need_gb": need_gb, "priority": priority,
                  "evicted": [], "protected": [{"name": n, "why": w} for n, w in pl.protected]}
        free = gpu_free_gb()
        verified = free is not None
        result["verified"] = verified

        def fits() -> bool:
            f = gpu_free_gb()
            return bool(f) and max(f) >= need_gb

        evicted: list[dict] = []
        if verified and max(free) >= need_gb:
            pass                                     # already room: no eviction, still record the lease
        elif verified:
            for v in pl.victims:
                if not unload(base, v.name):
                    continue
                evicted.append({"name": v.name, "size_gb": round(v.size_gb, 2), "reason": v.reason})
                gone = _wait_for(lambda: v.name not in {m["name"] for m in (resident_detail(base) or [])},
                                 wait_s, _sleep)
                if gone and _wait_for(fits, 2.0, _sleep):
                    break
            if not fits():
                _undo(base, evicted, _sleep)
                best = max(gpu_free_gb() or [0.0])
                result.update(evicted=[], reason=f"insufficient: {best:.1f} GB free on the best GPU after "
                                                 f"evicting every eligible model, need {need_gb:g} GB; restored")
                return result
        else:
            for v in pl.victims:                      # no telemetry: evict everything eligible, unverified
                if unload(base, v.name):
                    evicted.append({"name": v.name, "size_gb": round(v.size_gb, 2), "reason": v.reason})
        lease_id = uuid.uuid4().hex[:12]
        now = time.time()
        record = {"job": job, "priority": priority, "need_gb": need_gb, "granted_at": now,
                  "expires_at": now + max(1, int(ttl_s)), "restore": bool(restore), "verified": verified,
                  "evicted": [e["name"] for e in evicted], "evicted_detail": evicted}
        ledger = _read()
        ledger[lease_id] = record
        _write(ledger)
        result.update(granted=True, lease_id=lease_id, evicted=evicted,
                      free_gb_after=[round(x, 1) for x in (gpu_free_gb() or [])],
                      expires_at=record["expires_at"])
        return result


def _undo(base: str, evicted: list[dict], sleep: Callable[[float], None]) -> None:
    keep = _restore_keep_alive()
    for e in reversed(evicted):                       # most valuable (evicted last) first
        load(base, e["name"], keep)


def release(lease_id: str, base_url: Optional[str] = None) -> dict:
    """End a lease and load back the models it evicted (unless it was taken with restore=False)."""
    base = (base_url if base_url is not None else _base_url()).rstrip("/")
    with _lock:
        ledger = _read()
        rec = ledger.pop(lease_id, None)
        if rec is None:
            return {"released": False, "reason": "no such lease (already released or expired)"}
        _write(ledger)
        restored, failed = [], []
        if rec.get("restore", True) and base:
            keep = _restore_keep_alive()
            for e in reversed(rec.get("evicted_detail") or [{"name": n} for n in rec.get("evicted", [])]):
                (restored if load(base, e["name"], keep) else failed).append(e["name"])
        return {"released": True, "job": rec.get("job"), "restored": restored, "restore_failed": failed}


def reap(base_url: Optional[str] = None, now: Optional[float] = None,
         _sleep: Callable[[float], None] = time.sleep) -> list[str]:
    """Release every expired lease (restoring what it evicted). Returns the ids released."""
    now = time.time() if now is None else now
    ledger = _read()
    expired = [k for k, v in ledger.items() if v.get("expires_at", 0) <= now]
    for k in expired:
        release(k, base_url=base_url)
    return expired


def status(base_url: Optional[str] = None) -> dict:
    base = (base_url if base_url is not None else _base_url()).rstrip("/")
    reaped = reap(base_url=base)
    res = resident_detail(base) if base else None
    return {"leases": active_leases(), "reaped": reaped,
            "resident": res if res is not None else "unreachable",
            "gpu_free_gb": [round(x, 1) for x in (gpu_free_gb() or [])] or "unavailable"}


@contextlib.contextmanager
def lease(job: str, need_gb: float, **kw):
    """``with lease("evo-run", 9.0) as grant:`` -- releases (and restores) on exit even on error."""
    grant = acquire(job, need_gb, **kw)
    try:
        yield grant
    finally:
        if grant.get("granted"):
            release(grant["lease_id"])


def _main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "status"
    if cmd == "status":
        print(json.dumps(status(), indent=1))
    elif cmd == "plan" and len(argv) >= 2:
        base = _base_url()
        res = resident_detail(base) or []
        pri = argv[2] if len(argv) > 2 else "normal"
        pl = plan(res, float(argv[1]), pri, _pool_entries(), _primaries(), inflight_models())
        print(json.dumps({"victims": [v.__dict__ for v in pl.victims],
                          "protected": [{"name": n, "why": w} for n, w in pl.protected]}, indent=1))
    elif cmd == "acquire" and len(argv) >= 3:
        print(json.dumps(acquire(argv[1], float(argv[2]), argv[3] if len(argv) > 3 else "normal"), indent=1))
    elif cmd == "release" and len(argv) >= 2:
        print(json.dumps(release(argv[1]), indent=1))
    else:
        print("usage: model_lease.py status | plan NEED_GB [priority] | acquire JOB NEED_GB [priority] | release ID")
        return 2
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))

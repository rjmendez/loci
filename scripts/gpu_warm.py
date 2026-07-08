#!/usr/bin/env python3
"""GPU warm-keeper — pin the hot Ollama models resident so a warm single-stream
never re-eats the ~70s cold load.

Why this exists (session grounding):
  - [gen] Local generation shipped on qwen2.5:3b (mcp/llm_local.py), pinned via keep_alive.
  - [retrieval] Embeddings run via Ollama nomic-embed-text (768-dim), "warm on GPU".
  - [substrate/llm_local] COLD-LOAD is ~70s; without keep_alive every call re-eats it.
  - Ollama lives at OLLAMA_BASE_URL (fallback OLLAMA_URL) — same convention as
    mcp/embed_ops.py and mcp/llm_local.py.

What it does: issue a minimal /api/generate (qwen2.5:3b) and /api/embed (nomic-embed-text)
call with keep_alive set to a long TTL (-1 = never unload, or a configurable duration),
which forces Ollama to load + hold each model resident. Then report /api/ps residency and
basic GPU state.

Modes:
  - one-shot (default): pin both models once, print residency + GPU state, exit.
  - --loop: re-pin every N seconds forever (a lightweight keeper). With keep_alive=-1 a
    re-pin is cheap (models already resident) but the loop also re-loads anything that was
    evicted (e.g. another job grabbed VRAM), so it self-heals.

Fail-open [pattern:fail-open]: an unreachable / erroring Ollama NEVER raises. One-shot
prints a degraded note and exits 0; the loop logs the degraded tick and keeps going.

Injectable [pattern:injectable]: `post_fn(url, json, timeout) -> obj-with-.json()/.raise_for_status()`
defaults to None -> lazily resolved to `requests.post` at call time, so importing this
module never hard-requires `requests` and tests stub the HTTP layer with NO live Ollama.

Grounding is SILENT on: exact keep_alive TTL policy and re-pin interval — chosen here as
keep_alive=-1 (indefinite) and a 240s default loop interval; both are configurable.
Grounding is SILENT on a torch/nvidia-smi call being desired for GPU state, so GPU state is
best-effort: Ollama /api/ps (authoritative for what Ollama holds) plus an optional
nvidia-smi probe that is itself fail-open.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Callable, Optional

# Same base-URL convention as embed_ops.py / llm_local.py.
_OLLAMA = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_URL") or ""

# The two hot models to keep resident. [gen] qwen2.5:3b, [retrieval] nomic-embed-text.
_GEN_MODEL = os.environ.get("WARM_GEN_MODEL", "qwen2.5:3b")
_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# Generous timeout: a cold load is ~70s [substrate]; 120s covers cold-load + tiny gen.
_TIMEOUT = float(os.environ.get("OLLAMA_WARM_TIMEOUT", "120"))

# keep_alive: -1 = never unload. Configurable via env; accepts "-1", "30m", "2h", etc.
_KEEP_ALIVE = os.environ.get("WARM_KEEP_ALIVE", "-1")

# Default re-pin cadence for --loop (grounding silent; 240s is a light keeper).
_DEFAULT_INTERVAL = float(os.environ.get("WARM_INTERVAL", "240"))


def _resolve_post(post_fn: Optional[Callable]) -> Optional[Callable]:
    """Lazily resolve the HTTP poster. Returns None if requests is unavailable
    (fail-open: caller treats None as 'Ollama unreachable')."""
    if post_fn is not None:
        return post_fn
    try:
        import requests
        return requests.post
    except Exception:
        return None


def _coerce_keep_alive(value):
    """keep_alive of "-1" should go to Ollama as the int -1 (indefinite). Duration
    strings like "30m" pass through untouched."""
    if value in ("-1", -1):
        return -1
    return value


def pin_model(model: str, kind: str, keep_alive=_KEEP_ALIVE,
              post_fn: Optional[Callable] = None) -> dict:
    """Force one model resident via a minimal request with keep_alive set.

    kind: "gen" -> /api/generate (empty-ish prompt, num_predict=0 so it only loads);
          "embed" -> /api/embed (one tiny input).
    Returns {model, kind, ok, keep_alive, error?}. Fail-open: never raises.
    """
    result = {"model": model, "kind": kind, "ok": False,
              "keep_alive": _coerce_keep_alive(keep_alive)}
    if not _OLLAMA:
        result["error"] = "no OLLAMA_BASE_URL/OLLAMA_URL configured"
        return result
    post = _resolve_post(post_fn)
    if post is None:
        result["error"] = "requests unavailable"
        return result

    ka = _coerce_keep_alive(keep_alive)
    if kind == "embed":
        url = f"{_OLLAMA}/api/embed"
        body = {"model": model, "input": "warm", "keep_alive": ka}
    else:
        url = f"{_OLLAMA}/api/generate"
        # num_predict=0 -> load the model but emit no tokens; cheapest possible pin.
        body = {"model": model, "prompt": "warm", "stream": False,
                "keep_alive": ka, "options": {"num_predict": 0}}
    try:
        r = post(url, json=body, timeout=_TIMEOUT)
        r.raise_for_status()
        r.json()  # ensure a well-formed body; value itself is unused for a pin
        result["ok"] = True
    except Exception as e:  # fail-open [pattern:fail-open]
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def ollama_ps(post_fn: Optional[Callable] = None) -> dict:
    """Query /api/ps for what Ollama currently holds resident. Fail-open.

    Returns {ok, models:[{name, size_vram?, expires_at?}], error?}.
    """
    out = {"ok": False, "models": []}
    if not _OLLAMA:
        out["error"] = "no OLLAMA_BASE_URL/OLLAMA_URL configured"
        return out
    post = _resolve_post(post_fn)
    if post is None:
        out["error"] = "requests unavailable"
        return out
    # /api/ps is a GET in Ollama; we go through the injected poster for stub-ability and
    # fall back to requests.get only for the real (non-stubbed) path.
    try:
        try:
            import requests
            r = requests.get(f"{_OLLAMA}/api/ps", timeout=min(_TIMEOUT, 15))
        except Exception:
            # If the real requests import fails but a stub poster was injected, use it.
            r = post(f"{_OLLAMA}/api/ps", json=None, timeout=min(_TIMEOUT, 15))
        r.raise_for_status()
        data = r.json() or {}
        models = []
        for m in data.get("models", []) or []:
            models.append({
                "name": m.get("name") or m.get("model"),
                "size_vram": m.get("size_vram"),
                "expires_at": m.get("expires_at"),
            })
        out["ok"] = True
        out["models"] = models
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def gpu_state() -> dict:
    """Best-effort GPU state via nvidia-smi. Fail-open: returns {available:False} if
    nvidia-smi is missing or errors. Grounding is silent on wanting a torch probe here,
    so we keep this dependency-free and optional."""
    out = {"available": False, "gpus": []}
    exe = shutil.which("nvidia-smi")
    if not exe:
        out["note"] = "nvidia-smi not found"
        return out
    try:
        q = ("--query-gpu=index,name,memory.used,memory.total,utilization.gpu"
             " --format=csv,noheader,nounits")
        p = subprocess.run([exe] + q.split(), capture_output=True, text=True, timeout=10)
        if p.returncode != 0:
            out["note"] = "nvidia-smi returned non-zero"
            return out
        for line in p.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 5:
                out["gpus"].append({
                    "index": parts[0], "name": parts[1],
                    "mem_used_mib": parts[2], "mem_total_mib": parts[3],
                    "util_pct": parts[4],
                })
        out["available"] = bool(out["gpus"])
    except Exception as e:
        out["note"] = f"{type(e).__name__}: {e}"
    return out


def warm_once(keep_alive=_KEEP_ALIVE, post_fn: Optional[Callable] = None,
              include_gpu: bool = True) -> dict:
    """Pin both hot models once and gather residency + GPU state.

    Returns a well-formed report dict. degraded=True if either pin failed or Ollama is
    unreachable. NEVER raises [pattern:fail-open].
    """
    pins = [
        pin_model(_GEN_MODEL, "gen", keep_alive=keep_alive, post_fn=post_fn),
        pin_model(_EMBED_MODEL, "embed", keep_alive=keep_alive, post_fn=post_fn),
    ]
    ps = ollama_ps(post_fn=post_fn)
    degraded = (not all(p["ok"] for p in pins)) or (not ps["ok"])
    report = {
        "ts": time.time(),
        "ollama": _OLLAMA or None,
        "pins": pins,
        "ps": ps,
        "degraded": degraded,
    }
    if include_gpu:
        report["gpu"] = gpu_state()
    return report


def _print_report(report: dict) -> None:
    """Human-readable one-shot / per-tick summary to stdout."""
    if report.get("degraded"):
        bad = [f'{p["kind"]}:{p["model"]} ({p.get("error", "?")})'
               for p in report["pins"] if not p["ok"]]
        if not report["ps"]["ok"]:
            bad.append(f'ps ({report["ps"].get("error", "?")})')
        print(f"[gpu_warm] DEGRADED — Ollama at {report['ollama'] or '<unset>'} "
              f"unreachable or partial: {'; '.join(bad) or 'see report'}")
    else:
        pinned = ", ".join(f'{p["model"]} (keep_alive={p["keep_alive"]})'
                           for p in report["pins"])
        print(f"[gpu_warm] pinned: {pinned}")
    resident = report["ps"].get("models") or []
    if resident:
        print("[gpu_warm] /api/ps resident: " +
              ", ".join(f'{m["name"]}' + (f' [{m["size_vram"]}B vram]'
                        if m.get("size_vram") else "") for m in resident))
    elif report["ps"]["ok"]:
        print("[gpu_warm] /api/ps resident: (none)")
    gpu = report.get("gpu") or {}
    if gpu.get("available"):
        for g in gpu["gpus"]:
            print(f"[gpu_warm] GPU{g['index']} {g['name']}: "
                  f"{g['mem_used_mib']}/{g['mem_total_mib']} MiB, util {g['util_pct']}%")
    elif gpu.get("note"):
        print(f"[gpu_warm] GPU state: {gpu['note']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Keep hot Ollama models warm on GPU.")
    ap.add_argument("--loop", action="store_true",
                    help="re-pin forever (lightweight keeper) instead of one-shot")
    ap.add_argument("--interval", type=float, default=_DEFAULT_INTERVAL,
                    help=f"seconds between re-pins in --loop mode (default {_DEFAULT_INTERVAL})")
    ap.add_argument("--keep-alive", default=_KEEP_ALIVE,
                    help='keep_alive TTL sent to Ollama: "-1" (indefinite) or "30m","2h"')
    ap.add_argument("--json", action="store_true", help="emit the raw report as JSON")
    ap.add_argument("--no-gpu", action="store_true", help="skip the nvidia-smi probe")
    args = ap.parse_args(argv)

    def _tick():
        report = warm_once(keep_alive=args.keep_alive, include_gpu=not args.no_gpu)
        if args.json:
            print(json.dumps(report))
        else:
            _print_report(report)
        return report

    if not args.loop:
        _tick()
        return 0  # always exit 0 — fail-open [pattern:fail-open]

    print(f"[gpu_warm] keeper up: re-pinning every {args.interval:.0f}s "
          f"(keep_alive={args.keep_alive}). Ctrl-C to stop.")
    try:
        while True:
            _tick()
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("[gpu_warm] keeper stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

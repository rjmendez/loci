"""Batched-serving generation client — concurrent fan-out generation with Ollama fallback.

For high-concurrency workflow fan-out (N planning/research agents, per-prompt gates,
map-stage classification), the Ollama tier serializes: `llm_local.generate` issues one
POST per prompt and Ollama processes them essentially one-at-a-time. A batched,
OpenAI-compatible server (vLLM / TGI) can serve many small requests *concurrently* with
continuous batching, which is a large throughput win when you have dozens of short prompts.

Substrate facts this is built against (session grounding):
  - [gen] The local generation tier (mcp/llm_local.py) already exists and speaks to Ollama
    (qwen2.5:3b, pinned via keep_alive). This module's FALLBACK path lazily reuses it.
  - [hardware] Oxalis exposes TWO GPUs shared with the DAMA ant-trainer. GPU placement is
    governed by ONE authoritative policy (dama-gotchi/training/GPU_PACKING_POLICY.md): the
    2080 Ti is the ant-TRAINER GPU, the 4070 Ti hosts persistent inference. A batched-gen
    server must therefore NOT squat the 2080 Ti persistently (it would collide with training
    bursts) — it shares the 4070 Ti inference GPU or runs opportunistically and yields to
    training. See scripts/gpu_placement.md for the reconciled placement.
  - [pattern:fail-open] Every op fails open: on missing config / HTTP error / timeout /
    parse failure we return a well-formed degraded result and NEVER raise. A failed prompt
    yields {"text": "", "ok": False}, so the output list ALWAYS aligns 1:1 with `prompts`.
  - [pattern:injectable] The HTTP client is injected via `client_fn` (defaulting to None ->
    lazily resolved at call time), so importing this module never hard-requires `requests`,
    a live vLLM server, or the sibling llm_local module. Tests stub both paths.

Grounding is SILENT on: whether a vLLM/TGI server is actually deployed, its model name, and
the exact VLLM_BASE_URL value. This module therefore treats VLLM_BASE_URL as the sole switch:
unset -> go straight to the Ollama fallback; set -> try the batched path first, fall back on
any failure. The concrete deploy plan lives in scripts/vllm_serve.md (not yet executed).

Contract:
    generate_batch(prompts, model=None, max_tokens=256, fmt=None, client_fn=None)
        -> list[dict]   # each {"text": str, "ok": bool}, aligned to `prompts`
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from typing import Callable, Optional

# Max in-flight requests to the batched server. Continuous batching only engages when the
# server sees multiple concurrent requests, so we dispatch the batch across a thread pool.
_MAX_CONCURRENCY = int(os.environ.get("VLLM_MAX_CONCURRENCY", "16"))

# [hardware] The batched server's OpenAI-compatible base URL, e.g. http://100.73.200.19:8000
# Unset -> we never touch the network for the primary path and degrade to Ollama immediately.
_VLLM = os.environ.get("VLLM_BASE_URL") or ""

# Model the batched server serves. Grounding is silent on the exact tag; default matches the
# small instruct model recommended in scripts/vllm_serve.md. Override via VLLM_MODEL or arg.
_DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "Qwen2.5-3B-Instruct")

# Generous timeout: a batched server may queue many concurrent requests behind continuous
# batching. Grounding is silent on an exact value; 120s mirrors llm_local's cold-load budget.
_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "120"))


def _fail(n: int) -> list[dict]:
    """A fully-degraded, correctly-aligned result: n copies of {'text':'','ok':False}."""
    return [{"text": "", "ok": False} for _ in range(n)]


def _ok_text(text: str, fmt: Optional[str]) -> dict:
    """Wrap a generated string, validating JSON when fmt=='json' (mirrors llm_local)."""
    text = text or ""
    if fmt == "json":
        try:
            json.loads(text)
        except Exception:
            return {"text": text, "ok": False}
    return {"text": text, "ok": True}


def _via_ollama(prompts: list[str], model: Optional[str], max_tokens: int,
                fmt: Optional[str]) -> list[dict]:
    """FALLBACK path: sequential mcp/llm_local.generate per prompt. Fail-open per prompt.

    Lazily imports llm_local so this module never hard-requires it at import time
    [pattern:injectable]. If the import itself fails, we degrade the whole batch.
    """
    try:
        import llm_local  # lazy sibling import; may not be importable in all contexts
    except Exception:
        try:
            from . import llm_local  # type: ignore
        except Exception:
            return _fail(len(prompts))

    out: list[dict] = []
    for p in prompts:
        try:
            # llm_local.generate defaults model to qwen2.5:3b when we pass None-equivalent;
            # only forward `model` if the caller actually named one for the batched server.
            if model:
                res = llm_local.generate(p, model=model, fmt=fmt, max_tokens=max_tokens)
            else:
                res = llm_local.generate(p, fmt=fmt, max_tokens=max_tokens)
            out.append({"text": res.get("text", ""), "ok": bool(res.get("ok"))})
        except Exception:
            # Per-prompt isolation: one bad prompt never poisons the rest [pattern:fail-open].
            out.append({"text": "", "ok": False})
    return out


def _post_completions(client, url: str, model: str, prompt: str, max_tokens: int,
                      fmt: Optional[str]) -> dict:
    """POST one prompt to the OpenAI-compatible /v1/completions endpoint.

    One request per prompt, but generate_batch() dispatches these CONCURRENTLY across a
    thread pool so the server has multiple in-flight sequences to interleave — that is what
    engages vLLM/TGI continuous batching (a single blocking loop would not). Per-prompt
    requests (vs one list-prompt request) give clean per-prompt failure isolation and work
    identically against TGI.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    if fmt == "json":
        # vLLM supports OpenAI's response_format for JSON-guided decoding.
        body["response_format"] = {"type": "json_object"}
    r = client.post(f"{url}/v1/completions", json=body, timeout=_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    text = ((data.get("choices") or [{}])[0].get("text")) or ""
    return _ok_text(text, fmt)


def generate_batch(prompts: list[str],
                   model: Optional[str] = None,
                   max_tokens: int = 256,
                   fmt: Optional[str] = None,
                   client_fn: Optional[Callable[[], object]] = None) -> list[dict]:
    """Generate for many prompts, batched on vLLM/TGI when available, else Ollama fallback.

    Args:
        prompts: list of prompt strings. The result is ALWAYS the same length and order.
        model: model tag to serve. None -> VLLM_MODEL default on the batched path, and
               llm_local's own default (qwen2.5:3b) on the fallback path.
        max_tokens: max new tokens per prompt.
        fmt: 'json' requests JSON-guided output AND validates each body parses as JSON;
             a non-JSON body downgrades that prompt to ok=False (mirrors llm_local).
        client_fn: injectable zero-arg factory returning an HTTP client exposing
                   .post(url, json=..., timeout=...) -> response with .raise_for_status()
                   and .json() (i.e. a `requests`-like Session). None -> lazily use
                   `requests` [pattern:injectable]. Used ONLY for the batched path;
                   the fallback path goes through llm_local.

    Returns:
        list[dict] aligned to `prompts`, each {"text": str, "ok": bool}. Never raises.
    """
    prompts = list(prompts or [])
    if not prompts:
        return []
    # Normalize to strings so a stray non-str prompt can't blow up json serialization.
    prompts = [p if isinstance(p, str) else str(p) for p in prompts]

    # No batched server configured -> straight to the sequential Ollama fallback [gen].
    if not _VLLM:
        return _via_ollama(prompts, model, max_tokens, fmt)

    # Resolve the HTTP client lazily/injectably. If even that fails, degrade to Ollama.
    try:
        if client_fn is not None:
            client = client_fn()
        else:
            import requests  # lazy: importing this module must not require requests
            client = requests
    except Exception:
        return _via_ollama(prompts, model, max_tokens, fmt)

    served_model = model or _DEFAULT_MODEL

    # Dispatch the batch CONCURRENTLY so the batched server has multiple in-flight requests
    # to interleave — a plain blocking loop would serialize and never engage continuous
    # batching. Results are written back by index to preserve 1:1 order [pattern:fail-open].
    out: list[Optional[dict]] = [None] * len(prompts)
    hard_fail = False

    def _one(i: int, p: str):
        try:
            return i, _post_completions(client, _VLLM, served_model, p, max_tokens, fmt), False
        except Exception:
            # Per-prompt isolation on the batched path too [pattern:fail-open].
            return i, {"text": "", "ok": False}, True

    workers = max(1, min(len(prompts), _MAX_CONCURRENCY))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in concurrent.futures.as_completed(
                    [ex.submit(_one, i, p) for i, p in enumerate(prompts)]):
                i, res, failed = fut.result()
                out[i] = res
                hard_fail = hard_fail or failed
    except Exception:
        # Pool-level failure (never expected) -> degrade the whole batch to Ollama.
        return _via_ollama(prompts, model, max_tokens, fmt)

    any_ok = any(r and r["ok"] for r in out)
    # If the batched server produced NOTHING usable (e.g. server down -> every request
    # raised), fall back to Ollama for the whole batch rather than returning all-failed.
    if hard_fail and not any_ok:
        return _via_ollama(prompts, model, max_tokens, fmt)
    return [r if r is not None else {"text": "", "ok": False} for r in out]

"""Concurrent fan-out generation over the local Ollama tier.

For high-concurrency workflow fan-out (N planning/research agents, per-prompt gates,
map-stage classification) this dispatches many small generate calls concurrently, bounded
by OLLAMA_MAX_CONCURRENCY. Live-verified 2026-09-16 that the Ollama server genuinely
parallelizes concurrent requests to an already-loaded model rather than silently
serializing them one-at-a-time.

Substrate facts this is built against:
  - [gen] The local generation tier (mcp/llm_local.py) speaks to Ollama
    (qwen2.5:3b, pinned via keep_alive). This module lazily reuses it.
  - [pattern:fail-open] Every op fails open: on missing config / HTTP error / timeout /
    parse failure we return a well-formed degraded result and NEVER raise. A failed prompt
    yields {"text": "", "ok": False}, so the output list ALWAYS aligns 1:1 with `prompts`.
  - [pattern:injectable] The sibling llm_local module is imported lazily, so importing this
    module never hard-requires it or a live server.

History: this module used to prefer a vLLM/TGI batched endpoint and fall back to Ollama.
vLLM was removed from Loci (2026-10-04); `client_fn` and `endpoint_role` are still accepted
so existing callers keep working, and are ignored.

Contract:
    generate_batch(prompts, model=None, max_tokens=256, fmt=None, client_fn=None,
                   endpoint_role=None)
        -> list[dict]   # each {"text": str, "ok": bool}, aligned to `prompts`
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from typing import Callable, Optional

# Max in-flight requests to the Ollama host. Live-verified 2026-09-16: Ollama
# (0.34.1, Windows-native, 100.73.200.19:11434) genuinely parallelizes concurrent requests
# to an already-loaded model rather than silently queueing them one-at-a-time -- 4 concurrent
# calls to qwen2.5:3b completed in 1.82s wall-clock vs 4.08s sequential. Kept modest (default
# 6): Ollama here also shares two GPUs (2080 Ti 11GB / 4070 Ti 12GB) with only one large (5-18GB) model
# resident at a time, so heavy over-fanout against a large model risks request queuing/latency
# cliffs rather than a hard crash. Override via OLLAMA_MAX_CONCURRENCY for wider fan-out modes
# (e.g. swarm_escalate.py's --seeds) that have already sized their batch to the target model.
_OLLAMA_MAX_CONCURRENCY = int(os.environ.get("OLLAMA_MAX_CONCURRENCY", "6"))

def _fail(n: int) -> list[dict]:
    """A fully-degraded, correctly-aligned result: n copies of {'text':'','ok':False}."""
    return [{"text": "", "ok": False} for _ in range(n)]


def _via_ollama(prompts: list[str], model: Optional[str], max_tokens: int,
                fmt: Optional[str], think: bool = False) -> list[dict]:
    """Concurrent mcp/llm_local.generate calls, dispatched across a thread
    pool (bounded by OLLAMA_MAX_CONCURRENCY). Fail-open per prompt.

    Live-verified (2026-09-16): the Ollama server genuinely serves concurrent requests in
    parallel rather than silently serializing them -- a single blocking for-loop here would
    leave real throughput on the table for every caller (swarm_escalate.py's fan-out,
    local_deep_think.py's multi-model ideation) whenever routing goes through Ollama, which
    it always does for multi-model tier diversity.

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

    def _one(p: str) -> dict:
        try:
            # llm_local.generate defaults model to qwen2.5:3b when we pass None-equivalent;
            # only forward `model` if the caller actually named one.
            if model:
                res = llm_local.generate(p, model=model, fmt=fmt, max_tokens=max_tokens,
                                         think=think)
            else:
                res = llm_local.generate(p, fmt=fmt, max_tokens=max_tokens, think=think)
            return {"text": res.get("text", ""), "ok": bool(res.get("ok"))}
        except Exception:
            # Per-prompt isolation: one bad prompt never poisons the rest [pattern:fail-open].
            return {"text": "", "ok": False}

    if len(prompts) == 1:
        # Skip pool overhead for the common single-prompt call path.
        return [_one(prompts[0])]

    workers = max(1, min(len(prompts), _OLLAMA_MAX_CONCURRENCY))
    out: list[Optional[dict]] = [None] * len(prompts)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_one, p): i for i, p in enumerate(prompts)}
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    out[i] = fut.result()
                except Exception:
                    out[i] = {"text": "", "ok": False}
    except Exception:
        # Pool-level failure (never expected) -- degrade the whole batch rather than raise
        # [pattern:fail-open].
        return _fail(len(prompts))
    return [r if r is not None else {"text": "", "ok": False} for r in out]


def generate_batch(prompts: list[str],
                   model: Optional[str] = None,
                   max_tokens: int = 256,
                   fmt: Optional[str] = None,
                   client_fn: Optional[Callable[[], object]] = None,
                   think: bool = False,
                   endpoint_role: Optional[str] = None) -> list[dict]:
    """Generate for many prompts concurrently through the local Ollama tier.

    Args:
        prompts: list of prompt strings. The result is ALWAYS the same length and order.
        model: model tag to serve. None -> llm_local's own default.
        max_tokens: max new tokens per prompt. See llm_local.generate's think=True note:
               this budget is SHARED between hidden reasoning and the visible answer on
               thinking-capable models, so pass a larger value (~1000+) when think=True.
        fmt: 'json' requests JSON output AND validates each body parses as JSON;
             a non-JSON body downgrades that prompt to ok=False (mirrors llm_local).
        client_fn: ignored; kept so existing callers keep working (it injected the HTTP
                   client of the removed vLLM path).
        think: opt-in reasoning mode, forwarded to llm_local.generate.
        endpoint_role: ignored; kept for caller compatibility (it selected a per-role vLLM
               endpoint on the removed path).

    Returns:
        list[dict] aligned to `prompts`, each {"text": str, "ok": bool}. Never raises.
    """
    prompts = list(prompts or [])
    if not prompts:
        return []
    # Normalize to strings so a stray non-str prompt can't blow up serialization.
    prompts = [p if isinstance(p, str) else str(p) for p in prompts]
    return _via_ollama(prompts, model, max_tokens, fmt, think=think)

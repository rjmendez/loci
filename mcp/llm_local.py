"""Local-generation primitive — the on-GPU generation tier for Loci-native workflows.

The embedding path (embed_ops.py) is rock-solid; generation on the local GPU is the
newer, colder tier. This module is the single low-level `generate()` call that talks to
the Ollama /api/generate endpoint. It is deliberately dependency-light so sibling modules
can import `llm_local.generate` lazily as the injectable `gen_fn` without hard-requiring it.

Substrate facts this is built against (session grounding):
  - [substrate] Ollama lives at OLLAMA_BASE_URL (fallback OLLAMA_URL). The verified-good
    generation model is qwen2.5:3b (valid JSON, ~111 tok/s warm).
  - [substrate] COLD-LOAD is ~70s; you MUST pass keep_alive so the model stays resident,
    else every call re-eats the cold load. Hence keep_alive defaults to '30m' and is always
    sent in the request body.
  - [pattern:fail-open] Every op fails open: on timeout / HTTP error / bad JSON we return a
    well-formed degraded result ({'text':'','ok':False,...}) and NEVER raise.

Note: the shared gen_fn contract used by callers is
    gen_fn(prompt, *, fmt=None, max_tokens=256) -> {"text":str,"ok":bool}
This function is a superset of that contract (it also returns 'model' and accepts model/
temperature/keep_alive), so it can be passed directly as a gen_fn.
"""
from __future__ import annotations

import json
import os
from typing import Optional

# [substrate] read the base URL from env, same convention as embed_ops.py. When unset, the
# call-time _resolve_ollama() falls back to backends (local probe -> config) for portability.
# GENERATION-specific vars ONLY. OLLAMA_BASE_URL is the EMBEDDING endpoint, and
# reading it here re-created the exact conflation #222 set out to fix, one level
# up: #222 taught the RESOLVER to separate generation from embeddings, then left
# the embedding variable as the highest-precedence generation override.
#
# scripts/loci_groom.load_env() — which every groom pass calls — sets
# OLLAMA_BASE_URL to backends.ollama_url(), the in-cluster host that serves only
# nomic-embed-text. Measured in one process: bare import resolves to oxalis and
# generates; after load_env() it resolves to 10.42.0.1 and 404s. So under cron,
# 100% of generation went to a host with no generation model, and the vLLM
# fallback was silently rescuing it — until #224 made that fallback opt-in, which
# took the groom pass from 9/100 degraded to 100/100.
#
# A host with one capable Ollama still works: _resolve_ollama() falls through to
# backends.ollama_gen_url(), which itself falls back to ollama_url(). This only
# removes the env var's ability to OUTRANK that resolution.
# Read at CALL time, not import time. qdrant_ops._retention_days() already
# established this pattern here for the same reason: scripts/loci_groom.load_env()
# runs AFTER this module is imported, so an import-time capture reflects the
# environment before the process configured itself. That ordering is what let a
# stale value win, and a module-level constant cannot be tested without reload()
# — which made the test for it order-dependent in the suite.
def _gen_env() -> str:
    return (os.environ.get("LOCI_OLLAMA_GEN_URL")
            or os.environ.get("OLLAMA_GEN_URL") or "")


def _resolve_ollama() -> str:
    """The GENERATION endpoint, which is not necessarily the embedding one.

    This used to call backends.ollama_url() and inherit whichever Ollama answered
    a reachability probe first. That instance serves only nomic-embed-text here,
    so every generate() call failed against a server that was demonstrably up.
    """
    try:
        import backends
        return backends.ollama_gen_url()
    except Exception:
        return ""

# Timeout is generous because a cold model load can take ~70s even when we pin with
# keep_alive (grounding is silent on an exact value; 120s covers a cold load plus generation).
_TIMEOUT = float(os.environ.get("OLLAMA_GEN_TIMEOUT", "120"))


def generate(prompt: str,
             model: str = "",
             fmt: Optional[str] = None,
             max_tokens: int = 256,
             temperature: float = 0.2,
             keep_alive: str = "30m") -> dict:
    """Generate text from the local Ollama model. Fail-open, never raises.

    Args:
        prompt: the prompt string.
        model: Ollama model tag. Defaults to the verified-good qwen2.5:3b [substrate].
        fmt: if 'json', request structured JSON output AND validate the body parses as
             JSON; a non-JSON body downgrades the result to ok=False.
        max_tokens: mapped to Ollama options.num_predict.
        temperature: mapped to Ollama options.temperature.
        keep_alive: pins the model resident to avoid the ~70s cold load [substrate].
                    Always included in the request body — this is the critical bit.

    Returns:
        {'text': str, 'ok': bool, 'model': str}. On any failure text='' and ok=False.
    """
    if not model:
        try:
            import backends
            model = backends.ollama_gen_model()
        except Exception:
            model = "qwen2.5:3b"

    def fail(why: str) -> dict:
        """Degraded, but never silent.

        This used to be a bare `return fail` that discarded the exception, so a
        misconfigured backend produced {'text':'','ok':False} with nothing to act
        on. That cost a live diagnosis: the verify groom pass degraded for a day
        and the reason -- a model name no server recognised -- was sitting in a
        swallowed exception the whole time.
        """
        return {"text": "", "ok": False, "model": model, "why": why}

    base = _gen_env() or _resolve_ollama()   # gen env wins; else backends
    if not prompt:
        return fail("empty prompt")
    if not base:
        return fail("no Ollama endpoint resolved (OLLAMA_BASE_URL unset and backends "
                    "returned nothing)")

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,  # critical: keep the model resident [substrate]
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    if fmt == "json":
        body["format"] = "json"

    try:
        import requests
        r = requests.post(f"{base}/api/generate", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
        text = (r.json().get("response") or "")
    except Exception as exc:
        # Ollama could not serve this. Try the vLLM tier before giving up: on this
        # host Ollama carries only nomic-embed-text (an EMBEDDING model, no
        # generation model at all) while vLLM serves the generation model under a
        # different name -- Qwen/Qwen2.5-3B-Instruct, not the Ollama-style
        # qwen2.5:3b. Asking one server for the other's name fails on both.
        fallback = _try_vllm(prompt, fmt=fmt, max_tokens=max_tokens,
                             temperature=temperature)
        if fallback is not None:
            return fallback
        return fail(f"ollama {type(exc).__name__}: {exc}"[:300])

    if fmt == "json":
        # ok=True only if the body actually parses as JSON.
        try:
            json.loads(text)
        except Exception as exc:
            return {"text": text, "ok": False, "model": model,
                    "why": f"response was not valid JSON: {exc}"[:200]}

    return {"text": text, "ok": True, "model": model}


def _try_vllm(prompt: str, *, fmt: Optional[str], max_tokens: int,
              temperature: float) -> Optional[dict]:
    """Second tier. Returns a result dict, or None if vLLM is not usable either.

    batched_gen already resolves the vLLM endpoint AND the model name the server
    actually registers (backends.vllm_model()), which is the part llm_local was
    getting wrong. Reusing it keeps one definition of both.
    """
    # OPT-IN. This fallback was added when Ollama generation was broken; Ollama
    # works now, and the endpoint backends resolves is NOT Loci's: 127.0.0.1:18000
    # is /home/rjmendez/dama-vllm/vllm_tailscale_forward.py (pid 378), another
    # project's service, serving Qwen2.5-3B-Instruct at max_model_len=4096.
    # Grounded verify prompts exceed that, so firing into it 400s AND borrows
    # capacity Loci does not own. Measured verify latencies (153s, >300s) also
    # exceed _TIMEOUT=120s, so the failure path that reaches here is exactly the
    # one that would hit it hardest.
    #
    # Set LOCI_VLLM_FALLBACK=1 when Loci has a vLLM of its own.
    if os.environ.get("LOCI_VLLM_FALLBACK", "").strip() in ("", "0"):
        return None
    try:
        import batched_gen
    except Exception:
        return None
    try:
        out = batched_gen.generate_batch([prompt], max_tokens=max_tokens, fmt=fmt,
                                         temperature=temperature)
    except TypeError:
        # older signature without temperature
        try:
            out = batched_gen.generate_batch([prompt], max_tokens=max_tokens, fmt=fmt)
        except Exception:
            return None
    except Exception:
        return None
    if not out:
        return None
    first = out[0] or {}
    if not first.get("ok"):
        return None
    served = first.get("model")
    if not served:
        try:
            served = batched_gen._resolve_vllm_model()
        except Exception:
            served = "vllm"
    return {"text": first.get("text", ""), "ok": True, "model": served, "tier": "vllm"}

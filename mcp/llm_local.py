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

# [substrate] read the base URL from env, same convention as embed_ops.py.
_OLLAMA = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_URL") or ""

# Timeout is generous because a cold model load can take ~70s even when we pin with
# keep_alive (grounding is silent on an exact value; 120s covers a cold load plus generation).
_TIMEOUT = float(os.environ.get("OLLAMA_GEN_TIMEOUT", "120"))


def generate(prompt: str,
             model: str = "qwen2.5:3b",
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
    fail = {"text": "", "ok": False, "model": model}
    if not prompt or not _OLLAMA:
        return fail

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
        r = requests.post(f"{_OLLAMA}/api/generate", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
        text = (r.json().get("response") or "")
    except Exception:
        return fail

    if fmt == "json":
        # ok=True only if the body actually parses as JSON.
        try:
            json.loads(text)
        except Exception:
            return {"text": text, "ok": False, "model": model}

    return {"text": text, "ok": True, "model": model}

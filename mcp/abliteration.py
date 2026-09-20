"""Abliteration cloud tier — uncensored escalation provider.

OpenAI-compatible chat/completions wrapper with the same list-aligned contract as
openrouter.generate_batch so callers can swap providers without reshaping outputs.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger("loci-mcp.abliteration")

DEFAULT_BASE = "https://api.abliteration.ai/v1"
DEFAULT_MODEL = "abliterated-model"
_MAX_INFLIGHT = int(os.environ.get("LOCI_ABLITERATION_CONCURRENCY", "4"))
_TIMEOUT = float(os.environ.get("LOCI_ABLITERATION_TIMEOUT", "120"))


def _max_tokens_cap() -> int:
    try:
        return max(0, int(os.environ.get("LOCI_ABLITERATION_MAX_TOKENS_PER_CALL", "0")))
    except Exception:
        return 0


def credentials() -> tuple[str, str]:
    key = os.environ.get("ABLITERATION_API_KEY", "")
    base = os.environ.get("ABLITERATION_BASE_URL", "")
    if not (key and base):
        try:
            import backends
            cfg_base, cfg_key = backends.abliteration()
            base = base or cfg_base
            key = key or cfg_key
        except Exception as exc:
            logger.debug("abliteration: config read failed: %r", exc)
    return (base or DEFAULT_BASE), key


def available() -> bool:
    return bool(credentials()[1])


def _extract_json(text: str) -> Optional[str]:
    if not text:
        return None
    stripped = text.strip()
    try:
        json.loads(stripped)
        return stripped
    except ValueError:
        pass
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except ValueError:
                        break
    return None


def _one(session, base: str, key: str, model: str, prompt: str,
         max_tokens: int, fmt: Optional[str], temperature: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if fmt == "json":
        body["response_format"] = {"type": "json_object"}
    try:
        r = session.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=_TIMEOUT,
        )
        try:
            payload = r.json()
        except ValueError:
            return {"text": "", "ok": False, "status": r.status_code}
        err = payload.get("error")
        if err:
            return {"text": "", "ok": False, "status": err.get("code") or r.status_code}
        if r.status_code != 200:
            return {"text": "", "ok": False, "status": r.status_code}
        text = (payload.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if fmt == "json":
            extracted = _extract_json(text)
            if extracted is None:
                return {"text": text, "ok": False}
            text = extracted
        usage = payload.get("usage") or {}
        return {
            "text": text,
            "ok": bool(text.strip()),
            "model": payload.get("model", model),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
    except Exception:
        return {"text": "", "ok": False}


def generate_batch(prompts: list[str],
                   model: Optional[str] = None,
                   max_tokens: int = 256,
                   fmt: Optional[str] = None,
                   temperature: float = 0.2,
                   session_fn: Optional[Callable] = None) -> list[dict]:
    if not prompts:
        return []
    cap = _max_tokens_cap()
    if cap > 0 and max_tokens > cap:
        logger.warning("abliteration: guardrail refused request max_tokens=%s cap=%s",
                       max_tokens, cap)
        return [{"text": "", "ok": False, "why": f"max_tokens cap exceeded ({max_tokens}>{cap})"}
                for _ in prompts]
    base, key = credentials()
    if not key:
        return [{"text": "", "ok": False} for _ in prompts]
    if session_fn is None:
        def session_fn():
            import requests
            return requests.Session()
    try:
        session = session_fn()
    except Exception:
        return [{"text": "", "ok": False} for _ in prompts]

    chosen_model = model or DEFAULT_MODEL
    results = [{"text": "", "ok": False} for _ in prompts]
    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_INFLIGHT) as pool:
        futures = {
            pool.submit(_one, session, base, key, chosen_model, prompt, max_tokens, fmt, temperature): i
            for i, prompt in enumerate(prompts)
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = {"text": "", "ok": False}
    return results

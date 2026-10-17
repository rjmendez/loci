"""OpenRouter tier — the generation backend whose availability is not this node's.

The local tiers (vLLM on the GPU node, Ollama) share a failure domain: one WSL2
node whose device plugin advertises phantom GPUs and periodically reports none
healthy, taking running pods down with it. That is precisely when a passive
grooming pass would otherwise stall. A remote tier turns that outage into a
slower hour instead.

Contract, identical to ``batched_gen.generate_batch`` so the tiers are
interchangeable::

    generate_batch(prompts, model=...) -> list[dict]   # {"text": str, "ok": bool}

The result list is ALWAYS 1:1 with ``prompts`` — a failed prompt yields
``{"text": "", "ok": False}`` — so no caller ever has to realign by hand.

Never enabled implicitly. Findings carry internal addressing, host names and
cluster topology; shipping them to a third party is a decision a pass makes
explicitly, not a fallback it drifts into. See docs/companion-service.md.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger("loci-mcp.openrouter")

DEFAULT_BASE = "https://openrouter.ai/api/v1"

# Free-tier ladder, cheapest-to-run first. Free models rate-limit and come and go,
# so a pass names a *preference* and the caller falls down the list. Ordered by
# measured fitness for short structured-output work, not by parameter count.
FREE_LADDER = (
    "google/gemma-4-31b-it:free",
    "z-ai/glm-5.2:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free",
)

# Paid but negligible: ~$0.0000145 for a 700-in/40-out grooming call, i.e. the
# whole 2.4k-finding corpus for about four cents. Worth it when a pass needs a
# result rather than a best effort.
CHEAP_LADDER = (
    "mistralai/mistral-nemo",
    "qwen/qwen3.7-flash",
    "openai/gpt-oss-20b",
)

# Free first, then the negligible-cost tier as the safety net. The free pool is a
# SHARED upstream quota — "temporarily rate-limited upstream" is its normal state
# under load, not a fault — so a ladder that stops at free stalls exactly when the
# work is biggest.
DEFAULT_LADDER = FREE_LADDER + CHEAP_LADDER

_MAX_INFLIGHT = int(os.environ.get("LOCI_OPENROUTER_CONCURRENCY", "6"))
_TIMEOUT = float(os.environ.get("LOCI_OPENROUTER_TIMEOUT", "90"))


def credentials() -> tuple:
    """(base_url, api_key). Env first, then ~/.loci/backends.toml."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    base = os.environ.get("OPENROUTER_BASE_URL", "")
    if not (key and base):
        try:
            import backends
            cfg_base, cfg_key = backends.openrouter()
            base = base or cfg_base
            key = key or cfg_key
        except Exception as exc:
            logger.debug("openrouter: config read failed: %r", exc)
    return (base or DEFAULT_BASE), key


def available() -> bool:
    return bool(credentials()[1])


def _extract_json(text: str) -> Optional[str]:
    """The outermost JSON object in `text`, or None.

    A reasoning model asked for JSON writes several paragraphs and then the JSON.
    Demanding a bare object throws away a correct answer for a presentation
    difference, so find it — from the end, since the answer follows the thinking.
    """
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
            json=body, timeout=_TIMEOUT,
        )
        try:
            payload = r.json()
        except ValueError:
            logger.warning("openrouter: %s returned a non-JSON body (%s)", model, r.status_code)
            return {"text": "", "ok": False, "status": r.status_code}

        # OpenRouter reports upstream failures as HTTP 200 with an `error` object
        # and empty choices. Reading only the status code turns a rate-limit into
        # something indistinguishable from a model that answered with nothing.
        err = payload.get("error")
        if err:
            code = err.get("code") or r.status_code
            logger.warning("openrouter: %s failed (%s): %s",
                           model, code, str(err.get("message"))[:120])
            return {"text": "", "ok": False, "status": code}
        if r.status_code != 200:
            logger.warning("openrouter: %s returned %s: %s", model, r.status_code, r.text[:200])
            return {"text": "", "ok": False, "status": r.status_code}

        text = (payload.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if fmt == "json":
            # Reasoning models narrate before answering. Extracting the object is
            # not the same as accepting invented content — callers still validate.
            extracted = _extract_json(text)
            if extracted is None:
                logger.warning("openrouter: %s returned no JSON object under fmt=json", model)
                return {"text": text, "ok": False}
            text = extracted
        usage = payload.get("usage") or {}
        return {
            "text": text, "ok": bool(text.strip()), "model": payload.get("model", model),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
    except Exception as exc:
        logger.warning("openrouter: %s call failed: %r", model, exc)
        return {"text": "", "ok": False}


def generate_batch(prompts: list, model: Optional[str] = None, max_tokens: int = 256,
                   fmt: Optional[str] = None, temperature: float = 0.2,
                   ladder: Optional[tuple] = None,
                   session_fn: Optional[Callable] = None) -> list:
    """Generate for every prompt. Always returns len(prompts) results.

    ``ladder`` retries the prompts that failed against the next model in the list,
    which is what makes a free tier usable: a 429 on one free model is not an
    outage, it is a hint to ask a different one.
    """
    if not prompts:
        return []
    base, key = credentials()
    if not key:
        logger.warning("openrouter: no API key configured — returning no generations")
        return [{"text": "", "ok": False} for _ in prompts]

    if session_fn is None:
        def session_fn():
            import requests
            return requests.Session()
    try:
        session = session_fn()
    except Exception as exc:
        logger.warning("openrouter: no HTTP client: %r", exc)
        return [{"text": "", "ok": False} for _ in prompts]

    candidates = list(ladder or ([model] if model else DEFAULT_LADDER))
    results: list = [{"text": "", "ok": False} for _ in prompts]

    for candidate in candidates:
        todo = [i for i, r in enumerate(results) if not r.get("ok")]
        if not todo:
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_INFLIGHT) as pool:
            futures = {
                pool.submit(_one, session, base, key, candidate, prompts[i],
                            max_tokens, fmt, temperature): i
                for i in todo
            }
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception:
                    results[i] = {"text": "", "ok": False}

    return results

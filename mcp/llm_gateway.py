"""Client for an OpenAI-compatible LLM gateway (LiteLLM), so Loci's own generation and embedding
calls go through one authenticated, cloud-first, fallback-aware door instead of an Ollama native API.

Off unless a gateway URL is configured. Everything machine-specific lives in the private config or env,
never in code:

    [llm_gateway]
    url         = "https://gateway.example:4443/v1"   # env LOCI_LLM_GATEWAY_URL
    key         = "..."                               # env LOCI_LLM_GATEWAY_KEY
    model       = "<gateway model group>"             # env LOCI_LLM_GATEWAY_MODEL (chat)
    embed_model = "<gateway embedding group>"         # env LOCI_LLM_GATEWAY_EMBED_MODEL
    timeout     = 60                                  # seconds, per request
    fallback_raw = false                              # true: on a gateway failure, fall back to raw Ollama

Failures are loud: a result carries ``ok=False`` and a ``why``. A gateway that is configured but cannot
answer is never silently replaced by a raw Ollama call unless ``fallback_raw`` is set.

Requests carry ``metadata.loci_internal_call = true`` so a gateway plugin can tell Loci's own plumbing
(judges, classifiers, embeddings) from user traffic and skip grounding and memory write-back for it.

stdlib only (urllib), so it works inside the memcheck package as well as the server.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

_LOG = logging.getLogger("loci-mcp.llm_gateway")

INTERNAL_CALL_FLAG = "loci_internal_call"
_DEFAULT_TIMEOUT_S = 60.0


def _cfg(key: str, default=None):
    try:
        import backends
        return backends._cfg("llm_gateway", key, default)
    except Exception:
        return default


def _setting(env: str, key: str, default: str = "") -> str:
    value = os.environ.get(env)
    if value is None or not value.strip():
        value = _cfg(key, default)
    return str(value or "").strip()


def base_url() -> str:
    """The gateway's OpenAI-compatible base (ending in /v1), or '' when no gateway is configured."""
    url = _setting("LOCI_LLM_GATEWAY_URL", "url").rstrip("/")
    if not url:
        return ""
    return url if url.endswith("/v1") else f"{url}/v1"


def enabled() -> bool:
    return bool(base_url())


def chat_model() -> str:
    return _setting("LOCI_LLM_GATEWAY_MODEL", "model")


def role_group(role: str) -> str:
    """Gateway model group for a role, from `[llm_gateway.roles]` in the private config ('' when the role is unmapped).

    A mapped role is served by its own group, never by the default one: those groups hold the abliterated and
    specialist models and are local-only on purpose, so a failure is returned, not rerouted to a cloud model."""
    key = (role or "").strip().lower()
    roles = _cfg("roles", {})
    if not key or not isinstance(roles, dict):
        return ""
    return str(roles.get(key) or "").strip()


def embed_model() -> str:
    return _setting("LOCI_LLM_GATEWAY_EMBED_MODEL", "embed_model")


def fallback_raw() -> bool:
    raw = os.environ.get("LOCI_LLM_GATEWAY_FALLBACK_RAW")
    if raw is None or not raw.strip():
        raw = _cfg("fallback_raw", False)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def timeout_s() -> float:
    try:
        return float(_setting("LOCI_LLM_GATEWAY_TIMEOUT", "timeout") or _DEFAULT_TIMEOUT_S)
    except ValueError:
        return _DEFAULT_TIMEOUT_S


def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    key = _setting("LOCI_LLM_GATEWAY_KEY", "key")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _post(path: str, payload: dict, timeout: Optional[float]) -> dict:
    """POST to the gateway; return the parsed JSON object. Raises on any transport/HTTP/JSON failure."""
    request = urllib.request.Request(f"{base_url()}{path}", data=json.dumps(payload).encode("utf-8"),
                                     headers=_headers(), method="POST")
    with urllib.request.urlopen(request, timeout=timeout or timeout_s()) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"gateway returned {type(body).__name__}, not an object")
    return body


def _http_reason(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read().decode("utf-8", "replace")[:160]
        except Exception:
            detail = ""
        return f"HTTP {exc.code} {detail}".strip()
    return f"{type(exc).__name__}: {exc}"


def chat(prompt: str, *, max_tokens: int = 256, temperature: float = 0.2, json_mode: bool = False,
         timeout: Optional[float] = None, role: str = "") -> dict:
    """One chat completion. Returns {'text', 'ok', 'model', 'tier': 'gateway', 'why'?}; never raises."""
    model = role_group(role) or chat_model()
    out = {"text": "", "ok": False, "model": model, "tier": "gateway"}
    if not enabled():
        out["why"] = "no LLM gateway configured"
        return out
    if not model:
        out["why"] = "LLM gateway configured but no model group is set (LOCI_LLM_GATEWAY_MODEL / [llm_gateway] model)"
        return out
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
        "metadata": {INTERNAL_CALL_FLAG: True},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        body = _post("/chat/completions", payload, timeout)
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if isinstance(text, list):
            text = " ".join(str(p.get("text") or "") for p in text if isinstance(p, dict))
        if not str(text).strip():
            text = message.get("reasoning_content") or ""
        out["text"] = str(text)
        out["model"] = str(body.get("model") or model)
        out["ok"] = bool(str(text).strip())
        if not out["ok"]:
            out["why"] = "gateway returned an empty completion"
    except Exception as exc:
        out["why"] = f"gateway {_http_reason(exc)}"[:300]
        _LOG.warning("llm_gateway chat failed model=%s: %s", model, out["why"])
    return out


def embed(texts: list, *, timeout: Optional[float] = None) -> list:
    """Embed ``texts`` through the gateway. [] on any failure (the caller's fail-open convention)."""
    model = embed_model()
    if not texts or not enabled():
        return []
    if not model:
        _LOG.warning("llm_gateway embed: gateway configured but no embedding group set "
                     "(LOCI_LLM_GATEWAY_EMBED_MODEL / [llm_gateway] embed_model)")
        return []
    try:
        body = _post("/embeddings", {"model": model, "input": list(texts),
                                     "metadata": {INTERNAL_CALL_FLAG: True}}, timeout)
        rows = sorted(body.get("data") or [], key=lambda r: r.get("index", 0))
        vectors = [list(r["embedding"]) for r in rows]
    except Exception as exc:
        _LOG.warning("llm_gateway embed failed model=%s: %s", model, _http_reason(exc))
        return []
    if len(vectors) != len(texts):
        _LOG.warning("llm_gateway embed: %d vector(s) for %d input(s) — returning none", len(vectors), len(texts))
        return []
    return vectors

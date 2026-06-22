"""Shared LLM + embedding backend for memcheck (the deep_think → loci merge).

Loci historically had *no* chat-completion client — only embeddings. This module
brings deep_think's provider-calling layer into loci so the memory-validation
path can run an LLM contradiction judge (and the in-server reasoning tool can
fan out), while staying:

  - **stdlib-only** — urllib + json, no httpx/numpy/anthropic SDK. Adds zero hard
    dependencies to the loci MCP venv; cosine is computed in pure Python.
  - **fail-open** — every call returns ``None`` (or ``[]``) on any error instead
    of raising, so an advisory check never breaks a store/recall.
  - **opt-in** — nothing here is invoked on the default self-check path; the
    caller decides when an LLM/embedding endpoint is in play.

Config follows loci's ``.env`` conventions (no new required vars):

  Embeddings  OLLAMA_BASE_URL (no ``/v1`` suffix; ``/v1/embeddings`` appended),
              EMBED_MODEL / MNEMOSYNE_EMBEDDING_MODEL (default ``nomic-embed-text``).
  LLM         provider auto-detected: ANTHROPIC_API_KEY → anthropic,
              GITHUB_COPILOT_OAUTH_TOKEN → copilot, else local Ollama
              (``/api/generate`` at OLLAMA_BASE_URL). Model from MEMCHECK_LLM_MODEL,
              falling back to SWR_LLM_MODEL, default ``llama3.2:latest``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import urllib.request

__all__ = [
    "llm_available",
    "embeddings_available",
    "call_llm",
    "embed_texts",
    "cosine",
]

_log = logging.getLogger("memcheck.llm")

_DEFAULT_OLLAMA = "http://localhost:11434"


def _ollama_base() -> str:
    return (os.environ.get("OLLAMA_BASE_URL") or _DEFAULT_OLLAMA).rstrip("/")


def _embed_model() -> str:
    return (
        os.environ.get("EMBED_MODEL")
        or os.environ.get("MNEMOSYNE_EMBEDDING_MODEL")
        or "nomic-embed-text"
    )


def _llm_model() -> str:
    return (
        os.environ.get("MEMCHECK_LLM_MODEL")
        or os.environ.get("SWR_LLM_MODEL")
        or "llama3.2:latest"
    )


def _anthropic_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return k if k and k not in ("not-set",) else ""


def _copilot_token() -> str:
    for var in ("GITHUB_COPILOT_OAUTH_TOKEN", "GITHUB_TOKEN"):
        v = os.environ.get(var, "").strip()
        if v and v not in ("not-set",):
            return v
    return ""


def _provider() -> str:
    """Resolve the active LLM provider. Override with MEMCHECK_LLM_PROVIDER."""
    forced = os.environ.get("MEMCHECK_LLM_PROVIDER", "").strip().lower()
    if forced:
        return forced
    if _anthropic_key():
        return "anthropic"
    if _copilot_token():
        return "copilot"
    return "ollama"


def llm_available() -> bool:
    """True if some LLM endpoint should be reachable (a key is set, or Ollama)."""
    return _provider() in ("anthropic", "copilot", "ollama")


def embeddings_available() -> bool:
    """True if an embeddings endpoint is configured (always at least local Ollama)."""
    return bool(_ollama_base())


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict | None:
    """POST JSON and parse a JSON response. Returns None on any error (fail-open)."""
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json", **headers}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — fail-open boundary
        _log.debug("llm POST %s failed, degrading to None: %r", url, exc)
        return None


def call_llm(
    prompt: str,
    *,
    json_mode: bool = False,
    timeout: float = 60.0,
    max_tokens: int = 1024,
) -> str | None:
    """Single-shot completion against the resolved provider. None on any failure.

    json_mode hints the backend to emit JSON (Ollama ``format=json``; for cloud
    providers it is folded into the prompt by the caller). Fail-open: returns
    None rather than raising, so an advisory caller degrades to "no LLM signal".
    """
    if not prompt or not prompt.strip():
        return None
    provider = _provider()
    try:
        if provider == "anthropic":
            return _call_anthropic(prompt, timeout, max_tokens)
        if provider == "copilot":
            return _call_copilot(prompt, timeout, max_tokens)
        return _call_ollama(prompt, json_mode, timeout)
    except Exception as exc:  # noqa: BLE001 — belt-and-suspenders fail-open
        _log.debug("call_llm provider=%s failed: %r", provider, exc)
        return None


def _call_ollama(prompt: str, json_mode: bool, timeout: float) -> str | None:
    payload: dict = {"model": _llm_model(), "prompt": prompt, "stream": False}
    if json_mode:
        payload["format"] = "json"
    model = _llm_model()
    # qwen extended-thinking is noisy for a yes/no judge — disable when present.
    if "qwen" in model.lower():
        payload["think"] = False
    data = _post_json(f"{_ollama_base()}/api/generate", payload, {}, timeout)
    if not data:
        return None
    text = (data.get("response") or "").strip()
    return text or None


def _call_anthropic(prompt: str, timeout: float, max_tokens: int) -> str | None:
    model = os.environ.get("MEMCHECK_ANTHROPIC_MODEL", "claude-haiku-4-5")
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        {
            "x-api-key": _anthropic_key(),
            "anthropic-version": "2023-06-01",
        },
        timeout,
    )
    if not data:
        return None
    try:
        return (data["content"][0]["text"]).strip() or None
    except (KeyError, IndexError, TypeError):
        return None


def _call_copilot(prompt: str, timeout: float, max_tokens: int) -> str | None:
    model = os.environ.get("MEMCHECK_COPILOT_MODEL", "claude-sonnet-4.6")
    data = _post_json(
        "https://api.githubcopilot.com/chat/completions",
        {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        {
            "Authorization": f"Bearer {_copilot_token()}",
            "Copilot-Integration-Id": "vscode-chat",
        },
        timeout,
    )
    if not data:
        return None
    try:
        return (data["choices"][0]["message"]["content"]).strip() or None
    except (KeyError, IndexError, TypeError):
        return None


def embed_texts(texts: list[str], *, timeout: float = 60.0, batch: int = 16) -> list[list[float]]:
    """Embed texts via the local nomic endpoint. Returns [] on any failure.

    Mirrors ground_gate.py: OLLAMA_BASE_URL + ``/v1/embeddings``, EMBED_MODEL.
    Output is NOT normalized — use ``cosine`` which normalizes per-pair.
    """
    if not texts:
        return []
    url = f"{_ollama_base()}/v1/embeddings"
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [t[:2000] for t in texts[i : i + batch]]
        data = _post_json(url, {"model": _embed_model(), "input": chunk}, {}, timeout)
        if not data or "data" not in data:
            return []  # fail-open: partial embeds are useless for pairwise cosine
        try:
            out.extend(d["embedding"] for d in data["data"])
        except (KeyError, TypeError):
            return []
    return out


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in pure Python (no numpy). 0.0 on degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 0.0:
        return 0.0
    return dot / denom

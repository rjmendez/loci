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
import os as _os
import sys as _sys

# mcp/ is this file's grandparent; vecmath lives there.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from vecmath import cosine as _vecmath_cosine  # noqa: E402
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
    """Embedding endpoint. Generation resolves separately — see _ollama_gen_base."""
    return (os.environ.get("OLLAMA_BASE_URL") or _DEFAULT_OLLAMA).rstrip("/")


def _ollama_gen_base() -> str:
    """Generation endpoint, which is NOT necessarily the embedding one.

    Generation used _ollama_base(), i.e. OLLAMA_BASE_URL. On this host that
    resolves to an in-cluster Ollama serving ONLY nomic-embed-text, so every
    call_llm() returned None while llm_available() reported True. Everything
    gated on it — investigation_reflect's summaries, investigation_reason, the
    causal-inference LLM path, the contradiction polarity judge — degraded
    silently. Measured: 3 of 142 investigations had a summary.

    Falls back to _ollama_base() so a host with one capable Ollama needs no
    extra config.
    """
    env = os.environ.get("LOCI_OLLAMA_GEN_URL") or os.environ.get("OLLAMA_GEN_URL")
    if env:
        return env.rstrip("/")
    try:
        import backends
        url = backends.ollama_gen_url()
        if url:
            return url.rstrip("/")
    except Exception:
        pass
    return _ollama_base()


def _embed_model() -> str:
    return (
        os.environ.get("EMBED_MODEL")
        or os.environ.get("MNEMOSYNE_EMBEDDING_MODEL")
        or "nomic-embed-text"
    )


def _llm_model() -> str:
    """Generation model. The old default, llama3.2:latest, is not installed on
    any endpoint this deployment reaches — so even a correct URL failed."""
    explicit = os.environ.get("MEMCHECK_LLM_MODEL") or os.environ.get("SWR_LLM_MODEL")
    if explicit:
        return explicit
    try:
        import backends
        m = backends.ollama_gen_model()
        if m:
            return m
    except Exception:
        pass
    return "llama3.2:latest"


def _anthropic_key() -> str:
    """The key, or "" if it cannot be used as an HTTP header.

    .strip() removes OUTER whitespace only. A key wrapped across two lines in a
    shell rc file keeps an INTERNAL newline, which urllib rejects with a
    ValueError the caller then swallows — indistinguishable from "no key". Treat
    a key containing any whitespace as absent so provider selection falls through
    to something that works, and say so once.
    """
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not k or k in ("not-set",):
        return ""
    if any(c.isspace() for c in k):
        _log.warning("ANTHROPIC_API_KEY contains whitespace (a wrapped paste?) — "
                     "unusable as a header, treating as unset")
        return ""
    return k


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


# Findings carry internal addressing — IPs, hostnames, code paths — so leaving the
# box is a choice, not a default. Local Ollama is free, private and needs no
# credit, so it goes first; OpenRouter is the fallback that replaces Anthropic,
# which measured HTTP 400 "credit balance is too low" and cannot transact at all.
# Set LOCI_MEMCHECK_NO_REMOTE=1 to keep every call on this machine.
_DEFAULT_PROVIDER_ORDER = ("ollama", "openrouter", "anthropic", "copilot")


def _openrouter_enabled() -> bool:
    if os.environ.get("LOCI_MEMCHECK_NO_REMOTE", "").strip() not in ("", "0"):
        return False
    try:
        import openrouter
        return bool(openrouter.available())
    except Exception:
        return False


def _call_openrouter(prompt: str, json_mode: bool, timeout: float,
                     max_tokens: int) -> str | None:
    """Free/cheap ladder. Reuses mcp/openrouter.py so the ladder and its
    HTTP-200-with-error-envelope handling live in one place."""
    try:
        import openrouter
        out = openrouter.generate_batch([prompt], max_tokens=max_tokens,
                                        fmt="json" if json_mode else None)
    except Exception as exc:
        _log.warning("openrouter call raised: %r", exc)
        return None
    first = (out or [{}])[0] or {}
    if not first.get("ok"):
        _log.warning("openrouter declined: %s", str(first.get("why") or "")[:160])
        return None
    return first.get("text") or None


def _provider_order() -> list:
    """Resolution order, with an explicit override first.

    MEMCHECK_LLM_PROVIDER forces one to the front; the rest still follow, because
    committing to a single provider is what made every call return None while
    llm_available() reported True.
    """
    forced = os.environ.get("MEMCHECK_LLM_PROVIDER", "").strip().lower()
    order = [forced] if forced else []
    for p in _DEFAULT_PROVIDER_ORDER:
        if p not in order:
            order.append(p)
    return order


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
    model: str | None = None,
    options: dict | None = None,
) -> str | None:
    """Single-shot completion against the resolved provider. None on any failure.

    json_mode hints the backend to emit JSON (Ollama ``format=json``; for cloud
    providers it is folded into the prompt by the caller). Fail-open: returns
    None rather than raising, so an advisory caller degrades to "no LLM signal".
    """
    if not prompt or not prompt.strip():
        return None
    # Try the resolved provider, then FALL THROUGH. Selecting a provider on
    # key-presence and committing to it is how this starved: ANTHROPIC_API_KEY
    # was set, so _provider() returned "anthropic" for every call, and the API
    # answered "Your credit balance is too low" — a 400 the fail-open swallowed.
    # Everything gated on llm_available() reported ready and produced nothing.
    #
    # A configured provider that cannot transact is not an available provider.
    # Ollama is always last because it is local, free, and needs no credit.
    order = _provider_order()

    for provider in order:
        if provider == "anthropic" and not _anthropic_key():
            continue
        if provider == "copilot" and not _copilot_token():
            continue
        if provider == "openrouter" and not _openrouter_enabled():
            continue
        try:
            if provider == "anthropic":
                out = _call_anthropic(prompt, timeout, max_tokens)
            elif provider == "copilot":
                out = _call_copilot(prompt, timeout, max_tokens)
            elif provider == "openrouter":
                out = _call_openrouter(prompt, json_mode, timeout, max_tokens)
            else:
                out = _call_ollama(prompt, json_mode, timeout, model=model,
                                   options=options)
        except Exception as exc:  # noqa: BLE001 — fail-open, but say which one
            _log.warning("call_llm provider=%s raised, trying next: %r", provider, exc)
            continue
        if out:
            return out
        _log.warning("call_llm provider=%s returned nothing, trying next", provider)
    _log.warning("call_llm: every provider failed (tried %s)", ", ".join(order))
    return None


def _call_ollama(prompt: str, json_mode: bool, timeout: float,
                 model: str | None = None, options: dict | None = None) -> str | None:
    model = model or _llm_model()
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if json_mode:
        payload["format"] = "json"
    # qwen extended-thinking is noisy for a yes/no judge — disable when present.
    if "qwen" in model.lower():
        payload["think"] = False
    if options:
        payload["options"] = options
    data = _post_json(f"{_ollama_gen_base()}/api/generate", payload, {}, timeout)
    if not data:
        return None
    _warn_if_truncated(prompt, data, model)
    text = (data.get("response") or "").strip()
    return text or None


def _warn_if_truncated(prompt: str, data: dict, model: str) -> None:
    """Say so when the server silently dropped part of the prompt.

    Ollama truncates a prompt that exceeds the LOADED context window and returns
    done_reason="stop" with no error — the model then answers from whatever
    survived. Observed: prompt_eval_count=4095 against a 4096 window, and a
    confidently wrong answer about content that had been cut.

    The window is not ours to fix. It is set at load time by whoever loaded the
    model, this endpoint is shared, and the same model has been seen loaded at
    4096 and at 16384 while DECLARING 131072. Pinning num_ctx would force a
    reload and evict another tenant's model, so this reports rather than imposes.

    Detection is direct: compare the tokens the server says it evaluated against
    a rough estimate of what we sent. ~4 chars per token is crude, so the margin
    is generous — this must not cry wolf on ordinary prompts.
    """
    try:
        evaluated = int(data.get("prompt_eval_count") or 0)
    except (TypeError, ValueError):
        return
    if evaluated <= 0:
        return

    # Thresholding on a chars/4 token estimate does not work — it is off by ±25%
    # on ordinary text, so any threshold either misses real truncation or fires
    # on normal prompts. Truncation has a much sharper signature: the server
    # stops exactly AT the window, and windows are powers of two. Observed
    # 4095 against a 4096 window.
    estimated = max(1, len(prompt) // 4)
    for window in (2048, 4096, 8192, 16384, 32768, 65536):
        if abs(evaluated - window) <= 2 and estimated > window:
            _log.warning(
                "prompt TRUNCATED by the context window: model=%s evaluated %d "
                "tokens against an apparent %d-token window, for a ~%d-token "
                "prompt. The answer was formed from a PARTIAL prompt and may be "
                "confidently wrong. Load the model with a larger num_ctx, or "
                "shorten the grounding.",
                model, evaluated, window, estimated,
            )
            return


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
            # fail-open: partial embeds are useless for pairwise cosine
            _log.warning("embed_texts: %s returned no data for batch %d — "
                         "returning no vectors", url, i // batch)
            return []
        try:
            out.extend(d["embedding"] for d in data["data"])
        except (KeyError, TypeError) as exc:
            _log.warning("embed_texts: malformed embedding payload from %s: %r — "
                         "returning no vectors", url, exc)
            return []
    return out


def cosine(a, b):
    """Delegates to mcp/vecmath.py. None when the comparison is unanswerable;
    callers that need a float choose their own default rather than inheriting
    0.0, which reads as a confident 'not similar'."""
    return _vecmath_cosine(a, b)

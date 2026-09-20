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
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("loci-mcp.llm_local")

# GENERATION vars only, read at CALL time: OLLAMA_BASE_URL is the EMBEDDING endpoint, and loci_groom.load_env() rewrites it after import.
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

# A cold model load is ~70s even with keep_alive, so 120s covers a cold load plus generation.
_TIMEOUT = float(os.environ.get("OLLAMA_GEN_TIMEOUT", "120"))


def _flag_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_for_cloud(prompt: str, max_len: int = 12000) -> str:
    # Conservative redaction for obvious credential patterns before third-party egress.
    import re
    text = (prompt or "")[:max_len]
    patterns = [
        r"sk-or-v1-[A-Za-z0-9\-_]+",
        r"sk-[A-Za-z0-9\-_]{16,}",
        r"ghp_[A-Za-z0-9]{20,}",
        r"Bearer\s+[A-Za-z0-9\-_\.=]{16,}",
        r"(?i)(api[_-]?key\s*[:=]\s*)([^\s\"']+)",
    ]
    redacted = text
    for pat in patterns:
        redacted = re.sub(pat, lambda m: (m.group(1) if m.lastindex and m.lastindex >= 1 else "") + "[REDACTED]",
                          redacted)
    return redacted


def _cloud_refusal(why: str, *, provider: str = "", role: str = "", model: str = "") -> dict:
    out = {"text": "", "ok": False, "tier": "cloud-refused", "why": why[:300], "model": model}
    if provider:
        out["provider"] = provider
    if role:
        out["route_role"] = role
    return out


def _read_json_file(path: Path) -> dict:
    try:
        if path.exists():
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                return parsed
    except Exception:
        pass
    return {}


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _consume_cloud_daily_budget(*, max_tokens: int) -> Optional[str]:
    try:
        import backends
        call_budget = backends.cloud_daily_call_budget()
        token_budget = backends.cloud_daily_token_budget()
        state_path = Path(backends.cloud_budget_state_path())
    except Exception as exc:
        return f"cloud guardrail unavailable: {type(exc).__name__}: {exc}"[:220]

    if call_budget <= 0 and token_budget <= 0:
        return None

    day = datetime.now(timezone.utc).date().isoformat()
    try:
        state = _read_json_file(state_path)
        if state.get("day") != day:
            state = {"day": day, "calls": 0, "tokens": 0}
        calls = int(state.get("calls", 0))
        tokens = int(state.get("tokens", 0))
    except Exception as exc:
        return f"cloud guardrail state unreadable: {type(exc).__name__}: {exc}"[:220]

    if call_budget > 0 and (calls + 1) > call_budget:
        return f"cloud daily call budget exhausted ({calls}/{call_budget})"
    if token_budget > 0 and (tokens + max_tokens) > token_budget:
        return ("cloud daily token budget exhausted "
                f"({tokens}+{max_tokens}>{token_budget})")

    state["calls"] = calls + 1
    state["tokens"] = tokens + max_tokens
    try:
        _write_json_file(state_path, state)
    except Exception as exc:
        return f"cloud guardrail state unwritable: {type(exc).__name__}: {exc}"[:220]
    return None


def _normalize_role(role: Optional[str]) -> str:
    if role is None:
        return ""
    return str(role).strip().lower().replace("_", "-")


def _tmux_session_available(session_name: str) -> bool:
    """Check whether a configured tmux session is currently live."""
    session_name = (session_name or "").strip()
    if not session_name:
        return False
    try:
        import subprocess
        proc = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _tmux_offload_policy(role: Optional[str]) -> dict:
    """Resolve tmux offload policy for a generation role.

    The policy is advisory: if a mapped tmux lane exists and is live, we surface it as a
    higher-priority actor for expensive roles without disturbing the standard local/vLLM/cloud
    fallback flow when the feature is disabled or no tmux lane is available. In strict mode
    (require_mapped_session=True), a missing mapped lane becomes a hard fail instead of
    silently falling back.
    """
    try:
        import backends
    except Exception:
        return {
            "enabled": False,
            "role": _normalize_role(role),
            "priority": "normal",
            "session": None,
            "available": False,
            "require_mapped_session": False,
            "reason": "tmux backend unavailable",
        }

    normalized = _normalize_role(role)
    enabled = bool(backends.tmux_offload_enabled())
    if not enabled:
        return {
            "enabled": False,
            "role": normalized,
            "priority": "normal",
            "session": None,
            "available": False,
            "require_mapped_session": False,
            "reason": "feature disabled",
        }

    mappings = {str(k).lower(): str(v) for k, v in (backends.tmux_offload_role_sessions() or {}).items()}
    expensive = {str(v).strip().lower() for v in (backends.tmux_offload_expensive_roles() or set())}
    session_name = mappings.get(normalized, "") or ""
    is_expensive = normalized in expensive
    if session_name:
        available = _tmux_session_available(session_name)
        return {
            "enabled": True,
            "role": normalized,
            "priority": "expensive" if is_expensive else "normal",
            "session": session_name,
            "available": available,
            "require_mapped_session": bool(backends.tmux_offload_require_mapped_session()),
            "reason": "session available" if available else "session missing",
        }

    return {
        "enabled": True,
        "role": normalized,
        "priority": "expensive" if is_expensive else "normal",
        "session": None,
        "available": False,
        "require_mapped_session": bool(backends.tmux_offload_require_mapped_session()),
        "reason": "no mapped session for role",
    }


def generate(prompt: str,
             model: str = "",
             fmt: Optional[str] = None,
             max_tokens: int = 256,
             temperature: float = 0.2,
             keep_alive: str = "30m",
             think: bool = False,
             role: Optional[str] = None) -> dict:
    """Generate text from the local Ollama model. Fail-open, never raises.

    Args:
        prompt: the prompt string.
        model: Ollama model tag. Defaults to the verified-good qwen2.5:3b [substrate].
        fmt: if 'json', request structured JSON output AND validate the body parses as
             JSON; a non-JSON body downgrades the result to ok=False.
        max_tokens: mapped to Ollama options.num_predict. NOTE for think=True callers:
             on thinking-capable models this budget is SHARED between the hidden
             reasoning trace and the visible answer (live-verified 2026-09-16 against
             gemma4:26b/qwen3.8: num_predict=220 with think=True spent the whole budget
             on `thinking` and returned an EMPTY `response`). Pass a materially larger
             max_tokens (~1000+) whenever think=True, or the answer will come back blank.
        temperature: mapped to Ollama options.temperature.
        keep_alive: pins the model resident to avoid the ~70s cold load [substrate].
                    Always included in the request body — this is the critical bit.
        think: opt-in reasoning mode for thinking-capable models (default False, matching
             every existing caller's expectation that `response` holds the whole answer).
             Set True only for single, quality-critical calls (e.g. a deep-think synthesis
             or self-reflection pass) where the extra latency and larger max_tokens are
             worth the higher-quality result; do NOT enable for high-fanout/cheap tiers.

    Returns:
        {'text': str, 'ok': bool, 'model': str}. On any failure text='' and ok=False.
    """
    normalized_role = _normalize_role(role)
    tmux_policy = _tmux_offload_policy(normalized_role or _heuristic_route(prompt).get("role"))
    if tmux_policy.get("enabled") and tmux_policy.get("require_mapped_session") and tmux_policy.get("session") and not tmux_policy.get("available"):
        return {
            "text": "",
            "ok": False,
            "model": model or "",
            "tier": "tmux-offload-refused",
            "route_role": tmux_policy.get("role"),
            "tmux_session": tmux_policy.get("session"),
            "why": f"tmux offload required for role '{tmux_policy.get('role')}' but session '{tmux_policy.get('session')}' is unavailable",
        }
    if tmux_policy.get("enabled") and tmux_policy.get("available"):
        logger.info(
            "llm_local.tmux_offload role=%s session=%s priority=%s",
            tmux_policy.get("role"),
            tmux_policy.get("session"),
            tmux_policy.get("priority"),
        )
    model_was_unspecified = not bool(model)
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
        # Reasoning-mode models (e.g. Qwen3-family "thinking" variants) route their entire
        # JSON answer into a separate `thinking` field and leave `response` empty when
        # fmt="json" is set, which silently degraded every caller (compress_text,
        # classify_text, verify_finding) to ok=False against such a model. Disabling
        # thinking (the default) keeps the answer in `response`. Harmless no-op for
        # non-thinking models (verified live) and ignored outright by Ollama builds that
        # predate the option. Callers that explicitly opt into think=True accept the
        # response-may-land-in-`thinking` risk; the fallback below still recovers it.
        "think": bool(think),
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
        payload = r.json()
        text = (payload.get("response") or "")
        if not text.strip():
            # Defensive fallback: some models/builds still route JSON output into
            # `thinking` even with think=False sent. Recover the answer from there rather
            # than scoring a false failure. Mirrors scripts/ab_eval_local_model.py.
            thinking = payload.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                text = thinking
    except Exception as exc:
        route = _supervisor_route(prompt, fmt=fmt, max_tokens=max_tokens) if model_was_unspecified else None
        fallback = _try_vllm(prompt, fmt=fmt, max_tokens=max_tokens,
                             temperature=temperature, endpoint_role=(route or {}).get("role"))
        if fallback is not None:
            return fallback
        cloud = _try_cloud_tier(prompt, fmt=fmt, max_tokens=max_tokens,
                                temperature=temperature, route=route, model=model)
        if cloud is not None:
            return cloud
        return fail(f"ollama {type(exc).__name__}: {exc}"[:300])

    if fmt == "json":
        # ok=True only if the body actually parses as JSON.
        try:
            json.loads(text)
        except Exception as exc:
            return {"text": text, "ok": False, "model": model,
                    "why": f"response was not valid JSON: {exc}"[:200]}

    out = {"text": text, "ok": True, "model": model}
    tmux_policy = _tmux_offload_policy(_normalize_role(role) or _heuristic_route(prompt).get("role"))
    if tmux_policy.get("enabled") and tmux_policy.get("session"):
        out["tmux_session"] = tmux_policy["session"]
        out["route_role"] = tmux_policy.get("role")
        out["tmux_priority"] = tmux_policy.get("priority")
    return out


def _try_vllm(prompt: str, *, fmt: Optional[str], max_tokens: int,
              temperature: float, endpoint_role: Optional[str] = None) -> Optional[dict]:
    """Second tier. Returns a result dict, or None if vLLM is not usable either.

    batched_gen already resolves the vLLM endpoint AND the model name the server
    actually registers (backends.vllm_model()), which is the part llm_local was
    getting wrong. Reusing it keeps one definition of both.
    """
    # OPT-IN: the vLLM backends resolves is another project's service, not Loci's. Set LOCI_VLLM_FALLBACK=1 when Loci has its own.
    if os.environ.get("LOCI_VLLM_FALLBACK", "").strip() in ("", "0"):
        return None
    try:
        import batched_gen
    except Exception:
        return None
    try:
        out = batched_gen.generate_batch([prompt], max_tokens=max_tokens, fmt=fmt,
                                         think=False, endpoint_role=endpoint_role)
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
            served = batched_gen._resolve_vllm_model(endpoint_role)
        except Exception:
            served = "vllm"
    return {"text": first.get("text", ""), "ok": True, "model": served, "tier": "vllm"}


def _supervisor_route(prompt: str, *, fmt: Optional[str], max_tokens: int) -> Optional[dict]:
    """Use a stronger local supervisor to route unspecified calls across cloud tier options."""
    try:
        import backends
        if not backends.cloud_tier_enabled():
            return None
        supervisor_model = backends.cloud_supervisor_model()
    except Exception:
        return None

    base = _gen_env() or _resolve_ollama()
    if not base:
        return _heuristic_route(prompt)

    instruction = (
        "Return strict JSON only: {\"provider\":\"openrouter|abliteration|vllm|ollama\","
        "\"role\":\"triage|coding|reasoning|synthesis|redteam\","
        "\"reason\":\"short\"}. Pick cloud provider only when local first-tier would"
        " likely degrade for this prompt."
    )
    body = {
        "model": supervisor_model,
        "prompt": f"{instruction}\n\nPrompt:\n{prompt[:5000]}",
        "stream": False,
        "keep_alive": "30m",
        "think": False,
        "format": "json",
        "options": {"num_predict": min(max_tokens, 220), "temperature": 0.0},
    }
    try:
        import requests
        r = requests.post(f"{base}/api/generate", json=body, timeout=min(_TIMEOUT, 45))
        r.raise_for_status()
        text = (r.json() or {}).get("response") or ""
        parsed = json.loads(text)
        provider = str(parsed.get("provider", "")).strip().lower()
        role = str(parsed.get("role", "")).strip().lower()
        if provider not in {"openrouter", "abliteration", "vllm", "ollama"}:
            return _heuristic_route(prompt)
        if role not in {"triage", "coding", "reasoning", "synthesis", "redteam"}:
            role = "reasoning"
        route = {"provider": provider, "role": role}
        logger.info("llm_local.supervisor_route provider=%s role=%s", provider, role)
        return route
    except Exception:
        return _heuristic_route(prompt)


def _heuristic_route(prompt: str) -> dict:
    p = (prompt or "").lower()
    if any(k in p for k in ("red team", "redteam", "adversarial", "jailbreak", "exploit")):
        return {"provider": "abliteration", "role": "redteam"}
    if any(k in p for k in ("refactor", "code", "python", "typescript", "compile", "test")):
        return {"provider": "openrouter", "role": "coding"}
    if len(p) > 3500 or any(k in p for k in ("synthesize", "long context", "compare", "summarize")):
        return {"provider": "openrouter", "role": "synthesis"}
    return {"provider": "openrouter", "role": "triage"}


def _try_cloud_tier(prompt: str, *, fmt: Optional[str], max_tokens: int,
                    temperature: float, route: Optional[dict], model: str = "") -> Optional[dict]:
    """Third tier: provider-selected cloud fallback (OpenRouter + Abliteration)."""
    try:
        import backends
        if not backends.cloud_tier_enabled():
            return None
        if not _flag_on("LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT"):
            why = ("cloud fallback refused: prompt export is disabled; "
                   "set LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT=1")
            logger.warning("llm_local: %s", why)
            return _cloud_refusal(why, role=(route or {}).get("role", ""))
        role = (route or {}).get("role") or "triage"
        preferred = (route or {}).get("provider", "openrouter")
        openrouter_model = backends.openrouter_model(role)
        abliteration_model = backends.abliteration_model(role)
        max_per_call = backends.cloud_max_tokens_per_call()
        deny_roles = backends.cloud_deny_roles()
        allowed_roles = backends.cloud_allowed_roles()
        deny_providers = backends.cloud_deny_providers()
    except Exception:
        return None

    if max_per_call > 0 and max_tokens > max_per_call:
        return _cloud_refusal(
            f"cloud fallback refused: max_tokens={max_tokens} exceeds per-call cap={max_per_call}",
            role=role,
            model=model,
        )
    if role in deny_roles:
        return _cloud_refusal(
            f"cloud fallback refused: role '{role}' is denied by policy",
            role=role,
            model=model,
        )
    if allowed_roles and role not in allowed_roles:
        return _cloud_refusal(
            f"cloud fallback refused: role '{role}' is not in allowed set",
            role=role,
            model=model,
        )

    providers = ["openrouter", "abliteration"] if preferred != "abliteration" else ["abliteration", "openrouter"]
    providers = [p for p in providers if p not in deny_providers]
    if not providers:
        return _cloud_refusal(
            f"cloud fallback refused: provider deny-list removed all candidates ({sorted(deny_providers)})",
            role=role,
            model=model,
        )
    budget_refusal = _consume_cloud_daily_budget(max_tokens=max_tokens)
    if budget_refusal:
        return _cloud_refusal(f"cloud fallback refused: {budget_refusal}", role=role, model=model)

    prompt_for_cloud = _sanitize_for_cloud(prompt)
    for provider in providers:
        if provider == "openrouter":
            try:
                import openrouter
                out = openrouter.generate_batch([prompt_for_cloud], model=openrouter_model,
                                                max_tokens=max_tokens, fmt=fmt,
                                                temperature=temperature)
                first = (out or [{}])[0]
                if first.get("ok"):
                    return {"text": first.get("text", ""), "ok": True,
                            "model": first.get("model", openrouter_model),
                            "tier": "cloud-openrouter", "route_role": role}
            except Exception:
                continue
        else:
            try:
                import abliteration
                out = abliteration.generate_batch([prompt_for_cloud], model=abliteration_model,
                                                  max_tokens=max_tokens, fmt=fmt,
                                                  temperature=temperature)
                first = (out or [{}])[0]
                if first.get("ok"):
                    return {"text": first.get("text", ""), "ok": True,
                            "model": first.get("model", abliteration_model),
                            "tier": "cloud-abliteration", "route_role": role}
            except Exception:
                continue
    return None

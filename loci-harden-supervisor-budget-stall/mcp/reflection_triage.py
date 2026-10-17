"""Optional semantic triage for reflection-loop findings.

The reflection loop already extracts deterministic signatures (errors/warnings/tool/event
counts). This module adds an OPTIONAL local-model pass that classifies a candidate finding's
likely root-cause category plus a coarse novelty judgment. It is advisory only.

Design mirrors mcp/query_expand.py / mcp/guardian.py:

- Generation is injectable for tests. ``gen_fn`` defaults to a LAZY import of
  ``llm_local.generate`` so importing this module never hard-requires the local model lane.
- Fail-open: any model error, timeout, malformed JSON, or invalid labels returns a
  well-formed degraded result and never raises.
- Output is a plain dict suitable to tuck under finding metadata as
  ``{"llm_triage": {...}}``.
"""
from __future__ import annotations

from typing import Callable, Optional

from model_json import extract_json_object

GenFn = Callable[..., dict]

_CATEGORIES = {
    "real_regression": "real_regression",
    "regression": "real_regression",
    "bug": "real_regression",
    "flaky_or_nondeterministic": "flaky_or_nondeterministic",
    "flaky": "flaky_or_nondeterministic",
    "nondeterministic": "flaky_or_nondeterministic",
    "config_or_environment": "config_or_environment",
    "config": "config_or_environment",
    "environment": "config_or_environment",
    "env": "config_or_environment",
    "noise_or_benign": "noise_or_benign",
    "noise": "noise_or_benign",
    "benign": "noise_or_benign",
    "unknown": "unknown",
    "unclear": "unknown",
}

_NOVELTY = {
    "novel_signal": "novel_signal",
    "novel": "novel_signal",
    "known_pattern": "known_pattern",
    "known": "known_pattern",
    "repeat": "known_pattern",
    "repeated": "known_pattern",
    "unclear": "unclear",
    "unknown": "unclear",
}

_PROMPT_TMPL = (
    "You are triaging a self-reflection finding mined from local agent artifacts.\n"
    "Classify the likely root-cause category and whether the signal appears novel.\n"
    "This is advisory only: do NOT invent certainty.\n\n"
    "Allowed category labels:\n"
    "- real_regression\n"
    "- flaky_or_nondeterministic\n"
    "- config_or_environment\n"
    "- noise_or_benign\n"
    "- unknown\n\n"
    "Allowed novelty labels:\n"
    "- novel_signal\n"
    "- known_pattern\n"
    "- unclear\n\n"
    "Respond with ONLY a JSON object of this exact shape, no prose:\n"
    '{{"category":"real_regression|flaky_or_nondeterministic|config_or_environment|noise_or_benign|unknown",'
    '"novelty":"novel_signal|known_pattern|unclear"}}\n\n'
    "kind: {kind}\n"
    "path: {path}\n"
    "sampling: {sampling}\n"
    "top_events: {events}\n"
    "top_tools: {tools}\n"
    "visible_errors: {errors}\n"
    "visible_warnings: {warnings}\n"
)


def _lazy_generate(prompt: str, *, fmt: Optional[str] = None, max_tokens: int = 256) -> dict:
    """Default gen_fn: lazily import llm_local.generate and route via gen_model."""
    try:
        from llm_local import generate
        try:
            import backends
            model = backends.ollama_gen_model()
        except Exception:
            model = ""
        return generate(prompt, model=model, fmt=fmt, max_tokens=max_tokens, temperature=0.0)
    except Exception:
        return {"text": "", "ok": False}


def _normalize_label(raw: object, mapping: dict[str, str]) -> Optional[str]:
    key = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return mapping.get(key)


def _degraded(error: str | None = None, *, category: Optional[str] = None,
              novelty: Optional[str] = None) -> dict:
    return {
        "category": category,
        "novelty": novelty,
        "degraded": True,
        "ok": False,
        "error": (error or "")[:200] or None,
    }


def classify_reflection_observation(
    kind: str,
    path: str,
    *,
    sampling_mode: str = "full",
    events: Optional[dict] = None,
    tools: Optional[dict] = None,
    errors: Optional[dict] = None,
    warnings: Optional[dict] = None,
    gen_fn: Optional[GenFn] = None,
) -> dict:
    """Classify one reflection-loop observation; never raises.

    Returns ``{"category", "novelty", "degraded", "ok", "error"}``.
    """
    prompt = _PROMPT_TMPL.format(
        kind=str(kind or ""),
        path=str(path or "")[:400],
        sampling=str(sampling_mode or "full"),
        events=dict(list((events or {}).items())[:6]),
        tools=dict(list((tools or {}).items())[:6]),
        errors=dict(list((errors or {}).items())[:4]),
        warnings=dict(list((warnings or {}).items())[:4]),
    )
    fn = gen_fn or _lazy_generate
    try:
        result = fn(prompt, fmt="json", max_tokens=120)
    except Exception as exc:
        return _degraded(f"generate() raised: {exc}")

    if not isinstance(result, dict) or not result.get("ok"):
        why = (result or {}).get("why") if isinstance(result, dict) else "not ok"
        return _degraded(str(why or "empty response"))

    obj = extract_json_object(str(result.get("text", "")))
    if obj is None:
        return _degraded("unparseable response")

    category = _normalize_label(obj.get("category"), _CATEGORIES)
    novelty = _normalize_label(obj.get("novelty"), _NOVELTY)
    if not category or not novelty:
        return _degraded("invalid labels", category=category, novelty=novelty)

    return {
        "category": category,
        "novelty": novelty,
        "degraded": False,
        "ok": True,
        "error": None,
    }

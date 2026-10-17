"""Advisory-only LLM scan for undeclared implicit wiring obligations.

This module is intentionally separate from wiring_obligation_declare/list/resolve:
it NEVER reads or writes investigation state, and it never auto-declares or
auto-resolves anything. It only scans a snippet / diff hunk / file body and
returns candidate obligations a human or agent may choose to declare manually.

Design mirrors the house local-model wrappers:

- Generation is injectable via ``gen_fn`` and defaults to a lazy import of
  ``llm_local.generate``.
- The scanner is fail-open: on timeout / backend error / bad JSON / unparseable
  output it returns ``{"candidates": [], "degraded": True, ...}`` and never raises.
- The output is bounded, normalized JSON so the write-path tool can be called
  explicitly with reviewed arguments later.
"""
from __future__ import annotations

from typing import Callable, Optional

from model_json import extract_json_object

GenFn = Callable[..., dict]

_MAX_CONTENT_CHARS = 12_000
_MAX_EVIDENCE_CHARS = 280
_MAX_CANDIDATES = 6

_PROMPT_TEMPLATE = (
    "You review code for implicit follow-up obligations that are NOT already tracked.\n"
    "A candidate obligation exists only when calling a function or method creates a\n"
    "required later action for the caller or another cooperating component, such as:\n"
    "- caller must later release/unlock/close/reset/restore/stop/commit/rollback\n"
    "- the code returns a lease/handle/token/resource that implies a required counterpart call\n"
    "- a comment or docstring explicitly says the caller must do something after calling\n"
    "- the code sets a flag/mode/override that another path must undo\n\n"
    "Do NOT flag:\n"
    "- cleanup already handled inside the same function\n"
    "- resources safely wrapped by context managers or obvious try/finally cleanup\n"
    "- ordinary helper logic with no required external follow-up\n"
    "- vague comments with no concrete required action\n\n"
    "Return ONLY a JSON object with this exact shape:\n"
    '{{"candidates":[{{"description":"...",'
    '"evidence_excerpt":"...",'
    '"confidence":"low|medium|high",'
    '"suggested_declare_args":{{"class_name":"...",'
    '"method_name":"...",'
    '"expected_effect":"..."}}}}]}}\n'
    'If there are no credible candidates, return {{"candidates":[]}}.\n\n'
    "Subject path: {path}\n"
    "Subject context: {context}\n"
    "Code to inspect:\n"
    "```python\n{content}\n```"
)


def _lazy_generate(prompt: str, *, fmt: Optional[str] = None, max_tokens: int = 256) -> dict:
    """Default gen_fn: lazy llm_local import, fail-open like sibling modules."""
    try:
        import backends
        from llm_local import generate

        return generate(
            prompt,
            model=backends.ollama_verify_model(),
            fmt=fmt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception:
        return {"text": "", "ok": False}


def _degraded(error: str) -> dict:
    return {"candidates": [], "degraded": True, "error": error[:200]}


def _clean_text(value: object, *, cap: int) -> str:
    text = str(value or "").strip()
    if len(text) > cap:
        text = text[: cap - 3].rstrip() + "..."
    return text


def _clean_confidence(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"low", "medium", "high"}:
        return raw
    return "medium"


def _clean_suggested_args(raw: object, *, description: str) -> dict:
    data = raw if isinstance(raw, dict) else {}
    expected_effect = _clean_text(data.get("expected_effect", ""), cap=180)
    if not expected_effect:
        expected_effect = description
    return {
        "class_name": _clean_text(data.get("class_name", ""), cap=120),
        "method_name": _clean_text(data.get("method_name", ""), cap=120),
        "expected_effect": expected_effect,
    }


def _clean_candidate(raw: object) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    description = _clean_text(raw.get("description", ""), cap=220)
    if not description:
        return None
    return {
        "description": description,
        "evidence_excerpt": _clean_text(raw.get("evidence_excerpt", ""), cap=_MAX_EVIDENCE_CHARS),
        "confidence": _clean_confidence(raw.get("confidence")),
        "suggested_declare_args": _clean_suggested_args(
            raw.get("suggested_declare_args"), description=description
        ),
    }


def scan(
    content: str,
    *,
    path: str = "",
    context: str = "",
    gen_fn: Optional[GenFn] = None,
    max_candidates: int = _MAX_CANDIDATES,
) -> dict:
    """Scan code text for implicit obligations that might merit declaration.

    Returns ``{"candidates": [...], "degraded": bool, "error": str|None}``.
    This function is advisory-only and read-only: it does not touch the
    investigation store, and callers must explicitly decide whether to call
    ``wiring_obligation_declare`` with any suggestion.
    """
    text = (content or "").strip()
    if not text:
        return {"candidates": [], "degraded": False, "error": None}

    prompt = _PROMPT_TEMPLATE.format(
        path=(path or "").strip() or "(unspecified)",
        context=(context or "").strip() or "(none)",
        content=text[:_MAX_CONTENT_CHARS],
    )

    fn = gen_fn or _lazy_generate
    try:
        result = fn(prompt, fmt="json", max_tokens=500)
    except Exception as exc:
        return _degraded(f"generate() raised: {exc}")

    if not isinstance(result, dict) or not result.get("ok"):
        why = ""
        if isinstance(result, dict):
            why = str(result.get("why") or "")
        return _degraded(why or "generation unavailable")

    obj = extract_json_object(str(result.get("text", "")))
    if not isinstance(obj, dict):
        return _degraded("unparseable response")

    raw_candidates = obj.get("candidates")
    if raw_candidates is None:
        return _degraded("response missing candidates")
    if not isinstance(raw_candidates, list):
        return _degraded("response candidates was not a list")

    candidates: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in raw_candidates:
        candidate = _clean_candidate(item)
        if candidate is None:
            continue
        args = candidate["suggested_declare_args"]
        key = (
            candidate["description"].lower(),
            args["class_name"].lower(),
            args["method_name"].lower(),
            args["expected_effect"].lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
        if len(candidates) >= max(1, int(max_candidates)):
            break

    return {"candidates": candidates, "degraded": False, "error": None}

"""Semantic corroboration for conflict detection's same-investigation neighbours.

_detect_conflicts() already does the cheap part well: Qdrant finds same-investigation
neighbours whose cosine is high enough that they are plausibly about the same thing.
But the write/no-write decision after that was pure heuristics (gap -> observed,
assumed -> non-assumed, optional negation token mismatch), which means it could not
tell "same topic" from "same fact with opposite polarity". That loses semantic
contradictions and over-relies on wording.

This module is deliberately the same shape as guardian.py / verify.py: a thin,
fail-open wrapper around llm_local.generate(), with an injectable ``gen_fn`` for
tests and callers that already hold a warm client. It never raises. Any import
error, backend outage, timeout, empty reply, or unparseable JSON becomes
``{"verdict": None, "ok": False, ...}``, so the caller can preserve the existing
heuristic-only behaviour unchanged.
"""
from __future__ import annotations

from typing import Callable, Optional

_VALID_VERDICTS = frozenset(("contradict", "agree", "same_topic_no_conflict"))

_PROMPT_TEMPLATE = (
    "You are adjudicating whether two findings from the same investigation directly "
    "conflict.\n\n"
    "A contradiction means one finding asserts a specific fact and the other denies "
    "that SAME fact. Mere topic overlap, different sub-claims, missing detail, or "
    "different confidence does NOT count.\n\n"
    "Return ONLY valid JSON, with no prose or markdown:\n"
    '{{"verdict":"contradict|agree|same_topic_no_conflict","reason":"<short>"}}\n\n'
    "Prefer same_topic_no_conflict when the findings are related but do not directly "
    "oppose each other, or when the text is too ambiguous to prove a contradiction.\n\n"
    "FINDING A (type={type_a}):\n{finding_a}\n\n"
    "FINDING B (type={type_b}):\n{finding_b}\n"
)

_MAX_TEXT_CHARS = 1800


def _lazy_generate(prompt: str, *, fmt: Optional[str] = None, max_tokens: int = 256) -> dict:
    """Default gen_fn: route through verify_model lazily so import stays dependency-light.

    This is the same adversarial-reasoning family as verify_finding: the model is not
    free-chatting, it is judging whether two near-neighbour claims actually negate the
    same fact. Reusing backends.ollama_verify_model() keeps one operator knob for the
    stronger/slower local reasoning tier instead of inventing a second one for the same
    class of task. Fail-open to ``{"text":"", "ok":False}``.
    """
    try:
        from llm_local import generate
        try:
            import backends
            model = backends.ollama_verify_model()
        except Exception:
            model = ""
        return generate(prompt, model=model, fmt=fmt, max_tokens=max_tokens, temperature=0.0)
    except Exception:
        return {"text": "", "ok": False}


def judge_conflict(
    finding_a: str,
    finding_b: str,
    *,
    type_a: str = "",
    type_b: str = "",
    gen_fn: Optional[Callable[..., dict]] = None,
) -> dict:
    """Judge whether two findings directly contradict. Never raises.

    Args:
        finding_a: text of the first finding.
        finding_b: text of the second finding.
        type_a: optional record/type label for finding_a (observed/gap/assumed/...).
        type_b: optional record/type label for finding_b.
        gen_fn: injectable generation fn matching llm_local.generate's contract.
            None -> lazy llm_local.generate routed through ollama_verify_model().

    Returns:
        ``{"verdict": str|None, "reason": str, "ok": bool, "error": str|None}``.
        ``verdict`` is one of ``contradict``, ``agree``, ``same_topic_no_conflict`` when
        ``ok`` is True. Any failure degrades to ``verdict=None`` so callers can keep the
        existing heuristic result unchanged.
    """
    text_a = str(finding_a or "").strip()
    text_b = str(finding_b or "").strip()
    if not text_a or not text_b:
        return {"verdict": None, "reason": "", "ok": True, "error": None}

    prompt = _PROMPT_TEMPLATE.format(
        type_a=(type_a or "unknown"),
        type_b=(type_b or "unknown"),
        finding_a=text_a[:_MAX_TEXT_CHARS],
        finding_b=text_b[:_MAX_TEXT_CHARS],
    )
    fn = gen_fn or _lazy_generate
    try:
        result = fn(prompt, fmt="json", max_tokens=120)
    except Exception as exc:
        return {"verdict": None, "reason": "", "ok": False,
                "error": f"generate() raised: {exc}"[:200]}

    if not isinstance(result, dict):
        return {"verdict": None, "reason": "", "ok": False, "error": "generate() returned non-dict"}
    if not result.get("ok"):
        why = result.get("why") or "generation failed"
        return {"verdict": None, "reason": "", "ok": False, "error": str(why)[:200]}

    raw = str(result.get("text", "") or "").strip()
    if not raw:
        why = result.get("why") or "empty response"
        return {"verdict": None, "reason": "", "ok": False, "error": str(why)[:200]}

    try:
        from memcheck.checks.contradiction_llm import extract_json as _extract_json
    except Exception as exc:
        return {"verdict": None, "reason": "", "ok": False,
                "error": f"extract_json import failed: {exc}"[:200]}

    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        return {"verdict": None, "reason": "", "ok": False,
                "error": f"unparseable response: {raw[:80]!r}"}

    verdict = str(obj.get("verdict", "") or "").strip().lower()
    if verdict not in _VALID_VERDICTS:
        return {"verdict": None, "reason": "", "ok": False,
                "error": f"invalid verdict: {verdict or raw[:40]!r}"}

    reason = obj.get("reason")
    if not isinstance(reason, str):
        reason = "" if reason is None else str(reason)

    return {"verdict": verdict, "reason": reason.strip(), "ok": True, "error": None}

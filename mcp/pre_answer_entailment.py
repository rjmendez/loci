"""Advisory claim-entailment corroboration for investigation_pre_answer_check.

The pre-answer tool's deterministic lanes are intentionally cheap and fail-open:
lexical token overlap, dense-neighbour retrieval, a corroboration gate, and a
small negation heuristic. They are good at finding *candidate* evidence but not
at deciding whether that evidence really supports an analyst's EXACT wording.
A finding about host B, an alert rather than a compromise, or a tentative note
about "may have happened" can still share enough surface form to look
supportive.

This module adds an additive local-model check for that narrower question:
"does the cited evidence actually entail/support this exact claim, considering
subject, scope, time, modality, and negation?" It is deliberately advisory-only
-- callers must NEVER replace or flip the deterministic ``supported`` boolean
with this result. Its job is to surface corroborating metadata when a lexical
support call is potentially over-claiming.

Design mirrors guardian.py / verify.py:

- Thin wrapper around an injectable ``gen_fn`` so tests can stub the model.
- Default routing reuses verify._lazy_generate, which already targets
  backends.ollama_verify_model() for this stronger reasoning task.
- Fail-open: any import error, model error, or unparseable response returns a
  plain ``available=False`` dict rather than raising.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Optional

from model_json import extract_json_object

GenFn = Callable[..., dict]

_VALID_VERDICTS = ("confirmed", "refuted", "uncertain")
_MAX_EVIDENCE_ITEMS = 8
_MAX_EVIDENCE_CHARS = 1200
_FASTPATH_MIN_OVERLAP = 0.72
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
# Polarity cues. The reflex only compares tokens, so "No evidence that X" contains X
# verbatim; a support row whose negation cues differ from the claim's is left to the model.
_NEGATION_TOKENS = frozenset({
    "no", "not", "never", "none", "nothing", "nobody", "neither", "nor", "without",
    "false", "untrue", "cannot", "cant", "didn", "doesn", "don", "isn", "wasn", "weren",
    "aren", "hasn", "haven", "hadn", "won", "wouldn", "couldn", "shouldn",
    "refuted", "disproved", "disproven", "unconfirmed",
})

_PROMPT_TMPL = (
    "You are checking whether cited investigation evidence REALLY supports an EXACT claim.\n"
    "Judge only from the evidence shown. Consider subject identity, scope, time, modality,\n"
    "uncertainty, and negation. Evidence about a different host/user/file, about an alert\n"
    "instead of the claimed outcome, about a different time window, or expressed only as a\n"
    "possibility does NOT confirm the claim.\n\n"
    "Return ONLY a JSON object of this exact shape, with no prose outside it:\n"
    '{{"verdict": "confirmed|refuted|uncertain", "rationale": "brief why", "confidence": 0.0}}\n\n'
    "Use:\n"
    '  - "confirmed" only when the evidence itself supports the exact claim.\n'
    '  - "refuted" when the evidence points the other way or clearly mismatches\n'
    "    subject/scope/time/modality.\n"
    '  - "uncertain" when the evidence is relevant but insufficient, mixed, or ambiguous.\n\n'
    "CLAIM:\n{claim}\n\n"
    "CITED EVIDENCE:\n{evidence}\n"
)


def _coerce_verdict(raw) -> str:
    """Map model output to one of the supported verdicts; unknown -> uncertain."""
    if not isinstance(raw, str):
        return "uncertain"
    verdict = raw.strip().lower()
    return verdict if verdict in _VALID_VERDICTS else "uncertain"


def _coerce_confidence(raw) -> float:
    """Coerce confidence to [0,1]; invalid or non-finite values degrade to 0.0."""
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _is_nonfinite_confidence(raw) -> bool:
    """True for a confidence that parses as a number but is NaN, +/-inf or overflows."""
    try:
        return not math.isfinite(float(raw))
    except OverflowError:
        return True
    except (TypeError, ValueError):
        return False


def _unavailable(error: str = "", rationale: str = "") -> dict:
    """Well-formed fail-open result for an unavailable advisory entailment check."""
    err = error if isinstance(error, str) else ("" if error is None else str(error))
    why = rationale if isinstance(rationale, str) else ("" if rationale is None else str(rationale))
    return {
        "available": False,
        "verdict": None,
        "rationale": why.strip(),
        "confidence": 0.0,
        "degraded": True,
        "error": err.strip(),
    }


def _tokenize(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def _normalize_lexeme_text(text: str) -> str:
    return " ".join(m.group(0).lower() for m in _TOKEN_RE.finditer(str(text or "")))


def _lexical_overlap_ratio(claim: str, evidence_text: str) -> float:
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return 0.0
    evidence_tokens = _tokenize(evidence_text)
    if not evidence_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / float(len(claim_tokens))


def _is_prevalidated_support(item: dict) -> bool:
    """Best-effort marker check for support that was already validated upstream."""
    if not isinstance(item, dict):
        return False
    if bool(item.get("prevalidated")) or bool(item.get("validated")):
        return True
    status = str(item.get("validation_status") or "").strip().lower()
    if status in {"validated", "prevalidated", "verified", "tool_verified"}:
        return True
    strength = str(item.get("evidence_strength") or "").strip().lower()
    return strength in {"validated", "release_blocking", "generalizable"}


def _reflex_arc_fastpath(claim: str, evidence: list[dict]) -> dict | None:
    """Deterministic low-latency short-circuit for obvious entailment outcomes.

    Returns a full result payload when confidence gates are met, otherwise None.
    Fail-safe by design: thresholds are conservative and non-matches fall back
    to the model-backed check.
    """
    claim_text = str(claim or "").strip()
    if not claim_text:
        return None
    claim_norm = _normalize_lexeme_text(claim_text)
    supports = [row for row in (evidence or []) if isinstance(row, dict) and str(row.get("role") or "").lower() == "support"]
    contradictions = [row for row in (evidence or []) if isinstance(row, dict) and str(row.get("role") or "").lower() == "contradiction"]

    # Reflex rejection: strong lexical contradiction should not spend model latency.
    for row in contradictions:
        text = str(row.get("text") or row.get("snippet") or "").strip()
        if not text:
            continue
        overlap = _lexical_overlap_ratio(claim_text, text)
        if overlap >= _FASTPATH_MIN_OVERLAP:
            return {
                "available": True,
                "verdict": "refuted",
                "rationale": "reflex_fastpath: high-overlap contradiction evidence already present.",
                "confidence": 0.93,
                "degraded": False,
                "error": "",
            }

    # Reflex acceptance: exact or near-exact support match with no contradiction.
    if not contradictions:
        claim_negations = _tokenize(claim_text) & _NEGATION_TOKENS
        for row in supports:
            text = str(row.get("text") or row.get("snippet") or "").strip()
            if not text:
                continue
            if (_tokenize(text) & _NEGATION_TOKENS) != claim_negations:
                continue  # polarity differs: lexical overlap cannot tell support from denial
            text_norm = _normalize_lexeme_text(text)
            overlap = _lexical_overlap_ratio(claim_text, text)
            exactish = (claim_norm in text_norm) or (text_norm in claim_norm)
            if exactish and overlap >= 0.85:
                return {
                    "available": True,
                    "verdict": "confirmed",
                    "rationale": "reflex_fastpath: exact high-overlap support evidence; model bypassed.",
                    "confidence": 0.97,
                    "degraded": False,
                    "error": "",
                }
            if _is_prevalidated_support(row) and overlap >= _FASTPATH_MIN_OVERLAP:
                return {
                    "available": True,
                    "verdict": "confirmed",
                    "rationale": "reflex_fastpath: prevalidated support evidence with strong lexical overlap.",
                    "confidence": 0.9,
                    "degraded": False,
                    "error": "",
                }
    return None


def _render_evidence(evidence: list[dict]) -> str:
    """Render a small, bounded evidence bundle into prompt text."""
    blocks: list[str] = []
    for idx, item in enumerate((evidence or [])[:_MAX_EVIDENCE_ITEMS], start=1):
        if not isinstance(item, dict):
            continue
        body = str(item.get("text") or item.get("snippet") or "").strip()
        if not body:
            continue
        blocks.append(
            f"[{idx}] role={item.get('role') or 'support'} "
            f"id={item.get('evidence_id') or ''} "
            f"type={item.get('record_type') or ''} "
            f"source={item.get('source') or ''} "
            f"ts={item.get('ts') or ''}\n"
            f"{body[:_MAX_EVIDENCE_CHARS]}"
        )
    return "\n\n".join(blocks)


def check_claim_entailment(
    claim: str,
    evidence: list[dict],
    *,
    gen_fn: Optional[GenFn] = None,
) -> dict:
    """Advisory local-model check: does cited evidence entail/support this EXACT claim?

    Args:
        claim: the analyst claim being stress-tested.
        evidence: cited evidence rows (support / contradiction / baseline), each a plain
            dict carrying metadata plus ``text`` or ``snippet``.
        gen_fn: injectable generation callable matching verify._lazy_generate's
            contract. Defaults to verify._lazy_generate so model routing follows
            backends.ollama_verify_model().

    Returns:
        {"available": bool, "verdict": "confirmed"|"refuted"|"uncertain"|None,
         "rationale": str, "confidence": float, "degraded": bool, "error": str}.
        Fail-open: any error returns ``available=False`` and never raises. This
        result is advisory metadata only; callers must not let it replace their
        deterministic support verdict.
    """
    text = (claim or "").strip()
    rendered = _render_evidence(evidence)
    if not text or not rendered:
        return _unavailable()

    if gen_fn is None:
        try:
            import verify
            gen_fn = verify._lazy_generate
        except Exception as exc:
            return _unavailable(f"verify import failed: {exc}"[:200])

    try:
        reflex = _reflex_arc_fastpath(text, evidence)
    except Exception:
        reflex = None
    if isinstance(reflex, dict):
        return reflex

    prompt = _PROMPT_TMPL.format(claim=text, evidence=rendered)

    try:
        result = gen_fn(prompt, fmt="json", max_tokens=220)
    except Exception as exc:
        return _unavailable(f"generate() raised: {exc}"[:200])

    if not isinstance(result, dict) or not result.get("ok"):
        rationale = (result.get("text", "") if isinstance(result, dict) else "")
        error = (result.get("why", "") if isinstance(result, dict) else "") or "model unavailable"
        return _unavailable(str(error)[:200], rationale)

    raw = result.get("text", "")
    obj = extract_json_object(raw)
    if obj is None:
        return _unavailable(f"unparseable response: {str(raw)[:120]!r}")

    if _is_nonfinite_confidence(obj.get("confidence")):
        return _unavailable(f"non-finite confidence: {str(raw)[:120]!r}")

    rationale = obj.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = obj.get("reasoning")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = obj.get("refutation")
    if not isinstance(rationale, str):
        rationale = "" if rationale is None else str(rationale)

    return {
        "available": True,
        "verdict": _coerce_verdict(obj.get("verdict")),
        "rationale": rationale.strip(),
        "confidence": _coerce_confidence(obj.get("confidence")),
        "degraded": False,
        "error": "",
    }

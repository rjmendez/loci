"""Advisory local-model audit for Mnemosyne consolidation quality.

This checker is strictly additive: it reviews a completed sleep/consolidation
merge after the fact and reports whether the merged summary appears to preserve
the distinct, non-redundant information from the original working-memory
entries. It MUST never influence what Mnemosyne writes.

Design mirrors guardian.py / pre_answer_entailment.py:

- Thin wrapper around an injectable ``gen_fn`` so tests can stub the model.
- Default routing reuses llm_local.generate with backends.ollama_verify_model().
- Fail-open: any import error, model error, or unparseable response returns a
  degraded-but-well-formed advisory result rather than raising.
"""
from __future__ import annotations

from typing import Callable, Optional

from model_json import extract_json_object

GenFn = Callable[..., dict]

_VALID_VERDICTS = ("preserved", "lost_or_conflated", "uncertain")
_MAX_SOURCE_ITEMS = 8
_MAX_SOURCE_CHARS = 900
_MAX_SUMMARY_CHARS = 1400

_PROMPT_TMPL = (
    "You are auditing whether a memory-consolidation merge preserved distinct facts.\n"
    "Compare the ORIGINAL ENTRIES against the MERGED RESULT. Answer whether the merged\n"
    "result preserves the distinct, non-redundant information from both/all originals,\n"
    "or whether something specific appears lost, collapsed together incorrectly, or\n"
    "conflated.\n\n"
    "Return ONLY a JSON object of this exact shape, with no prose outside it:\n"
    '{{"verdict": "preserved|lost_or_conflated|uncertain", '
    '"concern": "brief specific concern or empty string", "confidence": 0.0}}\n\n'
    'Use "preserved" when the merged result covers the distinct substance of the inputs.\n'
    'Use "lost_or_conflated" when a specific detail, distinction, or polarity appears lost.\n'
    'Use "uncertain" when the merge is too compressed or ambiguous to tell.\n\n'
    "ORIGINAL ENTRIES:\n{sources}\n\n"
    "MERGED RESULT:\n{summary}\n"
)


def _lazy_generate(prompt: str, *, fmt: Optional[str] = None,
                   max_tokens: int = 256) -> dict:
    """Default advisory generator routed through the verify model."""
    import llm_local

    try:
        import backends
        model = backends.ollama_verify_model()
    except Exception:
        model = ""

    return llm_local.generate(
        prompt,
        model=model,
        fmt=fmt,
        max_tokens=max_tokens,
        temperature=0.0,
    )


def _coerce_verdict(raw) -> str:
    if not isinstance(raw, str):
        return "uncertain"
    verdict = raw.strip().lower()
    return verdict if verdict in _VALID_VERDICTS else "uncertain"


def _coerce_confidence(raw) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value != value:
        return 0.0
    return max(0.0, min(1.0, value))


def _unavailable(error: str = "", concern: str = "") -> dict:
    return {
        "available": False,
        "verdict": None,
        "concern": (concern or "").strip(),
        "confidence": 0.0,
        "degraded": True,
        "error": (error or "").strip(),
    }


def _render_sources(source_entries: list[dict]) -> str:
    blocks: list[str] = []
    for idx, item in enumerate((source_entries or [])[:_MAX_SOURCE_ITEMS], start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("content") or item.get("text") or "").strip()
        if not text:
            continue
        blocks.append(
            f"[{idx}] id={item.get('id') or ''} "
            f"source={item.get('source') or ''} "
            f"ts={item.get('timestamp') or ''}\n"
            f"{text[:_MAX_SOURCE_CHARS]}"
        )
    return "\n\n".join(blocks)


def audit_merge_quality(
    merged_summary: str,
    source_entries: list[dict],
    *,
    gen_fn: Optional[GenFn] = None,
) -> dict:
    """Advisory local-model spot-check for one completed Mnemosyne merge."""
    summary = str(merged_summary or "").strip()
    rendered_sources = _render_sources(source_entries)
    meaningful_sources = [
        item for item in (source_entries or [])
        if isinstance(item, dict) and str(item.get("content") or item.get("text") or "").strip()
    ]
    if not summary or not rendered_sources or len(meaningful_sources) < 2:
        return _unavailable()

    if gen_fn is None:
        try:
            gen_fn = _lazy_generate
        except Exception as exc:
            return _unavailable(f"generator setup failed: {exc}"[:200])

    prompt = _PROMPT_TMPL.format(
        sources=rendered_sources,
        summary=summary[:_MAX_SUMMARY_CHARS],
    )

    try:
        result = gen_fn(prompt, fmt="json", max_tokens=220)
    except Exception as exc:
        return _unavailable(f"generate() raised: {exc}"[:200])

    if not isinstance(result, dict) or not result.get("ok"):
        concern = result.get("text", "") if isinstance(result, dict) else ""
        error = (result.get("why", "") if isinstance(result, dict) else "") or "model unavailable"
        return _unavailable(str(error)[:200], str(concern)[:200])

    raw = result.get("text", "")
    obj = extract_json_object(raw)
    if obj is None:
        return _unavailable(f"unparseable response: {str(raw)[:120]!r}")

    concern = obj.get("concern")
    if not isinstance(concern, str):
        concern = obj.get("rationale")
    if not isinstance(concern, str):
        concern = "" if concern is None else str(concern)

    return {
        "available": True,
        "verdict": _coerce_verdict(obj.get("verdict")),
        "concern": concern.strip(),
        "confidence": _coerce_confidence(obj.get("confidence")),
        "degraded": False,
        "error": "",
    }

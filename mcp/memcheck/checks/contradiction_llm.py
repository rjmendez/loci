"""Semantic contradiction check — the deep_think -> loci merge for self-check.

The lexical ``run_contradiction`` keys on token overlap (Szymkiewicz-Simpson >= 0.5)
plus a negation regex. Empirically that both *misses* real contradictions whose
wording diverges (``"LIMITED to 1 expansion"`` vs ``"NOT limited to 1 expansion"``
share too little surface vocab to clear the bar) and *invents* false ones (two
findings about different tools that happen to share jargon, one containing
"should not"). Bag-of-words can't tell "same subject" from "shared words", nor
"negated phrasing" from "actually contradicts".

This replaces both heuristics with the two pieces of tech the merge brings in:

  Stage 1 - **subject gate** (deep_think_loci grounding tech): embed every
            finding once and pair only those whose cosine clears a threshold -
            i.e. findings genuinely *about the same thing*. This is what the
            token-overlap prefilter was a poor proxy for, and it drops the
            different-tools false positive.
  Stage 2 - **polarity judge** (deep_think alarm-scan feature): an LLM decides,
            per surviving pair, whether they make directly incompatible factual
            claims - ignoring wording/emphasis/scope. This catches the semantic
            negations the regex can't.

Pure logic with injected ``embed_fn`` / ``llm_fn`` (defaults wire to ``llm.py``),
so the pipeline is unit-testable with zero network. Fail-open throughout: any
embed/LLM failure degrades to the caller's existing (lexical) verdicts, never
raising.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from ..verdict import Verdict, make_signature, new_verdict, redact_excerpt
from .contradiction import (
    EWC_HIGH_THRESH,
    _MAX_FINDINGS,
    _finding_id,
    _protection_weight,
)

__all__ = [
    "run_contradiction_llm",
    "verify_and_merge",
    "DEFAULT_SUBJECT_THRESHOLD",
]

_log = logging.getLogger("memcheck.contradiction_llm")

# Cosine above which two findings are treated as "about the same subject" and
# handed to the polarity judge. Aligned with deep_think_loci's grounding gate,
# whose offline sweep put on-target similarity above ~0.60 and bleed below it.
DEFAULT_SUBJECT_THRESHOLD = 0.62

# Cap LLM judge calls per run so a large investigation can't fan out unbounded.
_MAX_JUDGE_PAIRS = 40


_JUDGE_PROMPT = """\
You are a contradiction detector. Two findings from the same investigation are \
below; they are already known to concern the same subject.

Decide whether they FACTUALLY CONTRADICT: one asserts a specific fact and the \
other denies that same fact (A says X is true, B says X is false). IGNORE any \
difference in wording, emphasis, framing, confidence, or scope - only a direct \
factual conflict counts.

FINDING A: {a}

FINDING B: {b}

Return ONLY valid JSON, no other text:
{{"contradict": true or false, "claim": "<the specific fact in dispute, or empty>"}}"""


def _wire(embed_fn, llm_fn, cosine_fn):
    """Late-bind the default llm backend so the module imports without it."""
    if embed_fn is None or llm_fn is None or cosine_fn is None:
        from .. import llm as _llm

        embed_fn = embed_fn or _llm.embed_texts
        llm_fn = llm_fn or _llm.call_llm
        cosine_fn = cosine_fn or _llm.cosine
    return embed_fn, llm_fn, cosine_fn


def _extract_json(text: str) -> dict | None:
    """Tolerant JSON extraction from a model reply (handles ``` fences)."""
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    if "```" in t:
        inner = t.split("```", 1)[1]
        if inner.startswith("json"):
            inner = inner[4:]
        inner = inner.split("```", 1)[0].strip()
        try:
            return json.loads(inner)
        except Exception:
            pass
    start = t.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(t[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start : i + 1])
                    except Exception:
                        break
    return None


def _prepare(findings: list[dict], max_findings: int) -> list[tuple[str, str, dict]]:
    """(id, text, raw) for the most-recent, non-empty findings."""
    source = (findings or [])[-max_findings:] if findings else []
    base = len(findings or []) - len(source)
    prepared: list[tuple[str, str, dict]] = []
    for offset, f in enumerate(source):
        if not isinstance(f, dict):
            continue
        text = str(f.get("text", "") or "")
        if not text.strip():
            continue
        prepared.append((_finding_id(f, base + offset), text, f))
    return prepared


def _gate_and_judge(
    prepared: list[tuple[str, str, dict]],
    vectors: list[list[float]],
    llm_fn: Callable[[str], Optional[str]],
    cosine_fn: Callable[[list[float], list[float]], float],
    subject_threshold: float,
    max_pairs: int,
) -> list[Verdict]:
    """Stage 1 (cosine subject gate) + Stage 2 (LLM polarity judge) over vectors."""
    # Stage 1: subject gate - collect same-subject candidate pairs by cosine.
    candidates: list[tuple[float, int, int]] = []
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            try:
                c = cosine_fn(vectors[i], vectors[j])
            except Exception:
                continue
            if c >= subject_threshold:
                candidates.append((c, i, j))
    if not candidates:
        return []
    # Judge the most-similar pairs first; cap to bound LLM spend.
    candidates.sort(key=lambda t: t[0], reverse=True)
    if len(candidates) > max_pairs:
        _log.debug("subject gate produced %d pairs; capping LLM judge to %d",
                   len(candidates), max_pairs)
        candidates = candidates[:max_pairs]

    # Stage 2: polarity judge - LLM confirms a true factual contradiction.
    verdicts: list[Verdict] = []
    for cos_val, i, j in candidates:
        id_a, text_a, raw_a = prepared[i]
        id_b, text_b, raw_b = prepared[j]
        prompt = _JUDGE_PROMPT.format(a=text_a[:1500], b=text_b[:1500])
        try:
            reply = llm_fn(prompt)
        except Exception as exc:  # noqa: BLE001 - fail-open per pair
            _log.debug("llm judge failed for pair (%s,%s): %r", id_a, id_b, exc)
            continue
        parsed = _extract_json(reply or "")
        if not parsed or not bool(parsed.get("contradict")):
            continue

        claim = str(parsed.get("claim") or "").strip()
        excerpt = redact_excerpt(f"{text_a} <> {text_b}")
        pair_key = tuple(sorted((id_a, id_b)))

        # EWC protection (shared with the lexical check): a single datum that
        # contradicts an *established* finding is provisional until corroborated.
        max_prot = max(_protection_weight(raw_a), _protection_weight(raw_b))
        provisional = max_prot >= EWC_HIGH_THRESH
        base_rationale = (
            f"LLM judge confirmed factual contradiction (subject cosine={cos_val:.2f})"
        )
        if claim:
            base_rationale += f"; disputed claim: {claim[:160]}"
        if provisional:
            rationale = (
                base_rationale
                + f" - PROVISIONAL: established finding (prot={max_prot:.2f})"
                " requires corroboration before the verdict is enforced"
            )
            confidence = 0.55
        else:
            rationale = base_rationale
            confidence = 0.80

        verdicts.append(
            new_verdict(
                subject_kind="memory",
                subject_signature=make_signature("memory", "|".join(pair_key)),
                subject_excerpt=excerpt,
                verdict_type="contradiction",
                decision="flag",
                confidence=confidence,
                rationale=rationale,
                source="llm",
                refs=[id_a, id_b],
                provisional=provisional,
            )
        )
    return verdicts


def run_contradiction_llm(
    findings: list[dict],
    *,
    embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    llm_fn: Optional[Callable[[str], Optional[str]]] = None,
    cosine_fn: Optional[Callable[[list[float], list[float]], float]] = None,
    subject_threshold: float = DEFAULT_SUBJECT_THRESHOLD,
    max_findings: int = _MAX_FINDINGS,
    max_pairs: int = _MAX_JUDGE_PAIRS,
) -> list[Verdict]:
    """Flag pairs of findings an LLM confirms are factual contradictions.

    Returns ``contradiction`` verdicts with ``source="llm"``. Fail-open: returns
    ``[]`` if embeddings or the LLM are unavailable.
    """
    embed_fn, llm_fn, cosine_fn = _wire(embed_fn, llm_fn, cosine_fn)
    prepared = _prepare(findings, max_findings)
    if len(prepared) < 2:
        return []
    try:
        vectors = embed_fn([p[1] for p in prepared])
    except Exception as exc:  # noqa: BLE001 - fail-open
        _log.debug("embed_fn failed, degrading to no LLM contradictions: %r", exc)
        return []
    if not vectors or len(vectors) != len(prepared):
        _log.debug("embeddings unavailable/mismatched (%s vs %s) - skipping",
                   len(vectors) if vectors else 0, len(prepared))
        return []
    return _gate_and_judge(
        prepared, vectors, llm_fn, cosine_fn, subject_threshold, max_pairs
    )


def verify_and_merge(
    findings: list[dict],
    lexical_verdicts: list[Verdict],
    *,
    embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    llm_fn: Optional[Callable[[str], Optional[str]]] = None,
    cosine_fn: Optional[Callable[[list[float], list[float]], float]] = None,
    subject_threshold: float = DEFAULT_SUBJECT_THRESHOLD,
    max_findings: int = _MAX_FINDINGS,
    max_pairs: int = _MAX_JUDGE_PAIRS,
) -> list[Verdict]:
    """Return the LLM-verified contradiction set, or the lexical set on failure.

    Semantics: when embeddings actually run, the LLM-confirmed set *supersedes*
    the lexical verdicts (it has both better candidate generation and a real
    judge) - dropping lexical false positives and adding semantic ones the regex
    missed. When embeddings/LLM are unreachable, the lexical verdicts are
    returned unchanged, so an endpoint outage never silently erases the cheap
    signal. This is the only place the two checks are reconciled.
    """
    embed_fn, llm_fn, cosine_fn = _wire(embed_fn, llm_fn, cosine_fn)
    prepared = _prepare(findings, max_findings)
    if len(prepared) < 2:
        return lexical_verdicts
    try:
        vectors = embed_fn([p[1] for p in prepared])
    except Exception as exc:  # noqa: BLE001 - fail-open
        _log.debug("verify_and_merge embed failed, keeping lexical: %r", exc)
        return lexical_verdicts
    if not vectors or len(vectors) != len(prepared):
        # Embeddings genuinely unavailable -> cannot verify; keep lexical.
        return lexical_verdicts
    return _gate_and_judge(
        prepared, vectors, llm_fn, cosine_fn, subject_threshold, max_pairs
    )

"""Contradiction check — "if you changed your read, say so".

Finding-vs-finding, negation-aware. When two findings share enough tokens
(they are about the same thing) but disagree on negation polarity (one asserts,
the other negates), they likely contradict. This surfaces silent revisions the
analyst should reconcile.

Pure: takes plain dicts, returns ``list[Verdict]``. Fail-safe: a malformed
record is skipped, never raised.

Performance: pairwise comparison is bounded. Findings are first reduced to a
token-set, then only pairs that clear a cheap token-overlap prefilter are
checked for polarity. The number of findings considered is capped
(``max_findings``) so a huge investigation never triggers an O(n^2) blowup on
the full set.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Optional

from ..verdict import Verdict, make_signature, new_verdict, redact_excerpt

__all__ = ["run_contradiction"]

# EWC-style finding protection (Kirkpatrick 2017; CLS McClelland 1995).
# A finding with confidence >= HIGH_PROTECTION_THRESH is considered established;
# a single contradicting datum is marked provisional rather than flagged outright.
_CONF_PROTECTION = {"high": 0.9, "medium": 0.6, "low": 0.3}
EWC_HIGH_THRESH = float(os.environ.get("MEMCHECK_EWC_HIGH_THRESH", "0.75"))

# Mirrors server.py's _NEGATION_RE / _TOKEN_RE for the injected-free path.
_DEFAULT_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|none|without|cannot|can't|didn't|isn't|aren't|won't)\b", re.I
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._:/-]{2,}", re.I)
_GENERIC_MATCH_TOKENS = {
    "host", "user", "device", "query", "result", "results", "output", "input", "tool",
    "found", "seen", "shows", "reported", "detected", "contacted", "event", "events",
    "record", "records", "row", "rows",
}

# Cap on the number of findings compared pairwise, newest-last preserved.
_MAX_FINDINGS = 300


def _default_tokenize(text: str) -> set[str]:
    return {
        token
        for token in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
        if token not in _GENERIC_MATCH_TOKENS
    }


def _finding_id(finding: dict, index: int) -> str:
    fid = finding.get("id") or finding.get("finding_id")
    if fid:
        return str(fid)
    inv = finding.get("investigation_id")
    return f"{inv}:{index}" if inv else f"finding:{index}"


def _overlap(a: set, b: set) -> float:
    """Symmetric token overlap (Szymkiewicz–Simpson), so order doesn't matter."""
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _protection_weight(finding: dict) -> float:
    """EWC importance proxy for a finding — higher = more established."""
    conf = str(finding.get("confidence", "") or "").lower()
    base = _CONF_PROTECTION.get(conf, 0.5)
    # "observed" findings are more established than "assumed"/"gap".
    rtype = str(finding.get("record_type") or finding.get("type", "") or "").lower()
    type_mult = {"observed": 1.1, "inferred": 1.0, "assumed": 0.85, "gap": 0.7}.get(rtype, 1.0)
    return min(1.0, base * type_mult)


def run_contradiction(
    findings: list[dict],
    *,
    min_overlap: float = 0.5,
    negation_re=None,
    tokenizer: Optional[Callable[[str], set]] = None,
    max_findings: int = _MAX_FINDINGS,
) -> list[Verdict]:
    """Flag pairs of findings that appear to contradict each other.

    Two findings contradict when their tokens overlap >= ``min_overlap`` AND
    exactly one of them is negated (negation-polarity mismatch). One verdict is
    emitted per contradicting pair; symmetric pairs are deduped.

    Parameters
    ----------
    findings:
        Plain finding dicts.
    min_overlap:
        Minimum symmetric token overlap for two findings to be "about the same".
    negation_re:
        Compiled regex used to detect negation (inject the server's ``_NEGATION_RE``).
    tokenizer:
        Optional injected tokenizer (the server's ``_tokenize``).
    max_findings:
        Cap on the number of findings compared pairwise.
    """
    neg_re = negation_re or _DEFAULT_NEGATION_RE
    tok = tokenizer or _default_tokenize

    # Build a compact, prefiltered view: (id, text, tokens, negated, raw_finding).
    # Keep the raw finding for EWC protection weight. Skip malformed/empty.
    # Keep the most recent ``max_findings``.
    prepared: list[tuple[str, str, set, bool, dict]] = []
    source_list = (findings or [])[-max_findings:] if findings else []
    base_index = len(findings or []) - len(source_list)
    for offset, finding in enumerate(source_list):
        index = base_index + offset
        if not isinstance(finding, dict):
            continue
        try:
            text = str(finding.get("text", "") or "")
            if not text.strip():
                continue
            tokens = tok(text)
            if not tokens:
                continue
            negated = bool(neg_re.search(text))
            prepared.append((_finding_id(finding, index), text, tokens, negated, finding))
        except Exception:
            continue

    verdicts: list[Verdict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for i in range(len(prepared)):
        id_a, text_a, tok_a, neg_a, raw_a = prepared[i]
        for j in range(i + 1, len(prepared)):
            id_b, text_b, tok_b, neg_b, raw_b = prepared[j]
            # Opposite negation polarity is the contradiction signal.
            if neg_a == neg_b:
                continue
            # Cheap prefilter then the real overlap bar (same metric here).
            if _overlap(tok_a, tok_b) < min_overlap:
                continue

            pair_key = tuple(sorted((id_a, id_b)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            try:
                excerpt = redact_excerpt(f"{text_a} <> {text_b}")

                # EWC protection: if the more-established finding has high
                # protection weight, the contradicting datum is provisional —
                # it needs confirmation before overwriting the established view.
                prot_a = _protection_weight(raw_a)
                prot_b = _protection_weight(raw_b)
                max_prot = max(prot_a, prot_b)
                provisional = max_prot >= EWC_HIGH_THRESH
                if provisional:
                    rationale = (
                        "findings appear to contradict (negation-polarity mismatch)"
                        f" — PROVISIONAL: established finding (prot={max_prot:.2f}) requires"
                        " corroboration before verdict is enforced"
                    )
                    # Soften confidence; provisional contradictions don't auto-block.
                    verdict_confidence = 0.40
                else:
                    rationale = "findings appear to contradict (negation-polarity mismatch)"
                    verdict_confidence = 0.55

                verdicts.append(
                    new_verdict(
                        subject_kind="memory",
                        subject_signature=make_signature("memory", "|".join(pair_key)),
                        subject_excerpt=excerpt,
                        verdict_type="contradiction",
                        decision="flag",
                        confidence=verdict_confidence,
                        rationale=rationale,
                        source="rule",
                        refs=[id_a, id_b],
                        provisional=provisional,
                    )
                )
            except Exception:
                continue

    return verdicts

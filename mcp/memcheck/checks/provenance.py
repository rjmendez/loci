"""Provenance check — "observed must cite a source".

The reasoning discipline (CLAUDE.md) requires every ``observed`` finding to
be traceable to a tool response. This check verifies that rule: for each
finding typed ``observed``, it looks for a supporting *receipt* among the
investigation's audit entries. A receipt supports a finding when:

- the tool/source names overlap (the finding's ``source`` names a tool that
  appears in an audit entry's ``tool`` field, or vice-versa), AND
- the finding text and the audit entry's text/summary share enough tokens
  (lexical overlap >= ``min_overlap``), AND
- (soft, optional) the receipt is not implausibly distant in time.

A finding with at least one clearing receipt is considered supported and emits
no verdict. A finding with NO clearing receipt emits a single advisory
``unsupported_observed`` warn verdict.

Pure: takes plain dicts, returns ``list[Verdict]``. Fail-safe: a malformed
record is skipped, never raised.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from ..verdict import Verdict, make_signature, new_verdict, redact_excerpt

__all__ = ["run_provenance"]

# Mirrors server.py's _TOKEN_RE / _GENERIC_MATCH_TOKENS so the internal
# fallback tokenizer behaves like the server's when no tokenizer is injected.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._:/-]{2,}", re.I)
_GENERIC_MATCH_TOKENS = {
    "host", "user", "device", "query", "result", "results", "output", "input", "tool",
    "found", "seen", "shows", "reported", "detected", "contacted", "event", "events",
    "record", "records", "row", "rows",
}


def _default_tokenize(text: str) -> set[str]:
    return {
        token
        for token in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
        if token not in _GENERIC_MATCH_TOKENS
    }


def _default_lexical_score(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a))


def _finding_id(finding: dict, index: int) -> str:
    """Stable id for a finding: prefer an explicit id, else derive from index."""
    fid = finding.get("id") or finding.get("finding_id")
    if fid:
        return str(fid)
    inv = finding.get("investigation_id")
    return f"{inv}:{index}" if inv else f"finding:{index}"


def _audit_text(entry: dict) -> str:
    """Best text to lexically match a receipt against."""
    parts = [
        entry.get("embedding_text"),
        entry.get("output"),
        entry.get("inputs"),
        entry.get("text"),
    ]
    return " ".join(str(p) for p in parts if p)


def _audit_names(entry: dict) -> set[str]:
    names = set()
    for key in ("tool", "tool_name", "source"):
        v = entry.get(key)
        if v:
            names.add(str(v).lower())
    return names


def run_provenance(
    findings: list[dict],
    audit_entries: list[dict],
    *,
    min_overlap: float = 0.45,
    tokenizer: Optional[Callable[[str], set]] = None,
    lexical_score: Optional[Callable[[set, set], float]] = None,
) -> list[Verdict]:
    """Flag ``observed`` findings that lack a matching audit receipt.

    Parameters
    ----------
    findings:
        Plain finding dicts (``type``/``record_type``, ``text``, ``source`` ...).
    audit_entries:
        Plain audit dicts (``tool``, ``inputs``, ``output``, ``embedding_text`` ...).
    min_overlap:
        Minimum lexical token overlap (finding-relative) for a receipt to count.
    tokenizer / lexical_score:
        Optional injected callables (the server's ``_tokenize`` / ``_lexical_match_score``).
        Internal defaults mirror the server's behavior when omitted.

    Returns
    -------
    A list of ``unsupported_observed`` warn verdicts — one per observed finding
    that no receipt supports. Supported findings emit nothing.
    """
    tok = tokenizer or _default_tokenize
    score = lexical_score or _default_lexical_score

    # Precompute receipt token-sets + names once.
    receipts: list[tuple[set, set]] = []
    for entry in audit_entries or []:
        if not isinstance(entry, dict):
            continue
        try:
            receipts.append((tok(_audit_text(entry)), _audit_names(entry)))
        except Exception:
            continue

    verdicts: list[Verdict] = []
    for index, finding in enumerate(findings or []):
        if not isinstance(finding, dict):
            continue
        try:
            ftype = finding.get("type") or finding.get("record_type")
            if ftype != "observed":
                continue
            text = str(finding.get("text", "") or "")
            if not text.strip():
                continue

            f_tokens = tok(text)
            f_source = str(finding.get("source", "") or "").lower()

            supported = False
            for r_tokens, r_names in receipts:
                # Name overlap: finding source names a known tool, or shares a
                # token with the receipt's names.
                name_ok = False
                if f_source and r_names:
                    f_tokens_name = set(re.split(r"[\W_]+", f_source))
                    if f_source in r_names or any(
                        n == f_source or bool(f_tokens_name & set(re.split(r"[\W_]+", n)))
                        for n in r_names
                    ):
                        name_ok = True
                # Token overlap clears the lexical bar.
                lex_ok = score(f_tokens, r_tokens) >= min_overlap
                # A receipt supports when names line up AND text overlaps, or
                # (no source named) when text overlap is strong on its own.
                if (name_ok and lex_ok) or (not f_source and lex_ok):
                    supported = True
                    break

            if supported:
                continue

            fid = _finding_id(finding, index)
            verdicts.append(
                new_verdict(
                    subject_kind="memory",
                    subject_signature=make_signature("memory", fid),
                    subject_excerpt=redact_excerpt(text),
                    verdict_type="unsupported_observed",
                    decision="warn",
                    confidence=0.6,
                    rationale="observed finding has no matching audit receipt",
                    source="rule",
                    refs=[fid],
                )
            )
        except Exception:
            # Fail-safe: a malformed finding is skipped, never raised.
            continue

    return verdicts

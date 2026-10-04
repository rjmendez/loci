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

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from ..verdict import Verdict, make_signature, new_verdict, redact_excerpt
from ._common import _default_tokenize, _finding_id

__all__ = ["run_provenance"]

_log = logging.getLogger("memcheck.provenance")

# A receipt written this long after the finding cannot have been its source.
_RECEIPT_MAX_LAG = timedelta(days=1)


def _default_lexical_score(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a))


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


def _cited_receipt_ids(finding: dict) -> list[str]:
    meta = finding.get("metadata")
    ids = meta.get("receipt_ids") if isinstance(meta, dict) else None
    if isinstance(ids, str):
        ids = [ids]
    return [str(i) for i in ids if i] if isinstance(ids, (list, tuple)) else []


def _parse_ts(value) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _cited_receipt_ok(finding: dict, entry: dict) -> bool:
    """A cited receipt is plausible when its tool matches the finding's source
    (when both are named) and it was not written long after the finding."""
    f_source = str(finding.get("source", "") or "").lower()
    names = _audit_names(entry)
    if f_source and names:
        f_toks = set(re.split(r"[\W_]+", f_source)) - {""}
        if not (f_source in names or any(
            f_toks & (set(re.split(r"[\W_]+", n)) - {""}) for n in names
        ) or f_source in _audit_text(entry).lower()):
            return False
    f_ts, r_ts = _parse_ts(finding.get("ts")), _parse_ts(entry.get("ts"))
    if f_ts and r_ts and r_ts - f_ts > _RECEIPT_MAX_LAG:
        return False
    return True


def run_provenance(
    findings: list[dict],
    audit_entries: list[dict],
    *,
    min_overlap: float = 0.45,
    tokenizer: Optional[Callable[[str], set]] = None,
    lexical_score: Optional[Callable[[set, set], float]] = None,
    global_audit_entries: Optional[list[dict]] = None,
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

    global_audit_entries:
        Optional receipts from the global daily log; consulted only for explicit
        ``metadata.receipt_ids`` citations.

    A finding that cites receipt ids in ``metadata.receipt_ids`` is supported when
    a cited id exists in ``audit_entries`` (or ``global_audit_entries``) and the
    receipt is plausible for it; the lexical rule below remains the fallback.

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
        except Exception as exc:
            _log.debug("provenance: skipping malformed audit entry: %s", exc)
            continue

    # Receipts addressable by id, for explicit citations.
    by_receipt_id: dict[str, dict] = {}
    for entry in list(global_audit_entries or []) + list(audit_entries or []):
        if isinstance(entry, dict) and entry.get("receipt_id"):
            by_receipt_id[str(entry["receipt_id"])] = entry

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

            supported = any(
                rid in by_receipt_id and _cited_receipt_ok(finding, by_receipt_id[rid])
                for rid in _cited_receipt_ids(finding)
            )
            for r_tokens, r_names in ([] if supported else receipts):
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
        except Exception as exc:
            # Fail-safe: a malformed finding is skipped, never raised.
            _log.debug("provenance: skipping malformed finding at index %s: %s", index, exc)
            continue

    return verdicts

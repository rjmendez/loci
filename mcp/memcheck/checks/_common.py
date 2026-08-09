"""Shared internals for the memcheck checks — package-private.

These four helpers existed as byte-identical copies across ``contagion``,
``contradiction`` and ``provenance``; they live here so the three modules
cannot drift apart. Nothing outside ``memcheck.checks`` should import this
module — it is deliberately absent from the package ``__all__``.

Note what is NOT here, on purpose: ``contradiction._overlap`` and
``provenance._default_lexical_score`` look alike but are different formulas
(symmetric Szymkiewicz-Simpson vs finding-relative), and
``contract_contradiction._TOKEN_RE`` is a different pattern. Unifying any of
those would silently move a check's threshold.
"""

from __future__ import annotations

import re

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


def _finding_id(finding: dict, index: int) -> str:
    """Stable id for a finding: prefer an explicit id, else derive from index."""
    fid = finding.get("id") or finding.get("finding_id")
    if fid:
        return str(fid)
    inv = finding.get("investigation_id")
    return f"{inv}:{index}" if inv else f"finding:{index}"

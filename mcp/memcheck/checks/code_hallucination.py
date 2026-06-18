"""Code-hallucination check — vendored static checker → verdicts.

Bridges the vendored LLM code-hallucination checker
(``memcheck.code_rules``) into the memcheck verdict model. Each static
``Issue`` becomes a ``subject_kind="code"`` ``Verdict`` carrying the rule code
(``LH001`` …) as its ``verdict_type`` and ``decision="warn"`` — these are
advisory code smells surfaced on the PostToolUse path, never hard blocks.

Loop-fuel for the future code→memory loop is stashed on the verdict: the
relative path, line, rule code, and the flagged identifier (when the rule names
one) all go into ``refs`` so a later pass can correlate a code smell back to the
symbol it touched without re-parsing.

Pure-ish + fail-safe: an unreadable file or a ``SyntaxError`` yields zero or a
single ``LH000`` verdict, never an exception. Never embeds a dev-local absolute
path — the excerpt/signature use the path relative to ``repo_root`` (or the
basename when no root is given).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..code_rules import (
    PATTERN_META,
    Issue,
    check_file,
    run_extended_checks,
)
from ..verdict import Verdict, make_signature, new_verdict, redact_excerpt

__all__ = ["run_code_checks"]

# Advisory confidence for a vendored static code smell (LH00x) — surfaced, not
# enforced. Extended-check verdicts (H2, SS3, …) instead carry the per-pattern
# confidence recorded in ``PATTERN_META``.
_CODE_CONFIDENCE = 0.6

# Vendored LH rule code -> the taxonomy pattern id it implements. Used to stamp
# the tier/confidence marker from PATTERN_META onto the vendored verdicts too,
# so the whole aggregated set is annotated consistently.
_LH_TO_PATTERN = {
    "LH001": "H1",
    "LH003": "H3",
    "LH007": "H7",
    "LH009": "H9",
}

# Pull the flagged identifier out of an Issue.message when the rule names one.
# LH001's message embeds the attribute as '_attr' in single quotes; this is the
# only rule that currently references a concrete symbol.
_QUOTED_NAME_RE = re.compile(r"'([^']+)'")


def _relpath(path: Path, repo_root: str | None) -> str:
    """Path relative to ``repo_root`` when possible, else the basename.

    Never returns a dev-local absolute path — the result is what lands in the
    verdict excerpt/signature, so it must be stable and portable.
    """
    if repo_root:
        try:
            return path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return path.name


def _flagged_symbol(issue: Issue) -> str | None:
    """Best-effort extraction of the identifier an Issue references, or None."""
    match = _QUOTED_NAME_RE.search(issue.message or "")
    return match.group(1) if match else None


def _confidence_and_marker(code: str) -> tuple[float, str | None]:
    """Resolve a verdict's confidence + tier/advisory marker from PATTERN_META.

    ``code`` is either a taxonomy pattern id (from the extended checks, e.g.
    ``"H2"``) or a vendored ``LHxxx`` rule code (mapped through
    ``_LH_TO_PATTERN``). Returns ``(confidence, marker)`` where ``marker`` is a
    ``"tier:N"`` / ``"advisory"`` ref token, or ``None`` when the code is not in
    the taxonomy (e.g. ``LH000`` syntax errors).
    """
    pattern_id = _LH_TO_PATTERN.get(code, code)
    meta = PATTERN_META.get(pattern_id)
    if meta is None:
        # Vendored LH00x with no taxonomy mapping (e.g. LH000 SyntaxError).
        return _CODE_CONFIDENCE, None
    confidence = float(meta.get("confidence") or 0.0) or _CODE_CONFIDENCE
    detection = meta.get("detection")
    if detection == "advisory":
        marker = "advisory"
    else:
        marker = f"tier:{meta.get('tier')}"
    return confidence, marker


def _issue_to_verdict(issue: Issue, relpath: str) -> Verdict:
    """Map one static :class:`Issue` to a ``subject_kind="code"`` Verdict."""
    code = issue.code
    line = issue.line
    confidence, marker = _confidence_and_marker(code)

    refs = [f"path:{relpath}", f"line:{line}", f"code:{code}"]
    symbol = _flagged_symbol(issue)
    if symbol:
        refs.append(f"symbol:{symbol}")
    if marker:
        refs.append(marker)

    return new_verdict(
        subject_kind="code",
        subject_signature=make_signature("code", f"{relpath}:{code}:{line}"),
        subject_excerpt=redact_excerpt(f"{relpath}:{line} {code}"),
        verdict_type=code,
        decision="warn",
        confidence=confidence,
        rationale=issue.message,
        source="rule",
        refs=refs,
    )


def run_code_checks(
    path: str | Path,
    *,
    repo_root: str | None = None,
) -> list[Verdict]:
    """Run the vendored + loci-owned extended static checks on ``path``.

    Aggregates two sources of :class:`Issue`:

    * the vendored ``check_file`` (LH001/LH003/LH007/LH009 — H1/H3/H7/H9), and
    * ``run_extended_checks`` (the rest of the taxonomy: H2, H4–H6, SD, SS, SL,
      SB, AC, TEC, PB, MF, OG, WG, MC, … — each Issue's ``code`` is the
      taxonomy pattern id).

    Each :class:`Issue` becomes one ``subject_kind="code"`` ``Verdict``:

    * ``verdict_type`` — the rule/pattern code (``"LH001"``, ``"H2"``, …, or
      ``"LH000"`` on a syntax error).
    * ``decision`` — ``"warn"`` (advisory; these never block).
    * ``confidence`` — the per-pattern confidence from ``PATTERN_META`` (or
      ~0.6 for vendored codes with no taxonomy mapping).
    * ``rationale`` — the Issue's message.
    * ``subject_excerpt`` — redacted ``"{relpath}:{line} {code}"``.
    * ``subject_signature`` — ``make_signature("code", "{relpath}:{code}:{line}")``.
    * ``refs`` — loop-fuel: ``["path:{relpath}", "line:{line}", "code:{code}"]``
      plus ``"symbol:{name}"`` when the rule names a flagged identifier and a
      ``"tier:N"``/``"advisory"`` marker from ``PATTERN_META``.

    Verdicts are deduped on ``subject_signature`` (which folds in path + code +
    line), so a smell flagged by both the vendored and extended layers surfaces
    once.

    Fail-safe: a missing/unreadable file or a parse failure yields zero or a
    single ``LH000`` verdict — this function never raises. Each underlying
    checker is independently guarded so one failing layer never suppresses the
    other.
    """
    p = Path(path)
    relpath = _relpath(p, repo_root)

    try:
        vendored_issues = check_file(p)
    except Exception:  # noqa: BLE001 — checker errors must not break the hook
        vendored_issues = []

    try:
        extended_issues = run_extended_checks(p)
    except Exception:  # noqa: BLE001 — extended layer is fail-safe too
        extended_issues = []

    verdicts: list[Verdict] = []
    seen_signatures: set[str] = set()
    for issue in (*vendored_issues, *extended_issues):
        try:
            verdict = _issue_to_verdict(issue, relpath)
        except Exception:  # noqa: BLE001 — skip a bad issue, never raise
            continue
        if verdict.subject_signature in seen_signatures:
            continue
        seen_signatures.add(verdict.subject_signature)
        verdicts.append(verdict)

    return verdicts

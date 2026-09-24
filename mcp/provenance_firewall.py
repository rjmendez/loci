"""Deterministic provenance firewall for findings used as evidence.

Finding storage already has a hot/warm/cold indexing tier.  This module's
provenance tier is separate and records evidentiary authority:

``human_authored`` > ``tool_verified`` / ``deterministic_derived`` >
``model_asserted``.

Backwards compatibility: legacy findings have no provenance field.  They are
still *displayed* as ``tool_verified`` so old investigations keep working, but
they carry ``provenance_defaulted=True`` and the firewall never counts a
defaulted row as independent evidence: nobody asserted that tier.  New
model-written findings should explicitly set
``metadata.evidence_provenance_tier = "model_asserted"``; Loci's own model,
reflection and declaration writers stamp it at write time.
"""
from __future__ import annotations

from typing import Any

HUMAN_AUTHORED = "human_authored"
TOOL_VERIFIED = "tool_verified"
DETERMINISTIC_DERIVED = "deterministic_derived"
MODEL_ASSERTED = "model_asserted"
LEGACY_UNTAGGED_DEFAULT = TOOL_VERIFIED

PROVENANCE_TIERS = frozenset({
    HUMAN_AUTHORED,
    TOOL_VERIFIED,
    DETERMINISTIC_DERIVED,
    MODEL_ASSERTED,
})
INDEPENDENT_EVIDENCE_TIERS = frozenset({
    HUMAN_AUTHORED,
    TOOL_VERIFIED,
    DETERMINISTIC_DERIVED,
})

_ALIASES = {
    "human": HUMAN_AUTHORED,
    "human-authored": HUMAN_AUTHORED,
    "human_authored": HUMAN_AUTHORED,
    "t0_human_authored": HUMAN_AUTHORED,
    "tool": TOOL_VERIFIED,
    "tool-verified": TOOL_VERIFIED,
    "tool_verified": TOOL_VERIFIED,
    "t1_tool_verified": TOOL_VERIFIED,
    "audit": TOOL_VERIFIED,
    "deterministic": DETERMINISTIC_DERIVED,
    "deterministic-derived": DETERMINISTIC_DERIVED,
    "deterministic_derived": DETERMINISTIC_DERIVED,
    "t2_deterministic_derived": DETERMINISTIC_DERIVED,
    "derived": DETERMINISTIC_DERIVED,
    "model": MODEL_ASSERTED,
    "model-asserted": MODEL_ASSERTED,
    "model_asserted": MODEL_ASSERTED,
    "t3_model_asserted": MODEL_ASSERTED,
    "llm": MODEL_ASSERTED,
    "llm_asserted": MODEL_ASSERTED,
    "ideate": MODEL_ASSERTED,
    "verify": MODEL_ASSERTED,
    "red-team": MODEL_ASSERTED,
    "red_team": MODEL_ASSERTED,
    "synthesize": MODEL_ASSERTED,
}


def _metadata(row: Any) -> dict:
    if not isinstance(row, dict):
        return {}
    meta = row.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _asserted_raw_tier(row: dict) -> Any:
    """The caller-asserted tier value normalize_provenance_tier reads first, or None."""
    meta = _metadata(row)
    return (
        row.get("evidence_provenance_tier")
        or row.get("provenance_tier")
        or meta.get("evidence_provenance_tier")
        or meta.get("provenance_tier")
        or row.get("evidence_kind")
        or meta.get("evidence_kind")
    )


def _is_known_tier(raw: Any) -> bool:
    key = str(raw or "").strip().lower().replace(" ", "_")
    return bool(key) and (key in _ALIASES or key in PROVENANCE_TIERS)


def normalize_provenance_tier(row: Any, *, default: str = LEGACY_UNTAGGED_DEFAULT) -> str:
    """Return a canonical provenance tier for a finding/evidence row; never raises."""
    try:
        if isinstance(row, str):
            raw = row
        elif isinstance(row, dict):
            raw = _asserted_raw_tier(row) or row.get("tier") or default
        else:
            raw = default
        key = str(raw or default).strip().lower().replace(" ", "_")
        return _ALIASES.get(key, key if key in PROVENANCE_TIERS else default)
    except Exception:
        return default


def provenance_fields(row: Any, *, default: str = LEGACY_UNTAGGED_DEFAULT) -> dict:
    """Small JSON-safe fields that callers can surface or store.

    A row already flagged ``provenance_defaulted=True`` (e.g. Mnemosyne
    metadata written from a defaulted finding) stays defaulted: the tier key it
    carries is the stored default, not an assertion. So does a row whose tier
    field holds a value that is not a known tier (``evidence_kind="log_excerpt"``):
    it normalises to the default, and nobody asserted that default.
    """
    tier = normalize_provenance_tier(row, default=default)
    explicit = isinstance(row, str)
    if isinstance(row, dict):
        meta = _metadata(row)
        explicit = _is_known_tier(_asserted_raw_tier(row)) and True not in (
            row.get("provenance_defaulted"), meta.get("provenance_defaulted"))
    return {
        "evidence_provenance_tier": tier,
        "provenance_defaulted": not explicit,
    }


def firewall_candidate_tier(row: Any) -> str:
    """Tier to gate a stored finding with when it is the *candidate* claim.

    A defaulted (untagged) finding is displayed as tool_verified, but nobody
    asserted that, so as a candidate it is gated like model_asserted: it needs
    linked independent evidence before a verifier or promotion accepts it.
    """
    fields = provenance_fields(row)
    return MODEL_ASSERTED if fields["provenance_defaulted"] else fields["evidence_provenance_tier"]


def assert_evidence_firewall(candidate: Any, evidence_rows: Any, *, phase: str = "") -> dict:
    """Check that model assertions are not verified only by model assertions.

    The rule is intentionally cheap: compute a frozenset of evidence tiers and
    require a set intersection with independent evidence when the candidate is
    model-asserted.  A row whose tier is only the legacy default
    (``provenance_defaulted``) is listed but never counts as independent, and
    callers must pass only evidence linked to the claim, not "any other
    finding".  Any unexpected shape fails open with ``allowed=True``.
    """
    try:
        candidate_tier = (firewall_candidate_tier(candidate) if isinstance(candidate, dict)
                          else normalize_provenance_tier(candidate))
        rows = evidence_rows if isinstance(evidence_rows, (list, tuple, set)) else []
        fields = [provenance_fields(row) for row in rows]
        evidence_tiers = frozenset(f["evidence_provenance_tier"] for f in fields)
        asserted_tiers = frozenset(
            f["evidence_provenance_tier"] for f in fields if not f["provenance_defaulted"]
        )
        independent = bool(asserted_tiers & INDEPENDENT_EVIDENCE_TIERS)
        allowed = candidate_tier != MODEL_ASSERTED or independent
        reason = (
            "ok"
            if allowed
            else "model_asserted_requires_independent_human_tool_or_deterministic_evidence"
        )
        return {
            "allowed": allowed,
            "phase": phase,
            "candidate_tier": candidate_tier,
            "evidence_tiers": sorted(evidence_tiers),
            "independent_evidence_tiers": sorted(asserted_tiers & INDEPENDENT_EVIDENCE_TIERS),
            "defaulted_evidence_count": sum(1 for f in fields if f["provenance_defaulted"]),
            "reason": reason,
            "degraded": False,
        }
    except Exception as exc:  # noqa: BLE001 - firewall must not crash callers
        return {
            "allowed": True,
            "phase": phase,
            "candidate_tier": LEGACY_UNTAGGED_DEFAULT,
            "evidence_tiers": [],
            "independent_evidence_tiers": [],
            "reason": f"provenance_firewall_failed_open:{type(exc).__name__}",
            "degraded": True,
        }


# Substrings of tool names whose output is model-generated text. An audit
# receipt for one of these records what a model said, not what a tool observed.
_MODEL_TOOL_MARKERS = (
    "llm", "swarm", "reason", "reflect", "adversarial", "verify_finding",
    "verify_all", "classify_text", "compress_text", "query_expand",
    "generate", "ideate", "synthes", "claude", "gpt", "chat", "completion",
)


def audit_provenance_fields(tool_name: Any) -> dict:
    """Provenance for an audit_log receipt, derived from the audited tool name.

    audit_log is written by the caller; Loci never observes the tool run. A
    receipt counts as ``tool_verified`` only when it names a non-model tool.
    Model tools map to ``model_asserted``; a missing tool name gets the legacy
    default with ``provenance_defaulted=True``, which is not independent.
    """
    name = str(tool_name or "").strip().lower()
    if not name:
        return {"evidence_provenance_tier": LEGACY_UNTAGGED_DEFAULT, "provenance_defaulted": True}
    if _ALIASES.get(name) == MODEL_ASSERTED or any(m in name for m in _MODEL_TOOL_MARKERS):
        return {"evidence_provenance_tier": MODEL_ASSERTED, "provenance_defaulted": False}
    return {"evidence_provenance_tier": TOOL_VERIFIED, "provenance_defaulted": False}

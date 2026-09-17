"""Deterministic provenance firewall for findings used as evidence.

Finding storage already has a hot/warm/cold indexing tier.  This module's
provenance tier is separate and records evidentiary authority:

``human_authored`` > ``tool_verified`` / ``deterministic_derived`` >
``model_asserted``.

Backwards compatibility: legacy findings have no provenance field.  They
default to ``tool_verified`` so old investigations keep working and existing
callers fail open instead of invalidating the store.  New model-written findings
should explicitly set ``metadata.evidence_provenance_tier = "model_asserted"``.
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


def normalize_provenance_tier(row: Any, *, default: str = LEGACY_UNTAGGED_DEFAULT) -> str:
    """Return a canonical provenance tier for a finding/evidence row; never raises."""
    try:
        if isinstance(row, str):
            raw = row
        elif isinstance(row, dict):
            meta = _metadata(row)
            raw = (
                row.get("evidence_provenance_tier")
                or row.get("provenance_tier")
                or meta.get("evidence_provenance_tier")
                or meta.get("provenance_tier")
                or row.get("evidence_kind")
                or meta.get("evidence_kind")
                or row.get("tier")
                or default
            )
        else:
            raw = default
        key = str(raw or default).strip().lower().replace(" ", "_")
        return _ALIASES.get(key, key if key in PROVENANCE_TIERS else default)
    except Exception:
        return default


def provenance_fields(row: Any, *, default: str = LEGACY_UNTAGGED_DEFAULT) -> dict:
    """Small JSON-safe fields that callers can surface or store."""
    tier = normalize_provenance_tier(row, default=default)
    explicit = False
    if isinstance(row, dict):
        meta = _metadata(row)
        explicit = any(
            row.get(k) is not None or meta.get(k) is not None
            for k in ("evidence_provenance_tier", "provenance_tier", "evidence_kind")
        )
    return {
        "evidence_provenance_tier": tier,
        "provenance_defaulted": not explicit,
    }


def assert_evidence_firewall(candidate: Any, evidence_rows: Any, *, phase: str = "") -> dict:
    """Check that model assertions are not verified only by model assertions.

    The rule is intentionally cheap: compute a frozenset of evidence tiers and
    require a set intersection with independent evidence when the candidate is
    model-asserted.  Any unexpected shape fails open with ``allowed=True``.
    """
    try:
        candidate_tier = normalize_provenance_tier(candidate)
        rows = evidence_rows if isinstance(evidence_rows, (list, tuple, set)) else []
        evidence_tiers = frozenset(normalize_provenance_tier(row) for row in rows)
        independent = bool(evidence_tiers & INDEPENDENT_EVIDENCE_TIERS)
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
            "independent_evidence_tiers": sorted(evidence_tiers & INDEPENDENT_EVIDENCE_TIERS),
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

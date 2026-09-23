from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

DATASET_SYMBOLS = ["fw", "mc", "BANC", "hb", "mv", "ol", "fafb", "l1em"]
DEFAULT_SCOPE_KEYS = (
    "dataset",
    "dataset_version",
    "sex",
    "life_stage",
    "annotation_completeness",
    "circuit_class",
    "experience_window",
    "species",  # default "Drosophila melanogaster"; use "multi-insect", "pan-arthropod", or "universal" for generalizable claims
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, Mapping):
                return {str(k): v for k, v in parsed.items()}
        except Exception:
            pass
    return {}


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


@dataclass(frozen=True)
class ComparativeClaim: 
    dataset: str = "unknown"
    dataset_version: str = "unknown"
    sex: str = "unspecified"
    life_stage: str = "unspecified"
    annotation_completeness: float = 0.0
    circuit_class: str = "unknown"
    experience_window: str = "unknown"
    species: str = "Drosophila melanogaster"  # Use "multi-insect", "pan-arthropod", or "universal" for generalizable claims
    confidence: float = 0.0
    evidence_strength: str = "structural"
    scope_notes: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "ComparativeClaim":
        scope = _coerce_mapping(payload)
        claim_scope = _coerce_mapping(scope.get("claim_scope") or scope.get("scope"))
        if not claim_scope:
            claim_scope = scope
        dataset = str(claim_scope.get("dataset") or scope.get("dataset") or "unknown")
        version = str(claim_scope.get("dataset_version") or scope.get("dataset_version") or "unknown")
        sex = str(claim_scope.get("sex") or scope.get("sex") or "unspecified")
        life_stage = str(claim_scope.get("life_stage") or scope.get("life_stage") or "unspecified")
        ann = claim_scope.get("annotation_completeness")
        try:
            annotation_completeness = float(ann) if ann is not None else 0.0
        except (TypeError, ValueError):
            annotation_completeness = 0.0
        circuit_class = str(claim_scope.get("circuit_class") or scope.get("circuit_class") or "unknown")
        experience_window = str(claim_scope.get("experience_window") or scope.get("experience_window") or "unknown")
        species = str(claim_scope.get("species") or scope.get("species") or "Drosophila melanogaster")
        provenance = dict(scope.get("provenance") if isinstance(scope.get("provenance"), Mapping) else {})
        evidence_strength = str(scope.get("evidence_strength") or scope.get("evidence") or "structural")
        confidence = float(scope.get("confidence", 0.0) or 0.0)
        scope_notes = tuple(str(v) for v in _coerce_list(scope.get("scope_notes")))
        return cls(
            dataset=dataset,
            dataset_version=version,
            sex=sex,
            life_stage=life_stage,
            annotation_completeness=annotation_completeness,
            circuit_class=circuit_class,
            experience_window=experience_window,
            species=species,
            confidence=confidence,
            evidence_strength=evidence_strength,
            scope_notes=scope_notes,
            provenance=provenance,
        )

    def scope_hash(self) -> str:
        return _stable_hash({
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "sex": self.sex,
            "life_stage": self.life_stage,
            "annotation_completeness": self.annotation_completeness,
            "circuit_class": self.circuit_class,
            "experience_window": self.experience_window,
            "species": self.species,
        })


class ComparativeClaimTieringEngine:
    """Deterministically tiers comparative FlyBrain claims by scope and evidence."""

    DATASET_WEIGHTS = {
        "fw": 1.0,
        "FlyWire": 1.0,
        "mc": 0.95,
        "Male CNS": 0.95,
        "BANC": 0.9,
        "hb": 0.75,
        "Hemibrain": 0.75,
        "mv": 0.7,
        "ol": 0.65,
        "fafb": 0.9,
        "l1em": 0.85,
    }
    EVIDENCE_WEIGHTS = {
        "structural": 0.35,
        "baseline": 0.5,
        "validated": 0.7,
        "release_blocking": 0.9,
        "generalizable": 0.96,
    }
    _CROSS_SEX_TOKENS = (
        "cross-sex",
        "across sex",
        "both",
        "male+female",
        "male/female",
        "all sexes",
    )
    _CROSS_STAGE_TOKENS = (
        "cross-stage",
        "across stage",
        "all stages",
        "larval+adult",
        "developmental",
    )
    _DEVELOPMENT_PLASTICITY_TOKENS = (
        "development",
        "develop",
        "larva",
        "pupa",
        "plastic",
        "learning",
        "trained",
        "experience-dependent",
    )

    def _dataset_weight(self, dataset: str) -> float:
        key = str(dataset).strip()
        return self.DATASET_WEIGHTS.get(key, self.DATASET_WEIGHTS.get(key.lower(), 0.5))

    def _scope_penalty(self, claim: ComparativeClaim) -> float:
        penalty = 0.0
        if not claim.dataset_version or claim.dataset_version == "unknown":
            penalty += 0.2
        if claim.sex in {"unspecified", "unknown"}:
            penalty += 0.1
        if claim.life_stage in {"unspecified", "unknown"}:
            penalty += 0.1
        if claim.experience_window in {"unspecified", "unknown"}:
            penalty += 0.1
        if claim.annotation_completeness <= 0.0:
            penalty += 0.1
        return min(penalty, 0.5)

    @staticmethod
    def _has_any_token(value: str, tokens: Sequence[str]) -> bool:
        text = str(value or "").strip().lower()
        return any(token in text for token in tokens)

    def _cross_sex_generalization(self, sex: str) -> bool:
        return self._has_any_token(sex, self._CROSS_SEX_TOKENS)

    def _cross_stage_generalization(self, life_stage: str) -> bool:
        return self._has_any_token(life_stage, self._CROSS_STAGE_TOKENS)

    def _development_plasticity_edge(self, claim: ComparativeClaim) -> bool:
        return self._has_any_token(claim.life_stage, self._DEVELOPMENT_PLASTICITY_TOKENS) or self._has_any_token(
            claim.experience_window, self._DEVELOPMENT_PLASTICITY_TOKENS
        )

    @staticmethod
    def _generalization_support(payload: Mapping[str, Any] | None) -> dict[str, bool]:
        raw = _coerce_mapping(payload)
        support = _coerce_mapping(raw.get("generalization_support"))
        if not support:
            support = _coerce_mapping(_coerce_mapping(raw.get("provenance")).get("generalization_support"))
        return {
            "cross_sex_validated": bool(support.get("cross_sex_validated") is True),
            "cross_stage_validated": bool(support.get("cross_stage_validated") is True),
        }

    @staticmethod
    def _decision_audit(decision: str, *, claim: ComparativeClaim, reason: str | None, tier: str, prior_tier: str | None, support: Mapping[str, bool]) -> dict[str, Any]:
        scope = {
            "dataset": claim.dataset,
            "dataset_version": claim.dataset_version,
            "sex": claim.sex,
            "life_stage": claim.life_stage,
            "annotation_completeness": claim.annotation_completeness,
            "circuit_class": claim.circuit_class,
            "experience_window": claim.experience_window,
            "species": claim.species,
        }
        return {
            "audit_type": "flybrain_claim_scope_decision",
            "decision": decision,
            "reason": reason,
            "tier": tier,
            "tier_before": prior_tier,
            "scope_hash": _stable_hash(scope),
            "claim_scope": scope,
            "generalization_support": {str(k): bool(v) for k, v in support.items()},
        }

    def tier(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        claim = ComparativeClaim.from_payload(payload)
        support = self._generalization_support(payload)
        cross_sex = self._cross_sex_generalization(claim.sex)
        cross_stage = self._cross_stage_generalization(claim.life_stage)
        edge_case = self._development_plasticity_edge(claim)
        evidence_weight = self.EVIDENCE_WEIGHTS.get(str(claim.evidence_strength).lower(), 0.35)
        scope_weight = self._dataset_weight(claim.dataset)
        base = (evidence_weight * 0.6) + (scope_weight * 0.3) + (claim.annotation_completeness * 0.25)
        score = max(0.0, min(1.0, base - self._scope_penalty(claim)))
        if claim.dataset_version == "unknown" or claim.experience_window in {"unspecified", "unknown"}:
            tier = "T0"
        elif score >= 0.9:
            tier = "T3"
        elif score >= 0.75:
            tier = "T2"
        elif score >= 0.55:
            tier = "T1"
        else:
            tier = "T0"
        prior_tier = tier
        ceiling_reasons: list[str] = []
        if cross_sex and not support["cross_sex_validated"]:
            ceiling_reasons.append("unsupported_cross_sex_generalization")
        if cross_stage and not support["cross_stage_validated"]:
            ceiling_reasons.append("unsupported_cross_stage_generalization")
        decision = "accepted"
        if ceiling_reasons:
            decision = "downgraded"
            tier = "T0" if edge_case else ("T1" if tier in {"T2", "T3"} else tier)
        audit = self._decision_audit(
            decision,
            claim=claim,
            reason=ceiling_reasons[0] if ceiling_reasons else ("scope_validated" if tier != "T0" else "scope_insufficient"),
            tier=tier,
            prior_tier=prior_tier,
            support=support,
        )
        return {
            "tier": tier,
            "score": round(score, 4),
            "scope_hash": claim.scope_hash(),
            "dataset": claim.dataset,
            "dataset_version": claim.dataset_version,
            "evidence_strength": claim.evidence_strength,
            "ceiling_reasons": ceiling_reasons,
            "decision": decision,
            "audit": audit,
            "audit_hook": audit,
            "provenance": {
                "scope": {k: getattr(claim, k) for k in DEFAULT_SCOPE_KEYS},
                "claim_scope_hash": claim.scope_hash(),
                "generalization_support": support,
            },
        }


class ValidationAnchorExpansionHook:
    """Expand validation anchors across dataset coverage and scope boundaries."""

    def __init__(self, dataset_symbols: Sequence[str] | None = None):
        self.dataset_symbols = tuple(dataset_symbols or DATASET_SYMBOLS)

    def expand_anchors(self, anchors: Sequence[Mapping[str, Any]], *, dataset_symbols: Sequence[str] | None = None) -> list[dict[str, Any]]:
        symbols = tuple(dataset_symbols or self.dataset_symbols)
        expanded: list[dict[str, Any]] = []
        for anchor in anchors:
            source = dict(anchor or {})
            base = {
                "anchor_id": source.get("anchor_id") or source.get("id") or _stable_hash(source),
                "circuit_class": source.get("circuit_class") or source.get("name") or "unknown",
                "evidence_strength": source.get("evidence_strength") or "validated",
                "validation_basis": source.get("validation_basis") or "behavioral evidence",
            }
            for symbol in symbols:
                expanded.append({
                    **base,
                    "dataset_symbol": symbol,
                    "scope_key": _stable_hash({
                        "anchor_id": base["anchor_id"],
                        "dataset_symbol": symbol,
                        "evidence_strength": base["evidence_strength"],
                    }),
                    "coverage_note": f"Anchor {base['anchor_id']} is valid within dataset {symbol} under the preserved scope contract.",
                })
        return expanded


class OpenToolingAdapterMap:
    """A minimal capability map that routes work to the best adapter."""

    def __init__(self):
        self._adapters: dict[str, dict[str, Any]] = {}

    def register_adapter(self, adapter_name: str, *, capabilities: Sequence[str], metadata: Mapping[str, Any] | None = None):
        self._adapters[str(adapter_name)] = {
            "name": str(adapter_name),
            "capabilities": tuple(sorted({str(v) for v in capabilities})),
            "metadata": dict(metadata or {}),
        }

    def route(self, required_capabilities: Sequence[str], *, preferred: Sequence[str] = ()) -> dict[str, Any] | None:
        required = {str(v) for v in required_capabilities}
        ordered = sorted(self._adapters.items(), key=lambda kv: (-(len(required & set(kv[1]["capabilities"]))), kv[0]))
        for _, info in ordered:
            caps = set(info["capabilities"])
            if required.issubset(caps):
                score = len(required & caps) + (0.05 * len(set(preferred) & caps))
                return {"adapter": info["name"], "capabilities": list(info["capabilities"]), "score": round(score, 4)}
        return None

    def as_map(self) -> dict[str, Any]:
        return {name: {"capabilities": data["capabilities"], "metadata": data["metadata"]} for name, data in self._adapters.items()}


class HemibrainFlyWireCompatLayer:
    """Translate scope differences between Hemibrain and FlyWire data sources."""

    _compat_map = {
        "hb": {"canonical": "hemibrain", "scope": "adult-region", "dataset_version": "neuprint_JRC_Hemibrain_1point2point1"},
        "hemibrain": {"canonical": "hemibrain", "scope": "adult-region", "dataset_version": "neuprint_JRC_Hemibrain_1point2point1"},
        "fw": {"canonical": "flywire", "scope": "adult-female-brain", "dataset_version": "flywire783"},
        "flywire": {"canonical": "flywire", "scope": "adult-female-brain", "dataset_version": "flywire783"},
        "fafb": {"canonical": "fafb", "scope": "adult-brain", "dataset_version": "catmaid_fafb"},
    }

    def normalize_scope(self, scope: Mapping[str, Any] | None) -> dict[str, Any]:
        payload = _coerce_mapping(scope)
        dataset = str(payload.get("dataset") or payload.get("dataset_symbol") or "unknown")
        detail = self._compat_map.get(dataset.lower(), self._compat_map.get(dataset, {"canonical": dataset.lower(), "scope": "unknown", "dataset_version": "unknown"}))
        normalized = {
            "dataset": detail["canonical"],
            "dataset_symbol": dataset,
            "dataset_version": str(payload.get("dataset_version") or detail.get("dataset_version") or "unknown"),
            "sex": str(payload.get("sex") or "unspecified"),
            "life_stage": str(payload.get("life_stage") or "adult"),
            "annotation_completeness": float(payload.get("annotation_completeness", 0.0) or 0.0),
            "circuit_class": str(payload.get("circuit_class") or "unknown"),
            "experience_window": str(payload.get("experience_window") or "unknown"),
            "scope_note": detail["scope"],
        }
        return normalized

    def translate_scope(self, scope: Mapping[str, Any] | None) -> dict[str, Any]:
        normalized = self.normalize_scope(scope)
        dataset = normalized["dataset"]
        sex = normalized["sex"]
        stage = normalized["life_stage"]
        compatibility = {
            "flywire": dataset == "flywire",
            "hemibrain": dataset == "hemibrain",
            "sex_compatible": sex in {"male", "female", "unspecified"},
            "stage_compatible": stage in {"adult", "unspecified"},
        }
        return {
            "normalized": normalized,
            "compatibility": compatibility,
            "translation_hash": _stable_hash(normalized),
            "scope_warning": "Dataset-local only unless explicit cross-dataset validation is present." if dataset in {"hemibrain", "flywire"} else "Scope translated without a direct FlyWire/Hemibrain equivalence claim.",
        }


class ResearchPriorityScoring:
    """Score and prioritize research paths by payoff, scope, confidence, and compatibility."""

    def score(self, path: Mapping[str, Any] | None) -> dict[str, Any]:
        payload = _coerce_mapping(path)
        payoff = float(payload.get("payoff", 0.0) or 0.0)
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        scope_compatibility = float(payload.get("scope_compatibility", 0.0) or 0.0)
        evidence_strength = float(payload.get("evidence_strength", 0.0) or 0.0)
        risk = float(payload.get("risk", 0.0) or 0.0)
        latency = float(payload.get("latency", 0.0) or 0.0)
        raw = (0.35 * payoff) + (0.3 * confidence) + (0.2 * scope_compatibility) + (0.15 * evidence_strength) - (0.1 * risk) - (0.05 * latency)
        score = max(0.0, min(1.0, raw))
        return {
            "score": round(score, 4),
            "priority": "high" if score >= 0.75 else "medium" if score >= 0.45 else "low",
            "components": {
                "payoff": payoff,
                "confidence": confidence,
                "scope_compatibility": scope_compatibility,
                "evidence_strength": evidence_strength,
                "risk": risk,
                "latency": latency,
            },
            "path_hash": _stable_hash(payload),
        }


comparative_claim_tier_engine = ComparativeClaimTieringEngine()
validation_anchor_expansion_hook = ValidationAnchorExpansionHook()
open_tooling_adapter_map = OpenToolingAdapterMap()
hemibrain_flywire_compat_layer = HemibrainFlyWireCompatLayer()
research_priority_scoring = ResearchPriorityScoring()

__all__ = [
    "ComparativeClaim",
    "ComparativeClaimTieringEngine",
    "ValidationAnchorExpansionHook",
    "OpenToolingAdapterMap",
    "HemibrainFlyWireCompatLayer",
    "ResearchPriorityScoring",
    "comparative_claim_tier_engine",
    "validation_anchor_expansion_hook",
    "open_tooling_adapter_map",
    "hemibrain_flywire_compat_layer",
    "research_priority_scoring",
    "DATASET_SYMBOLS",
    "DEFAULT_SCOPE_KEYS",
    "_stable_hash",
]

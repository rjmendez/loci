#!/usr/bin/env python3
"""Deterministic stigmergic gate for swarm escalation.

This module is deliberately not a synthesis layer and not an ant simulation. It consumes
cheap fan-out findings, groups corroborating claims, applies time decay over three
coordination channels (trail, resource, alarm), and returns the subset that still needs
supervisor/escalation judgment. It is pure Python, dependency-injectable by passing
plain mappings/objects, and fail-open: callers can keep their original triage if this
layer errors.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

_CONFIDENCE_WEIGHT = {"high": 0.9, "medium": 0.6, "low": 0.3}
_UNSAFE_MARKERS = ("jailbreak", "prompt injection", "ignore previous", "unsafe", "exploit")
_REFUTE_MARKERS = ("refute", "contradict", "false", "unsupported", "not true", "does not")
_STOPWORDS = {
    "about", "again", "against", "answer", "check", "does", "find", "from", "have",
    "into", "this", "that", "there", "their", "what", "when", "where", "which", "with",
    "would", "could", "should", "the", "and", "for", "are", "but", "not", "you", "all",
}


@dataclass(frozen=True)
class StigmergicConfig:
    enabled: bool = False
    rho_trail: float = 0.08
    rho_alarm: float = 0.50
    rho_resource: float = 0.20
    ttl_minutes: float = 60.0
    trail_threshold: float = 0.60
    resource_threshold: float = 0.50
    alarm_threshold: float = 0.60
    yes_ratio_threshold: float = 2.0 / 3.0
    refute_ratio_threshold: float = 1.0 / 3.0
    min_independent_agents: int = 2
    claim_similarity_threshold: float = 0.50
    revision_id: str = ""


@dataclass
class PheromoneFinding:
    claim_key: str
    agent_id: str
    subtask: str
    answer: str
    state: str = "open"
    trail: float = 0.0
    alarm: float = 0.0
    resource: float = 0.0
    created_ms: int = 0
    votes: list[dict] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    neighbors: list[str] = field(default_factory=list)
    index: int = -1
    revision_id: str = ""


@dataclass
class PheromoneCluster:
    cluster_id: str
    finding_indices: list[int]
    claim_keys: list[str]
    state: str
    trail: float
    resource: float
    alarm: float
    yes_ratio: float
    refute_ratio: float
    independent_agents: int
    refs: list[str]
    reasons: list[str]


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9_./:-]+", str(text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _normalize_claim(text: str) -> str:
    tokens = sorted(_tokens(text))
    return " ".join(tokens[:16])


def _refs_from_text(*parts: str) -> list[str]:
    refs: set[str] = set()
    for text in parts:
        for match in re.findall(r"\b[\w./-]+\.(?:py|js|ts|tsx|rs|go|java|md)(?::\d+)?\b", str(text or "")):
            refs.add(match.lower())
        for match in re.findall(r"https?://[^\s)\]}>'\"]+", str(text or "")):
            refs.add(match.rstrip(".,").lower())
    return sorted(refs)


def _confidence(value: Any) -> float:
    return _CONFIDENCE_WEIGHT.get(str(value or "").strip().lower(), 0.3)


def _age_minutes(created_ms: int, now_ms: int) -> float:
    if created_ms <= 0:
        return 0.0
    return max(0.0, (now_ms - created_ms) / 60000.0)


def _decay(value: float, rho: float, age_minutes: float) -> float:
    return max(0.0, float(value or 0.0)) * math.exp(-float(rho) * age_minutes)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "none"}


def _coerce_votes(raw_votes: Any, *, agent_id: str, confidence: float, alarm: float) -> list[dict]:
    if isinstance(raw_votes, list) and raw_votes:
        votes = [vote for vote in raw_votes if isinstance(vote, dict)]
        if votes:
            return votes
    vote = "refute" if alarm >= 0.50 else "yes"
    return [{"voter_id": agent_id, "vote": vote, "confidence": confidence}]


def _deposit(item: Any) -> tuple[float, float, float]:
    answer = str(_get(item, "answer", "") or "")
    subtask = str(_get(item, "subtask", "") or "")
    confidence = _confidence(_get(item, "confidence", "low"))
    ok = _boolish(_get(item, "ok", True))
    parse_ok = _boolish(_get(item, "parse_ok", ok)) and bool(answer.strip())
    refs = _get(item, "refs", None) or _refs_from_text(subtask, answer)
    text = f" {subtask} {answer} ".lower()

    trail = confidence * 0.35 if ok and parse_ok else 0.0
    resource = 0.0
    if parse_ok:
        resource += 0.25
    if len(answer.strip()) >= 20:
        resource += 0.10
    if refs:
        resource += 0.10
    if any(marker in text for marker in ("evidence", "source", "because", "observed", "measured")):
        resource += 0.10
    resource *= max(0.5, confidence)

    alarm = 0.0
    if not ok or not parse_ok:
        alarm += 0.50
    if any(marker in text for marker in _REFUTE_MARKERS):
        alarm += 0.30
    if any(marker in text for marker in _UNSAFE_MARKERS):
        alarm += 0.50
    return min(1.0, trail), min(1.0, resource), min(1.0, alarm)


def coerce_pheromone_findings(findings: list[Any], *, now_ms: Optional[int] = None) -> list[PheromoneFinding]:
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    out: list[PheromoneFinding] = []
    for idx, item in enumerate(list(findings or [])):
        subtask = str(_get(item, "subtask", "") or "")
        answer = str(_get(item, "answer", "") or "")
        claim_key = str(_get(item, "claim_key", "") or "").strip() or _normalize_claim(subtask)
        agent_id = str(
            _get(item, "agent_id", "")
            or _get(item, "seed", "")
            or f"{_get(item, 'model', 'agent')}#{idx}"
        )
        raw_refs = _get(item, "refs", None)
        refs = sorted({str(ref).strip().lower() for ref in raw_refs if str(ref).strip()}) if isinstance(raw_refs, list) else _refs_from_text(subtask, answer)
        trail, resource, alarm = _deposit(item)
        created_ms = int(_get(item, "created_ms", 0) or now)
        votes = _coerce_votes(_get(item, "votes", None), agent_id=agent_id, confidence=_confidence(_get(item, "confidence", "low")), alarm=alarm)
        out.append(PheromoneFinding(
            claim_key=claim_key,
            agent_id=agent_id,
            subtask=subtask,
            answer=answer,
            state=str(_get(item, "state", "open") or "open"),
            trail=float(_get(item, "trail", trail) if _get(item, "trail", None) is not None else trail),
            alarm=float(_get(item, "alarm", alarm) if _get(item, "alarm", None) is not None else alarm),
            resource=float(_get(item, "resource", resource) if _get(item, "resource", None) is not None else resource),
            created_ms=created_ms,
            votes=votes,
            refs=refs,
            neighbors=[str(n) for n in (_get(item, "neighbors", None) or [])],
            index=idx,
            revision_id=str(_get(item, "revision_id", "") or ""),
        ))
    return out


def _same_evidence(left: PheromoneFinding, right: PheromoneFinding) -> bool:
    return bool(set(left.refs) & set(right.refs))


def _linked(left: PheromoneFinding, right: PheromoneFinding, config: StigmergicConfig) -> bool:
    if left.claim_key and left.claim_key == right.claim_key:
        return True
    if _same_evidence(left, right):
        return True
    if left.refs and right.refs:
        return False
    if left.neighbors and (right.claim_key in left.neighbors or str(right.index) in left.neighbors):
        return True
    if right.neighbors and (left.claim_key in right.neighbors or str(left.index) in right.neighbors):
        return True
    return _jaccard(_tokens(left.claim_key), _tokens(right.claim_key)) >= config.claim_similarity_threshold


def _groups(items: list[PheromoneFinding], config: StigmergicConfig) -> list[list[PheromoneFinding]]:
    groups: list[list[PheromoneFinding]] = []
    for item in items:
        matched = None
        for group in groups:
            if any(_linked(item, other, config) for other in group):
                matched = group
                break
        if matched is None:
            groups.append([item])
        else:
            matched.append(item)
    return groups


def _vote_stats(group: list[PheromoneFinding]) -> tuple[float, float]:
    yes = 0.0
    refute = 0.0
    for item in group:
        for vote in item.votes or []:
            weight = float(vote.get("confidence", 1.0) or 1.0) if isinstance(vote, dict) else 1.0
            label = str(vote.get("vote", "") if isinstance(vote, dict) else vote).strip().lower()
            if label in {"yes", "agree", "support"}:
                yes += max(0.0, weight)
            elif label in {"no", "refute", "contradict", "reject"}:
                refute += max(0.0, weight)
    total = yes + refute
    if total <= 0:
        return 0.0, 0.0
    return yes / total, refute / total


def _cluster_state(group: list[PheromoneFinding], config: StigmergicConfig, *, now_ms: int) -> PheromoneCluster:
    trail = resource = alarm = 0.0
    expired = False
    revision_mismatch = False
    refs: set[str] = set()
    for item in group:
        age = _age_minutes(item.created_ms, now_ms)
        if config.ttl_minutes > 0 and age > config.ttl_minutes:
            expired = True
        if config.revision_id and item.revision_id and item.revision_id != config.revision_id:
            revision_mismatch = True
        trail += _decay(item.trail, config.rho_trail, age)
        resource += _decay(item.resource, config.rho_resource, age)
        alarm += _decay(item.alarm, config.rho_alarm, age)
        refs.update(item.refs)
    trail = min(1.0, trail)
    resource = min(1.0, resource)
    alarm = min(1.0, alarm)
    yes_ratio, refute_ratio = _vote_stats(group)
    independent = len({item.agent_id for item in group if item.agent_id})
    reasons: list[str] = []

    if expired:
        reasons.append("ttl_expired")
    if revision_mismatch:
        reasons.append("revision_boundary")
    if refute_ratio > config.refute_ratio_threshold:
        reasons.append("refute_quorum")
    if alarm > config.alarm_threshold:
        reasons.append("alarm_threshold")
    if independent < config.min_independent_agents:
        reasons.append("under_corroborated")
    if trail < config.trail_threshold:
        reasons.append("trail_below_threshold")
    if resource < config.resource_threshold:
        reasons.append("resource_below_threshold")
    if yes_ratio < config.yes_ratio_threshold:
        reasons.append("yes_ratio_below_threshold")

    if "refute_quorum" in reasons or "alarm_threshold" in reasons:
        state = "alarm"
    elif expired or revision_mismatch:
        state = "no_quorum"
    elif not reasons:
        state = "accepted"
    elif len(group) == 1 or "under_corroborated" in reasons or "resource_below_threshold" in reasons:
        state = "degraded"
    else:
        state = "no_quorum"

    return PheromoneCluster(
        cluster_id=f"cluster-{min(item.index for item in group)}",
        finding_indices=[item.index for item in group],
        claim_keys=sorted({item.claim_key for item in group if item.claim_key}),
        state=state,
        trail=round(trail, 4),
        resource=round(resource, 4),
        alarm=round(alarm, 4),
        yes_ratio=round(yes_ratio, 4),
        refute_ratio=round(refute_ratio, 4),
        independent_agents=independent,
        refs=sorted(refs),
        reasons=reasons,
    )


def consensus_gate(findings: list[Any], config: Optional[StigmergicConfig] = None, *, now_ms: Optional[int] = None) -> dict:
    cfg = config or StigmergicConfig(enabled=True)
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    pheromones = coerce_pheromone_findings(findings, now_ms=now)
    clusters = [_cluster_state(group, cfg, now_ms=now) for group in _groups(pheromones, cfg)]
    escalate_states = {"degraded", "alarm", "no_quorum"}
    flagged = sorted({idx for cluster in clusters if cluster.state in escalate_states for idx in cluster.finding_indices})
    accepted = sorted({idx for cluster in clusters if cluster.state == "accepted" for idx in cluster.finding_indices})
    return {
        "enabled": bool(cfg.enabled),
        "model_calls": 0,
        "flagged_indices": flagged,
        "accepted_indices": accepted,
        "clusters": [asdict(cluster) for cluster in clusters],
        "stats": {
            "finding_count": len(pheromones),
            "cluster_count": len(clusters),
            "accepted_clusters": sum(1 for c in clusters if c.state == "accepted"),
            "escalated_clusters": sum(1 for c in clusters if c.state in escalate_states),
        },
    }


def apply_stigmergic_consensus(findings: list[Any], triage: Optional[dict] = None,
                               config: Optional[StigmergicConfig] = None, *,
                               now_ms: Optional[int] = None) -> dict:
    """Return a replacement triage that escalates only uncertain pheromone clusters.

    Existing triage reasons are preserved as annotations, but accepted clusters are removed
    from escalation. If this function raises, callers should keep the original triage.
    """
    cfg = config or StigmergicConfig(enabled=True)
    base = triage or {}
    consensus = consensus_gate(findings, cfg, now_ms=now_ms)
    reasons: dict[str, list[str]] = {}
    base_reasons = base.get("escalation_reasons") if isinstance(base, dict) else {}
    for idx in consensus["flagged_indices"]:
        cluster_reasons: list[str] = []
        for cluster in consensus["clusters"]:
            if idx in cluster["finding_indices"]:
                cluster_reasons.extend(f"stigmergic:{reason}" for reason in cluster.get("reasons", []))
                cluster_reasons.append(f"stigmergic:{cluster['state']}")
        inherited = list((base_reasons or {}).get(str(idx), []))
        reasons[str(idx)] = sorted(set(inherited + cluster_reasons))
    triage_out = dict(base)
    triage_out["flagged_indices"] = list(consensus["flagged_indices"])
    triage_out["flagged_count"] = len(consensus["flagged_indices"])
    triage_out["escalation_reasons"] = reasons
    triage_out["stigmergic_consensus"] = consensus
    return {"triage": triage_out, "consensus": consensus}

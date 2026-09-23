from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


_CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}
_STOPWORDS = {
    "a", "an", "and", "are", "for", "how", "in", "is", "of", "on", "or", "the",
    "this", "to", "what", "when", "where", "which", "why", "with",
}
_NEGATIVE_MARKERS = (" no ", " not ", " never ", " absent ", " missing ", " false ", " unsupported ")
_POSITIVE_MARKERS = (
    " yes ", " present ", " true ", " supported ", " enabled ", " works ",
    " require ", " requires ",
)


@dataclass
class DeliberationOpinion:
    agent_id: str
    subtask: str
    claim: str
    confidence: str = "low"
    evidence: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


class ParallelDeliberationController:
    """Deterministic arbitration for multiple agents deliberating in parallel.

    The controller is intentionally fail-open: malformed/empty inputs degrade to a
    conservative "no_reliable_opinion" response rather than crashing the swarm.
    Provenance is preserved for every claim so downstream review can audit the full
    deliberation history and any conflict resolution.
    """

    def deliberate(self, topic: str, opinions: list[dict[str, Any]] | list[DeliberationOpinion]) -> dict[str, Any]:
        normalized = self._normalize_opinions(topic, opinions)
        if not normalized:
            return {
                "topic": topic,
                "degraded": True,
                "decision": "no_reliable_opinion",
                "summary": "Parallel deliberation had no usable claims; fail-open to a conservative default.",
                "participants": 0,
                "conflicts": [],
                "subtasks": [],
                "provenance": [],
            }

        grouped: dict[str, list[DeliberationOpinion]] = defaultdict(list)
        for opinion in normalized:
            grouped[str(opinion.subtask or topic or "unknown_subtask")].append(opinion)

        decisions: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []

        for subtask in sorted(grouped.keys(), key=lambda value: str(value).lower()):
            candidates = sorted(
                grouped[subtask],
                key=lambda item: (
                    -_confidence_rank(item.confidence),
                    str(item.agent_id).lower(),
                    str(item.claim or "").lower(),
                ),
            )
            winner = candidates[0]
            rival_conflicts = []
            for rival in candidates[1:]:
                if self._is_conflict(winner.claim, rival.claim):
                    rival_conflicts.append({
                        "agent_id": rival.agent_id,
                        "claim": rival.claim,
                        "confidence": rival.confidence,
                        "evidence": rival.evidence,
                    })
                    conflicts.append({
                        "subtask": subtask,
                        "winner_agent": winner.agent_id,
                        "winner_claim": winner.claim,
                        "winner_confidence": winner.confidence,
                        "losing_agent": rival.agent_id,
                        "losing_claim": rival.claim,
                        "losing_confidence": rival.confidence,
                        "rationale": "highest_confidence_wins",
                    })

            decision = {
                "subtask": subtask,
                "winner": {
                    "agent_id": winner.agent_id,
                    "claim": winner.claim,
                    "confidence": winner.confidence,
                    "evidence": winner.evidence,
                },
                "agents": [
                    {
                        "agent_id": item.agent_id,
                        "claim": item.claim,
                        "confidence": item.confidence,
                        "evidence": item.evidence,
                    }
                    for item in candidates
                ],
                "conflict": bool(rival_conflicts),
                "rationale": "consensus" if not rival_conflicts else "highest_confidence_and_deterministic_tie_breaker",
                "agent_count": len(candidates),
            }
            decisions.append(decision)
            provenance.extend(
                {
                    "agent_id": item.agent_id,
                    "subtask": subtask,
                    "claim": item.claim,
                    "confidence": item.confidence,
                    "evidence": item.evidence,
                    "source": item.provenance.get("source") or item.provenance.get("model") or "agent",
                    "provenance": item.provenance,
                }
                for item in candidates
            )

        return {
            "topic": topic,
            "degraded": False,
            "decision": "consensus" if not conflicts else "arbitrated",
            "summary": (
                "Parallel deliberation converged on a single answer across all subtasks."
                if not conflicts
                else "Parallel deliberation resolved conflicts using deterministic confidence ordering."
            ),
            "participants": len(normalized),
            "conflicts": conflicts,
            "subtasks": decisions,
            "provenance": provenance,
        }

    def _normalize_opinions(self, topic: str, opinions: list[dict[str, Any]] | list[DeliberationOpinion]) -> list[DeliberationOpinion]:
        cleaned: list[DeliberationOpinion] = []
        try:
            for idx, opinion in enumerate(opinions or []):
                if isinstance(opinion, DeliberationOpinion):
                    item = opinion
                elif isinstance(opinion, dict):
                    agent_id = str(opinion.get("agent_id") or opinion.get("agent") or f"agent_{idx}")
                    subtask = str(opinion.get("subtask") or opinion.get("task") or topic or "unknown_subtask").strip()
                    if not subtask:
                        subtask = str(topic or "unknown_subtask")
                    claim = str(opinion.get("claim") or opinion.get("answer") or opinion.get("decision") or "").strip()
                    if not claim and isinstance(opinion.get("output"), dict):
                        claim = str(opinion["output"].get("answer") or opinion["output"].get("claim") or "").strip()
                    if not claim and isinstance(opinion.get("finding"), dict):
                        claim = str(opinion["finding"].get("answer") or opinion["finding"].get("claim") or "").strip()
                    confidence = _normalize_confidence(
                        str(opinion.get("confidence") or opinion.get("score") or "low"),
                        default="low",
                    )
                    evidence = str(opinion.get("evidence") or opinion.get("source") or opinion.get("rationale") or "").strip()
                    provenance = opinion.get("provenance") if isinstance(opinion.get("provenance"), dict) else {}
                    item = DeliberationOpinion(
                        agent_id=agent_id,
                        subtask=subtask,
                        claim=claim,
                        confidence=confidence,
                        evidence=evidence,
                        provenance={"source": opinion.get("source") or provenance.get("source") or "agent", **provenance},
                    )
                else:
                    continue
                if not item.claim or not str(item.claim).strip():
                    continue
                cleaned.append(item)
            return cleaned
        except Exception:
            return []

    def _is_conflict(self, left: str, right: str) -> bool:
        left_text = str(left or "").strip()
        right_text = str(right or "").strip()
        if not left_text or not right_text or left_text.lower() == right_text.lower():
            return False

        left_tokens = _tokenize(left_text)
        right_tokens = _tokenize(right_text)
        if not left_tokens or not right_tokens:
            return True

        similarity = _jaccard(left_tokens, right_tokens)
        if similarity >= 0.75:
            return False

        left_polarity = _answer_polarity(left_text)
        right_polarity = _answer_polarity(right_text)
        if left_polarity != "unknown" and right_polarity != "unknown":
            return left_polarity != right_polarity
        return similarity < 0.5


def _normalize_confidence(value: str, *, default: str = "low") -> str:
    cleaned = str(value or "").strip().lower()
    return cleaned if cleaned in {"low", "medium", "high"} else default


def _confidence_rank(value: str) -> int:
    return _CONFIDENCE_RANK.get(str(value or "").lower(), 1)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return {token for token in tokens if len(token) > 2 and token not in _STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _answer_polarity(text: str) -> str:
    cooked = f" {re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', str(text or '').lower())).strip()} "
    if " does not " in cooked or " do not " in cooked or " not require " in cooked:
        return "negative"
    negative = any(marker in cooked for marker in _NEGATIVE_MARKERS)
    positive = any(marker in cooked for marker in _POSITIVE_MARKERS)
    if negative and not positive:
        return "negative"
    if positive and not negative:
        return "positive"
    return "unknown"

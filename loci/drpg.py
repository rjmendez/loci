"""
DRPG (Dynamic Retrieval-based Policy Generation) evolution loop.

Reads step-level reward trajectories persisted by eval_hook_stub,
scores them, and promotes high-quality policies back to Loci memory.

Architecture:
- TrajectoryStore: reads/writes JSONL trajectory files per policy_id
- TrajectoryScorer: aggregates 7D reward vectors into a scalar quality score
- PolicyPromoter: surfaces promote/demote/keep decisions given trajectory score
- DrpgSuggestionGenerator: generates next-step discovery/refinement suggestions
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loci.policy import Policy, PolicyBackend
from loci.mtime_cache import MtimeCache

# ---------------------------------------------------------------------------
# Module-level mtime-keyed trajectory cache
# ---------------------------------------------------------------------------

_TRAJ_CACHE: MtimeCache[List[Dict[str, Any]]] = MtimeCache(maxsize=64)


class TrajectoryStore:
    """Persists and retrieves step-level reward trajectories per policy_id."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, policy_id: str) -> Path:
        return self.base_dir / f"{policy_id}.jsonl"

    def push(self, policy_id: str, step_feedback_dict: Dict[str, Any]) -> None:
        """Append one step's feedback dict to the trajectory file."""
        entry = dict(step_feedback_dict)
        entry.setdefault("_ts", datetime.utcnow().isoformat() + "Z")
        with open(self._path(policy_id), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        # Invalidate cache so next read picks up the new step
        _TRAJ_CACHE.invalidate(policy_id)

    def read(self, policy_id: str) -> List[Dict[str, Any]]:
        """Return all persisted steps for a policy_id. Uses mtime-keyed cache."""
        path = self._path(policy_id)
        if not path.exists():
            return []
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return []
        cached = _TRAJ_CACHE.get(policy_id, mtime)
        if cached is not None:
            return list(cached)
        steps = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        steps.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        _TRAJ_CACHE.set(policy_id, steps, mtime)
        return list(steps)

    def list_policies(self) -> List[str]:
        """Return policy IDs that have trajectory files."""
        return [p.stem for p in self.base_dir.glob("*.jsonl")]


class TrajectoryScorer:
    """Aggregates a list of step dicts into a scalar quality score."""

    def score(self, steps: List[Dict[str, Any]]) -> float:
        """
        Weighted sum over all steps of:
            r_ret - abs(r_ovp) - abs(r_bkp) + r_ref - abs(r_step) + r_cor

        Normalized by step count. Returns 0.0 for empty input.
        """
        if not steps:
            return 0.0

        total = 0.0
        for step in steps:
            rv = step.get("reward", {})
            r_ret = float(rv.get("r_ret", 0.0))
            r_ovp = float(rv.get("r_ovp", 0.0))
            r_bkp = float(rv.get("r_bkp", 0.0))
            r_ref = float(rv.get("r_ref", 0.0))
            r_step = float(rv.get("r_step", 0.0))
            r_cor = float(rv.get("r_cor", 0.0))
            total += r_ret - abs(r_ovp) - abs(r_bkp) + r_ref - abs(r_step) + r_cor

        return total / len(steps)


class PolicyPromoter:
    """Decides promote / demote / keep given a trajectory quality score."""

    PROMOTE_THRESHOLD = 0.7
    DEMOTE_THRESHOLD = 0.2

    def decide(self, score: float, policy: Policy) -> Dict[str, Any]:
        """
        Returns {"action": "promote"|"demote"|"keep", "score": float, "reason": str}.
        Thresholds: score > 0.7 → promote, score < 0.2 → demote, else keep.
        """
        if score > self.PROMOTE_THRESHOLD:
            return {
                "action": "promote",
                "score": score,
                "reason": f"score {score:.4f} exceeds promote threshold {self.PROMOTE_THRESHOLD}",
            }
        if score < self.DEMOTE_THRESHOLD:
            return {
                "action": "demote",
                "score": score,
                "reason": f"score {score:.4f} below demote threshold {self.DEMOTE_THRESHOLD}",
            }
        return {
            "action": "keep",
            "score": score,
            "reason": f"score {score:.4f} within acceptable range [{self.DEMOTE_THRESHOLD}, {self.PROMOTE_THRESHOLD}]",
        }


class DrpgSuggestionGenerator:
    """Inspects step sequences for patterns and returns actionable suggestions."""

    def generate(self, steps: List[Dict[str, Any]], policy: Policy) -> List[Dict[str, Any]]:
        """
        Returns a list of {"suggestion": str, "confidence": float, "affects": str} dicts.

        Patterns detected:
        - Many BACKTRACK actions → suggest fewer hops
        - High average r_ovp magnitude → suggest dedup filter
        - Many REFUSE actions → suggest moving to refinement phase
        """
        if not steps:
            return []

        suggestions: List[Dict[str, Any]] = []
        total = len(steps)

        backtrack_count = sum(
            1 for s in steps if str(s.get("action", "")).upper() == "BACKTRACK"
        )
        refuse_count = sum(
            1 for s in steps if str(s.get("action", "")).upper() == "REFUSE"
        )
        avg_ovp = (
            sum(abs(float(s.get("reward", {}).get("r_ovp", 0.0))) for s in steps) / total
        )

        if backtrack_count / total > 0.3:
            suggestions.append({
                "suggestion": "Reduce max_retrieval_hops: high backtrack rate indicates over-exploration",
                "confidence": min(0.95, 0.5 + backtrack_count / total),
                "affects": "max_retrieval_hops",
            })

        if avg_ovp > 0.15:
            suggestions.append({
                "suggestion": "Enable semantic dedup filter: high subquery overlap detected",
                "confidence": min(0.9, avg_ovp * 4),
                "affects": "dedup_filter",
            })

        if refuse_count / total > 0.4:
            suggestions.append({
                "suggestion": "Transition to refinement phase: high refusal rate signals discovery saturation",
                "confidence": min(0.9, 0.5 + refuse_count / total),
                "affects": "discovery_mode",
            })

        return suggestions


def evolve_policy(
    policy_id: str,
    backend: PolicyBackend,
    trajectory_store: TrajectoryStore,
) -> Dict[str, Any]:
    """
    Full DRPG pipeline:
      1. Read trajectory from store
      2. Score via TrajectoryScorer
      3. Decide promote/demote/keep via PolicyPromoter
      4. If promote: update policy metadata and re-store
      5. Return summary dict

    Returns:
        {
            "policy_id": str,
            "action": "promote"|"demote"|"keep",
            "score": float,
            "suggestions": [...],
            "steps_analyzed": int,
        }
    """
    steps = trajectory_store.read(policy_id)
    scorer = TrajectoryScorer()
    score = scorer.score(steps)

    policy: Optional[Policy] = backend.load(policy_id)
    if policy is None:
        return {
            "policy_id": policy_id,
            "action": "error",
            "score": score,
            "suggestions": [],
            "steps_analyzed": len(steps),
            "error": "policy_not_found",
        }

    promoter = PolicyPromoter()
    decision = promoter.decide(score, policy)
    action = decision["action"]

    if action == "promote":
        policy.metadata.source = "drpg-evolved"
        policy.metadata.version += 1
        policy.metadata.updated_at = datetime.utcnow().isoformat() + "Z"
        backend.store(policy)

    suggester = DrpgSuggestionGenerator()
    suggestions = suggester.generate(steps, policy)

    return {
        "policy_id": policy_id,
        "action": action,
        "score": score,
        "suggestions": suggestions,
        "steps_analyzed": len(steps),
    }


def _default_trajectory_store() -> TrajectoryStore:
    traj_dir = os.environ.get("LOCI_TRAJECTORY_DIR", str(Path.home() / ".loci" / "trajectories"))
    return TrajectoryStore(Path(traj_dir))

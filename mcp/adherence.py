"""
Policy Adherence Confidence tracking.

Records whether policy enforcement actually changed model behavior:
- Was the requested model rewritten by tier routing?
- Did the rewritten call achieve similar r_cor (correctness)?
- Are policies being systematically bypassed?

Stored as JSONL per policy: ~/.loci/adherence/{policy_id}.jsonl
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loci.jsonl_bounds import append_bounded, tail_lines
from loci.policy import loci_home


def default_adherence_dir() -> Path:
    """$LOCI_ADHERENCE_DIR, else <LOCI_HOME or ~/.loci>/adherence, resolved at call time (never at import)."""
    override = os.getenv("LOCI_ADHERENCE_DIR")
    if override:
        return Path(override).expanduser()
    return loci_home() / "adherence"


@dataclass
class PolicyAdherenceRecord:
    """One call's adherence data."""
    policy_id: str
    call_id: str
    tier_requested: str       # tier from policy
    model_requested: str      # model caller asked for
    model_used: str           # model actually used (post-routing)
    was_downgraded: bool      # tier routing fired
    r_cor: float              # correctness reward (0.0-1.0)
    r_step: float             # step cost
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyAdherenceRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AdherenceStore:
    """Stores and queries policy adherence records."""

    def __init__(self, adherence_dir: Optional[Path] = None):
        self.adherence_dir = Path(adherence_dir or default_adherence_dir())
        self.adherence_dir.mkdir(parents=True, exist_ok=True)

    def record(self, rec: PolicyAdherenceRecord) -> None:
        """Append one adherence record. Fail-open.

        The per-policy file is bounded: past 5000 rows or 1 MiB it is rotated to
        ``<policy_id>.jsonl.<k>`` (plain JSONL) and a fresh file is started.
        """
        path = self.adherence_dir / f"{rec.policy_id}.jsonl"
        try:
            append_bounded(path, json.dumps(rec.to_dict()))
        except (OSError, ValueError):
            pass

    def read(self, policy_id: str, limit: int = 100) -> List[PolicyAdherenceRecord]:
        """Read the most recent N adherence records for a policy.

        Only the last ``limit`` lines of the file are read (cost bounded by ``limit``, not file
        size); a malformed line among them is skipped, so fewer than ``limit`` records can come back.
        """
        path = self.adherence_dir / f"{policy_id}.jsonl"
        records = []
        for raw in tail_lines(path, limit):
            try:
                records.append(PolicyAdherenceRecord.from_dict(json.loads(raw)))
            except (json.JSONDecodeError, TypeError):
                continue
        return records

    def compute_report(self, policy_id: str, min_records: int = 5) -> Dict[str, Any]:
        """
        Compute adherence confidence report for a policy.

        Returns:
        {
            "policy_id": str,
            "record_count": int,
            "adherence_rate": float,          # fraction of calls where routing fired
            "downgrade_effectiveness": float, # avg r_cor on downgraded calls vs baseline
            "confidence": float,              # overall adherence confidence (0-1)
            "recommendation": str,            # "effective" | "ineffective" | "monitoring"
            "avg_r_cor_downgraded": float,
            "avg_r_cor_baseline": float,
            "avg_r_step": float,
        }
        """
        records = self.read(policy_id)
        if len(records) < min_records:
            return {
                "policy_id": policy_id,
                "record_count": len(records),
                "recommendation": "insufficient_data",
                "confidence": 0.0,
            }

        downgraded = [r for r in records if r.was_downgraded]
        baseline = [r for r in records if not r.was_downgraded]

        adherence_rate = len(downgraded) / len(records)
        avg_r_cor_down = sum(r.r_cor for r in downgraded) / len(downgraded) if downgraded else 0.0
        avg_r_cor_base = sum(r.r_cor for r in baseline) / len(baseline) if baseline else avg_r_cor_down
        avg_r_step = sum(r.r_step for r in records) / len(records)

        # Effectiveness: downgraded calls maintain >=85% of baseline correctness
        effectiveness = avg_r_cor_down / avg_r_cor_base if avg_r_cor_base > 0 else 0.0
        confidence = min(1.0, (len(records) / 20) * effectiveness)

        recommendation = (
            "effective" if effectiveness >= 0.85 and adherence_rate > 0.1
            else "ineffective" if effectiveness < 0.7 and len(downgraded) > 3
            else "monitoring"
        )

        return {
            "policy_id": policy_id,
            "record_count": len(records),
            "adherence_rate": round(adherence_rate, 3),
            "downgrade_effectiveness": round(effectiveness, 3),
            "confidence": round(confidence, 3),
            "recommendation": recommendation,
            "avg_r_cor_downgraded": round(avg_r_cor_down, 3),
            "avg_r_cor_baseline": round(avg_r_cor_base, 3),
            "avg_r_step": round(avg_r_step, 4),
        }

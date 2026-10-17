"""
Memory-augmented response caching.

Checks whether a high-confidence recent Loci memory hit can serve
as a cached answer, avoiding a model call entirely.

Uses the same Qdrant collection as the grounding hook but adds:
- TTL check against finding timestamp
- Confidence threshold gating
- Cache hit recording for r_mem reward signal
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheHit:
    """A valid cache hit from Loci memory."""
    finding_id: str
    score: float
    content: str
    age_hours: float
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "score": self.score,
            "content": self.content[:500],  # truncate for logging
            "age_hours": round(self.age_hours, 2),
            "source": self.source,
        }


def check_memory_cache(
    findings: List[Dict[str, Any]],
    confidence_threshold: float = 0.85,
    ttl_hours: float = 24.0,
) -> Optional[CacheHit]:
    """
    Check if any finding qualifies as a cache hit.

    A finding qualifies if:
    - score >= confidence_threshold
    - created_at within ttl_hours (if timestamp present; no timestamp = always valid)

    Returns the best qualifying hit, or None.
    """
    now = datetime.now(timezone.utc)
    best: Optional[CacheHit] = None

    for finding in findings:
        score = finding.get("score", 0.0)
        if score < confidence_threshold:
            continue

        # Check age
        age_hours = 0.0
        ts_str = finding.get("created_at") or finding.get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                age_hours = (now - ts).total_seconds() / 3600
                if age_hours > ttl_hours:
                    continue
            except ValueError:
                pass  # no valid timestamp — treat as always valid

        content = finding.get("content") or finding.get("text") or ""
        if not content:
            continue

        hit = CacheHit(
            finding_id=finding.get("id", ""),
            score=score,
            content=content,
            age_hours=age_hours,
            source=finding.get("source", "loci_memory"),
        )
        if best is None or hit.score > best.score:
            best = hit

    return best


def format_cache_response(hit: CacheHit) -> str:
    """Format a cache hit into a response string with provenance."""
    return (
        f"[LOCI-CACHE hit={hit.score:.2f} age={hit.age_hours:.1f}h "
        f"src={hit.source} id={hit.finding_id}]\n{hit.content}"
    )

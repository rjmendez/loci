"""The memcheck decision engine — record / recall / promote / forget, fail-open.

Port of the Rust ``EnforcementMemory`` semantics: record verdicts, recall the
most-similar prior verdicts, and promote a confident, recurring, sufficiently-
similar verdict into a deterministic decision. Everything is **fail-open** — a
backend error degrades to "no recall" / "not recorded" and is debug-logged, but
never raises out of the engine. A check failure must never block a store/recall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .backend import InMemoryBackend, VerdictBackend
from .verdict import Verdict

__all__ = [
    "EmlConfig",
    "RecalledDecision",
    "VerdictEngine",
]

_log = logging.getLogger("memcheck")

# Decisions that count as a "block" for promotion purposes (advisory-first: the
# locked posture uses flag/warn, but quarantine is included for the future
# active mode).
_BLOCKING = ("flag", "warn", "quarantine")


@dataclass
class EmlConfig:
    """Promotion / recall tuning. Mirrors the Rust ``EmlConfig`` defaults."""

    block_threshold: float = 0.92
    min_confidence: float = 0.75
    promote_after: int = 3
    top_k: int = 5


@dataclass
class RecalledDecision:
    """A deterministic decision promoted from recalled verdicts."""

    decision: str
    verdict_type: str
    rationale: str
    subject_signature: str
    confidence: float
    similarity: float


class VerdictEngine:
    """Holds a backend + config and applies fail-open promote-after-N logic."""

    def __init__(self, backend: VerdictBackend, config: Optional[EmlConfig] = None) -> None:
        self.backend = backend
        self.config = config or EmlConfig()

    @classmethod
    def in_memory(cls, config: Optional[EmlConfig] = None) -> "VerdictEngine":
        """Convenience constructor backed by an in-memory store."""
        return cls(InMemoryBackend(), config)

    async def _record_backend(self, verdict: Verdict, embedding: Optional[list[float]]) -> None:
        """Call the backend's record, passing the embedding when supported."""
        rwe = getattr(self.backend, "record_with_embedding", None)
        if embedding is not None and callable(rwe):
            await rwe(verdict, embedding)
        else:
            await self.backend.record(verdict)

    async def record(
        self, verdict: Verdict, embedding: Optional[list[float]] = None
    ) -> None:
        """Record a verdict. Fail-open: a backend error is debug-logged, not raised."""
        try:
            await self._record_backend(verdict, embedding)
        except Exception as exc:  # noqa: BLE001 — fail-open boundary
            _log.debug("memcheck record failed, degrading to not-recorded: %r", exc)

    async def recall_decision(
        self, query_text: str, embedding: list[float], kind: str
    ) -> Optional[RecalledDecision]:
        """Recall a deterministic decision, or None if nothing qualifies.

        Qualifying verdicts have similarity >= block_threshold, a blocking
        decision, and confidence >= min_confidence. The effective occurrence
        count is ``max(number of qualifying verdicts, max occurrences among
        them)`` — so a single verdict already seen ``promote_after`` times
        promotes, and so does a fresh cluster of ``promote_after`` distinct
        qualifying verdicts. Below the threshold, returns None. Fail-open.
        """
        try:
            scored = await self.backend.recall(
                query_text, embedding, kind, self.config.top_k
            )
        except Exception as exc:  # noqa: BLE001 — fail-open boundary
            _log.debug("memcheck recall failed, degrading to no-recall: %r", exc)
            return None

        qualifying = [
            s
            for s in scored
            if s.similarity >= self.config.block_threshold
            and s.verdict.decision in _BLOCKING
            and s.verdict.confidence >= self.config.min_confidence
        ]
        if not qualifying:
            return None

        max_occ = max(s.verdict.occurrences for s in qualifying)
        effective_occurrences = max(len(qualifying), max_occ)
        if effective_occurrences < self.config.promote_after:
            return None

        # recall() returns sorted desc by similarity, so the first qualifying is
        # the closest.
        best = qualifying[0]
        return RecalledDecision(
            decision=best.verdict.decision,
            verdict_type=best.verdict.verdict_type,
            rationale=best.verdict.rationale,
            subject_signature=best.verdict.subject_signature,
            confidence=best.verdict.confidence,
            similarity=best.similarity,
        )

    async def forget(self, subject_excerpt: str, kind: str) -> int:
        """Remove matching verdicts. Fail-open: returns 0 on backend error."""
        try:
            return await self.backend.forget(subject_excerpt, kind)
        except Exception as exc:  # noqa: BLE001 — fail-open boundary
            _log.debug("memcheck forget failed, degrading to 0: %r", exc)
            return 0

    async def stats(self) -> dict:
        """Backend stats. Fail-open: returns zeroed stats marked ``ok: False`` on error.

        The zeros are invented, not measured, and they are exactly what a memcheck
        that has never fired also reports — so the caller is told which one it got.
        """
        try:
            return await self.backend.stats()
        except Exception as exc:  # noqa: BLE001 — fail-open boundary
            _log.debug("memcheck stats failed, degrading to zeros: %r", exc)
            return {"total_verdicts": 0, "recurring_blocks": 0,
                    "ok": False, "error": repr(exc)}

    def summary(self) -> str:
        """Human-readable one-liner describing the engine configuration."""
        c = self.config
        return (
            f"VerdictEngine(backend={type(self.backend).__name__}, "
            f"block_threshold={c.block_threshold}, min_confidence={c.min_confidence}, "
            f"promote_after={c.promote_after}, top_k={c.top_k})"
        )

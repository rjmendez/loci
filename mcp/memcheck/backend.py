"""Backend trait + in-memory backend for the memcheck core.

A ``VerdictBackend`` stores verdicts and recalls semantically-similar prior
verdicts by cosine similarity over an embedding. ``InMemoryBackend`` is the
test/reference implementation; the qdrant and mnemosyne backends (in their own
modules) implement the same ABC against persistent stores.

Pure stdlib — cosine is computed in Python so the core has no numpy dependency.
"""

from __future__ import annotations

import abc
import asyncio
import math
from dataclasses import dataclass

from .verdict import Verdict

__all__ = [
    "ScoredVerdict",
    "VerdictBackend",
    "InMemoryBackend",
    "cosine_similarity",
]

# Two subjects are coalesced into one verdict when their embeddings are at least
# this similar (and they share a subject_kind).
_COALESCE_THRESHOLD = 0.97


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors.

    Returns 0.0 for empty vectors, length mismatch, or a zero-magnitude vector
    (so callers never divide by zero and an absent embedding simply never
    matches).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        mag_a += x * x
        mag_b += y * y
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (math.sqrt(mag_a) * math.sqrt(mag_b))


@dataclass
class ScoredVerdict:
    """A verdict paired with its similarity to a recall query."""

    verdict: Verdict
    similarity: float


class VerdictBackend(abc.ABC):
    """Storage + recall interface for verdicts.

    All methods are async so persistent backends can do network I/O. The engine
    treats every backend method as potentially-failing and catches at its own
    boundary (fail-open), so backends may raise.
    """

    @abc.abstractmethod
    async def record(self, verdict: Verdict) -> None:
        """Persist a verdict, coalescing near-duplicate subjects."""
        raise NotImplementedError

    @abc.abstractmethod
    async def recall(
        self,
        query_text: str,
        embedding: list[float],
        kind: str,
        top_k: int,
    ) -> list[ScoredVerdict]:
        """Return up to ``top_k`` verdicts of ``kind`` most similar to ``embedding``."""
        raise NotImplementedError

    @abc.abstractmethod
    async def stats(self) -> dict:
        """Return ``{"total_verdicts": int, "recurring_blocks": int}``."""
        raise NotImplementedError

    async def forget(self, subject_excerpt: str, kind: str) -> int:
        """Remove verdicts of ``kind`` whose excerpt matches exactly.

        Concrete default of 0 so external/partial implementations can skip it
        without being forced to implement a removal path.
        """
        return 0


class InMemoryBackend(VerdictBackend):
    """List-backed reference backend, guarded by an ``asyncio.Lock``.

    Intended for tests and as the canonical semantics the persistent backends
    must mirror.
    """

    def __init__(self) -> None:
        self._verdicts: list[Verdict] = []
        self._embeddings: dict[str, list[float]] = {}  # verdict.id -> embedding
        self._lock = asyncio.Lock()

    async def record(self, verdict: Verdict) -> None:
        # The embedding must be supplied for coalescing; pull it from the caller
        # via record_with_embedding when available, else fall back to no-embed.
        await self._record(verdict, embedding=getattr(verdict, "_embedding", None))

    async def record_with_embedding(self, verdict: Verdict, embedding: list[float]) -> None:
        """Record carrying the embedding used for near-duplicate coalescing."""
        await self._record(verdict, embedding=embedding)

    async def _record(self, verdict: Verdict, embedding: list[float] | None) -> None:
        async with self._lock:
            if embedding:
                for existing in self._verdicts:
                    if existing.subject_kind != verdict.subject_kind:
                        continue
                    existing_emb = self._embeddings.get(existing.id)
                    if not existing_emb:
                        continue
                    if cosine_similarity(existing_emb, embedding) >= _COALESCE_THRESHOLD:
                        # Coalesce: bump occurrence count + last_seen, keep the
                        # strongest (highest-confidence) decision/rationale.
                        existing.occurrences += 1
                        existing.last_seen = verdict.last_seen
                        if verdict.confidence > existing.confidence:
                            existing.decision = verdict.decision
                            existing.rationale = verdict.rationale
                            existing.verdict_type = verdict.verdict_type
                            existing.confidence = verdict.confidence
                            existing.source = verdict.source
                        return
            self._verdicts.append(verdict)
            if embedding:
                self._embeddings[verdict.id] = embedding

    async def recall(
        self,
        query_text: str,
        embedding: list[float],
        kind: str,
        top_k: int,
    ) -> list[ScoredVerdict]:
        async with self._lock:
            scored = [
                ScoredVerdict(
                    verdict=v,
                    similarity=cosine_similarity(self._embeddings.get(v.id, []), embedding),
                )
                for v in self._verdicts
                if v.subject_kind == kind
            ]
        scored.sort(key=lambda s: s.similarity, reverse=True)
        return scored[:top_k]

    async def stats(self) -> dict:
        async with self._lock:
            total = len(self._verdicts)
            recurring = sum(
                1
                for v in self._verdicts
                if v.occurrences >= 2 and v.decision in ("flag", "warn", "quarantine")
            )
        return {"total_verdicts": total, "recurring_blocks": recurring}

    async def forget(self, subject_excerpt: str, kind: str) -> int:
        async with self._lock:
            keep: list[Verdict] = []
            removed = 0
            for v in self._verdicts:
                if v.subject_kind == kind and v.subject_excerpt == subject_excerpt:
                    removed += 1
                    self._embeddings.pop(v.id, None)
                else:
                    keep.append(v)
            self._verdicts = keep
        return removed

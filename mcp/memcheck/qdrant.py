"""Qdrant-backed VerdictBackend (increment 1).

Persists verdicts in a dedicated qdrant collection (``hermes_verdicts`` by
default), separate from ``ironclaw_enforcement`` per the locked design decision
but with the same schema so the collections are interchangeable.

Dependency injection
--------------------
This module never imports ``qdrant-client`` / ``fastembed`` at module top —
``import memcheck`` must stay dependency-free. The constructor takes injected
dependencies so the backend is testable without a live qdrant:

- ``client``  — a qdrant client object. Only a small slice of the
  ``qdrant-client`` sync API is used: ``retrieve``, ``upsert``,
  ``query_points``, ``delete`` and ``count``. The client's model types
  (``PointStruct``, ``Filter``, ...) are imported lazily inside the methods
  that build them, so a fake client in tests can avoid them entirely by
  exercising the same code paths with the real ``qdrant_client.models`` if
  installed, or the backend's filter is passed through opaquely.
- ``embed`` — a callable ``str -> list[float] | None`` turning text into a
  dense vector (``server.py``'s ``_embed``).

In production these come from ``server.py``'s ``_get_qdrant()`` (the
``QdrantClient``) and ``_embed()``. Here they are simply injected.

Point id: ``uuid5(NAMESPACE, subject_signature)`` so a given subject always
maps to one stable point. Verdict fields go in the point payload via
``Verdict.to_payload()``.

record(): read-modify-write on occurrences — retrieve the existing point by
the uuid5 id, and if present bump ``occurrences`` + ``last_seen`` and keep the
strongest (highest-confidence) decision/rationale before re-upserting; else
upsert fresh. Idempotent + bounded (same content -> same id).

recall(): embed the query (or accept a precomputed embedding), vector-search
filtered to ``subject_kind == kind``, map each hit to
``ScoredVerdict(Verdict.from_payload(hit.payload), hit.score)``, sort desc.

stats(): collection point count -> ``total_verdicts``; count points whose
payload ``occurrences >= 2`` and ``decision in {flag,warn,quarantine}`` ->
``recurring_blocks`` (best-effort scroll, capped).

forget(): compute ``make_signature(kind, subject_excerpt)`` -> point id ->
delete that point; return 1 if it existed else 0.

Fail behavior: hard errors may propagate (the ENGINE wraps backends fail-open).
Logic is not swallowed.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Callable, Optional

from .backend import ScoredVerdict, VerdictBackend
from .verdict import Verdict, make_signature

__all__ = ["QdrantBackend"]

_log = logging.getLogger("memcheck")

# Stable namespace for uuid5(subject_signature) point ids.
VERDICT_NAMESPACE = uuid.UUID("6d656d63-6865-636b-0000-646f6e757400")
COLLECTION = "hermes_verdicts"

# Decisions that count as a recurring "block" for stats.
_BLOCKING = ("flag", "warn", "quarantine")

# Cap on the scroll scan when computing recurring_blocks, so stats() stays
# bounded on a large collection.
_STATS_SCAN_CAP = 10_000


class QdrantBackend(VerdictBackend):
    """Persistent verdict store over qdrant.

    Parameters
    ----------
    client:
        A qdrant client object (e.g. ``qdrant_client.QdrantClient``). Injected
        so this module never hard-imports qdrant. The sync methods
        ``retrieve``, ``upsert``, ``query_points``, ``delete`` and ``count``
        are used; each is run via ``asyncio.to_thread`` so the async contract
        holds without blocking the event loop.
    collection:
        Target collection name (defaults to ``hermes_verdicts``).
    embed:
        Callable ``str -> list[float] | None`` producing the dense vector for a
        text (``server.py``'s ``_embed``).
    vector_name:
        Named vector to store/search against. Defaults to ``"dense"`` to match
        the server's collection layout; pass ``None`` for an unnamed (default)
        vector collection.
    """

    def __init__(
        self,
        client=None,
        *,
        collection: str = COLLECTION,
        embed: Optional[Callable[[str], Optional[list[float]]]] = None,
        vector_name: Optional[str] = "dense",
    ) -> None:
        self._client = client
        self._collection = collection
        self._embed = embed
        self._vector_name = vector_name

    @staticmethod
    def point_id(subject_signature: str) -> str:
        """Stable uuid5 point id for a subject signature."""
        return str(uuid.uuid5(VERDICT_NAMESPACE, subject_signature))

    # -- internals ----------------------------------------------------------

    def _embed_text(self, text: str) -> Optional[list[float]]:
        if self._embed is None:
            return None
        return self._embed(text)

    def _vector_payload(self, vector: list[float]):
        """Shape a vector for upsert: named dict when a vector name is set."""
        if self._vector_name:
            return {self._vector_name: vector}
        return vector

    async def _retrieve(self, point_id: str) -> Optional[dict]:
        """Return the stored payload for ``point_id``, or None if absent."""
        points = await asyncio.to_thread(
            self._client.retrieve,
            collection_name=self._collection,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return None
        payload = getattr(points[0], "payload", None)
        return dict(payload) if payload else None

    async def _upsert(self, point_id: str, vector: list[float], payload: dict) -> None:
        from qdrant_client.models import PointStruct

        point = PointStruct(
            id=point_id, vector=self._vector_payload(vector), payload=payload
        )
        await asyncio.to_thread(
            self._client.upsert, collection_name=self._collection, points=[point]
        )

    # -- VerdictBackend -----------------------------------------------------

    async def record(self, verdict: Verdict) -> None:
        """Record a verdict, embedding its excerpt for the dense vector."""
        await self.record_with_embedding(verdict, None)

    async def record_with_embedding(
        self, verdict: Verdict, embedding: Optional[list[float]]
    ) -> None:
        """Record carrying a precomputed embedding (preferred by the engine).

        Read-modify-write keyed on the stable uuid5 point id: if a point for
        this subject already exists, bump ``occurrences`` + ``last_seen`` and
        keep the higher-confidence decision/rationale, then re-upsert; else
        upsert fresh.
        """
        vector = embedding if embedding else self._embed_text(verdict.subject_excerpt)
        if not vector:
            # No vector available — cannot index. The engine is fail-open; a
            # missing embedding simply means this verdict is not stored.
            _log.debug(
                "memcheck qdrant: no embedding for subject %s; skipping upsert",
                verdict.subject_signature,
            )
            return

        pid = self.point_id(verdict.subject_signature)
        payload = verdict.to_payload()

        existing = await self._retrieve(pid)
        if existing is not None:
            prior = Verdict.from_payload(existing)
            # Bump occurrence accounting on the incoming verdict before storing.
            payload["occurrences"] = max(prior.occurrences, verdict.occurrences) + 1
            payload["first_seen"] = prior.first_seen or verdict.first_seen
            payload["last_seen"] = verdict.last_seen or prior.last_seen
            # Keep the strongest (highest-confidence) decision/rationale.
            if prior.confidence >= verdict.confidence:
                payload["decision"] = prior.decision
                payload["rationale"] = prior.rationale
                payload["verdict_type"] = prior.verdict_type
                payload["confidence"] = prior.confidence
                payload["source"] = prior.source
            # Preserve the original id so the point identity is stable.
            payload["id"] = prior.id

        await self._upsert(pid, vector, payload)

    async def recall(
        self, query_text: str, embedding: list[float], kind: str, top_k: int
    ) -> list[ScoredVerdict]:
        """Vector-search the collection for verdicts of ``kind``."""
        vector = embedding if embedding else self._embed_text(query_text)
        if not vector:
            return []

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = Filter(
            must=[FieldCondition(key="subject_kind", match=MatchValue(value=kind))]
        )

        kwargs = dict(
            collection_name=self._collection,
            query=vector,
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        )
        if self._vector_name:
            kwargs["using"] = self._vector_name

        result = await asyncio.to_thread(self._client.query_points, **kwargs)
        points = getattr(result, "points", result)

        scored: list[ScoredVerdict] = []
        for p in points:
            payload = getattr(p, "payload", None)
            if not payload:
                continue
            scored.append(
                ScoredVerdict(
                    verdict=Verdict.from_payload(dict(payload)),
                    similarity=float(getattr(p, "score", 0.0) or 0.0),
                )
            )
        scored.sort(key=lambda s: s.similarity, reverse=True)
        return scored[:top_k]

    async def stats(self) -> dict:
        """Total point count + count of recurring blocking verdicts."""
        count_result = await asyncio.to_thread(
            self._client.count, collection_name=self._collection, exact=True
        )
        total = int(getattr(count_result, "count", count_result) or 0)

        # Best-effort scan for recurring blocks, capped so this stays bounded.
        recurring = 0
        next_offset = None
        scanned = 0
        while scanned < _STATS_SCAN_CAP:
            points, next_offset = await asyncio.to_thread(
                self._client.scroll,
                collection_name=self._collection,
                limit=min(256, _STATS_SCAN_CAP - scanned),
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for p in points:
                scanned += 1
                payload = getattr(p, "payload", None) or {}
                if (
                    int(payload.get("occurrences", 1)) >= 2
                    and payload.get("decision") in _BLOCKING
                ):
                    recurring += 1
            if next_offset is None:
                break

        return {"total_verdicts": total, "recurring_blocks": recurring}

    async def forget(self, subject_excerpt: str, kind: str) -> int:
        """Delete the point for ``(kind, subject_excerpt)``.

        Recomputes ``make_signature(kind, subject_excerpt)`` -> point id, checks
        existence, deletes, and returns 1 if it existed else 0.
        """
        pid = self.point_id(make_signature(kind, subject_excerpt))
        existing = await self._retrieve(pid)
        if existing is None:
            return 0
        from qdrant_client.models import PointIdsList

        await asyncio.to_thread(
            self._client.delete,
            collection_name=self._collection,
            points_selector=PointIdsList(points=[pid]),
        )
        return 1

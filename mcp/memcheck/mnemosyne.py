"""Mnemosyne-backed VerdictBackend.

Mirror verdicts into the mnemosyne shared-memory layer so they survive in the
same substrate the MCP server uses for findings. This module reuses ``server.py``
helpers *by injection* rather than importing the mnemosyne client, so that
``import memcheck`` stays dependency-free.

Injected callables (mirroring ``server.py``)
--------------------------------------------
- ``remember(content: str, *, importance: float = 0.6, metadata: dict | None = None) -> bool``
  (``server.py:_mnemo_remember``, ~line 148) — store a memory; returns a
  bool success.
- ``recall(query: str, *, top_k: int = 10, investigation_id: str | None = None) -> list[dict]``
  (``server.py:_mnemo_recall``, ~line 202) — semantic recall returning a
  list of result dicts (content + metadata + maybe a score).

Promotion note
--------------
Mnemosyne stores each ``remember`` as a distinct memory (it does not coalesce),
so recurrence is the *number of matching records* recalled — the engine's
promotion logic counts distinct matching records, which is the intended
behavior here.

forget() is a documented NO-OP
------------------------------
The mnemosyne MCP surface exposes no delete tool (confirmed: remember / recall /
get_memory / get_stats / sleep / scratchpad only), so ``forget`` cannot remove
anything. It logs a warning and returns 0. See ``forget`` below.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .backend import ScoredVerdict, VerdictBackend
from .verdict import Verdict

__all__ = ["MnemosyneBackend"]

_log = logging.getLogger("memcheck")


def _as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float coercion that never raises."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_metadata(result: Any) -> dict:
    """Pull the verdict metadata dict out of a mnemosyne result envelope.

    Defensive about shape (mirroring the Rust ``MnemosyneMcpBackend``): the
    metadata may sit at ``result["metadata"]`` directly, or be nested under a
    ``memory``/``item`` wrapper. Returns ``{}`` when nothing dict-shaped is
    found.
    """
    if not isinstance(result, dict):
        return {}
    meta = result.get("metadata")
    if isinstance(meta, dict):
        return meta
    # Some envelopes nest the memory object one level down.
    for key in ("memory", "item", "payload", "data"):
        inner = result.get(key)
        if isinstance(inner, dict):
            inner_meta = inner.get("metadata")
            if isinstance(inner_meta, dict):
                return inner_meta
    return {}


def _extract_score(result: Any) -> float:
    """Pull a similarity/score off a result envelope, defaulting to 0.0."""
    if not isinstance(result, dict):
        return 0.0
    for key in ("score", "similarity"):
        if key in result and result[key] is not None:
            return _as_float(result[key], 0.0)
    return 0.0


class MnemosyneBackend(VerdictBackend):
    """Verdict store mirrored into mnemosyne.

    Parameters
    ----------
    remember, recall:
        The ``server.py`` helper callables, injected so this module has no
        hard import dependency on the mnemosyne client. See the module docstring
        for their signatures. When ``None``, the corresponding operation is a
        safe no-op (the engine treats backends as fail-open).
    """

    def __init__(
        self,
        *,
        remember: Optional[Callable[..., bool]] = None,
        recall: Optional[Callable[..., list]] = None,
    ) -> None:
        self._remember = remember
        self._recall = recall

    async def record(self, verdict: Verdict) -> None:
        """Store the verdict as a mnemosyne memory.

        Content is the verdict excerpt; ``importance`` scales with confidence;
        the full verdict payload rides in ``metadata`` (tagged ``memcheck: True``
        so recall can filter to enforcement memories).
        """
        if self._remember is None:
            return
        metadata = {**verdict.to_payload(), "memcheck": True}
        self._remember(
            content=verdict.subject_excerpt,
            importance=verdict.confidence,
            metadata=metadata,
        )

    async def recall(
        self, query_text: str, embedding: list[float], kind: str, top_k: int
    ) -> list[ScoredVerdict]:
        """Recall prior verdicts of ``kind`` semantically similar to the query.

        Calls the injected recall callable, keeps only results whose metadata is
        tagged ``memcheck == True`` and whose ``subject_kind == kind``,
        reconstructs each via :meth:`Verdict.from_payload`, and pairs it with the
        result's score (mnemosyne's hybrid score is used as the similarity proxy;
        ``embedding`` is accepted for interface parity but mnemosyne already
        ranks). Returns a list sorted by similarity, descending.
        """
        if self._recall is None:
            return []
        results = self._recall(query_text, top_k=top_k)
        if not results:
            return []

        scored: list[ScoredVerdict] = []
        for result in results:
            metadata = _extract_metadata(result)
            if metadata.get("memcheck") is not True:
                continue
            if metadata.get("subject_kind") != kind:
                continue
            try:
                verdict = Verdict.from_payload(metadata)
            except (KeyError, TypeError, ValueError):
                # Malformed payload — skip rather than fail the whole recall.
                continue
            scored.append(
                ScoredVerdict(verdict=verdict, similarity=_extract_score(result))
            )

        scored.sort(key=lambda s: s.similarity, reverse=True)
        return scored

    async def stats(self) -> dict:
        """Return the standard stats shape.

        Mnemosyne does not expose enforcement-scoped counts, so this is a
        best-effort zeroed snapshot. Stats are observability, never a hot path.
        """
        _log.debug(
            "memcheck: mnemosyne backend does not expose enforcement-scoped "
            "counts; returning zeroed stats"
        )
        return {"total_verdicts": 0, "recurring_blocks": 0}

    async def forget(self, subject_excerpt: str, kind: str) -> int:
        """No-op: mnemosyne exposes no delete tool, so nothing can be removed.

        The mnemosyne MCP surface is remember / recall / get_memory / get_stats /
        sleep / scratchpad only — there is no deletion path. This logs a warning
        and returns 0 so callers can treat it as "nothing removed".
        """
        _log.warning(
            "memcheck: mnemosyne-backed forget is unsupported (no delete tool); "
            "ignoring forget(kind=%s)",
            kind,
        )
        return 0

"""Tests for memcheck/qdrant.py — QdrantBackend with a FakeQdrantClient.

Exercises point_id determinism, RMW occurrences bump, confidence keep-max,
subject_kind filtering, forget() present/absent, and the no-embedding skip path.
No live Qdrant is required.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

# Stub out qdrant_client before any imports so the lazy imports in qdrant.py work
import types as _types
import sys as _sys

def _install_qdrant_stub():
    """Install a minimal qdrant_client stub so qdrant.py's lazy imports succeed."""
    if "qdrant_client" in _sys.modules:
        return

    @dataclass
    class PointStruct:
        id: str
        vector: Any
        payload: dict

    @dataclass
    class PointIdsList:
        points: list

    @dataclass
    class Filter:
        must: list = field(default_factory=list)
        must_not: list = field(default_factory=list)

    @dataclass
    class FieldCondition:
        key: str
        match: Any = None

    @dataclass
    class MatchValue:
        value: Any = None

    models_mod = _types.ModuleType("qdrant_client.models")
    models_mod.PointStruct = PointStruct
    models_mod.PointIdsList = PointIdsList
    models_mod.Filter = Filter
    models_mod.FieldCondition = FieldCondition
    models_mod.MatchValue = MatchValue

    qdrant_mod = _types.ModuleType("qdrant_client")
    qdrant_mod.models = models_mod

    _sys.modules["qdrant_client"] = qdrant_mod
    _sys.modules["qdrant_client.models"] = models_mod

_install_qdrant_stub()

from memcheck.qdrant import QdrantBackend, VERDICT_NAMESPACE, COLLECTION
from memcheck.verdict import make_signature, new_verdict
import uuid


# ---------------------------------------------------------------------------
# Minimal fake qdrant client — dict-backed, no qdrant_client import needed.
# ---------------------------------------------------------------------------

@dataclass
class _FakePoint:
    id: str
    payload: dict
    score: float = 1.0
    vector: Any = None


class _FakeQdrantClient:
    """Dict-backed fake for the QdrantBackend's narrow client interface."""

    def __init__(self):
        self._store: dict[str, _FakePoint] = {}

    def retrieve(self, *, collection_name, ids, with_payload=True):
        return [self._store[i] for i in ids if i in self._store]

    def upsert(self, *, collection_name, points):
        for p in points:
            pid = str(p.id)
            self._store[pid] = _FakePoint(
                id=pid,
                payload=dict(p.payload),
                vector=p.vector,
            )

    def query_points(self, *, collection_name, query, query_filter=None,
                     limit=10, with_payload=True, using=None):
        # Return all stored points (fake scoring = 0.9); filter by subject_kind
        # if a Filter object with must conditions is present.
        kind_filter = None
        if query_filter is not None:
            for cond in (query_filter.must or []):
                key = getattr(cond, "key", None)
                val = getattr(getattr(cond, "match", None), "value", None)
                if key == "subject_kind":
                    kind_filter = val

        results = []
        for p in self._store.values():
            if kind_filter and p.payload.get("subject_kind") != kind_filter:
                continue
            results.append(_FakePoint(id=p.id, payload=dict(p.payload), score=0.9))

        class _QueryResult:
            points = results[:limit]
        return _QueryResult()

    def delete(self, *, collection_name, points_selector):
        for pid in (points_selector.points or []):
            self._store.pop(str(pid), None)

    def count(self, *, collection_name, exact=True):
        class _CountResult:
            count = len(self._store)
        return _CountResult()

    def scroll(self, *, collection_name, limit=10, offset=None,
               with_payload=True, with_vectors=False):
        items = list(self._store.values())
        start = 0
        if offset is not None:
            try:
                start = items.index(offset)
            except ValueError:
                start = 0
        chunk = items[start : start + limit]
        next_offset = items[start + limit] if start + limit < len(items) else None
        return chunk, next_offset


def _fake_embed(text: str) -> list[float]:
    return [0.1] * 384


def _make_backend(embed=_fake_embed) -> QdrantBackend:
    return QdrantBackend(
        client=_FakeQdrantClient(),
        collection=COLLECTION,
        embed=embed,
        vector_name=None,  # unnamed vector — simpler for the fake
    )


def _run(coro):
    return asyncio.run(coro)


class TestPointIdDeterminism(unittest.TestCase):
    def test_same_input_same_id(self):
        sig = make_signature("action", "Bash ls /tmp")
        id1 = QdrantBackend.point_id(sig)
        id2 = QdrantBackend.point_id(sig)
        self.assertEqual(id1, id2)

    def test_different_input_different_id(self):
        id1 = QdrantBackend.point_id(make_signature("action", "Bash ls /tmp"))
        id2 = QdrantBackend.point_id(make_signature("action", "Bash rm -rf /"))
        self.assertNotEqual(id1, id2)

    def test_namespace_is_uuid5(self):
        sig = "test"
        expected = str(uuid.uuid5(VERDICT_NAMESPACE, sig))
        self.assertEqual(QdrantBackend.point_id(sig), expected)


class TestRecordWithEmbedding(unittest.TestCase):
    def _verdict(self, subject="test subject", confidence=0.8, decision="flag", occurrences=1):
        v = new_verdict(
            subject_kind="action",
            subject_signature=make_signature("action", subject),
            subject_excerpt=subject,
            verdict_type="observed_action",
            decision=decision,
            confidence=confidence,
            rationale="test",
            source="rule",
        )
        # Override occurrences for RMW tests
        object.__setattr__(v, "occurrences", occurrences) if hasattr(type(v), "__setattr__") else None
        v = v.__class__(**{**v.__dict__, "occurrences": occurrences})
        return v

    def test_upsert_on_fresh_record(self):
        b = _make_backend()
        v = self._verdict("fresh action")
        _run(b.record_with_embedding(v, _fake_embed("fresh action")))
        pid = QdrantBackend.point_id(v.subject_signature)
        self.assertIn(pid, b._client._store)

    def test_occurrences_bump_on_duplicate(self):
        b = _make_backend()
        v = self._verdict("dup action")
        _run(b.record_with_embedding(v, _fake_embed("dup action")))
        _run(b.record_with_embedding(v, _fake_embed("dup action")))
        pid = QdrantBackend.point_id(v.subject_signature)
        stored = b._client._store[pid].payload
        self.assertGreaterEqual(stored["occurrences"], 2)

    def test_confidence_max_kept(self):
        b = _make_backend()
        v_low = self._verdict("conf action", confidence=0.3, decision="warn")
        v_high = self._verdict("conf action", confidence=0.95, decision="flag")
        _run(b.record_with_embedding(v_low, _fake_embed("conf action")))
        _run(b.record_with_embedding(v_high, _fake_embed("conf action")))
        pid = QdrantBackend.point_id(v_low.subject_signature)
        stored = b._client._store[pid].payload
        # The higher confidence verdict's fields should be kept
        self.assertGreaterEqual(stored["confidence"], 0.95 - 1e-6)

    def test_no_embedding_skips_upsert(self):
        """record_with_embedding with embed=None and no precomputed embedding must skip upsert."""
        b = QdrantBackend(client=_FakeQdrantClient(), collection=COLLECTION, embed=None, vector_name=None)
        v = self._verdict("skip action")
        _run(b.record_with_embedding(v, None))
        pid = QdrantBackend.point_id(v.subject_signature)
        self.assertNotIn(pid, b._client._store)


class TestRecall(unittest.TestCase):
    def test_recall_filters_by_subject_kind(self):
        b = _make_backend()
        action_v = new_verdict(
            subject_kind="action",
            subject_signature=make_signature("action", "ls /"),
            subject_excerpt="ls /",
            verdict_type="observed_action",
            decision="flag",
            confidence=0.9,
            rationale="r",
            source="rule",
        )
        claim_v = new_verdict(
            subject_kind="claim",
            subject_signature=make_signature("claim", "some claim"),
            subject_excerpt="some claim",
            verdict_type="claim_check",
            decision="warn",
            confidence=0.7,
            rationale="r",
            source="llm",
        )
        _run(b.record_with_embedding(action_v, _fake_embed("ls /")))
        _run(b.record_with_embedding(claim_v, _fake_embed("some claim")))

        results = _run(b.recall("ls /", _fake_embed("ls /"), "action", 10))
        kinds = [sv.verdict.subject_kind for sv in results]
        self.assertIn("action", kinds)
        self.assertNotIn("claim", kinds)

    def test_recall_returns_scored_verdicts(self):
        b = _make_backend()
        v = new_verdict(
            subject_kind="action",
            subject_signature=make_signature("action", "recall test"),
            subject_excerpt="recall test",
            verdict_type="observed_action",
            decision="flag",
            confidence=1.0,
            rationale="r",
            source="rule",
        )
        _run(b.record_with_embedding(v, _fake_embed("recall test")))
        results = _run(b.recall("recall test", _fake_embed("recall test"), "action", 5))
        self.assertGreater(len(results), 0)
        self.assertIsNotNone(results[0].verdict)

    def test_recall_no_embed_returns_empty(self):
        b = QdrantBackend(client=_FakeQdrantClient(), collection=COLLECTION, embed=None, vector_name=None)
        results = _run(b.recall("query", None, "action", 5))
        self.assertEqual(results, [])


class TestForget(unittest.TestCase):
    def test_forget_existing_returns_1(self):
        b = _make_backend()
        excerpt = "forget me"
        kind = "action"
        v = new_verdict(
            subject_kind=kind,
            subject_signature=make_signature(kind, excerpt),
            subject_excerpt=excerpt,
            verdict_type="observed_action",
            decision="flag",
            confidence=0.8,
            rationale="r",
            source="rule",
        )
        _run(b.record_with_embedding(v, _fake_embed(excerpt)))
        result = _run(b.forget(excerpt, kind))
        self.assertEqual(result, 1)

    def test_forget_absent_returns_0(self):
        b = _make_backend()
        result = _run(b.forget("nonexistent subject", "action"))
        self.assertEqual(result, 0)

    def test_forget_actually_removes_point(self):
        b = _make_backend()
        excerpt = "remove this"
        kind = "action"
        v = new_verdict(
            subject_kind=kind,
            subject_signature=make_signature(kind, excerpt),
            subject_excerpt=excerpt,
            verdict_type="observed_action",
            decision="flag",
            confidence=0.8,
            rationale="r",
            source="rule",
        )
        _run(b.record_with_embedding(v, _fake_embed(excerpt)))
        _run(b.forget(excerpt, kind))
        # Should not find it anymore
        recalled = _run(b.recall(excerpt, _fake_embed(excerpt), kind, 10))
        sigs = [sv.verdict.subject_signature for sv in recalled]
        self.assertNotIn(v.subject_signature, sigs)


class TestStats(unittest.TestCase):
    def test_stats_total_count(self):
        b = _make_backend()
        for i in range(3):
            v = new_verdict(
                subject_kind="action",
                subject_signature=make_signature("action", f"action {i}"),
                subject_excerpt=f"action {i}",
                verdict_type="observed_action",
                decision="flag",
                confidence=0.8,
                rationale="r",
                source="rule",
            )
            _run(b.record_with_embedding(v, _fake_embed(f"action {i}")))
        stats = _run(b.stats())
        self.assertEqual(stats["total_verdicts"], 3)
        self.assertIn("recurring_blocks", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)

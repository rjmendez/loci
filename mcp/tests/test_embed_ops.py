"""Tests for embed_ops — the embedding-tier semantic ops. Embeddings are stubbed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import embed_ops as E  # noqa: E402


# Deterministic 2-D "embeddings": items in the same group point the same way.
_VEC = {
    "cat a": [1.0, 0.0], "cat a again": [0.999, 0.001],   # near-duplicates
    "dog b": [0.0, 1.0], "dog b too": [0.001, 0.999],     # a different near-dup pair
    "bird c": [0.7, 0.7],                                  # its own thing
}


def _stub(texts):
    return [_VEC[t] for t in texts]


def test_dedup_clusters_near_duplicates():
    items = ["cat a", "dog b", "cat a again", "dog b too", "bird c"]
    r = E.dedup(items, threshold=0.95, embed_fn=_stub)
    assert r["degraded"] is False
    assert len(r["clusters"]) == 3          # {cats}, {dogs}, {bird}
    assert r["dropped"] == 2
    assert set(r["kept"]) == {"cat a", "dog b", "bird c"}  # first-seen representative
    cats = next(c for c in r["clusters"] if c["text"] == "cat a")
    assert cats["member_indices"] == [0, 2]


def test_dedup_high_threshold_keeps_everything():
    items = ["cat a", "bird c", "dog b"]
    r = E.dedup(items, threshold=0.999999, embed_fn=_stub)
    assert r["dropped"] == 0 and len(r["clusters"]) == 3


def test_dedup_dicts_use_text_key():
    items = [{"id": 1, "msg": "cat a"}, {"id": 2, "msg": "cat a again"}]
    r = E.dedup(items, threshold=0.95, key="msg", embed_fn=_stub)
    assert r["dropped"] == 1
    assert r["kept"] == [{"id": 1, "msg": "cat a"}]


def test_dedup_fail_open_no_embeddings():
    items = ["cat a", "cat a again"]
    r = E.dedup(items, threshold=0.9, embed_fn=lambda t: [])  # embeddings unavailable
    assert r["degraded"] is True
    assert r["dropped"] == 0 and len(r["clusters"]) == 2  # nothing merged


def test_dedup_singleton_and_empty():
    assert E.dedup([], embed_fn=_stub)["kept"] == []
    assert E.dedup(["cat a"], embed_fn=_stub)["dropped"] == 0


def test_relevance_orders_by_topic():
    r = E.relevance(["cat a", "dog b"], "cat a", embed_fn=_stub)
    assert r["degraded"] is False
    assert r["scores"][0] > r["scores"][1]   # 'cat a' more relevant to topic 'cat a'
    assert abs(r["scores"][0] - 1.0) < 1e-6


def test_relevance_fail_open():
    r = E.relevance(["cat a"], "cat a", embed_fn=lambda t: [])
    assert r["degraded"] is True and r["scores"] == [None]
    assert E.relevance(["x"], "")["degraded"] is True  # empty topic

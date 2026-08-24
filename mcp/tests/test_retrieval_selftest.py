"""retrieval_selftest against a real in-memory Qdrant.

The failure this exists to catch — a collection written at one embedding width
and queried at another — is invisible per-query: it raises, the caller catches
per-collection, and a broken collection looks exactly like one that had no
match. These build that store for real rather than mocking it.
"""
import json
import unittest
from unittest import mock

import pytest

qdrant_client = pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient                       # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams   # noqa: E402

import server                                                # noqa: E402

DIM = 8


def _store():
    """Three collections: our width, a foreign width, and an empty one."""
    c = QdrantClient(location=":memory:")
    c.create_collection("hermes_memory",
                        vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)})
    c.create_collection("legacy_chunks",
                        vectors_config=VectorParams(size=DIM * 2, distance=Distance.COSINE))
    c.create_collection("empty_shelf",
                        vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)})
    c.upsert("hermes_memory", points=[
        PointStruct(id=1, vector={"dense": [0.1] * DIM}, payload={"text": "a finding"}),
    ])
    c.upsert("legacy_chunks", points=[
        PointStruct(id=1, vector=[0.1] * (DIM * 2), payload={"text": "a chunk"}),
    ])
    return c


_UNSET = object()


def _run(client, vec=_UNSET):
    embedding = [0.1] * DIM if vec is _UNSET else vec
    with mock.patch.object(server, "_get_qdrant", lambda: (client, "hermes_memory")), \
         mock.patch.object(server, "_embed", lambda _q: embedding):
        return json.loads(server.retrieval_selftest("anything"))


def _by_name(out):
    return {r["collection"]: r for r in out["collections"]}


class TestRetrievalSelftest(unittest.TestCase):
    def test_a_foreign_width_collection_is_named_not_silently_empty(self):
        out = _run(_store())
        rows = _by_name(out)
        self.assertEqual(rows["legacy_chunks"]["status"], "width_mismatch")
        self.assertIn("16-dim", rows["legacy_chunks"]["detail"])
        self.assertIn("8", rows["legacy_chunks"]["detail"])
        self.assertEqual(out["status"], "degraded")

    def test_a_healthy_collection_reports_hits(self):
        rows = _by_name(_run(_store()))
        self.assertEqual(rows["hermes_memory"]["status"], "ok")
        self.assertEqual(rows["hermes_memory"]["hits"], 1)

    def test_an_empty_collection_is_not_a_fault(self):
        out = _run(_store())
        self.assertEqual(_by_name(out)["empty_shelf"]["status"], "empty")
        self.assertEqual(out["status"], "degraded")  # from legacy_chunks alone

    def test_all_collections_broken_rolls_up_to_unhealthy(self):
        c = QdrantClient(location=":memory:")
        c.create_collection("a", vectors_config=VectorParams(size=DIM * 2, distance=Distance.COSINE))
        c.create_collection("b", vectors_config=VectorParams(size=DIM * 4, distance=Distance.COSINE))
        c.upsert("a", points=[PointStruct(id=1, vector=[0.1] * (DIM * 2))])
        c.upsert("b", points=[PointStruct(id=1, vector=[0.1] * (DIM * 4))])
        out = _run(c)
        self.assertEqual(out["status"], "unhealthy")

    def test_remediations_are_deduplicated_across_collections(self):
        c = QdrantClient(location=":memory:")
        for name in ("a", "b", "c"):
            c.create_collection(name, vectors_config=VectorParams(size=DIM * 2, distance=Distance.COSINE))
            c.upsert(name, points=[PointStruct(id=1, vector=[0.1] * (DIM * 2))])
        out = _run(c)
        self.assertEqual(len(out["collections"]), 3)
        self.assertEqual(len(out["remediations"]), 1)

    def test_a_healthy_store_is_ok(self):
        c = QdrantClient(location=":memory:")
        c.create_collection("only", vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)})
        c.upsert("only", points=[PointStruct(id=1, vector={"dense": [0.1] * DIM})])
        out = _run(c)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["embedder_dim"], DIM)

    def test_no_embedder_is_reported_rather_than_read_as_empty(self):
        out = _run(_store(), vec=None)
        statuses = {r["collection"]: r["status"] for r in out["collections"]}
        self.assertEqual(statuses["hermes_memory"], "error")
        self.assertEqual(statuses["empty_shelf"], "empty")
        self.assertTrue(any("OLLAMA_BASE_URL" in r for r in out["remediations"]))

    def test_qdrant_down_is_unhealthy_not_ok(self):
        with mock.patch.object(server, "_get_qdrant", lambda: (None, None)):
            out = json.loads(server.retrieval_selftest("anything"))
        self.assertEqual(out["status"], "unhealthy")
        self.assertEqual(out["collections"], [])


if __name__ == "__main__":
    unittest.main()

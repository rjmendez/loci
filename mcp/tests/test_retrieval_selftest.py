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
    c.create_collection("loci_memory",
                        vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)})
    c.create_collection("legacy_chunks",
                        vectors_config=VectorParams(size=DIM * 2, distance=Distance.COSINE))
    c.create_collection("empty_shelf",
                        vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)})
    c.upsert("loci_memory", points=[
        PointStruct(id=1, vector={"dense": [0.1] * DIM}, payload={"text": "a finding"}),
    ])
    c.upsert("legacy_chunks", points=[
        PointStruct(id=1, vector=[0.1] * (DIM * 2), payload={"text": "a chunk"}),
    ])
    return c


_UNSET = object()


def _run(client, vec=_UNSET):
    """Probe every collection in the store, explicitly.

    The tool's DEFAULT scope is only what the server retrieves from; these cases
    are about probe classification and rollup, so they name their collections and
    thereby opt them in — see TestScope for the scoping behaviour itself.
    """
    embedding = [0.1] * DIM if vec is _UNSET else vec
    names = sorted(c.name for c in client.get_collections().collections)
    with mock.patch.object(server, "_get_qdrant", lambda: (client, "loci_memory")), \
         mock.patch.object(server, "_embed", lambda _q: embedding):
        return json.loads(server.retrieval_selftest("anything", collections=names))


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
        self.assertEqual(rows["loci_memory"]["status"], "ok")
        self.assertEqual(rows["loci_memory"]["hits"], 1)

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
        self.assertEqual(statuses["loci_memory"], "error")
        self.assertEqual(statuses["empty_shelf"], "empty")
        self.assertTrue(any("OLLAMA_BASE_URL" in r for r in out["remediations"]))

    def test_qdrant_down_is_unhealthy_not_ok(self):
        with mock.patch.object(server, "_get_qdrant", lambda: (None, None)):
            out = json.loads(server.retrieval_selftest("anything"))
        self.assertEqual(out["status"], "unhealthy")
        self.assertEqual(out["collections"], [])


if __name__ == "__main__":
    unittest.main()


class _NamedVectorParams:
    def __init__(self, size):
        self.size = size


class _Cfg:
    def __init__(self, vectors, sparse=None):
        self.config = type("C", (), {"params": type("P", (), {
            "vectors": vectors, "sparse_vectors": sparse})()})()
        self.points_count = 1


class TestDenseVectorName(unittest.TestCase):
    """The call sites that take a collection name from the caller run against
    collections this server did not create, whose dense vector is not necessarily
    called `dense`. Asking for a name that is not there is a 400, which the
    retrieval path reports as 'no results'."""

    def setUp(self):
        import qdrant_ops
        qdrant_ops._dense_name_cache.clear()

    def _name(self, vectors):
        import qdrant_ops
        client = mock.Mock()
        client.get_collection.return_value = _Cfg(vectors)
        return qdrant_ops._dense_vector_name(client, "c")

    def test_dense_is_preferred_when_present(self):
        self.assertEqual(self._name({"dense": _NamedVectorParams(768),
                                     "other": _NamedVectorParams(768)}), "dense")

    def test_a_differently_named_vector_is_found(self):
        self.assertEqual(
            self._name({"fast-nomic-embed-text-v1.5": _NamedVectorParams(768)}),
            "fast-nomic-embed-text-v1.5")

    def test_the_one_matching_the_embedder_width_wins(self):
        import qdrant_ops
        got = self._name({"small": _NamedVectorParams(384),
                          "big": _NamedVectorParams(qdrant_ops.VECTOR_DIM)})
        self.assertEqual(got, "big")

    def test_an_unnamed_flat_collection_returns_none(self):
        self.assertIsNone(self._name(_NamedVectorParams(768)))

    def test_no_vectors_returns_none(self):
        self.assertIsNone(self._name({}))

    def test_a_failing_get_collection_returns_none_rather_than_raising(self):
        import qdrant_ops
        client = mock.Mock()
        client.get_collection.side_effect = RuntimeError("gone")
        self.assertIsNone(qdrant_ops._dense_vector_name(client, "c"))

    def test_the_answer_is_cached_not_refetched(self):
        import qdrant_ops
        client = mock.Mock()
        client.get_collection.return_value = _Cfg({"dense": _NamedVectorParams(768)})
        for _ in range(3):
            qdrant_ops._dense_vector_name(client, "c")
        self.assertEqual(client.get_collection.call_count, 1)


class TestProbeUsesTheResolvedName(unittest.TestCase):
    def test_a_collection_whose_vector_is_not_called_dense_probes_ok(self):
        import qdrant_ops
        qdrant_ops._dense_name_cache.clear()
        c = QdrantClient(location=":memory:")
        c.create_collection("odd", vectors_config={
            "fast-nomic-embed-text-v1.5": VectorParams(size=DIM, distance=Distance.COSINE)})
        c.upsert("odd", points=[PointStruct(
            id=1, vector={"fast-nomic-embed-text-v1.5": [0.1] * DIM})])
        with mock.patch.object(server, "_get_qdrant", lambda: (c, "odd")), \
             mock.patch.object(server, "_embed", lambda _q: [0.1] * DIM):
            out = json.loads(server.retrieval_selftest("anything", collections=["odd"]))
        row = {r["collection"]: r for r in out["collections"]}["odd"]
        self.assertEqual(row["status"], "ok", row.get("detail"))
        self.assertEqual(row["hits"], 1)


class TestScope(unittest.TestCase):
    """A store accumulates collections nobody queries. Rolling their width
    mismatches into the verdict makes the tool cry wolf, and a diagnostic you
    learn to ignore is worse than no diagnostic."""

    def _store(self):
        c = QdrantClient(location=":memory:")
        c.create_collection("loci_memory", vectors_config={
            "dense": VectorParams(size=DIM, distance=Distance.COSINE)})
        c.upsert("loci_memory", points=[
            PointStruct(id=1, vector={"dense": [0.1] * DIM})])
        # junk at a foreign width — feature vectors, leftovers, someone else's corpus
        for name, w in (("old_junk", DIM * 2), ("ant_features", DIM * 4)):
            c.create_collection(name, vectors_config=VectorParams(
                size=w, distance=Distance.COSINE))
            c.upsert(name, points=[PointStruct(id=1, vector=[0.1] * w)])
        return c

    def _run(self, **kw):
        import qdrant_ops
        qdrant_ops._dense_name_cache.clear()
        c = self._store()
        with mock.patch.object(server, "_get_qdrant", lambda: (c, "loci_memory")), \
             mock.patch.object(server, "_embed", lambda _q: [0.1] * DIM), \
             mock.patch.object(server, "QDRANT_COLLECTION_PREFIX", "loci_memory"), \
             mock.patch.object(server, "_CODE_CHUNKS_COLLECTION", ""):
            return json.loads(server.retrieval_selftest("anything", **kw))

    def test_default_scope_ignores_collections_the_server_never_queries(self):
        out = self._run()
        self.assertEqual(out["status"], "ok")
        self.assertEqual([r["collection"] for r in out["collections"]], ["loci_memory"])

    def test_scope_all_inventories_everything_but_health_stays_ok(self):
        out = self._run(scope="all")
        self.assertEqual(len(out["collections"]), 3)
        self.assertEqual(out["status"], "ok", "junk must not make the store unhealthy")
        self.assertIn("not queried by this server", out["summary"])

    def test_scope_all_still_reports_the_junk_for_inventory(self):
        out = self._run(scope="all")
        by = {r["collection"]: r for r in out["collections"]}
        self.assertEqual(by["old_junk"]["status"], "width_mismatch")
        self.assertFalse(by["old_junk"]["queried_by_server"])
        self.assertTrue(by["loci_memory"]["queried_by_server"])

    def test_remediations_only_cover_collections_that_are_queried(self):
        out = self._run(scope="all")
        self.assertEqual(out["remediations"], [],
                         "advice about collections nobody asks is noise")

    def test_an_explicit_list_overrides_scope(self):
        out = self._run(collections=["old_junk"])
        self.assertEqual([r["collection"] for r in out["collections"]], ["old_junk"])
        self.assertEqual(out["scope"], "explicit")
        self.assertEqual(out["status"], "unhealthy",
                         "asked about it explicitly, so it counts")

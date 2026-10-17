"""Rows must carry an id, and both cross-encoders must see the same passage.

Two defects that shipped together and hid each other. `_rag_search_collections`
dedups on `(origin, id)`; when `_qdrant_search_collection` returned rows without
an `id`, every row of a collection keyed to `(origin, None)` and the whole
collection folded to one hit. Measured on the live store before the fix:
`dama_gotchi_code` returned 5 rows with 1 distinct id, all None. `loci_memory`
escaped only because a finding stores its own id in the payload.
"""
import unittest
from unittest import mock

import qdrant_ops
import server


class _P:
    def __init__(self, pid, score, payload):
        self.id = pid
        self.score = score
        self.payload = payload


class _Res:
    def __init__(self, points):
        self.points = points


class _Client:
    def __init__(self, points, named=True):
        self._points = points
        self._named = named

    def get_collection(self, name):
        from types import SimpleNamespace
        vec = {"dense": SimpleNamespace(size=qdrant_ops.VECTOR_DIM)} if self._named \
            else SimpleNamespace(size=qdrant_ops.VECTOR_DIM)
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(
            vectors=vec, sparse_vectors=None)), points_count=len(self._points))

    def query_points(self, **kw):
        return _Res(list(self._points))


def _search(points, payload_has_id=False):
    client = _Client(points)
    with mock.patch.object(qdrant_ops, "_get_qdrant", lambda: (client, "c")), \
         mock.patch.object(qdrant_ops, "_embed", lambda _t: [0.1] * qdrant_ops.VECTOR_DIM), \
         mock.patch.object(qdrant_ops, "_embed_sparse", lambda _t: None), \
         mock.patch.object(qdrant_ops, "_ce_rerank", lambda q, rows, k: (rows[:k], False)):
        qdrant_ops._dense_name_cache.clear()
        return qdrant_ops._qdrant_search_collection("q", collection_name="c", limit=5)


class TestRowIdentity(unittest.TestCase):
    def test_a_payload_without_an_id_still_yields_distinct_rows(self):
        pts = [_P(f"pt{i}", 0.9 - i / 100, {"text": f"chunk {i}"}) for i in range(5)]
        rows = _search(pts)
        ids = [r.get("id") for r in rows]
        self.assertEqual(len(set(ids)), 5, "each row must be distinguishable")
        self.assertNotIn(None, ids)

    def test_a_payload_that_carries_its_own_id_keeps_it(self):
        # findings store their id in the payload; the point id must not clobber it
        pts = [_P("point-uuid", 0.9, {"id": "finding-uuid", "text": "a finding"})]
        self.assertEqual(_search(pts)[0]["id"], "finding-uuid")

    def test_dedup_no_longer_folds_a_collection_to_one_hit(self):
        rows = _search([_P(f"pt{i}", 0.9, {"text": f"c{i}", "origin": "col"}) for i in range(5)])
        best = {}
        for h in rows:                      # mirrors _rag_search_collections
            best[(h.get("origin"), h.get("id"))] = h
        self.assertEqual(len(best), 5)


class TestBothCrossEncodersShareTheBudget(unittest.TestCase):
    def test_the_rag_re_pass_uses_the_same_budget_as_the_first(self):
        seen = {}

        class _CE:
            def predict(self, pairs):
                seen["len"] = len(pairs[0][1])
                return [0.5] * len(pairs)

        rows = [{"text": "x" * 5000}, {"text": "y" * 5000}]
        with mock.patch.object(server, "_get_cross_encoder", lambda: _CE()):
            server._rag_cross_encode(rows, "q")
        self.assertEqual(seen["len"], qdrant_ops.RERANK_MAX_CHARS)
        self.assertGreater(seen["len"], 512, "512 scored below using no reranker at all")

    def test_no_512_literal_survives_in_either_module(self):
        import inspect
        for mod in (qdrant_ops, server):
            src = inspect.getsource(mod)
            self.assertNotIn("[:512]", src, f"{mod.__name__} still truncates at 512")


if __name__ == "__main__":
    unittest.main()

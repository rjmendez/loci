"""rag_context_search has to say when it did not actually search.

mode was hardcoded to "rag_hybrid", so a run where every collection raised
returned the same shape as a genuine zero-hit run — the caller could not tell a
broken embedder from "nothing is known about this topic".
"""
import json
import unittest
from unittest import mock

import server


def _run(failing: set, collections=("loci_memory", "agent_core_chunks")):
    """Search `collections`, raising for any name in `failing`."""
    def _fake(sq, collection_name=None, limit=None, query_filter=None):
        if collection_name in failing:
            raise RuntimeError("Wrong input: Vector dimension error")
        return [{"origin": collection_name, "id": "p1", "score": 0.7, "text": "a finding"}]

    with mock.patch.object(server, "_qdrant_search_collection", _fake), \
         mock.patch.object(server, "_get_qdrant", lambda: (object(), "loci_memory")), \
         mock.patch.object(server, "_rag_cross_encode", lambda *a, **k: None), \
         mock.patch.object(server, "_rag_record_access", lambda *a, **k: None):
        raw = server.rag_context_search(
            "contractor access", collections=list(collections), expand_query=False
        )
    return json.loads(raw)


class TestRagFailureMode(unittest.TestCase):
    def test_all_collections_failing_is_not_reported_as_a_successful_search(self):
        out = _run({"loci_memory", "agent_core_chunks"})
        self.assertEqual(out["mode"], "rag_failed")
        self.assertEqual(out["collections_failed"], ["agent_core_chunks", "loci_memory"])

    def test_one_collection_failing_is_degraded(self):
        out = _run({"agent_core_chunks"})
        self.assertEqual(out["mode"], "rag_degraded")
        self.assertEqual(out["collections_failed"], ["agent_core_chunks"])

    def test_a_clean_search_still_says_rag_hybrid(self):
        out = _run(set())
        self.assertEqual(out["mode"], "rag_hybrid")
        self.assertEqual(out["collections_failed"], [])


if __name__ == "__main__":
    unittest.main()

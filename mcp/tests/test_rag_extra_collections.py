"""LOCI_RAG_EXTRA_COLLECTIONS: extra default collections for rag_context_search and the health probe.

The log index (ops_logs) was only reachable by passing collections=["ops_logs"]; a plain query searched
the memory and code collections and returned 20 unrelated code chunks. The extras are opt-in per
deployment (unset = unchanged), keep order, drop duplicates, and a missing collection is a reported
failure, not a silent skip.
"""
import json
import os
import unittest
from unittest import mock

import server


def _search(extra=None, **kw):
    """Run rag_context_search with no explicit collections and record what was searched."""
    seen = []

    def _fake(sq, collection_name=None, limit=None, query_filter=None):
        seen.append(collection_name)
        if collection_name in kw.get("failing", ()):
            raise RuntimeError("Not found: Collection `%s` doesn't exist" % collection_name)
        return [{"origin": collection_name, "id": "p-" + collection_name, "score": 0.7, "text": "a finding"}]

    env = {"LOCI_RAG_EXTRA_COLLECTIONS": extra} if extra is not None else {}
    with mock.patch.dict(os.environ, env, clear=False), \
         mock.patch.object(server, "_qdrant_search_collection", _fake), \
         mock.patch.object(server, "_get_qdrant", lambda: (object(), server.QDRANT_COLLECTION_PREFIX)), \
         mock.patch.object(server, "_rag_cross_encode", lambda *a, **k: None), \
         mock.patch.object(server, "_rag_record_access", lambda *a, **k: None):
        if extra is None:
            os.environ.pop("LOCI_RAG_EXTRA_COLLECTIONS", None)
        out = json.loads(server.rag_context_search("fail-open breaker", expand_query=False))
    return out, seen


class DefaultCollectionList(unittest.TestCase):
    def setUp(self):
        self._patch = mock.patch.object(server, "_CODE_CHUNKS_COLLECTION", "code_chunks")
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def test_unset_is_the_old_behavior(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOCI_RAG_EXTRA_COLLECTIONS", None)
            self.assertEqual(server._rag_default_collections(),
                             [server.QDRANT_COLLECTION_PREFIX, "code_chunks"])

    def test_extras_are_appended_in_order_trimmed_and_deduplicated(self):
        value = " ops_logs , ,code_chunks,ops_logs,other_idx "
        with mock.patch.dict(os.environ, {"LOCI_RAG_EXTRA_COLLECTIONS": value}):
            self.assertEqual(server._rag_default_collections(),
                             [server.QDRANT_COLLECTION_PREFIX, "code_chunks", "ops_logs", "other_idx"])

    def test_works_without_a_code_collection(self):
        with mock.patch.object(server, "_CODE_CHUNKS_COLLECTION", ""), \
             mock.patch.dict(os.environ, {"LOCI_RAG_EXTRA_COLLECTIONS": "ops_logs"}):
            self.assertEqual(server._rag_default_collections(), [server.QDRANT_COLLECTION_PREFIX, "ops_logs"])


class SearchUsesTheExtras(unittest.TestCase):
    def setUp(self):
        self._patch = mock.patch.object(server, "_CODE_CHUNKS_COLLECTION", "code_chunks")
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def test_a_plain_query_searches_the_extra_collection(self):
        out, seen = _search(extra="ops_logs")
        self.assertIn("ops_logs", out["collections_searched"])
        self.assertIn("ops_logs", set(seen))

    def test_without_the_env_ops_logs_is_not_searched(self):
        out, seen = _search(extra=None)
        self.assertNotIn("ops_logs", out["collections_searched"])
        self.assertNotIn("ops_logs", set(seen))

    def test_an_explicit_collections_argument_still_wins(self):
        with mock.patch.dict(os.environ, {"LOCI_RAG_EXTRA_COLLECTIONS": "ops_logs"}):
            seen = []

            def _fake(sq, collection_name=None, limit=None, query_filter=None):
                seen.append(collection_name)
                return []

            with mock.patch.object(server, "_qdrant_search_collection", _fake), \
                 mock.patch.object(server, "_get_qdrant", lambda: (object(), "x")), \
                 mock.patch.object(server, "_rag_cross_encode", lambda *a, **k: None), \
                 mock.patch.object(server, "_rag_record_access", lambda *a, **k: None):
                out = json.loads(server.rag_context_search("q", collections=["only_this"], expand_query=False))
        self.assertEqual(out["collections_searched"], ["only_this"])
        self.assertEqual(set(seen), {"only_this"})

    def test_a_missing_extra_collection_is_reported_not_skipped(self):
        out, _ = _search(extra="ops_logs", failing={"ops_logs"})
        self.assertEqual(out["collections_failed"], ["ops_logs"])
        self.assertEqual(out["mode"], "rag_degraded")


if __name__ == "__main__":
    unittest.main()

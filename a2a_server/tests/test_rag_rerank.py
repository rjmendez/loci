"""Tests for rag_search's cross-encoder rerank hook.

rag_search fans out across collections and merges on raw cosine score, which is not
comparable across collections. _rerank() replaces that ordering when RERANK_HTTP_URL is
set. It must FAIL OPEN in every degraded case -- search returning slightly worse ordering
is acceptable, search erroring is not -- so most of these assert the fallback.
"""

import asyncio
import importlib.util
import os
import pathlib
import unittest
from unittest import mock

os.environ.setdefault("HERMES_A2A_TOKEN", "test-token-abc123")
os.environ.setdefault("HERMES_A2A_URL", "http://localhost:8201")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MNEMOSYNE_EMBEDDING_API_URL", "http://localhost:11434/v1")

_server_path = pathlib.Path(__file__).parent.parent / "server.py"
_spec = importlib.util.spec_from_file_location("a2a_rerank_impl", _server_path)
a2a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a)

URL = "http://127.0.0.1:8082/rerank"


def hits(n=3):
    return [
        {"collection": "c", "id": str(i), "score": 0.9 - i * 0.1, "content": f"doc {i}"}
        for i in range(n)
    ]


class _Resp:
    def __init__(self, status=200, payload=None, boom=False):
        self.status = status
        self._payload = payload or {}
        self._boom = boom

    async def json(self):
        if self._boom:
            raise ValueError("not json")
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session:
    def __init__(self, resp=None, raise_on_post=None):
        self._resp = resp
        self._raise = raise_on_post

    def post(self, *a, **kw):
        if self._raise:
            raise self._raise
        return self._resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def with_session(sess):
    return mock.patch.object(a2a.aiohttp, "ClientSession", lambda *a, **kw: sess)


class TestRerankDisabled(unittest.TestCase):
    def test_no_url_leaves_order_untouched(self):
        h = hits()
        before = [x["id"] for x in h]
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(run(a2a._rerank("q", h)))
        self.assertEqual([x["id"] for x in h], before)

    def test_blank_url_is_treated_as_unset(self):
        with mock.patch.dict(os.environ, {"RERANK_HTTP_URL": "   "}):
            self.assertFalse(run(a2a._rerank("q", hits())))

    def test_single_hit_is_not_sent(self):
        with mock.patch.dict(os.environ, {"RERANK_HTTP_URL": URL}):
            self.assertFalse(run(a2a._rerank("q", hits(1))))

    def test_all_empty_content_is_not_sent(self):
        h = [{"collection": "c", "id": "0", "score": 0.5, "content": "  "},
             {"collection": "c", "id": "1", "score": 0.4, "content": ""}]
        with mock.patch.dict(os.environ, {"RERANK_HTTP_URL": URL}):
            self.assertFalse(run(a2a._rerank("q", h)))


class TestRerankApplies(unittest.TestCase):
    def _reranked(self, scores):
        payload = {"results": [{"index": i, "relevance_score": s} for i, s in scores]}
        h = hits(3)
        with mock.patch.dict(os.environ, {"RERANK_HTTP_URL": URL}), \
             with_session(_Session(_Resp(200, payload))):
            ok = run(a2a._rerank("q", h))
        return ok, h

    def test_reorders_by_relevance(self):
        # cosine order is 0,1,2; the cross-encoder prefers the reverse.
        ok, h = self._reranked([(0, -9.0), (1, -3.0), (2, -1.0)])
        self.assertTrue(ok)
        self.assertEqual([x["id"] for x in h], ["2", "1", "0"])

    def test_preserves_original_cosine_score(self):
        _, h = self._reranked([(0, -9.0), (1, -3.0), (2, -1.0)])
        top = h[0]
        self.assertEqual(top["cosine_score"], 0.7)     # hit id=2 started at 0.9-0.2
        self.assertEqual(top["rerank_score"], -1.0)

    def test_no_hits_are_lost(self):
        _, h = self._reranked([(0, -9.0), (1, -3.0), (2, -1.0)])
        self.assertEqual(sorted(x["id"] for x in h), ["0", "1", "2"])

    def test_already_correct_order_is_stable(self):
        ok, h = self._reranked([(0, -1.0), (1, -3.0), (2, -9.0)])
        self.assertTrue(ok)
        self.assertEqual([x["id"] for x in h], ["0", "1", "2"])


class TestRerankFailsOpen(unittest.TestCase):
    def _fails_open(self, session):
        h = hits(3)
        before = [x["id"] for x in h]
        with mock.patch.dict(os.environ, {"RERANK_HTTP_URL": URL}), with_session(session):
            ok = run(a2a._rerank("q", h))
        self.assertFalse(ok)
        self.assertEqual([x["id"] for x in h], before)

    def test_non_200_keeps_cosine_order(self):
        self._fails_open(_Session(_Resp(503, {})))

    def test_connection_error_keeps_cosine_order(self):
        self._fails_open(_Session(raise_on_post=OSError("connection refused")))

    def test_unparseable_body_keeps_cosine_order(self):
        self._fails_open(_Session(_Resp(200, boom=True)))

    def test_missing_results_key_keeps_cosine_order(self):
        self._fails_open(_Session(_Resp(200, {"nope": []})))

    def test_partial_coverage_is_rejected_rather_than_dropping_hits(self):
        # 2 scores for 3 hits: accepting this would silently lose a result.
        payload = {"results": [{"index": 0, "relevance_score": -1.0},
                               {"index": 1, "relevance_score": -2.0}]}
        self._fails_open(_Session(_Resp(200, payload)))

    def test_duplicate_indices_are_rejected(self):
        payload = {"results": [{"index": 0, "relevance_score": -1.0},
                               {"index": 0, "relevance_score": -2.0},
                               {"index": 1, "relevance_score": -3.0}]}
        self._fails_open(_Session(_Resp(200, payload)))

    def test_out_of_range_index_is_rejected(self):
        payload = {"results": [{"index": 0, "relevance_score": -1.0},
                               {"index": 1, "relevance_score": -2.0},
                               {"index": 99, "relevance_score": -3.0}]}
        self._fails_open(_Session(_Resp(200, payload)))

    def test_malformed_entries_are_rejected(self):
        payload = {"results": [{"index": 0, "relevance_score": "high"},
                               {"index": 1},
                               {"relevance_score": -3.0}]}
        self._fails_open(_Session(_Resp(200, payload)))


if __name__ == "__main__":
    unittest.main()

import json
import time
from types import SimpleNamespace
from unittest import mock

import qdrant_ops
import server


class _Point:
    id = "p1"
    score = 0.9
    payload = {"id": "p1", "text": "dense hit"}


class _QueryResult:
    points = [_Point()]


class _Vector:
    def __init__(self, size):
        self.size = size


class _Client:
    def __init__(self, dim=768):
        self.dim = dim
        self.queries = []

    def get_collection(self, name):  # noqa: ARG002
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors={"dense": _Vector(self.dim)}, sparse_vectors={})
            ),
            points_count=1,
        )

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return _QueryResult()


class _SlowSecondQueryClient(_Client):
    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        if len(self.queries) > 1:
            time.sleep(0.02)
        return _QueryResult()


class _EmbeddingResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"embedding": [0.1] * 768}]}


def test_query_embedder_resolves_backends_config_at_call_time(monkeypatch):
    """RAG must not latch a missing OLLAMA_BASE_URL from import time."""
    import backends
    import requests

    client = _Client()
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.setattr(qdrant_ops, "_OLLAMA_BASE", "")
    monkeypatch.setattr(qdrant_ops, "_EMBED_MODEL", "")
    qdrant_ops._embed_cache.clear()
    qdrant_ops._dense_name_cache.clear()
    qdrant_ops._collection_shape_cache.clear()
    monkeypatch.setattr(backends, "ollama_url", lambda *a, **k: "http://embedder:11434")
    monkeypatch.setattr(backends, "embed_model", lambda: "nomic-embed-text")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _EmbeddingResponse())
    monkeypatch.setattr(qdrant_ops, "_get_qdrant", lambda: (client, "hermes_memory"))
    monkeypatch.setattr(qdrant_ops, "_embed_sparse", lambda q: None)
    monkeypatch.setattr(qdrant_ops, "_ce_rerank", lambda q, rows, limit: (rows[:limit], False))

    rows = qdrant_ops._qdrant_search_collection("hello", "hermes_memory", limit=1)

    assert rows and rows[0]["text"] == "dense hit"
    assert client.queries and client.queries[0]["using"] == "dense"


def test_embedding_failure_reports_backend_reason(monkeypatch):
    monkeypatch.setattr(qdrant_ops, "_get_qdrant", lambda: (_Client(), "hermes_memory"))
    monkeypatch.setattr(qdrant_ops, "_embed", lambda q: None)
    monkeypatch.setattr(
        qdrant_ops,
        "_embed_last_error",
        "embedding_unavailable: http://embedder:11434/v1/embeddings model=nomic failed after 10.0s: boom",
    )

    with mock.patch.object(server, "_get_qdrant", lambda: (object(), "hermes_memory")), \
         mock.patch.object(server, "_qdrant_search_collection", qdrant_ops._qdrant_search_collection), \
         mock.patch.object(server, "_rag_cross_encode", lambda *a, **k: None), \
         mock.patch.object(server, "_rag_record_access", lambda *a, **k: None):
        out = json.loads(server.rag_context_search(
            "hello", collections=["hermes_memory"], expand_query=False
        ))

    assert out["mode"] == "rag_failed"
    assert "http://embedder:11434/v1/embeddings" in out["collection_errors"][0]
    assert "model=nomic" in out["collection_errors"][0]


def test_missing_code_chunks_collection_degrades_but_main_searches(monkeypatch):
    def _search(q, collection_name=None, **kwargs):  # noqa: ARG001
        if collection_name == "missing_code":
            raise RuntimeError("collection_unavailable: get_collection failed: not found")
        return [{"origin": collection_name, "id": "m1", "score": 0.8, "text": "main hit"}]

    with mock.patch.object(server, "_CODE_CHUNKS_COLLECTION", "missing_code"), \
         mock.patch.object(server, "QDRANT_COLLECTION_PREFIX", "hermes_memory"), \
         mock.patch.object(server, "_get_qdrant", lambda: (object(), "hermes_memory")), \
         mock.patch.object(server, "_qdrant_search_collection", _search), \
         mock.patch.object(server, "_rag_cross_encode", lambda *a, **k: None), \
         mock.patch.object(server, "_rag_record_access", lambda *a, **k: None):
        out = json.loads(server.rag_context_search("hello", expand_query=False))

    assert out["mode"] == "rag_degraded"
    assert out["collections_searched"] == ["hermes_memory", "missing_code"]
    assert out["collections_failed"] == ["missing_code"]
    assert out["result_count"] == 1


def test_rag_deadline_degrades_instead_of_crashing(monkeypatch):
    client = _SlowSecondQueryClient()
    monkeypatch.setattr(qdrant_ops, "_get_qdrant", lambda: (client, "hermes_memory"))
    monkeypatch.setattr(qdrant_ops, "_embed", lambda q: [0.1] * 768)
    monkeypatch.setattr(qdrant_ops, "_embed_sparse", lambda q: None)
    monkeypatch.setattr(qdrant_ops, "_RAG_DEADLINE_SECONDS", 0.01)
    monkeypatch.setattr(qdrant_ops, "_RAG_DEADLINE_AT", None)
    monkeypatch.setattr(qdrant_ops, "_RAG_LAST_COLLECTION", None)
    monkeypatch.setattr(qdrant_ops, "_RAG_COLLECTION_RERANK", False)
    qdrant_ops._dense_name_cache.clear()
    qdrant_ops._collection_shape_cache.clear()

    with mock.patch.object(server, "_get_qdrant", lambda: (object(), "hermes_memory")), \
         mock.patch.object(server, "_qdrant_search_collection", qdrant_ops._qdrant_search_collection), \
         mock.patch.object(server, "_rag_cross_encode", lambda *a, **k: None), \
         mock.patch.object(server, "_rag_record_access", lambda *a, **k: None):
        out = json.loads(server.rag_context_search(
            "hello", collections=["hermes_memory", "dama_gotchi_code"], expand_query=False
        ))

    assert out["mode"] == "rag_degraded"
    assert out["collections_failed"] == ["dama_gotchi_code"]
    assert "deadline_exceeded" in out["collection_errors"][0]
    assert out["result_count"] == 1


def test_collection_search_does_not_pay_duplicate_rerank_by_default(monkeypatch):
    client = _Client()
    monkeypatch.setattr(qdrant_ops, "_get_qdrant", lambda: (client, "hermes_memory"))
    monkeypatch.setattr(qdrant_ops, "_embed", lambda q: [0.1] * 768)
    monkeypatch.setattr(qdrant_ops, "_embed_sparse", lambda q: None)
    monkeypatch.setattr(qdrant_ops, "_RAG_COLLECTION_RERANK", False)
    monkeypatch.setattr(
        qdrant_ops,
        "_ce_rerank",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("duplicate rerank")),
    )
    qdrant_ops._dense_name_cache.clear()
    qdrant_ops._collection_shape_cache.clear()

    rows = qdrant_ops._qdrant_search_collection("hello", "hermes_memory", limit=1)

    assert rows and rows[0]["text"] == "dense hit"


def test_query_expansion_defaults_off_for_mcp_latency(monkeypatch):
    monkeypatch.setenv("LOCI_RAG_EXPAND", "0")
    with mock.patch.object(server, "_get_qdrant", lambda: (object(), "hermes_memory")), \
         mock.patch.object(server, "_qdrant_search_collection", lambda *a, **k: []), \
         mock.patch.object(server, "_rag_expand_queries",
                           lambda *a, **k: (_ for _ in ()).throw(AssertionError("expanded"))), \
         mock.patch.object(server, "_rag_cross_encode", lambda *a, **k: None), \
         mock.patch.object(server, "_rag_record_access", lambda *a, **k: None):
        out = json.loads(server.rag_context_search("hello"))

    assert out["mode"] == "rag_hybrid"


def test_dimension_mismatch_is_clear(monkeypatch):
    client = _Client(dim=384)
    monkeypatch.setattr(qdrant_ops, "_get_qdrant", lambda: (client, "bad_code"))
    monkeypatch.setattr(qdrant_ops, "_embed", lambda q: [0.1] * 768)
    qdrant_ops._dense_name_cache.clear()
    qdrant_ops._collection_shape_cache.clear()

    try:
        qdrant_ops._qdrant_search_collection("hello", "bad_code", limit=1)
    except RuntimeError as exc:
        msg = str(exc)
    else:
        raise AssertionError("dimension mismatch should raise")

    assert "dimension_mismatch" in msg
    assert "bad_code" in msg
    assert "dense dim(s)=[384]" in msg
    assert "embedder produced 768" in msg

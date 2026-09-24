"""Reusable success-path fakes for the MCP integration tests.

The hermetic conftest (testsupport/loci_hermetic.py) points Qdrant, Ollama and
vLLM at 127.0.0.1:1. That isolation is correct, but on its own it means every
Qdrant-, embedder- or model-gated success branch is unreachable, and a test
that accepts the degraded/error branch as "valid JSON" proves nothing.

These fakes replace *dependencies* only, never the unit under test:

``in_memory_qdrant()``
    A real ``qdrant_client.QdrantClient(":memory:")``, handed to the real
    ``qdrant_ops._get_qdrant`` (which then creates the collection, payload
    indexes, etc. exactly as in production), plus a deterministic bag-of-words
    dense embedder in place of the Ollama HTTP call. Texts that share words
    get a higher cosine; identical texts get 1.0.

``fake_rag_retrieval(rows_by_collection)``
    Scripts the per-collection search under ``rag_context_search`` and records
    each call; the ranking/decay/assembly code under test runs for real.

``fake_verdict_backend()`` / ``FakeVerdictBackend``
    Installs a recording loci_verdicts backend so a test can assert both
    "persisted exactly these verdicts" and "was not called at all" (an empty
    list) without raising into fail-open code.

``fake_conflict_judge(verdict)``
    Stand-in for the LLM contradiction judge used by conflict detection;
    records the (new, neighbour) pairs it was asked about.

``fake_lazy_generate(responses)``
    Stand-in for a ``_lazy_generate``/``gen_fn`` model call: answers from a
    list (or a callable), records each prompt, and never touches a model.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import re
from typing import Callable
from unittest import mock

_TOKEN_RE = re.compile(r"[a-z0-9]+")
FAKE_QDRANT_URL = "http://fake-qdrant.invalid:6333"


def bow_embed(text: str, dim: int) -> list[float]:
    """Deterministic, L2-normalised bag-of-words embedding of ``text``.

    Each lowercase token is hashed to one dimension, so cosine similarity is
    the token-overlap cosine: stable across runs and processes (sha1, not
    ``hash()``), and able to tell a related text from an unrelated one.
    """
    vec = [0.0] * dim
    for tok in _TOKEN_RE.findall((text or "").lower()):
        idx = int.from_bytes(hashlib.sha1(tok.encode()).digest()[:4], "big") % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        # An all-zero vector is rejected by cosine collections; use a fixed unit vector.
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


class InMemoryQdrantHandle:
    """What ``in_memory_qdrant()`` yields: the live client plus call records."""

    def __init__(self, client, collection: str, dim: int):
        self.client = client
        self.collection = collection
        self.dim = dim
        self.embedded: list[str] = []

    def embed(self, text: str, use_cache: bool = True) -> list[float]:  # noqa: ARG002
        self.embedded.append(text)
        return bow_embed(text, self.dim)

    def points(self, collection: str | None = None) -> list:
        """Every point in ``collection`` (default: the main findings collection)."""
        pts, _ = self.client.scroll(
            collection_name=collection or self.collection,
            limit=10_000, with_payload=True, with_vectors=False,
        )
        return list(pts)


@contextlib.contextmanager
def in_memory_qdrant():
    """Route every Qdrant + dense-embed call in server/qdrant_ops to in-memory fakes.

    The real ``_get_qdrant`` builds the collection on a ``QdrantClient(':memory:')``;
    only the network client constructor and the Ollama HTTP embed are replaced.
    """
    import qdrant_client
    import qdrant_ops
    import server

    real_client_cls = qdrant_client.QdrantClient
    client = real_client_cls(location=":memory:")

    def _factory(*_a, **_kw):
        return client

    handle = InMemoryQdrantHandle(client, qdrant_ops.QDRANT_COLLECTION_PREFIX, qdrant_ops.VECTOR_DIM)
    real_ready = qdrant_ops._endpoint_ready

    def _ready(url: str) -> bool:
        # Only the fake Qdrant URL is "up"; every other endpoint keeps the real probe.
        return True if url == FAKE_QDRANT_URL else real_ready(url)

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict("os.environ", {"QDRANT_URL": FAKE_QDRANT_URL}))
        stack.enter_context(mock.patch.object(qdrant_ops, "_endpoint_ready", _ready))
        # The reranker is a multi-GB model download; ranking stays the fake cosine.
        stack.enter_context(mock.patch.object(qdrant_ops, "_get_cross_encoder", lambda: None))
        stack.enter_context(mock.patch.object(server, "_get_cross_encoder", lambda: None))
        stack.enter_context(mock.patch.object(qdrant_client, "QdrantClient", _factory))
        stack.enter_context(mock.patch.object(qdrant_ops, "_qdrant_client", None))
        stack.enter_context(mock.patch.object(qdrant_ops, "_qdrant_failed_at", None))
        stack.enter_context(mock.patch.object(qdrant_ops, "_embed", handle.embed))
        stack.enter_context(mock.patch.object(server, "_embed", handle.embed))
        stack.enter_context(mock.patch.dict(qdrant_ops._embed_cache, {}, clear=True))
        got_client, col = qdrant_ops._get_qdrant()
        assert got_client is client, "in_memory_qdrant: _get_qdrant did not take the fake client"
        handle.collection = col
        # server re-exports _get_qdrant; qdrant_ops' own calls resolve the module global.
        stack.enter_context(mock.patch.object(server, "_get_qdrant", qdrant_ops._get_qdrant))
        try:
            yield handle
        finally:
            client.close()


class RagRetrievalHandle:
    """What ``fake_rag_retrieval()`` yields: the scripted rows plus call records."""

    def __init__(self, rows_by_collection: dict):
        self.rows_by_collection = rows_by_collection
        self.calls: list[dict] = []

    def search(self, query, collection_name=None, limit=10, query_filter=None, **_kw):
        self.calls.append({"query": query, "collection": collection_name,
                           "limit": limit, "filter": query_filter})
        rows = self.rows_by_collection.get(collection_name, [])
        if isinstance(rows, Exception):
            raise rows
        # Fresh dicts each call: rag_context_search rescores rows in place.
        return [dict(r) for r in rows][:limit]


@contextlib.contextmanager
def fake_rag_retrieval(rows_by_collection: dict):
    """Script the RAG retrieval layer under ``server.rag_context_search``.

    Replaces only the per-collection search (``_qdrant_search_collection``),
    reports Qdrant as reachable, and disables the cross-encoder model so the
    ranking under test is the one ``rag_context_search`` computes itself.
    A collection mapped to an Exception raises it (collection failure).
    """
    import server

    handle = RagRetrievalHandle(rows_by_collection)
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(server, "_qdrant_search_collection", handle.search))
        stack.enter_context(mock.patch.object(
            server, "_get_qdrant", lambda: (object(), server.QDRANT_COLLECTION_PREFIX)))
        stack.enter_context(mock.patch.object(server, "_get_cross_encoder", lambda: None))
        yield handle


class FakeVerdictBackend:
    """Recording stand-in for the loci_verdicts QdrantBackend (verdict_ops).

    ``record`` is async like memcheck's backend. ``recorded`` lists every
    Verdict handed to it; a test that expects no persistence asserts it is
    empty rather than making the backend raise into fail-open code.
    """

    def __init__(self):
        self.recorded: list = []

    async def record(self, verdict) -> None:
        self.recorded.append(verdict)


@contextlib.contextmanager
def fake_verdict_backend():
    """Install a FakeVerdictBackend as verdict_ops' lazily-built backend."""
    import verdict_ops

    backend = FakeVerdictBackend()
    with mock.patch.object(verdict_ops, "_get_verdict_backend", lambda: backend):
        yield backend


@contextlib.contextmanager
def fake_conflict_judge(verdict: "str | None | Callable[[dict, dict], dict]" = None):
    """Replace the LLM contradiction judge behind conflict detection.

    ``verdict`` is a fixed verdict string ("contradict", "consistent", ...), None
    (judge unavailable), or a callable ``(new, neighbor) -> dict``. The handle's
    ``calls`` records every ``(new_text, neighbor_text)`` pair judged.
    """
    import server

    class _Judge:
        calls: list = []

        def __call__(self, new_finding, neighbor, *, gen_fn=None):  # noqa: ARG002
            self.calls.append((str(new_finding.get("text", "")), str(neighbor.get("text", ""))))
            if callable(verdict):
                return verdict(new_finding, neighbor)
            return {"verdict": verdict, "reason": f"scripted:{verdict}", "ok": verdict is not None}

    judge = _Judge()
    judge.calls = []
    with mock.patch.object(server, "_judge_conflict_pair", judge):
        yield judge


def fake_lazy_generate(responses: "list[str] | Callable[[str], str]"):
    """Build a ``_lazy_generate`` replacement. ``.prompts`` records every call."""
    queue = None if callable(responses) else list(responses)

    def _gen(prompt, *args, **kwargs):  # noqa: ARG001
        _gen.prompts.append(prompt)
        if queue is None:
            return responses(prompt)
        if not queue:
            raise AssertionError("fake_lazy_generate: more calls than scripted responses")
        return queue.pop(0)

    _gen.prompts = []
    return _gen

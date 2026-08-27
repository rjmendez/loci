"""The verdict collection has ONE vector contract, and every writer honours it.

Four call sites write ``loci_verdicts`` — the memcheck PreToolUse hook, the
``verdict_ops`` claim path, and ``server.py``'s record and forget paths — and
``memory_health`` reports on it. If any of them disagrees about the dimension,
whichever creates the collection first pins it and the rest fail on every
upsert. These tests pin the agreement on the actual upserted payload, not on
the constants.

Hermetic: a fake qdrant client throughout, no network, no model load.
"""
from __future__ import annotations

import sys
import types
import unittest

import pytest
from pathlib import Path
from unittest import mock

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

from memcheck.vectors import (
    COLLECTION,
    EMBED_DIM,
    VECTOR_NAME,
    VerdictDimensionMismatch,
    ensure_collection,
    hash_embed,
)


def _vector_params(size: int):
    """Stand-in for qdrant's VectorParams (only .size is read)."""
    return types.SimpleNamespace(size=size, distance="Cosine")


class _NotFound(RuntimeError):
    """Stands in for qdrant_client's UnexpectedResponse(status_code=404)."""

    status_code = 404


def _collection_info(dim: int, named: bool = True):
    """Stand-in for qdrant's CollectionInfo at a given dense dimension."""
    vectors = {VECTOR_NAME: _vector_params(dim)} if named else _vector_params(dim)
    return types.SimpleNamespace(
        config=types.SimpleNamespace(params=types.SimpleNamespace(vectors=vectors))
    )


class FakeQdrantClient:
    """Records create_collection / upsert so tests can assert on the payload."""

    def __init__(self, existing_dim=None):
        self._existing_dim = existing_dim
        self.created: list[dict] = []
        self.upserts: list[tuple] = []

    # -- schema -------------------------------------------------------------
    def get_collection(self, collection):
        if self._existing_dim is None:
            # The real client raises UnexpectedResponse carrying status_code;
            # ensure_collection treats only a 404 as absent, so a double that
            # raises a bare error is testing a case that cannot happen.
            raise _NotFound(f"collection {collection!r} not found")
        return _collection_info(self._existing_dim)

    def create_collection(self, collection_name, vectors_config, **kw):
        size = vectors_config[VECTOR_NAME].size
        self.created.append({"collection": collection_name, "size": size})
        self._existing_dim = size

    def collection_exists(self, collection):
        return self._existing_dim is not None

    # -- points -------------------------------------------------------------
    def retrieve(self, collection_name=None, ids=None, with_payload=True, **kw):
        return []

    def upsert(self, collection_name=None, points=None, **kw):
        for p in points or []:
            self.upserts.append((collection_name, p))

    def count(self, collection, exact=False):
        return types.SimpleNamespace(count=0)


def _upserted_vector_len(client) -> int:
    """Length of the dense vector in the single recorded upsert."""
    assert client.upserts, "nothing was upserted"
    _col, point = client.upserts[-1]
    vec = point.vector
    if isinstance(vec, dict):
        vec = vec[VECTOR_NAME]
    return len(vec)


class TestEnsureCollectionOwnsCreation(unittest.TestCase):
    """ensure_collection is the sole creator and the loud mismatch boundary."""

    def test_creates_at_the_contract_dimension(self):
        client = FakeQdrantClient(existing_dim=None)
        ensure_collection(client, COLLECTION)
        self.assertEqual(
            client.created, [{"collection": COLLECTION, "size": EMBED_DIM}]
        )

    def test_existing_collection_at_right_dim_is_left_alone(self):
        client = FakeQdrantClient(existing_dim=EMBED_DIM)
        ensure_collection(client, COLLECTION)
        self.assertEqual(client.created, [])

    def test_dimension_mismatch_raises_naming_both_dims(self):
        """A wrong-dim collection fails HERE, not as a 400 on the first upsert."""
        client = FakeQdrantClient(existing_dim=768)
        with self.assertRaises(VerdictDimensionMismatch) as ctx:
            ensure_collection(client, COLLECTION)
        msg = str(ctx.exception)
        self.assertIn("768", msg)
        self.assertIn(str(EMBED_DIM), msg)
        self.assertIn(COLLECTION, msg)

    def test_mismatch_is_logged_at_error(self):
        client = FakeQdrantClient(existing_dim=768)
        with self.assertLogs("memcheck", level="ERROR") as logs:
            with self.assertRaises(VerdictDimensionMismatch):
                ensure_collection(client, COLLECTION)
        self.assertTrue(any("768" in line for line in logs.output))


class TestEveryWriterAgreesOnDimension(unittest.TestCase):
    """Each production writer must upsert an EMBED_DIM vector."""

    def test_hash_embed_is_the_contract_dimension(self):
        self.assertEqual(len(hash_embed("anything")), EMBED_DIM)

    def test_memcheck_hook_backend(self):
        import memcheck.cli as cli

        client = FakeQdrantClient(existing_dim=None)
        with mock.patch.dict("os.environ", {"QDRANT_URL": "http://fake:6333"}), \
             mock.patch("qdrant_client.QdrantClient", lambda **kw: client):
            backend = cli._build_qdrant_backend()
        self.assertEqual(backend._collection, COLLECTION)
        self.assertEqual(len(backend._embed("x")), EMBED_DIM)
        self.assertEqual(client.created[0]["size"], EMBED_DIM)

    def test_verdict_ops_backend(self):
        """verdict_ops used the server's 768-dim _embed against a 384-dim collection."""
        import verdict_ops

        client = FakeQdrantClient(existing_dim=EMBED_DIM)
        with mock.patch.object(verdict_ops, "_verdict_backend", None), \
             mock.patch.object(verdict_ops, "_verdict_backend_failed", False), \
             mock.patch.dict("os.environ", {"QDRANT_URL": "http://fake:6333"}), \
             mock.patch("qdrant_client.QdrantClient", lambda **kw: client):
            backend = verdict_ops._get_verdict_backend()
        self.assertIsNotNone(backend)
        self.assertEqual(backend._collection, COLLECTION)
        vec = backend._embed("x")
        self.assertIsNotNone(
            vec, "backend wired an embedder that yielded no vector for this collection"
        )
        self.assertEqual(len(vec), EMBED_DIM)

    def test_server_record_verdicts_upserts_contract_dimension(self):
        """Asserts on the upserted PAYLOAD, not on which embedder was wired."""
        import server
        from memcheck.verdict import new_verdict

        client = FakeQdrantClient(existing_dim=EMBED_DIM)
        v = new_verdict(
            subject_kind="memory",
            subject_signature="sig-contract-test",
            subject_excerpt="a claim under test",
            verdict_type="claim_supported",
            decision="allow",
            confidence=0.9,
            rationale="test",
            source="rule",
        )
        with mock.patch.object(server, "_get_qdrant", lambda: (client, "loci_memory")):
            ok = server._record_verdicts([v])
        self.assertTrue(ok)
        self.assertEqual(_upserted_vector_len(client), EMBED_DIM)

    def test_hook_end_to_end_upserts_contract_dimension(self):
        """The whole PreToolUse path, asserted on the upserted vector."""
        import memcheck.cli as cli
        from memcheck.engine import EmlConfig, VerdictEngine

        client = FakeQdrantClient(existing_dim=None)
        with mock.patch.dict("os.environ", {"QDRANT_URL": "http://fake:6333"}), \
             mock.patch("qdrant_client.QdrantClient", lambda **kw: client), \
             mock.patch.object(cli, "_append_audit_line", lambda *a, **k: None):
            engine = VerdictEngine(cli._build_qdrant_backend(), EmlConfig())
            cli.process_action(
                {"tool_name": "Bash", "tool_input": {"command": "ls -la"}}, engine
            )
        self.assertEqual(_upserted_vector_len(client), EMBED_DIM)

    def test_server_forget_path_does_not_create_the_collection(self):
        import server

        client = FakeQdrantClient(existing_dim=None)
        with mock.patch.object(server, "_get_qdrant", lambda: (client, "loci_memory")):
            server._forget_finding_verdicts({"text": "some finding"})
        self.assertEqual(client.created, [])


class TestHealthProbeUsesTheVerdictContract(unittest.TestCase):
    """memory_health must not judge a 384-dim collection by the 768-dim embedder."""

    def _client_with(self, verdicts_dim):
        info = {COLLECTION: verdicts_dim, "loci_memory": 768}

        class C:
            def get_collections(self):
                names = [types.SimpleNamespace(name=n) for n in info]
                return types.SimpleNamespace(collections=names)

            def get_collection(self, c):
                return _collection_info(info[c])

            def count(self, c, exact=False):
                return types.SimpleNamespace(count=0)

        return C()

    def test_correct_verdicts_dim_is_ok_and_not_compared_to_embedder(self):
        import server

        dims: dict = {}
        status, report, _hint = server._health_probe_qdrant_collections(
            self._client_with(EMBED_DIM), "loci_memory", dims
        )
        self.assertEqual(status, "ok")
        # The 384-dim verdict collection must never reach probe 6's embedder check.
        self.assertNotIn(COLLECTION, dims)
        status6, _d, _h = server._health_probe_dimension_consistency(768, dims)
        self.assertEqual(status6, "ok")

    def test_wrong_verdicts_dim_fails_loudly(self):
        import server

        dims: dict = {}
        status, _report, hint = server._health_probe_qdrant_collections(
            self._client_with(768), "loci_memory", dims
        )
        self.assertEqual(status, "fail")
        self.assertIn("768", hint)
        self.assertIn(str(EMBED_DIM), hint)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# ensure_collection: absence, reachability, and the vector NAME
#
# An earlier draft treated every get_collection failure as "absent" and followed
# it with a create, so an unreachable qdrant cost two timeouts on a PreToolUse
# hook budgeted around one second. It also validated the dimension but not the
# vector name, so an unnamed or differently-named vector passed.
# ---------------------------------------------------------------------------

class _Err(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _Client:
    def __init__(self, on_get):
        self._on_get = on_get
        self.calls = []

    def get_collection(self, name):
        self.calls.append("get")
        if isinstance(self._on_get, Exception):
            raise self._on_get
        return self._on_get

    def create_collection(self, **kw):
        self.calls.append("create")


def _info(vectors):
    params = type("P", (), {"vectors": vectors})()
    config = type("C", (), {"params": params})()
    return type("I", (), {"config": config})()


def _params(size):
    from qdrant_client import models as q
    return q.VectorParams(size=size, distance=q.Distance.COSINE)


def test_an_unreachable_qdrant_is_not_treated_as_absent():
    from memcheck import vectors as V
    c = _Client(_Err(500))
    with pytest.raises(Exception) as exc:
        V.ensure_collection(c)
    assert not isinstance(exc.value, V.VerdictDimensionMismatch)
    assert c.calls == ["get"], "must not follow an unreachable get with a create"


def test_a_404_still_creates_the_collection():
    from memcheck import vectors as V
    c = _Client(_Err(404))
    V.ensure_collection(c)
    assert c.calls == ["get", "create"]


def test_a_correct_collection_is_left_alone():
    from memcheck import vectors as V
    c = _Client(_info({V.VECTOR_NAME: _params(V.EMBED_DIM)}))
    V.ensure_collection(c)
    assert c.calls == ["get"]


@pytest.mark.parametrize("vectors, label", [
    (None, "sparse-only"),
    ({"text": None}, "a differently-named vector"),
])
def test_a_collection_without_the_named_dense_vector_is_refused(vectors, label):
    from memcheck import vectors as V
    if vectors == {"text": None}:
        vectors = {"text": _params(V.EMBED_DIM)}
    with pytest.raises(V.VerdictDimensionMismatch, match="no named"):
        V.ensure_collection(_Client(_info(vectors)))


def test_an_unnamed_vector_is_refused():
    from memcheck import vectors as V
    with pytest.raises(V.VerdictDimensionMismatch, match="no named"):
        V.ensure_collection(_Client(_info(_params(V.EMBED_DIM))))


def test_the_wrong_dimension_is_refused_naming_both():
    from memcheck import vectors as V
    with pytest.raises(V.VerdictDimensionMismatch) as exc:
        V.ensure_collection(_Client(_info({V.VECTOR_NAME: _params(768)})))
    assert "768" in str(exc.value) and str(V.EMBED_DIM) in str(exc.value)

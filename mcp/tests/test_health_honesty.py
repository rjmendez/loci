"""Health and fallback paths must report failures instead of reading clean.

Each test is a regression for one audit finding (ids in the test docstrings).
Everything runs against in-memory Qdrant, a throwaway local HTTP stub, or
monkeypatched tool functions; nothing touches a live store.
"""
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

qdrant_client = pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import (  # noqa: E402
    Distance, PointStruct, SparseVector, SparseVectorParams, VectorParams,
)

import backends  # noqa: E402
import qdrant_ops  # noqa: E402
import server  # noqa: E402

DIM = 8


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _stub_embedder(dim=768):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            self.rfile.read(n)
            body = json.dumps({"data": [{"embedding": [0.1] * dim}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


@pytest.fixture
def fresh_transport(monkeypatch):
    """Isolate the embed cache, readiness cache and breakers for one test."""
    monkeypatch.setattr(qdrant_ops, "_embed_cache", {})
    monkeypatch.setattr(qdrant_ops, "_endpoint_ready_cache", {})
    monkeypatch.setattr(qdrant_ops, "_transport_breakers", {})


def _patch_client(monkeypatch, client, col="loci_memory"):
    """Point both the mutating and the read-only accessors at ``client``."""
    monkeypatch.setattr(server, "_get_qdrant", lambda: (client, col))
    monkeypatch.setattr(server, "_qdrant_client_readonly", lambda: (client, col), raising=False)
    monkeypatch.setattr(server, "QDRANT_COLLECTION_PREFIX", col)
    monkeypatch.setattr(server, "_CODE_CHUNKS_COLLECTION", "")


# --------------------------------------------------------------------------- #
# embed cache masking an outage
# --------------------------------------------------------------------------- #
def test_memory_health_dense_probe_goes_live_after_embedder_dies(monkeypatch, fresh_transport):
    """embed-health-from-cache / health-embed-cache-masks-outage."""
    srv, url = _stub_embedder()
    monkeypatch.setattr(qdrant_ops, "_OLLAMA_BASE", url)
    try:
        first = server._health_probe_embeddings_dense({})
        assert first[0] == "ok", first
    finally:
        srv.shutdown()
        srv.server_close()
    qdrant_ops._endpoint_ready_cache.clear()
    second = server._health_probe_embeddings_dense({})
    assert second[0] == "fail", f"dense probe reported {second!r} from the embed cache"
    assert "memory_health probe" not in qdrant_ops._embed_cache, "a probe must not fill the cache"


def test_retrieval_selftest_default_query_is_not_served_from_cache(monkeypatch, fresh_transport):
    """health-embed-cache-masks-outage: the default query 'system architecture' was cached."""
    c = QdrantClient(location=":memory:")
    c.create_collection("loci_memory",
                        vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)})
    c.upsert("loci_memory", points=[PointStruct(id=1, vector={"dense": [0.1] * DIM})])
    _patch_client(monkeypatch, c)
    qdrant_ops._embed_cache["system architecture"] = [0.1] * DIM
    monkeypatch.setattr(qdrant_ops, "_OLLAMA_BASE", "http://127.0.0.1:1")  # embedder is down
    out = json.loads(server.retrieval_selftest())
    assert out["status"] != "ok", out
    assert out["embedder_dim"] is None


def test_selftest_remediation_names_a_down_embedder_not_a_missing_url(monkeypatch):
    monkeypatch.setattr(qdrant_ops, "_OLLAMA_BASE", "http://127.0.0.1:1")
    c = QdrantClient(location=":memory:")
    c.create_collection("x", vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)})
    c.upsert("x", points=[PointStruct(id=1, vector={"dense": [0.1] * DIM})])
    row = qdrant_ops.probe_collection(None, c, "x")
    assert row["status"] == "error"
    assert "configured but returned no vector" in row["remediation"]


# --------------------------------------------------------------------------- #
# health checks that mutated Qdrant / trusted a cached client
# --------------------------------------------------------------------------- #
def _shared_fake_qdrant(monkeypatch):
    shared = QdrantClient(location=":memory:")

    class Fake:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, n):
            return getattr(shared, n)

    monkeypatch.setattr(qdrant_client, "QdrantClient", Fake)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.invalid:6333")
    monkeypatch.setattr(qdrant_ops, "_qdrant_client", None)
    monkeypatch.setattr(qdrant_ops, "_qdrant_failed_at", None)
    return shared


def test_memory_health_qdrant_probes_never_create_the_main_collection(monkeypatch):
    """health-checks-mutate-and-mask-missing-collection (memory_health half)."""
    shared = _shared_fake_qdrant(monkeypatch)
    sink = {"client": None, "main_col": None, "embed_dim": None}
    s1 = server._health_check("qdrant_reachable",
                              lambda: server._health_probe_qdrant_reachable("http://qdrant.invalid:6333", sink))
    s2 = server._health_check("qdrant_collections",
                              lambda: server._health_probe_qdrant_collections(sink["client"], sink["main_col"], {}))
    assert [c.name for c in shared.get_collections().collections] == [], "health created a collection"
    assert s1["status"] == "ok"
    assert s2["status"] == "fail" and s2["detail"]["main_present"] is False


def test_memory_health_cached_client_is_probed_live(monkeypatch):
    """tcp-only-health (memory_health half): a cached client said 'connected' with no request."""
    class Dead:
        def get_collections(self):
            raise ConnectionError("qdrant stopped answering")

    monkeypatch.setenv("QDRANT_URL", "http://qdrant.invalid:6333")
    monkeypatch.setattr(qdrant_ops, "_qdrant_client", (Dead(), "loci_memory"))
    sink = {"client": None, "main_col": None, "embed_dim": None}
    s1 = server._health_check("qdrant_reachable",
                              lambda: server._health_probe_qdrant_reachable("http://qdrant.invalid:6333", sink))
    assert s1["status"] == "fail", s1
    assert sink["client"] is None


def test_retrieval_selftest_reports_missing_main_collection(monkeypatch):
    """health-checks-mutate-and-mask-missing-collection (retrieval_selftest half)."""
    c = QdrantClient(location=":memory:")
    c.create_collection("some_other", vectors_config={})
    _patch_client(monkeypatch, c)
    monkeypatch.setattr(server, "_embed", lambda _q: [0.1] * DIM)
    monkeypatch.setattr(server, "_embed_uncached", lambda _q: [0.1] * DIM, raising=False)
    out = json.loads(server.retrieval_selftest())
    assert out["status"] == "unhealthy", out
    rows = {r["collection"]: r for r in out["collections"]}
    assert rows["loci_memory"]["status"] == "missing"
    assert [c.name for c in c.get_collections().collections] == ["some_other"]


# --------------------------------------------------------------------------- #
# qdrant_ops search-path correctness
# --------------------------------------------------------------------------- #
def test_transient_get_collection_error_is_not_cached(monkeypatch):
    """dense-name-cache-poisoned."""
    real = QdrantClient(location=":memory:")
    real.create_collection("named", vectors_config={"dense": VectorParams(size=4, distance=Distance.COSINE)})
    real.upsert("named", points=[PointStruct(id=1, vector={"dense": [1, 0, 0, 0]}, payload={})])

    class Flaky:
        calls = 0

        def __getattr__(self, n):
            return getattr(real, n)

        def get_collection(self, name):
            Flaky.calls += 1
            if Flaky.calls == 2:
                raise TimeoutError("transient read timeout")
            return real.get_collection(name)

    monkeypatch.setattr(qdrant_ops, "VECTOR_DIM", 4)
    monkeypatch.setattr(qdrant_ops, "_dense_name_cache", {})
    c = Flaky()
    qdrant_ops.probe_collection([1, 0, 0, 0], c, "named")  # hits the transient timeout
    second = qdrant_ops.probe_collection([1, 0, 0, 0], c, "named")
    assert second["status"] == "ok", second
    assert qdrant_ops._dense_name_cache.get("named") == "dense"


def test_search_collection_uses_the_sparse_index_and_keeps_cosine_scores(monkeypatch):
    """hybrid-never-runs-sparse-check-wrong-field."""
    from qdrant_client.models import SparseIndexParams

    real = QdrantClient(location=":memory:")
    real.create_collection(
        "hyb",
        vectors_config={"dense": VectorParams(size=4, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())},
    )
    real.upsert("hyb", points=[
        PointStruct(id=1, vector={"dense": [1, 0, 0, 0],
                                  "sparse": SparseVector(indices=[7], values=[1.0])},
                    payload={"text": "cve-2024-0001"}),
        PointStruct(id=2, vector={"dense": [0, 1, 0, 0],
                                  "sparse": SparseVector(indices=[9], values=[1.0])},
                    payload={"text": "unrelated"}),
    ])
    calls = []

    class Spy:
        def __getattr__(self, n):
            return getattr(real, n)

        def query_points(self, **kw):
            calls.append(kw)
            return real.query_points(**kw)

    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setattr(qdrant_ops, "_get_qdrant", lambda: (Spy(), "hyb"))
    monkeypatch.setattr(qdrant_ops, "_dense_name_cache", {})
    monkeypatch.setattr(qdrant_ops, "_embed", lambda *a, **k: [1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(qdrant_ops, "_embed_sparse",
                        lambda t: SparseVector(indices=[7], values=[1.0]))
    monkeypatch.setattr(qdrant_ops, "_get_cross_encoder", lambda: None)

    rows = qdrant_ops._qdrant_search_collection("cve-2024-0001", "hyb", limit=2)
    assert calls, "no query was issued"
    usings = [getattr(p, "using", None) for p in (calls[-1].get("prefetch") or [])]
    assert "sparse" in usings, f"sparse index present but hybrid branch not taken: {calls[-1]}"
    assert rows[0]["text"] == "cve-2024-0001"
    # Score stays on the cosine scale callers threshold against (memory_surface 0.25).
    assert rows[0]["score"] > 0.9


# --------------------------------------------------------------------------- #
# memory_surface fabricated score
# --------------------------------------------------------------------------- #
def _docs(n):
    return json.dumps({"results": [{"title": f"doc{i}", "summary": f"docs guidance {i}"}
                                   for i in range(n)]})


def test_memory_surface_docs_fallback_is_flagged_and_unscored(monkeypatch):
    """memory-surface-fabricated-score (Qdrant down)."""
    monkeypatch.setattr(server, "_get_qdrant", lambda: (None, None))
    monkeypatch.setattr(server, "docs_search", lambda *a, **k: _docs(2))
    out = json.loads(server.memory_surface("auth token expiry", top_k=5))
    assert out["count"] == 2
    assert out.get("degraded") is True and out.get("fallback") == "docs_search"
    for hit in out["surfaced"]:
        assert hit["score"] != 0.95, "fabricated similarity score"
        assert hit["origin"] == "docs_search"


def test_memory_surface_docs_hits_do_not_displace_real_findings(monkeypatch):
    """memory-surface-fabricated-score (Qdrant up)."""
    monkeypatch.setattr(server, "_get_qdrant", lambda: (object(), "loci_memory"))
    monkeypatch.setattr(server, "_qdrant_search_collection", lambda *a, **k: [
        {"id": f"f-{i}", "investigation_id": "inv", "text": f"finding {i}", "score": 0.6}
        for i in range(2)])
    monkeypatch.setattr(server, "docs_search", lambda *a, **k: _docs(3))
    out = json.loads(server.memory_surface("auth token expiry", top_k=2))
    ids = [h["finding_id"] for h in out["surfaced"]]
    assert ids == ["f-0", "f-1"], f"docs guidance displaced real findings: {ids}"


# --------------------------------------------------------------------------- #
# memory_self_check swallowed failures
# --------------------------------------------------------------------------- #
def test_self_check_crashed_contradiction_check_is_degraded(monkeypatch, tmp_path):
    """self-check-swallowed-failure-reads-clean."""
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path))
    inv = tmp_path / "inv-hc"
    inv.mkdir()
    rows = [
        {"id": "f-a", "record_type": "observed",
         "text": "The UWB anchor on node3 firmware build is calibrated and ranging offsets are stable"},
        {"id": "f-b", "record_type": "observed",
         "text": "The UWB anchor on node3 firmware build is not calibrated and ranging offsets are stable"},
    ]
    (inv / "findings.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(server, "_get_qdrant", lambda: (None, None))

    clean = json.loads(server.memory_self_check("inv-hc", record=False))
    assert clean.get("degraded") is False and clean.get("degraded_checks") == []

    def boom(*a, **k):
        raise RuntimeError("contradiction engine crashed")

    monkeypatch.setattr(server, "run_contradiction", boom)
    out = json.loads(server.memory_self_check("inv-hc", record=False))
    assert out["counts"]["contradiction"] == 0
    assert out.get("degraded") is True, "a crashed check must not read as a clean store"
    joined = " ".join(out["degraded_checks"])
    assert "contradiction" in joined and "crashed" in joined
    assert "hallucination_candidates" in joined  # computed without one of its inputs


def test_self_check_reports_llm_verify_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path))
    (tmp_path / "inv-llm").mkdir()
    (tmp_path / "inv-llm" / "findings.jsonl").write_text(
        json.dumps({"id": "f", "record_type": "observed", "text": "x is y"}) + "\n")
    monkeypatch.setattr(server, "_get_qdrant", lambda: (None, None))
    import memcheck.checks.contradiction_llm as cl

    def boom(*a, **k):
        raise RuntimeError("llm unreachable")

    monkeypatch.setattr(cl, "verify_and_merge", boom)
    out = json.loads(server.memory_self_check("inv-llm", record=False, llm_verify=True))
    assert out.get("llm_verify_applied") is False
    assert out["degraded"] is True


# --------------------------------------------------------------------------- #
# loci_health: HTTP answers, generation endpoint
# --------------------------------------------------------------------------- #
def _health_env(monkeypatch, http):
    monkeypatch.setattr(backends, "_alive", lambda url, timeout=1.0: True)
    monkeypatch.setattr(backends, "_http_probe", http, raising=False)
    monkeypatch.setattr(backends, "ollama_url", lambda *a, **k: "http://embed-host:11434")
    monkeypatch.setattr(backends, "ollama_gen_url", lambda *a, **k: "http://gen-host:11434")
    monkeypatch.setattr(backends, "vllm_url", lambda *a, **k: "http://vllm-host:8000")
    monkeypatch.setattr(backends, "qdrant", lambda: ("http://qdrant-host:6333", ""))
    monkeypatch.delenv("LOCI_TMUX_COMPANION_REQUIRED", raising=False)


def test_loci_health_tcp_accept_without_http_answer_is_not_reachable(monkeypatch):
    """tcp-only-health: a Qdrant that accepts TCP but never answers HTTP."""
    monkeypatch.setenv("QDRANT_URL", "http://qdrant-host:6333")
    _health_env(monkeypatch, lambda url, path="", timeout=1.0, headers=None:
                (("qdrant-host" not in url), {"models": []}))
    out = json.loads(server.loci_health())
    assert out["qdrant_reachable"] is False
    assert out["status"] == "unhealthy"
    assert any(f.startswith("qdrant") for f in out["failures"])


def test_loci_health_probes_the_generation_endpoint(monkeypatch):
    """loci-health-ignores-generation-endpoint."""
    monkeypatch.setenv("LOCI_OLLAMA_GEN_URL", "http://gen-host:11434")
    _health_env(monkeypatch, lambda url, path="", timeout=1.0, headers=None:
                (("gen-host" not in url), {"models": []}))
    out = json.loads(server.loci_health())
    assert out["ollama_reachable"] is True
    assert out["ollama_gen_reachable"] is False
    assert out["status"] == "unhealthy"
    assert any(f.startswith("ollama_gen") for f in out["failures"])


def test_loci_health_flags_a_gen_endpoint_without_the_gen_model(monkeypatch):
    monkeypatch.setenv("LOCI_OLLAMA_GEN_URL", "http://gen-host:11434")
    monkeypatch.setenv("LOCI_OLLAMA_GEN_MODEL", "qwen3.8:latest")
    _health_env(monkeypatch, lambda url, path="", timeout=1.0, headers=None:
                (True, {"models": [{"name": "nomic-embed-text:latest"}]}))
    out = json.loads(server.loci_health())
    assert out["ollama_gen_reachable"] is True
    assert out["ollama_gen_model_present"] is False
    assert out["status"] == "unhealthy"


def test_http_probe_never_raises_and_reads_json():
    srv, url = _stub_embedder(dim=2)
    srv.shutdown()
    srv.server_close()
    assert backends._http_probe(url, "/api/tags", timeout=0.5) == (False, None)
    assert backends._http_probe("", "/x") == (False, None)


# --------------------------------------------------------------------------- #
# ladybug: stale writer pid, failed Cypher read as empty
# --------------------------------------------------------------------------- #
def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_stale_writer_pid_is_not_reported(tmp_path):
    """stale-ladybug-writer-pid."""
    pytest.importorskip("ladybug")
    from graph.ladybug_store import LadybugStore

    s = LadybugStore(str(tmp_path / "graphdb"))
    with open(s._lease_path, "w") as f:
        f.write(f"{_dead_pid()} 1790120072")
    assert s.lock_holder_pid() is None
    with open(s._lease_path, "w") as f:
        f.write(f"{os.getpid()} 1790120072")
    assert s.lock_holder_pid() == os.getpid()


def test_code_graph_query_failure_is_an_error_not_zero_rows(tmp_path, monkeypatch):
    """code-graph-query-error-as-empty."""
    pytest.importorskip("ladybug")
    import graph_tools
    from graph.ladybug_store import LadybugStore

    store = LadybugStore(str(tmp_path / "cq"))
    assert store.available()
    assert store.upsert_investigation("inv", "creates the db and schema")
    monkeypatch.setattr(graph_tools, "_get_ladybug", lambda *a, **k: store)
    ok = json.loads(graph_tools.code_graph_query("MATCH (s:CodeSymbol) RETURN s.id"))
    assert ok == {"row_count": 0, "rows": []}
    bad = json.loads(graph_tools.code_graph_query("MATCH (f:NoSuchTable) RETURN f"))
    assert "error" in bad and "row_count" not in bad, bad

    # A query that never ran because no read session opened (lease timeout) is also an error.
    monkeypatch.setattr(store, "_session", _no_session)
    missed = json.loads(graph_tools.code_graph_query("MATCH (s:CodeSymbol) RETURN s.id"))
    assert "error" in missed, missed


@contextmanager
def _no_session(*a, **k):
    yield None

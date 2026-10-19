"""The four fail-open latches retry transient failures instead of disabling a feature until restart.

Sites: reranker._get_model, qdrant_ops._get_sparse_embedder, verdict_ops._get_verdict_backend,
backends._ollama_list. All use retry_latch.RetryLatch on an injected clock; the heavy dependencies
(sentence-transformers, fastembed, Qdrant, the ollama binary) are fakes, nothing is downloaded.
"""
import collections
import json
import logging
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backends as B  # noqa: E402
import qdrant_ops as Q  # noqa: E402
import reranker as R  # noqa: E402
import server  # noqa: E402
import verdict_ops as V  # noqa: E402
from memcheck.vectors import COLLECTION, EMBED_DIM, VECTOR_NAME  # noqa: E402
from retry_latch import RetryLatch  # noqa: E402


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# =========================================================================== reranker

class _Encoders:
    """Fake sentence_transformers.CrossEncoder: each construction pops the next behaviour
    (an exception to raise, or None to succeed) and is recorded."""

    def __init__(self, behaviours, delay=0.0):
        self.behaviours = list(behaviours)
        self.created = []
        self.delay = delay
        self._lock = threading.Lock()

    def make_class(self):
        outer = self

        class CrossEncoder:
            def __init__(self, name, max_length, device):
                with outer._lock:
                    outer.created.append((name, device))
                    behaviour = outer.behaviours.pop(0) if outer.behaviours else None
                if outer.delay:
                    time.sleep(outer.delay)
                if behaviour is not None:
                    raise behaviour
        return CrossEncoder


@pytest.fixture
def rr(monkeypatch):
    clock = Clock()
    torch = types.ModuleType("torch")
    torch.__version__ = "2.11.0+cu128"
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(R, "_model", None)
    monkeypatch.setattr(R, "_model_name", None)
    monkeypatch.setattr(R, "_device", None)
    monkeypatch.setattr(R, "_torch_build", None)
    monkeypatch.setattr(R, "_latch", RetryLatch("reranker", base_s=30.0, cap_s=600.0, clock=clock,
                                                wall=lambda: 1_700_000_000.0))
    monkeypatch.delenv("RERANK_HTTP_URL", raising=False)
    monkeypatch.setenv("RERANK_MODEL", "some/reranker")

    def install(behaviours, delay=0.0):
        enc = _Encoders(behaviours, delay)
        st = types.ModuleType("sentence_transformers")
        st.CrossEncoder = enc.make_class()
        monkeypatch.setitem(sys.modules, "sentence_transformers", st)
        return enc
    return types.SimpleNamespace(clock=clock, install=install)


def test_a_transient_load_failure_retries_at_the_backoff_and_recovers(rr, caplog):
    enc = rr.install([OSError("not enough memory"), None])
    with caplog.at_level(logging.WARNING, logger="loci-mcp"):
        assert R._get_model() is None
    assert enc.created == [("some/reranker", "cuda")]
    assert _warnings(caplog) == ["reranker failed (attempt 1), retrying in 30s: OSError: not enough memory"]
    lat = R.health()["latch"]
    assert (lat["ok"], lat["permanent"], lat["attempts"], lat["last_error"], lat["next_retry_in_s"]) == (
        False, False, 1, "OSError: not enough memory", 30.0)
    assert R.health()["state"] == "unavailable"

    rr.clock.t += 29.9                       # refused inside the window: no second load
    assert R._get_model() is None
    assert len(enc.created) == 1

    rr.clock.t += 0.1                        # exactly at the backoff: the retry happens and succeeds
    model = R._get_model()
    assert model is not None and len(enc.created) == 2
    h = R.health()
    assert h["state"] == "loaded" and h["device"] == "cuda"
    assert (h["latch"]["ok"], h["latch"]["last_error"], h["latch"]["attempts"]) == (True, "", 0)
    assert R._get_model() is model           # and it is cached again
    assert len(enc.created) == 2


def test_the_backoff_grows_exponentially_across_repeated_failures(rr):
    enc = rr.install([OSError("a"), OSError("b"), OSError("c")])
    waits = []
    for _ in range(3):
        assert R._get_model() is None
        waits.append(R.health()["latch"]["next_retry_in_s"])
        rr.clock.t += waits[-1]
    assert waits == [30.0, 60.0, 120.0]
    assert len(enc.created) == 3
    assert R._get_model() is not None        # the fourth load succeeds


def test_an_import_error_stays_permanent_and_is_logged_at_warning(rr, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)   # `import` raises ImportError
    with caplog.at_level(logging.DEBUG, logger="loci-mcp"):
        assert R._get_model() is None
    msgs = [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "loci-mcp"]
    assert len(msgs) == 1 and msgs[0][0] == logging.WARNING
    assert msgs[0][1] == ("reranker unavailable permanently (attempt 1): ModuleNotFoundError: "
                          "import of sentence_transformers halted; None in sys.modules")
    rr.clock.t += 1e9
    st = types.ModuleType("sentence_transformers")
    st.CrossEncoder = _Encoders([]).make_class()
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)      # even once it is installed
    assert R._get_model() is None
    lat = R.health()["latch"]
    assert (lat["permanent"], lat["next_retry_in_s"], lat["attempts"]) == (True, None, 1)


def test_concurrent_callers_trigger_exactly_one_load(rr):
    enc = rr.install([None], delay=0.05)
    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(R._get_model())
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(enc.created) == 1
    assert len({id(m) for m in results}) == 1 and results[0] is not None


def test_concurrent_callers_after_a_failure_do_not_each_retry(rr):
    enc = rr.install([OSError("boom")], delay=0.05)
    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(R._get_model())
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [None] * 8
    assert len(enc.created) == 1
    assert R.health()["latch"]["attempts"] == 1


def test_reconfiguring_the_model_retries_at_once(rr, monkeypatch):
    enc = rr.install([OSError("boom"), None])
    assert R._get_model() is None
    monkeypatch.setenv("RERANK_MODEL", "other/reranker")
    assert R._get_model() is not None                       # no wait: the failure was for the other model
    assert [n for n, _ in enc.created] == ["some/reranker", "other/reranker"]


def test_a_healthy_dependency_loads_once_and_reports_ok(rr):
    enc = rr.install([None])
    m1 = R._get_model()
    m2 = R._get_model()
    assert m1 is m2 and m1 is not None
    assert enc.created == [("some/reranker", "cuda")]
    h = R.health()
    assert (h["state"], h["device"], h["latch"]["ok"], h["latch"]["failures_total"]) == ("loaded", "cuda", True, 0)


def test_the_http_backend_is_unaffected(rr, monkeypatch):
    monkeypatch.setenv("RERANK_HTTP_URL", "http://127.0.0.1:1/rerank")
    assert isinstance(R._get_model(), R._HttpCrossEncoder)
    assert R.health()["device"] == "http"


# =========================================================================== sparse embedder

class _FakeSparseResult:
    def __init__(self, idx, vals):
        self.indices = types.SimpleNamespace(tolist=lambda: list(idx))
        self.values = types.SimpleNamespace(tolist=lambda: list(vals))


@pytest.fixture
def sp(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(Q, "_sparse_model", None)
    monkeypatch.setattr(Q, "_sparse_latch", RetryLatch("sparse_embedder", base_s=60.0, cap_s=600.0, clock=clock,
                                                       wall=lambda: 1_700_000_000.0))
    monkeypatch.setattr(Q, "_embed_sparse_cache", {})
    monkeypatch.setitem(Q._health_events, "sparse", collections.deque(maxlen=200))

    def install(behaviours, embed=None):
        created = []
        pending = list(behaviours)

        class SparseTextEmbedding:
            def __init__(self, name, **kw):
                created.append(name)
                b = pending.pop(0) if pending else None
                if b is not None:
                    raise b

            def embed(self, texts):
                if isinstance(embed, BaseException):
                    raise embed
                return [_FakeSparseResult([1, 5], [0.5, 0.25])]
        mod = types.ModuleType("fastembed")
        mod.SparseTextEmbedding = SparseTextEmbedding
        monkeypatch.setitem(sys.modules, "fastembed", mod)
        return created
    return types.SimpleNamespace(clock=clock, install=install)


def test_a_transient_sparse_load_failure_retries_at_exactly_60s(sp, caplog):
    created = sp.install([OSError("huggingface.co unreachable"), None])
    with caplog.at_level(logging.WARNING, logger="loci-mcp"):
        assert Q._get_sparse_embedder() is None
    assert _warnings(caplog) == [
        "sparse_embedder failed (attempt 1), retrying in 60s: OSError: huggingface.co unreachable"]
    h = Q.sparse_health()
    assert (h["ok"], h["loaded"], h["attempts"], h["next_retry_in_s"]) == (False, False, 1, 60.0)
    sp.clock.t += 59.9
    assert Q._get_sparse_embedder() is None
    assert created == ["Qdrant/bm25"]
    sp.clock.t += 0.1
    model = Q._get_sparse_embedder()
    assert model is not None and len(created) == 2
    h = Q.sparse_health()
    assert (h["ok"], h["loaded"], h["last_error"]) == (True, True, "")
    assert Q._get_sparse_embedder() is model and len(created) == 2


def test_a_missing_fastembed_is_permanent(sp, monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    assert Q._get_sparse_embedder() is None
    sp.clock.t += 1e9
    created = sp.install([])
    assert Q._get_sparse_embedder() is None and created == []
    h = Q.sparse_health()
    assert (h["permanent"], h["next_retry_in_s"]) == (True, None)


def test_concurrent_callers_load_the_sparse_model_once(sp):
    created = sp.install([None])
    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(Q._get_sparse_embedder())
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(created) == 1
    assert len({id(m) for m in results}) == 1 and results[0] is not None


def test_embedding_without_a_sparse_model_is_counted_and_warned_once(sp, caplog):
    sp.install([OSError("offline")])
    with caplog.at_level(logging.WARNING, logger="loci-mcp"):
        assert Q._embed_sparse("hello") is None
        assert Q._embed_sparse("hello") is None
    msgs = [m for m in _warnings(caplog) if m.startswith("sparse embedder unavailable")]
    assert msgs == ["sparse embedder unavailable: hybrid search is dense-only and points are written "
                    "without a sparse vector (OSError: offline)"]       # the second is rate-limited
    t = Q.transport_health()["sparse"]
    assert (t["ok"], t["failed"], t["consecutive_failures"]) == (0, 2, 2)


def test_a_working_sparse_embedder_is_counted_ok_and_returns_the_vector(sp, caplog):
    sp.install([])
    with caplog.at_level(logging.WARNING, logger="loci-mcp"):
        vec = Q._embed_sparse("hello")
    assert (list(vec.indices), list(vec.values)) == ([1, 5], [0.5, 0.25])
    t = Q.transport_health()["sparse"]
    assert (t["ok"], t["failed"], t["consecutive_failures"]) == (1, 0, 0)
    assert _warnings(caplog) == []


def test_a_per_call_sparse_embed_failure_is_a_warning_and_counted(sp, caplog):
    sp.install([], embed=RuntimeError("onnx session broke"))
    with caplog.at_level(logging.DEBUG, logger="loci-mcp"):
        assert Q._embed_sparse("hello") is None
    msgs = [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "loci-mcp"]
    assert msgs == [(logging.WARNING, "sparse embed failed (RuntimeError: onnx session broke); this call is dense-only")]
    assert Q.transport_health()["sparse"]["failed"] == 1
    assert Q.sparse_health()["loaded"] is True       # the model itself stays loaded: only this call failed


class _RecordingClient:
    def __init__(self):
        self.upserts = []

    def upsert(self, col, points):
        self.upserts.append((col, points))


def _upsert(monkeypatch, sparse_vec, payload):
    client = _RecordingClient()
    monkeypatch.setattr(Q, "_get_qdrant", lambda: (client, "col"))
    monkeypatch.setattr(Q, "_embed", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(Q, "_embed_sparse", lambda text: sparse_vec)
    monkeypatch.delenv("HERMES_AGENT_ID", raising=False)
    monkeypatch.delenv("LOCI_NAMESPACE", raising=False)
    assert Q._qdrant_upsert("p1", "text", payload) is True
    (col, points), = client.upserts
    assert col == "col"
    return points[0]


def test_a_point_written_without_a_sparse_vector_is_marked_sparse_false(sp, monkeypatch):
    point = _upsert(monkeypatch, None, {"a": 1})
    assert point.payload == {"a": 1, "sparse": False}
    assert set(point.vector) == {"dense"}


def test_a_point_written_with_a_sparse_vector_is_unmarked(sp, monkeypatch):
    from qdrant_client.models import SparseVector
    point = _upsert(monkeypatch, SparseVector(indices=[1], values=[0.5]), {"a": 1})
    assert point.payload == {"a": 1}
    assert set(point.vector) == {"dense", "sparse"}


# =========================================================================== verdict backend

class _Info:
    def __init__(self, dim):
        self.config = types.SimpleNamespace(params=types.SimpleNamespace(
            vectors={VECTOR_NAME: types.SimpleNamespace(size=dim, distance="Cosine")}))


class _VerdictClient:
    def __init__(self, fail_first=0, dim=EMBED_DIM):
        self.fail_first = fail_first
        self.dim = dim
        self.calls = 0

    def get_collection(self, collection):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ConnectionError("qdrant timed out")
        return _Info(self.dim)


@pytest.fixture
def vb(monkeypatch):
    import qdrant_client
    clock = Clock()
    monkeypatch.setattr(V, "_verdict_backend", None)
    monkeypatch.setattr(V, "_verdict_backend_failed", False)
    monkeypatch.setattr(V, "_verdict_latch", RetryLatch("verdict_backend", base_s=30.0, cap_s=600.0, clock=clock,
                                                        wall=lambda: 1_700_000_000.0, log=V.logger))
    monkeypatch.setenv("QDRANT_URL", "http://fake:6333")
    constructed = []

    def install(client):
        def factory(**kw):
            constructed.append(kw)
            return client
        monkeypatch.setattr(qdrant_client, "QdrantClient", factory)
    return types.SimpleNamespace(clock=clock, install=install, constructed=constructed)


def test_a_transient_qdrant_error_retries_at_30s_and_recovers(vb, caplog):
    client = _VerdictClient(fail_first=1)
    vb.install(client)
    with caplog.at_level(logging.WARNING, logger="loci-mcp"):
        assert V._get_verdict_backend() is None
    assert _warnings(caplog) == [
        "verdict_backend failed (attempt 1), retrying in 30s: ConnectionError: qdrant timed out"]
    assert V._verdict_backend_failed is False
    assert V._record_claim_verdicts("inv", [], record=True) == {"recorded": 0, "qdrant": "unavailable"}
    assert client.calls == 1                              # refused during the backoff without touching Qdrant
    vb.clock.t += 29.9
    assert V._get_verdict_backend() is None and client.calls == 1
    vb.clock.t += 0.1
    backend = V._get_verdict_backend()
    assert backend is not None and backend._collection == COLLECTION
    h = V.verdict_backend_health()
    assert (h["ok"], h["loaded"], h["last_error"], h["attempts"]) == (True, True, "", 0)
    assert V._get_verdict_backend() is backend and len(vb.constructed) == 2


def test_a_dimension_mismatch_is_permanent(vb, caplog):
    client = _VerdictClient(dim=768)
    vb.install(client)
    with caplog.at_level(logging.WARNING, logger="loci-mcp"):
        assert V._get_verdict_backend() is None
    assert any(m.startswith("verdict_backend unavailable permanently (attempt 1): VerdictDimensionMismatch: ")
               for m in _warnings(caplog))
    vb.clock.t += 1e9
    assert V._get_verdict_backend() is None
    assert len(vb.constructed) == 1                       # never retried
    assert V._verdict_backend_failed is True
    h = V.verdict_backend_health()
    assert (h["permanent"], h["next_retry_in_s"], h["loaded"]) == (True, None, False)


def test_an_import_error_is_permanent_for_the_verdict_backend(vb, monkeypatch):
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    assert V._get_verdict_backend() is None
    assert V.verdict_backend_health()["permanent"] is True


def test_a_healthy_qdrant_builds_the_backend_first_time(vb):
    vb.install(_VerdictClient())
    backend = V._get_verdict_backend()
    assert backend is not None
    h = V.verdict_backend_health()
    assert (h["ok"], h["attempts"], h["failures_total"]) == (True, 0, 0)
    assert vb.constructed == [{"url": "http://fake:6333", "api_key": None, "timeout": 5}]


def test_no_qdrant_url_is_not_a_failure(vb, monkeypatch):
    monkeypatch.delenv("QDRANT_URL")
    assert V._get_verdict_backend() is None
    assert V.verdict_backend_health()["ok"] is True


# =========================================================================== ollama list

_LIST_OUT = ("NAME        ID            SIZE      MODIFIED\n"
             "qwen2.5:3b  357c53fb659c  1.9 GB    3 weeks ago\n")


@pytest.fixture
def ol(monkeypatch):
    clock = Clock(0.0)
    monkeypatch.setattr(B, "_ollama_list_clock", clock)
    monkeypatch.setattr(B, "_ollama_list_latch", RetryLatch("ollama_list", base_s=5.0, cap_s=60.0, clock=clock,
                                                            permanent_types=(), wall=lambda: 1_700_000_000.0))
    monkeypatch.setattr(B, "_ollama_list_cache", None)
    monkeypatch.setattr(B, "_OLLAMA_LIST_TTL_S", 300.0)
    runs = []
    script = []
    state = types.SimpleNamespace(delay=0.0)

    def fake_run(cmd, **kw):
        runs.append((tuple(cmd), kw.get("timeout")))
        if state.delay:
            time.sleep(state.delay)
        step = script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step
    monkeypatch.setattr(subprocess, "run", fake_run)
    return types.SimpleNamespace(clock=clock, runs=runs, script=script, state=state)


def _ok(out=_LIST_OUT):
    return subprocess.CompletedProcess(["ollama", "list"], 0, out, "")


def test_a_failed_listing_is_not_cached_and_recovers(ol, caplog):
    ol.script += [FileNotFoundError("ollama"), _ok()]
    with caplog.at_level(logging.WARNING, logger="loci-mcp"):
        assert B._ollama_list() == {}
    assert _warnings(caplog) == ["ollama_list failed (attempt 1), retrying in 5s: FileNotFoundError: ollama"]
    ol.clock.t = 4.9
    assert B._ollama_list() == {}
    assert len(ol.runs) == 1                              # inside the 5 s window: no new subprocess
    ol.clock.t = 5.0
    assert B._ollama_list() == {"qwen2.5:3b": 1_900_000_000}
    assert len(ol.runs) == 2
    h = B.ollama_list_health()
    assert (h["ok"], h["last_error"], h["attempts"]) == (True, "", 0)


def test_a_nonzero_exit_is_a_failure_not_an_empty_inventory(ol):
    ol.script += [subprocess.CompletedProcess(["ollama", "list"], 1, "", "could not connect"), _ok()]
    assert B._ollama_list() == {}
    assert B.ollama_list_health()["last_error"] == "RuntimeError: ollama list exited 1: could not connect"
    ol.clock.t = 5.0
    assert B._ollama_list() == {"qwen2.5:3b": 1_900_000_000}


def test_a_timeout_is_a_failure_too(ol):
    ol.script += [subprocess.TimeoutExpired(["ollama", "list"], 2.0), _ok()]
    assert B._ollama_list() == {}
    assert B.ollama_list_health()["last_error"].startswith("TimeoutExpired: ")
    ol.clock.t = 5.0
    assert len(B._ollama_list()) == 1


def test_a_successful_listing_is_cached_for_the_ttl_only(ol):
    ol.script += [_ok(), _ok("NAME  ID  SIZE  MODIFIED\nllama3:8b  abc  4.7 GB  1 day ago\n")]
    assert B._ollama_list() == {"qwen2.5:3b": 1_900_000_000}
    ol.clock.t = 299.9
    assert B._ollama_list() == {"qwen2.5:3b": 1_900_000_000}
    assert len(ol.runs) == 1
    ol.clock.t = 300.0
    assert B._ollama_list() == {"llama3:8b": 4_700_000_000}   # a newly pulled model is seen without a restart
    assert len(ol.runs) == 2


def test_the_returned_inventory_is_a_copy(ol):
    ol.script += [_ok()]
    first = B._ollama_list()
    first.clear()                                         # mutating the fresh result ...
    second = B._ollama_list()                             # ... does not poison the cache
    assert second == {"qwen2.5:3b": 1_900_000_000}
    second.clear()                                        # nor does mutating a cache hit
    assert B._ollama_list() == {"qwen2.5:3b": 1_900_000_000}
    assert len(ol.runs) == 1


def test_concurrent_callers_spawn_one_listing(ol):
    ol.script += [_ok()]
    ol.state.delay = 0.05
    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(B._ollama_list())
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ol.runs) == 1
    assert results == [{"qwen2.5:3b": 1_900_000_000}] * 8


def test_local_tags_follow_the_recovered_inventory(ol):
    ol.script += [OSError("busy"), _ok()]
    assert B._ollama_local_tags() == set()
    ol.clock.t = 5.0
    assert B._ollama_local_tags() == {"qwen2.5:3b"}


def test_reset_cache_forgets_the_listing_and_the_failure(ol):
    ol.script += [_ok(), OSError("x"), _ok("NAME  ID  SIZE  MODIFIED\nb:1b  abc  1 GB  now\n")]
    assert B._ollama_list() == {"qwen2.5:3b": 1_900_000_000}
    B._reset_cache()
    assert B._ollama_list() == {}                         # the second run fails
    B._reset_cache()                                      # clears the backoff as well
    assert B._ollama_list() == {"b:1b": 1_000_000_000}


def test_gen_model_is_not_stuck_on_the_fallback_after_one_failed_listing(ol, monkeypatch):
    monkeypatch.delenv("LOCI_OLLAMA_GEN_MODEL", raising=False)
    monkeypatch.setattr(B, "_pool_pick", lambda role: "")
    ol.script += [OSError("busy"), _ok("NAME  ID  SIZE  MODIFIED\nllama3:8b  abc  4.7 GB  1 day ago\n")]
    assert B.ollama_gen_model() == B.ONE_GPU_FALLBACK_MODEL   # degraded while the listing fails ...
    ol.clock.t = 5.0
    assert B.ollama_gen_model() == "llama3:8b"                # ... and recovers without a restart


# =========================================================================== loci_health

def test_loci_health_shows_a_failing_latch_with_its_error_and_countdown(rr, sp, ol, vb):
    rr.install([OSError("not enough memory")])
    assert R._get_model() is None
    out = json.loads(server.loci_health())
    assert out["latches"]["reranker"]["last_error"] == "OSError: not enough memory"
    assert out["latches"]["reranker"]["next_retry_in_s"] == 30.0
    assert out["latches"]["sparse_embedder"]["ok"] is True
    assert "reranker failing (1 attempt(s) since 2023-11-14T22:13:20+00:00): OSError: not enough memory; " \
           "next retry in 30.0s" in out["warnings"]


def test_loci_health_is_quiet_when_every_latch_is_healthy(rr, sp, ol, vb):
    out = json.loads(server.loci_health())
    assert sorted(out["latches"]) == ["ollama_list", "reranker", "sparse_embedder", "verdict_backend"]
    assert all(v["ok"] for v in out["latches"].values())
    assert not any("failing" in w or "unavailable permanently" in w for w in out.get("warnings", []))


# ---------------------------------------------------------------- the production configuration itself

def test_the_production_latches_carry_the_documented_schedules():
    # every fixture above swaps in a latch with a fake clock and the same numbers, so the schedules actually shipped at each
    # site need their own assertion (a mutation of any of them passed the whole file)
    cfg = lambda l: (l.base_s, l.cap_s, l.factor, l.permanent_types)           # noqa: E731
    assert cfg(R._latch) == (30.0, 600.0, 2.0, (ImportError,))
    assert cfg(Q._sparse_latch) == (60.0, 600.0, 2.0, (ImportError,))
    assert cfg(V._verdict_latch) == (30.0, 600.0, 2.0, (ImportError,))         # a dimension mismatch is made permanent explicitly
    assert cfg(B._ollama_list_latch) == (5.0, 60.0, 2.0, ())                   # an inventory outage is never permanent
    assert [l.name for l in (R._latch, Q._sparse_latch, V._verdict_latch, B._ollama_list_latch)] == [
        "reranker", "sparse_embedder", "verdict_backend", "ollama_list"]

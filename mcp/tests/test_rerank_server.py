"""scripts/rerank_server.py: the llama.cpp /rerank protocol that mcp/reranker.py's _HttpCrossEncoder speaks.

The server is real (a ThreadingHTTPServer on an ephemeral loopback port) and the client under test is Loci's own
_HttpCrossEncoder, so a drift between the two ends of the protocol turns a test red. Only the model is faked: a
recording scorer that scores a document by its length, which makes every expected number exact.
"""
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MCP_DIR = _ROOT / "mcp"
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import reranker as R  # noqa: E402

_spec = importlib.util.spec_from_file_location("rerank_server", _ROOT / "scripts" / "rerank_server.py")
rerank_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rerank_server)


class LengthScorer:
    """Scores a (query, doc) pair as len(doc) and records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, pairs):
        self.calls.append([list(p) for p in pairs])
        return [float(len(p[1])) for p in pairs]


@pytest.fixture
def served():
    scorer = LengthScorer()
    srv = rerank_server.make_server(scorer, "127.0.0.1", 0, model_name="fake-model", device="fake-dev")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base, scorer
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(url, payload):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.load(resp)


def _post_error(url, payload):
    with pytest.raises(urllib.error.HTTPError) as info:
        _post(url, payload)
    return info.value.code, json.load(info.value)


def test_loci_http_client_gets_one_exact_score_per_document(served):
    base, scorer = served
    scores = R._HttpCrossEncoder(base + "/rerank").predict([["q", "a"], ["q", "abcd"], ["q", "ab"]])
    assert scores == [1.0, 4.0, 2.0]
    assert scorer.calls == [[["q", "a"], ["q", "abcd"], ["q", "ab"]]]


def test_response_names_the_model_and_indexes_each_result(served):
    base, _ = served
    status, body = _post(base + "/rerank", {"query": "q", "documents": ["xx", "y"]})
    assert status == 200
    assert body == {"model": "fake-model", "results": [
        {"index": 0, "relevance_score": 2.0}, {"index": 1, "relevance_score": 1.0}]}


def test_empty_document_list_answers_empty_without_scoring(served):
    base, scorer = served
    status, body = _post(base + "/rerank", {"query": "q", "documents": []})
    assert (status, body) == (200, {"model": "fake-model", "results": []})
    assert scorer.calls == []


@pytest.mark.parametrize("payload,message", [
    (b"not json", "body is not JSON"),
    (["a"], "body must be a JSON object"),
    ({"documents": ["a"]}, "query must be a string"),
    ({"query": "q", "documents": "a"}, "documents must be a list of strings"),
    ({"query": "q", "documents": ["a", 3]}, "documents must be a list of strings"),
])
def test_malformed_request_is_a_400_naming_the_fault_and_never_scores(served, payload, message):
    base, scorer = served
    code, body = _post_error(base + "/rerank", payload)
    assert code == 400
    assert body["error"].startswith(message)
    assert scorer.calls == []


def test_the_same_server_scores_a_well_formed_request_after_a_bad_one(served):
    base, scorer = served
    _post_error(base + "/rerank", {"query": 1, "documents": ["a"]})
    status, body = _post(base + "/rerank", {"query": "q", "documents": ["abc"]})
    assert (status, body["results"]) == (200, [{"index": 0, "relevance_score": 3.0}])
    assert scorer.calls == [[["q", "abc"]]]


def test_a_scorer_failure_is_a_500_with_the_error_not_a_dropped_connection():
    def broken(pairs):
        raise RuntimeError("CUDA out of memory")

    srv = rerank_server.make_server(broken, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        code, body = _post_error(f"http://127.0.0.1:{srv.server_address[1]}/rerank", {"query": "q", "documents": ["a"]})
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)
    assert (code, body) == (500, {"error": "RuntimeError: CUDA out of memory"})


@pytest.mark.parametrize("preload,expected_loads", [(True, [("m1", "d1")]), (False, [])])
def test_preload_loads_the_model_at_construction_and_not_otherwise(monkeypatch, preload, expected_loads):
    loads = []

    def fake_load(model_name, device):
        loads.append((model_name, device))
        return LengthScorer()

    monkeypatch.setattr(rerank_server, "load_predict", fake_load)
    srv = rerank_server.make_server(None, "127.0.0.1", 0, model_name="m1", device="d1", preload=preload)
    srv.server_close()
    assert loads == expected_loads


def test_health_reports_model_and_device_and_unknown_routes_404(served):
    base, _ = served
    with urllib.request.urlopen(base + "/health", timeout=5) as resp:
        assert json.load(resp) == {"status": "ok", "model": "fake-model", "device": "fake-dev"}
    with pytest.raises(urllib.error.HTTPError, match="404") as info:
        urllib.request.urlopen(base + "/nope", timeout=5)
    assert json.load(info.value) == {"error": "no route /nope"}

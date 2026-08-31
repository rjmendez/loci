"""The shadow-eval gate must score arms at the passage budget production uses.

_make_retrieve truncated each candidate to a hard-coded 512 chars before handing
it to reranker.rerank(), which truncates nothing of its own. Production feeds the
cross-encoder qdrant_ops.RERANK_MAX_CHARS (1024). The repo measured 512 as worse
than using no reranker at all on a 1,048-char median finding, so every arm was
ranked in a regime the running pipeline never sees — and judge_eval.py imports
this same function, which is what flipped the production RERANK_MODEL default.
"""
import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "shadow_eval.py"
sys.path.insert(0, str(REPO / "mcp"))

import qdrant_ops  # noqa: E402


@pytest.fixture
def se():
    spec = importlib.util.spec_from_file_location("shadow_eval", SCRIPT)
    assert spec and spec.loader, f"cannot load {SCRIPT}"
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Point:
    def __init__(self, pid, text):
        self.id = pid
        self.payload = {"text": text}


class _Result:
    def __init__(self, points):
        self.points = points


class _Client:
    def __init__(self, points):
        self._points = points

    def query_points(self, **kw):
        return _Result(self._points)


@pytest.fixture
def stub_server(monkeypatch, se):
    """Replace the Qdrant/Ollama halves of mcp/server.py, keep _make_retrieve real."""
    import server as S
    text = "x" * 5000
    monkeypatch.setattr(S, "_get_qdrant", lambda: (_Client([_Point("p1", text)]), "col"))
    monkeypatch.setattr(S, "_embed", lambda q: [0.1])
    return text


def test_retrieved_passages_are_cut_at_the_production_rerank_budget(se, stub_server):
    cands = se._make_retrieve(5)("some query", exclude_id="other")
    assert len(cands) == 1
    assert len(cands[0]["text"]) == qdrant_ops.RERANK_MAX_CHARS
    assert qdrant_ops.RERANK_MAX_CHARS == 1024, "the swept default moved; re-check the gate"


def test_the_budget_is_imported_not_restated(se, stub_server, monkeypatch):
    """Overriding the module constant must move the harness with it — a second
    literal is exactly how the two copies drifted apart."""
    monkeypatch.setattr(qdrant_ops, "RERANK_MAX_CHARS", 777)
    cands = se._make_retrieve(5)("some query", exclude_id="other")
    assert len(cands[0]["text"]) == 777

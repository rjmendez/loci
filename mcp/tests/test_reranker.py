"""Tests for reranker — the pluggable GPU reranking unit.

Scores are stubbed via an injected model_fn; NO model is downloaded and no GPU/network
is touched (mirrors test_embed_ops.py style + the [pattern:injectable] contract).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import reranker as R  # noqa: E402


# A stub scorer: score = length of the doc string. Deterministic, no model.
def _by_length(query, docs):
    return [float(len(d)) for d in docs]


def test_orders_by_score_desc():
    docs = ["aa", "aaaa", "a", "aaa"]
    r = R.rerank("q", docs, model_fn=_by_length)
    # Highest length first.
    assert [x["text"] for x in r] == ["aaaa", "aaa", "aa", "a"]
    assert [x["index"] for x in r] == [1, 3, 0, 2]  # index into the INPUT list
    assert r[0]["score"] == 4.0
    assert all("degraded" not in x for x in r)


def test_top_k_caps_results():
    docs = ["a", "aaaa", "aa", "aaa"]
    r = R.rerank("q", docs, top_k=2, model_fn=_by_length)
    assert len(r) == 2
    assert [x["text"] for x in r] == ["aaaa", "aaa"]


def test_ties_keep_original_order():
    docs = ["xx", "yy", "zz"]  # all length 2
    r = R.rerank("q", docs, model_fn=_by_length)
    assert [x["index"] for x in r] == [0, 1, 2]  # stable sort


def test_fail_open_passthrough_when_unavailable(monkeypatch):
    # No model_fn injected + force the real model to be unavailable -> original order, score=None.
    monkeypatch.setattr(R, "_real_model_fn", lambda q, d: None)
    docs = ["b", "cccc", "aa"]
    r = R.rerank("q", docs)  # model_fn=None
    assert [x["text"] for x in r] == ["b", "cccc", "aa"]  # unchanged
    assert all(x["score"] is None for x in r)
    assert all(x["degraded"] is True for x in r)


def test_fail_open_respects_top_k():
    r = R.rerank("q", ["a", "b", "c"], top_k=2, model_fn=lambda q, d: None)
    assert len(r) == 2
    assert all(x["score"] is None for x in r)
    assert [x["text"] for x in r] == ["a", "b"]


def test_fail_open_on_model_fn_raise():
    def _boom(q, d):
        raise RuntimeError("model exploded")
    r = R.rerank("q", ["a", "bb"], model_fn=_boom)
    assert [x["text"] for x in r] == ["a", "bb"]
    assert all(x["score"] is None and x["degraded"] for x in r)


def test_fail_open_on_wrong_length_scores():
    # A scorer that returns the wrong number of scores is treated as unavailable.
    r = R.rerank("q", ["a", "bb", "ccc"], model_fn=lambda q, d: [1.0])
    assert [x["text"] for x in r] == ["a", "bb", "ccc"]
    assert all(x["score"] is None for x in r)


def test_empty_docs():
    assert R.rerank("q", [], model_fn=_by_length) == []
    assert R.rerank("q", []) == []


def test_model_id_reads_env(monkeypatch, tmp_path):
    # Isolate from any real gitignored ~/.loci/backends.toml so we test the CODE default.
    monkeypatch.delenv("RERANK_MODEL", raising=False)
    monkeypatch.setenv("LOCI_CONFIG", str(tmp_path / "no-such.toml"))
    import backends as B
    B._reset_cache()
    # Default flipped to bge on judge-eval evidence (+14% nDCG@10).
    assert R._model_id() == "BAAI/bge-reranker-v2-m3"
    monkeypatch.setenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert R._model_id() == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    B._reset_cache()

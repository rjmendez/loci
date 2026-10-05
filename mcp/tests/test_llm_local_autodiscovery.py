"""Auto-discovery must never substitute a model that does not fit one GPU."""
from __future__ import annotations

import logging
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import llm_local as L  # noqa: E402

GB = 10**9


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


def _setup(monkeypatch, tags, ps=(), installed=()):
    """Mock Ollama: generate 404s for tags not in `installed`, else 200."""
    import backends
    monkeypatch.setattr(backends, "ollama_gen_model", lambda: "cfg:missing")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test")
    monkeypatch.delenv("LOCI_OLLAMA_AUTO_MAX_GB", raising=False)
    monkeypatch.setattr(L, "_gen_env", lambda: "http://ollama.test")
    monkeypatch.setattr(L, "_supervisor_route", lambda *a, **k: None)
    monkeypatch.setattr(L, "_try_cloud_tier", lambda *a, **k: None)
    posts = []

    def get(url, timeout=None):
        if url.endswith("/api/tags"):
            return _Resp({"models": [{"name": n, **({} if s is None else {"size": s})} for n, s in tags]})
        if url.endswith("/api/ps"):
            return _Resp({"models": [{"name": n[0], "size": n[1]} if isinstance(n, tuple) else {"name": n}
                                     for n in ps]})
        raise AssertionError(url)

    def post(url, json=None, timeout=None):  # noqa: A002
        posts.append(json["model"])
        if json["model"] not in installed:
            raise RuntimeError("404 model not found")
        return _Resp({"response": "ok"})

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=get, post=post))
    return posts


def test_small_first_tag_substitutes_with_warning(monkeypatch, caplog):
    posts = _setup(monkeypatch, [("small:3b", 2 * GB), ("big:27b", 16 * GB)], installed={"small:3b"})
    with caplog.at_level(logging.WARNING, logger=L._LOG.name):
        r = L.generate("hi")
    assert r["ok"] and r["model"] == "small:3b"
    assert posts == ["cfg:missing", "small:3b"]
    msg = " ".join(rec.getMessage() for rec in caplog.records if "substitution" in rec.getMessage())
    assert "cfg:missing" in msg and "small:3b" in msg and "GiB" in msg


def test_large_first_tag_skipped(monkeypatch):
    posts = _setup(monkeypatch, [("big:27b", 16 * GB), ("small:3b", 2 * GB)], installed={"small:3b", "big:27b"})
    r = L.generate("hi")
    assert r["ok"] and r["model"] == "small:3b"
    assert "big:27b" not in posts


def test_only_large_tags_fail_open_without_loading(monkeypatch):
    posts = _setup(monkeypatch, [("big:27b", 16 * GB), ("bigger:70b", 40 * GB), ("nomic-embed:1", GB // 10)],
                   installed={"big:27b", "bigger:70b"})
    r = L.generate("hi")
    assert r["ok"] is False
    assert posts == ["cfg:missing"]
    assert "cfg:missing" in r["why"] and "no eligible fallback" in r["why"]


def test_resident_tag_preferred(monkeypatch):
    posts = _setup(monkeypatch, [("a:3b", GB), ("b:3b", GB), ("c:7b", 4 * GB)],
                   ps=["b:3b"], installed={"a:3b", "b:3b", "c:7b"})
    r = L.generate("hi")
    assert r["ok"] and r["model"] == "b:3b"
    assert posts == ["cfg:missing", "b:3b"]


def test_cap_is_configurable(monkeypatch):
    posts = _setup(monkeypatch, [("big:27b", 16 * GB)], installed={"big:27b"})
    monkeypatch.setenv("LOCI_OLLAMA_AUTO_MAX_GB", "20")
    r = L.generate("hi")
    assert r["ok"] and r["model"] == "big:27b"


def test_explicit_model_unchanged(monkeypatch):
    posts = _setup(monkeypatch, [("small:3b", GB)], installed={"mine:70b"})
    r = L.generate("hi", model="mine:70b")
    assert r["ok"] and r["model"] == "mine:70b"
    assert posts == ["mine:70b"]


def test_present_configured_model_unchanged(monkeypatch):
    posts = _setup(monkeypatch, [("small:3b", GB)], installed={"cfg:missing"})
    r = L.generate("hi")
    assert r["ok"] and r["model"] == "cfg:missing"
    assert posts == ["cfg:missing"]


def test_sizeless_resident_small_is_eligible(monkeypatch):
    posts = _setup(monkeypatch, [("mystery:7b", None)], ps=[("mystery:7b", 4 * GB)], installed={"mystery:7b"})
    r = L.generate("hi")
    assert r["ok"] and r["model"] == "mystery:7b"
    assert posts == ["cfg:missing", "mystery:7b"]


def test_sizeless_non_resident_is_skipped(monkeypatch):
    posts = _setup(monkeypatch, [("mystery:7b", None)], installed={"mystery:7b"})
    r = L.generate("hi")
    assert r["ok"] is False
    assert posts == ["cfg:missing"]
    assert "no eligible fallback" in r["why"]


def test_sizeless_resident_above_cap_is_skipped(monkeypatch):
    posts = _setup(monkeypatch, [("mystery:70b", None), ("small:3b", 2 * GB)],
                   ps=[("mystery:70b", 40 * GB)], installed={"mystery:70b", "small:3b"})
    r = L.generate("hi")
    assert r["ok"] and r["model"] == "small:3b"
    assert "mystery:70b" not in posts

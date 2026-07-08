"""Tests for query_expand — RAG query expansion. Generation is stubbed; no live Ollama."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import query_expand as Q  # noqa: E402


# --- stub gen_fn factories: match the shared contract gen_fn(prompt, *, fmt, max_tokens) ---

def _ok(text):
    def _fn(prompt, *, fmt=None, max_tokens=256):
        assert fmt == "json"           # expand() must request JSON format
        assert isinstance(prompt, str) and "Query:" in prompt
        return {"text": text, "ok": True}
    return _fn


def _not_ok(prompt, *, fmt=None, max_tokens=256):
    return {"text": "irrelevant", "ok": False}   # caller should fall back


def _raises(prompt, *, fmt=None, max_tokens=256):
    raise RuntimeError("boom")


_CANNED = '{"queries": ["how to reset the base station", "restart RTK base"], ' \
          '"keywords": ["RTK", "base station", "reset", "RTK"]}'

_PROSE = (
    "Sure! Here is the expansion you asked for:\n"
    "```json\n"
    '{"queries": ["alt phrasing one", "alt phrasing two"], '
    '"keywords": ["alpha", "beta"]}\n'
    "```\n"
    "Hope that helps."
)

_PROSE_NOFENCE = (
    'Here you go: {"queries": ["only alt"], "keywords": ["gamma"]} -- thanks!'
)


def test_happy_path_parses_dedups_and_caps():
    r = Q.expand("reset base", gen_fn=_ok(_CANNED), n_queries=3, n_keywords=6)
    assert r["degraded"] is False
    # original query leads, alternates follow
    assert r["queries"][0] == "reset base"
    assert "how to reset the base station" in r["queries"]
    assert len(r["queries"]) == 3
    # keywords de-duped (RTK appeared twice) and preserved
    assert r["keywords"].count("RTK") == 1
    assert "base station" in r["keywords"]


def test_caps_are_respected():
    r = Q.expand("q", gen_fn=_ok(_CANNED), n_queries=2, n_keywords=1)
    assert len(r["queries"]) == 2      # original + 1 alt
    assert len(r["keywords"]) == 1


def test_embedded_in_prose_with_fences():
    r = Q.expand("orig", gen_fn=_ok(_PROSE), n_queries=3, n_keywords=6)
    assert r["degraded"] is False
    assert r["queries"][0] == "orig"
    assert "alt phrasing one" in r["queries"]
    assert r["keywords"] == ["alpha", "beta"]


def test_embedded_in_prose_no_fence_brace_scan():
    r = Q.expand("orig", gen_fn=_ok(_PROSE_NOFENCE))
    assert r["degraded"] is False
    assert "only alt" in r["queries"]
    assert r["keywords"] == ["gamma"]


def test_not_ok_fails_open_to_original_query():
    r = Q.expand("my query", gen_fn=_not_ok)
    assert r == {"queries": ["my query"], "keywords": [], "degraded": True}


def test_gen_fn_exception_fails_open():
    r = Q.expand("my query", gen_fn=_raises)
    assert r["degraded"] is True and r["queries"] == ["my query"]


def test_garbage_output_fails_open():
    r = Q.expand("my query", gen_fn=_ok("this is not json at all, no braces here"))
    assert r["degraded"] is True and r["queries"] == ["my query"]


def test_malformed_json_object_fails_open():
    r = Q.expand("my query", gen_fn=_ok('{"queries": [oops not valid'))
    assert r["degraded"] is True and r["queries"] == ["my query"]


def test_valid_json_but_empty_lists_is_degraded():
    r = Q.expand("my query", gen_fn=_ok('{"queries": [], "keywords": []}'))
    # nothing added beyond the seed and no keywords -> degraded, but still runnable
    assert r["degraded"] is True and r["queries"] == ["my query"]


def test_empty_query_short_circuits():
    r = Q.expand("   ", gen_fn=_ok(_CANNED))
    assert r == {"queries": [], "keywords": [], "degraded": True}


def test_non_string_list_items_are_coerced_and_filtered():
    r = Q.expand("orig", gen_fn=_ok('{"queries": ["a", "", "  "], "keywords": [1, 2, "x"]}'))
    assert r["queries"] == ["orig", "a"]        # blanks dropped, original leads
    assert r["keywords"] == ["1", "2", "x"]     # coerced to strings


def test_default_gen_fn_is_lazy_and_fails_open(monkeypatch):
    # With no llm_local importable, the lazy default must fail-open, not raise.
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "llm_local":
            raise ImportError("llm_local not written yet")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    r = Q.expand("my query")   # no gen_fn -> lazy import path
    assert r["degraded"] is True and r["queries"] == ["my query"]

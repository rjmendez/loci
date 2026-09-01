"""Characterization tests for scripts/hooks/pre_llm_grounding.py.

This hook runs before *every* LLM call in a Claude Code session, so its
contract is mostly about what it emits on stdout and when it silently
gets out of the way. These tests pin the behaviour AS IT IS TODAY --
including several things that are arguably wrong (see the module-level
BUG comments) -- so that a refactor can be checked against them.

No network, no Qdrant, no Ollama: every outbound call goes through
urllib.request.urlopen, which is patched, or through a module-level
function that is patched.

The module does a lot of work at import time (reads ~/.hermes/.env,
derives ~20 constants from the environment, side-loads
scripts/spreading_activation.py). So tests load a *fresh* copy of the
module under a fully controlled environment via the `hook` fixture, and
tests that care about constant derivation call load_hook() themselves.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import pathlib
import builtins
import sys
import urllib.error
import time
import types
from unittest import mock

import pytest

HOOK_PATH = pathlib.Path(__file__).resolve().parent.parent / "pre_llm_grounding.py"

# HOME points at a nonexistent directory so .env loading and ~/.hermes defaults stay inert.
BASE_ENV = {
    "HOME": "/nonexistent-home-for-pre-llm-grounding-tests",
    "HERMES_HOME": "/nonexistent-home-for-pre-llm-grounding-tests/.hermes",
    "QDRANT_URL": "http://qdrant.invalid:6333",
    "QDRANT_API_KEY": "test-key",
    "OLLAMA_BASE_URL": "http://ollama.invalid:11434",
}


def load_hook(env: dict | None = None):
    """Exec a fresh copy of the hook under a controlled environment."""
    e = dict(BASE_ENV)
    e.update(env or {})
    with mock.patch.dict(os.environ, e, clear=True):
        spec = importlib.util.spec_from_file_location("_pre_llm_grounding_uut", HOOK_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def hook():
    """Fresh module per test -- tests mutate module globals freely."""
    return load_hook()


class Resp:
    """Minimal stand-in for the object urlopen returns."""

    def __init__(self, obj=None, raw=None):
        self._raw = raw if raw is not None else json.dumps(obj).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass


class _Calls(list):
    """A list that also knows how to stop the patcher that fills it."""

    stop = staticmethod(lambda: None)


def fake_urlopen(hook, responder):
    """Patch urllib's urlopen; returns a call recorder with a .stop()."""
    calls = _Calls()

    def _open(req, timeout=None):
        calls.append({"req": req, "timeout": timeout,
                      "url": req.full_url,
                      "body": json.loads(req.data.decode()) if req.data else None,
                      "headers": dict(req.header_items())})
        out = responder(req) if callable(responder) else responder
        if isinstance(out, Exception):
            raise out
        return out

    patcher = mock.patch.object(hook.urllib.request, "urlopen", _open)
    patcher.start()
    calls.stop = patcher.stop
    return calls


def run_main(hook, payload, env=None):
    """Drive main() with a JSON payload on stdin. Returns (exit_code|None, stdout)."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    out = io.StringIO()
    code = None
    with mock.patch.object(hook.sys, "stdin", io.StringIO(raw)), \
            mock.patch.object(hook.sys, "stdout", out), \
            mock.patch.dict(os.environ, env or {}):
        try:
            hook.main()
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


def context_of(stdout: str) -> str:
    """main() emits exactly one JSON object with a single 'context' key."""
    obj = json.loads(stdout)
    assert list(obj) == ["context"]
    return obj["context"]


# ---------------------------------------------------------------------------
# import-time environment handling
# ---------------------------------------------------------------------------

def test_constants_default_when_env_absent():
    h = load_hook({"OLLAMA_BASE_URL": "", "QDRANT_URL": ""})
    assert h.QDRANT_URL == ""
    # OLLAMA_URL is None (not "") when the base is unset/empty -- _embed keys off it.
    assert h.OLLAMA_URL is None
    assert h.EMBED_MODEL == "nomic-embed-text"
    assert (h.RECALL_TOP_K, h.MIN_IMPORTANCE, h.MIN_SCORE) == (3, 0.2, 0.55)
    assert (h.MAX_CONTENT_CHARS, h.RULES_MAX_CHARS) == (200, 1200)
    assert (h.EMBED_TIMEOUT, h.QDRANT_TIMEOUT, h.WORKERS) == (3.0, 2.0, 8)
    assert (h.RANKER_W_RELEVANCE, h.RANKER_W_RECENCY,
            h.RANKER_W_TRUST, h.RANKER_W_TYPE) == (0.50, 0.20, 0.15, 0.15)
    assert h.RECENCY_HALFLIFE_DAYS == 7 and h.MMR_LAMBDA == 0.75
    assert (h.PHERO_BETA, h.PHERO_HALFLIFE_H,
            h.PHERO_DEPOSIT, h.PHERO_EPSILON) == (0.08, 24.0, 1.0, 0.05)
    assert h.SA_ENABLED is True and h.SA_TIMEOUT_MS == 25.0


def test_ollama_url_gets_v1_suffix():
    h = load_hook({"OLLAMA_BASE_URL": "http://o:11434"})
    assert h.OLLAMA_URL == "http://o:11434/v1"


def test_sa_enabled_false_words():
    for word in ("false", "FALSE", "0", "no"):
        assert load_hook({"HOOK_SA_ENABLED": word}).SA_ENABLED is False
    # anything else is truthy, including "off"
    assert load_hook({"HOOK_SA_ENABLED": "off"}).SA_ENABLED is True


def test_collections_base_set_and_field_mapping():
    h = load_hook({"GROUNDING_EXTRA_COLLECTIONS": ""})
    assert h.COLLECTIONS == [
        ("mnemosyne", "content", "importance", True),
        ("loci_sessions", "content_preview", None, True),
        ("loci_memory", "text", "numeric_confidence", True),
    ]


def test_collections_extra_names_are_stripped_and_mapped():
    h = load_hook({"GROUNDING_EXTRA_COLLECTIONS":
                   " ecc_skills , agent_core_chunks ,, brand_new_col "})
    assert h.COLLECTIONS[3:] == [
        ("ecc_skills", "content_preview", None, True),
        ("agent_core_chunks", "text", None, False),   # note: unnamed vector
        ("brand_new_col", "text", None, True),        # unknown -> default fields
    ]


def test_grounding_extra_collections_does_not_inherit_extra_rag_collections():
    """.env.example claimed a fallback that does not exist."""
    h = load_hook({"EXTRA_RAG_COLLECTIONS": "my_codebase,my_docs"})
    assert [c[0] for c in h.COLLECTIONS] == ["mnemosyne", "loci_sessions", "loci_memory"]


def test_env_file_is_loaded_with_setdefault_semantics(tmp_path):
    home = tmp_path / "home"
    (home / ".hermes").mkdir(parents=True)
    (home / ".hermes" / ".env").write_text(
        "# a comment\n"
        "\n"
        "no_equals_here\n"
        "QDRANT_URL=http://from-file:6333\n"
        "  MNEMOSYNE_EMBEDDING_MODEL = spaced-model  \n"
        "HOOK_RECALL_TOP_K=9\n"
    )
    h = load_hook({"HOME": str(home), "HERMES_HOME": str(home / ".hermes"),
                   "QDRANT_URL": "http://from-real-env:6333"})
    # real env wins (setdefault), file fills the gaps, whitespace is stripped
    assert h.QDRANT_URL == "http://from-real-env:6333"
    assert h.EMBED_MODEL == "spaced-model"
    assert h.RECALL_TOP_K == 9


def test_hermes_profile_selects_a_profile_subdirectory(tmp_path):
    home = tmp_path / "home"
    prof = home / ".hermes" / "profiles" / "p1"
    prof.mkdir(parents=True)
    (prof / ".env").write_text("HOOK_RECALL_MIN_SCORE=0.91\n")
    h = load_hook({"HOME": str(home), "HERMES_HOME": str(home / ".hermes"),
                   "HERMES_PROFILE": "p1"})
    assert h.MIN_SCORE == 0.91
    # RULES_DIR defaults under the profile dir too
    assert h.RULES_DIR == str(prof / "rules")


# ---------------------------------------------------------------------------
# _embed_headers / _embed
# ---------------------------------------------------------------------------

def test_embed_headers_without_key():
    assert load_hook()._embed_headers() == {"Content-Type": "application/json"}


def test_embed_headers_authorization_is_bearer_prefixed():
    h = load_hook({"EMBED_API_KEY": "sk-123"})
    assert h._embed_headers() == {"Content-Type": "application/json",
                                  "Authorization": "Bearer sk-123"}


def test_embed_headers_custom_header_is_raw_key():
    h = load_hook({"EMBED_API_KEY": "sk-123", "EMBED_API_KEY_HEADER": "X-Api-Key"})
    assert h._embed_headers() == {"Content-Type": "application/json",
                                  "X-Api-Key": "sk-123"}


def test_embed_returns_none_without_ollama_url_and_makes_no_request():
    h = load_hook({"OLLAMA_BASE_URL": ""})
    calls = fake_urlopen(h, Resp({}))
    try:
        assert h._embed("anything") is None
        assert calls == []
    finally:
        calls.stop()


def test_embed_posts_openai_shaped_body_and_returns_vector(hook):
    calls = fake_urlopen(hook, Resp({"data": [{"embedding": [0.1, 0.2, 0.3]}]}))
    try:
        assert hook._embed("hello world") == [0.1, 0.2, 0.3]
    finally:
        calls.stop()
    c = calls[0]
    assert c["url"] == "http://ollama.invalid:11434/v1/embeddings"
    assert c["req"].get_method() == "POST"
    assert c["body"] == {"model": "nomic-embed-text", "input": ["hello world"]}
    assert c["timeout"] == 3.0


@pytest.mark.parametrize("resp", [
    OSError("connection refused"),
    Resp(raw=b"not json"),
    Resp({"data": []}),          # IndexError
    Resp({"error": "nope"}),     # KeyError
])
def test_embed_swallows_every_failure_and_returns_none(hook, resp):
    calls = fake_urlopen(hook, resp)
    try:
        assert hook._embed("q") is None
    finally:
        calls.stop()


# ---------------------------------------------------------------------------
# _search_collection
# ---------------------------------------------------------------------------

def _hit_response(points):
    return Resp({"result": points})


def test_search_collection_named_vector_request_shape(hook):
    calls = fake_urlopen(hook, _hit_response([]))
    try:
        hook._search_collection("mnemosyne", [1.0, 2.0], "content", "importance", True, top_k=4)
    finally:
        calls.stop()
    c = calls[0]
    assert c["url"] == "http://qdrant.invalid:6333/collections/mnemosyne/points/search"
    # A named vector is {"name": ..., "vector": ...}; {"dense": [...]} is a 400.
    assert c["body"] == {"vector": {"name": "dense", "vector": [1.0, 2.0]}, "limit": 4,
                         "with_payload": True, "with_vector": False}
    assert c["headers"]["Api-key"] == "test-key"
    assert c["timeout"] == 2.0


def test_search_collection_unnamed_vector_sends_bare_list(hook):
    calls = fake_urlopen(hook, _hit_response([]))
    try:
        hook._search_collection("agent_core_chunks", [1.0], "text", None, False)
    finally:
        calls.stop()
    assert calls[0]["body"]["vector"] == [1.0]
    assert calls[0]["body"]["limit"] == 5   # default top_k


def test_search_collection_normalises_hits(hook):
    pts = [{"id": "p1", "score": 0.9,
            "payload": {"content": "alpha", "importance": 0.8, "extra": 1}}]
    calls = fake_urlopen(hook, _hit_response(pts))
    try:
        hits = hook._search_collection("mnemosyne", [1.0], "content", "importance", True)
    finally:
        calls.stop()
    assert hits == [{
        "collection": "mnemosyne",
        "point_id": "p1",
        "score": 0.9,
        "importance": 0.8,
        "fused": pytest.approx(0.72),
        "content": "alpha",
        "payload": {"content": "alpha", "importance": 0.8, "extra": 1},
    }]


def test_search_collection_min_score_is_inclusive_at_the_threshold(hook):
    pts = [{"id": "a", "score": 0.55, "payload": {"content": "keep"}},
           {"id": "b", "score": 0.5499, "payload": {"content": "drop"}},
           {"id": "c", "payload": {"content": "no score -> 0.0 -> drop"}}]
    calls = fake_urlopen(hook, _hit_response(pts))
    try:
        hits = hook._search_collection("c1", [1.0], "content", None, True)
    finally:
        calls.stop()
    assert [h["content"] for h in hits] == ["keep"]


def test_search_collection_content_field_fallback_order(hook):
    pts = [{"id": "a", "score": 0.9,
            "payload": {"content": "from-content", "description": "from-desc"}},
           {"id": "b", "score": 0.9, "payload": {"description": "only-desc"}},
           {"id": "c", "score": 0.9, "payload": {"nothing": "usable"}}]
    calls = fake_urlopen(hook, _hit_response(pts))
    try:
        hits = hook._search_collection("c1", [1.0], "missing_field", None, True)
    finally:
        calls.stop()
    # 'text' -> 'content' -> 'content_preview' -> 'description'; c has none -> dropped
    assert [h["content"] for h in hits] == ["from-content", "only-desc"]


def test_search_collection_importance_defaults_and_falsy_coercion(hook):
    pts = [{"id": "a", "score": 0.9, "payload": {"text": "x"}},              # field missing
           {"id": "b", "score": 0.9, "payload": {"text": "y", "conf": 0}},   # 0 -> 0.5 (or-clause)
           {"id": "c", "score": 0.9, "payload": {"text": "z", "conf": None}},
           {"id": "d", "score": 0.9, "payload": {"text": "w", "conf": "0.25"}}]
    calls = fake_urlopen(hook, _hit_response(pts))
    try:
        hits = hook._search_collection("c1", [1.0], "text", "conf", True)
        no_field = hook._search_collection("c1", [1.0], "text", None, True)
    finally:
        calls.stop()
    assert [h["importance"] for h in hits] == [0.5, 0.5, 0.5, 0.25]
    assert all(h["importance"] == 0.5 for h in no_field)


def test_search_collection_non_numeric_importance_raises(hook):
    """The importance field must be numeric: float() on a label raises straight
    out of the search (the try/except only wraps the HTTP call). That is why
    loci_memory reads numeric_confidence -- see the tests below."""
    pts = [{"id": "a", "score": 0.9, "payload": {"text": "x", "confidence": "high"}}]
    calls = fake_urlopen(hook, _hit_response(pts))
    try:
        with pytest.raises(ValueError):
            hook._search_collection("loci_memory", [1.0], "text", "confidence", True)
    finally:
        calls.stop()


def test_loci_memory_reads_the_numeric_confidence_the_producer_writes(hook):
    """A loci_memory finding carries BOTH fields: 'confidence' is the
    high/medium/low label _multi_signal_score reads, and 'numeric_confidence' is
    the float. Reading the label as importance made float() raise, so the lane
    contributed nothing on every prompt -- silently in the main session, and
    fatally in a subagent, where only _SearchFailed is caught."""
    pts = [{"id": "a", "score": 0.9,
            "payload": {"text": "x", "confidence": "high", "numeric_confidence": 0.9}}]
    calls = fake_urlopen(hook, _hit_response(pts))
    try:
        col, _, impf, named = hook.COLLECTIONS[2]
        assert col == "loci_memory"
        hits = hook._search_collection(col, [1.0], "text", impf, named)
    finally:
        calls.stop()
    assert [h["importance"] for h in hits] == [0.9]
    # and the label is still there for the ranker that wants a label
    assert hook._multi_signal_score(hits[0], time.time()) > 0


def test_search_collection_raises_on_transport_error(hook):
    # Returning [] would make a failed search indistinguishable from an empty one.
    calls = fake_urlopen(hook, OSError("down"))
    try:
        with pytest.raises(hook._SearchFailed):
            hook._search_collection("c1", [1.0], "text", None, True)
    finally:
        calls.stop()


def test_search_collection_raises_on_http_error(hook):
    calls = fake_urlopen(hook, urllib.error.HTTPError(
        "u", 400, "Bad Request", {}, io.BytesIO(b'{"status":{"error":"x"}}')))
    try:
        with pytest.raises(hook._SearchFailed):
            hook._search_collection("c1", [1.0], "text", None, True)
    finally:
        calls.stop()


def test_search_collection_null_result_raises_typeerror(hook):
    """BUG: Qdrant error envelopes carry {"result": null}; `.get("result", [])`
    returns None and the for-loop raises. Swallowed in the fan-out (futures are
    guarded) but fatal on the subagent path, which calls this directly."""
    calls = fake_urlopen(hook, Resp({"status": {"error": "boom"}, "result": None}))
    try:
        with pytest.raises(TypeError):
            hook._search_collection("c1", [1.0], "text", None, True)
    finally:
        calls.stop()


def test_search_collection_missing_result_key_is_empty(hook):
    calls = fake_urlopen(hook, Resp({"status": "ok"}))
    try:
        assert hook._search_collection("c1", [1.0], "text", None, True) == []
    finally:
        calls.stop()


# ---------------------------------------------------------------------------
# _beam_fallback
# ---------------------------------------------------------------------------

def test_beam_fallback_returns_empty_when_mnemosyne_missing(hook):
    # Clearing sys.modules is not enough: the import re-succeeds from disk.
    saved = list(sys.path)
    real_import = builtins.__import__

    def no_mnemosyne(name, *a, **kw):
        if name.startswith("mnemosyne"):
            raise ImportError(f"no module named {name}")
        return real_import(name, *a, **kw)

    blocked = {m: None for m in list(sys.modules) if m.startswith("mnemosyne")}
    try:
        with mock.patch.dict(sys.modules, blocked), \
                mock.patch.object(builtins, "__import__", no_mnemosyne):
            for m in blocked:
                del sys.modules[m]
            assert hook._beam_fallback("some query") == []
        # side effect: it prepends two hard-coded ~/.hermes paths to sys.path
        assert sys.path[0].endswith("/.hermes/mnemosyne")
        assert sys.path[1].endswith("site-packages")
    finally:
        sys.path[:] = saved


def _install_fake_beam(recall_result, raises=None):
    """Register a fake mnemosyne.core.beam so the fallback import succeeds."""
    pkg = types.ModuleType("mnemosyne")
    core = types.ModuleType("mnemosyne.core")
    beam = types.ModuleType("mnemosyne.core.beam")
    calls = []

    class BeamMemory:
        def __init__(self, session_id=None):
            calls.append({"session_id": session_id})

        def recall(self, query, **kw):
            calls.append({"query": query, **kw})
            if raises:
                raise raises
            return recall_result

    beam.BeamMemory = BeamMemory
    pkg.core = core
    core.beam = beam
    return {"mnemosyne": pkg, "mnemosyne.core": core, "mnemosyne.core.beam": beam}, calls


def test_beam_fallback_maps_and_filters_by_min_importance(hook):
    rows = [{"content": "kept", "score": 0.8, "importance": 0.5},
            {"content": "dropped", "score": 0.9, "importance": 0.1},
            {"content": "edge", "score": 0.6, "importance": 0.2}]
    mods, calls = _install_fake_beam(rows)
    saved = list(sys.path)
    with mock.patch.dict(sys.modules, mods), \
            mock.patch.dict(os.environ, {"HERMES_AGENT_ID": "agent-7"}):
        try:
            out = hook._beam_fallback("why is x broken")
        finally:
            sys.path[:] = saved
    assert out == [
        {"collection": "mnemosyne_beam", "score": 0.8, "importance": 0.5,
         "fused": pytest.approx(0.4), "content": "kept"},
        {"collection": "mnemosyne_beam", "score": 0.6, "importance": 0.2,
         "fused": pytest.approx(0.12), "content": "edge"},
    ]
    # beam hits carry no payload/point_id keys -- downstream must tolerate that
    assert "payload" not in out[0] and "point_id" not in out[0]
    assert calls[0] == {"session_id": "agent-7"}
    assert calls[1] == {"query": "why is x broken", "top_k": 3,
                        "vec_weight": 0.5, "fts_weight": 0.3, "importance_weight": 0.2}


def test_beam_fallback_swallows_recall_errors(hook):
    mods, _ = _install_fake_beam(None, raises=RuntimeError("db locked"))
    saved = list(sys.path)
    with mock.patch.dict(sys.modules, mods):
        try:
            assert hook._beam_fallback("q") == []
        finally:
            sys.path[:] = saved


# ---------------------------------------------------------------------------
# _load_rules_summary
# ---------------------------------------------------------------------------

def test_rules_summary_empty_when_dir_missing(hook, tmp_path):
    hook.RULES_DIR = str(tmp_path / "nope")
    assert hook._load_rules_summary() == ""


def test_rules_summary_only_h2_headers_sorted_by_filename(hook, tmp_path):
    (tmp_path / "b.md").write_text("# Title\n## Second\ntext\n### Third\n")
    (tmp_path / "a.md").write_text("##   Spaced Header  \n## Another\n")
    (tmp_path / "c.md").write_text("no headers at all\n")
    (tmp_path / "d.txt").write_text("## ignored non-md\n")
    hook.RULES_DIR = str(tmp_path)
    assert hook._load_rules_summary() == (
        "ACTIVE RULES FILES:\n"
        "[rules/a.md]: Spaced Header, Another\n"
        "[rules/b.md]: Second"
    )


def test_rules_summary_returns_empty_when_no_file_has_headers(hook, tmp_path):
    (tmp_path / "a.md").write_text("# only h1\nbody\n")
    hook.RULES_DIR = str(tmp_path)
    assert hook._load_rules_summary() == ""


def test_rules_budget_skips_oversized_entries_but_keeps_scanning(hook, tmp_path):
    (tmp_path / "a.md").write_text("## " + "x" * 200 + "\n")
    (tmp_path / "b.md").write_text("## small\n")
    hook.RULES_DIR = str(tmp_path)
    hook.RULES_MAX_CHARS = 60
    out = hook._load_rules_summary()
    assert "xxxx" not in out
    assert out == "ACTIVE RULES FILES:\n[rules/b.md]: small"


# ---------------------------------------------------------------------------
# _keyword_rerank
# ---------------------------------------------------------------------------

def test_keyword_rerank_noop_for_short_queries(hook):
    hits = [{"content": "database schema", "fused": 0.1}]
    out = hook._keyword_rerank(hits, "schema")
    assert out is hits            # same object, not even re-sorted
    assert hits[0]["fused"] == 0.1


def test_keyword_rerank_bonus_and_reordering(hook):
    hits = [{"content": "nothing relevant", "fused": 0.30},
            {"content": "the DATABASE schema migration plan", "fused": 0.25}]
    out = hook._keyword_rerank(hits, "database schema migration")
    # 3 tokens all >3 chars, all present -> +0.15 (case-insensitive match)
    assert out[0]["content"] == "the DATABASE schema migration plan"
    assert out[0]["fused"] == pytest.approx(0.40)
    assert out[1]["fused"] == pytest.approx(0.30)   # untouched, bonus 0


def test_keyword_rerank_ignores_tokens_of_four_chars_or_less(hook):
    hits = [{"content": "the bug is odd", "fused": 0.0}]
    hook._keyword_rerank(hits, "the bug odd")   # 3 tokens, none longer than 3
    assert hits[0]["fused"] == 0.0


def test_keyword_rerank_bonus_is_capped_at_quarter(hook):
    words = "alpha bravo charlie delta echo foxtrot golf"
    hits = [{"content": words, "fused": 0.0}]
    hook._keyword_rerank(hits, words)
    assert hits[0]["fused"] == pytest.approx(0.25)   # 7 * 0.05 clamped


def test_keyword_rerank_defaults_missing_fused_to_zero(hook):
    hits = [{"content": "alpha bravo"}]
    hook._keyword_rerank(hits, "alpha bravo")
    assert hits[0]["fused"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# _multi_signal_score / _effective_pheromone
# ---------------------------------------------------------------------------

def test_multi_signal_score_all_defaults(hook):
    now = 1_000_000.0
    hit = {"score": 0.8, "importance": 0.5, "fused": 0.4, "payload": {}}
    # 0.5*0.4 + 0.2*0.5 (unknown age) + 0.15*0.6 (unknown conf) + 0.15*0.7 (unknown type)
    assert hook._multi_signal_score(hit, now) == pytest.approx(0.495)


def test_multi_signal_score_recency_halflife(hook):
    now = 1_000_000.0
    seven_days = now - 7 * 86400
    hit = {"score": 1.0, "importance": 1.0, "fused": 0.0,
           "payload": {"created_at_ts": seven_days}}
    # recency == 0.5 exactly at one half-life
    assert hook._multi_signal_score(hit, now) == pytest.approx(0.2 * 0.5 + 0.15 * 0.6 + 0.15 * 0.7)


def test_multi_signal_score_future_timestamp_clamps_to_full_recency(hook):
    now = 1_000_000.0
    hit = {"score": 0.0, "importance": 0.0, "fused": 0.0,
           "payload": {"created_at_ts": now + 99999}}
    assert hook._multi_signal_score(hit, now) == pytest.approx(0.2 + 0.15 * 0.6 + 0.15 * 0.7)


def test_multi_signal_score_falls_back_to_ts_epoch(hook):
    now = 1_000_000.0
    a = {"score": 0.0, "importance": 0.0, "fused": 0.0,
         "payload": {"ts_epoch": now}}
    b = {"score": 0.0, "importance": 0.0, "fused": 0.0,
         "payload": {"created_at_ts": 0, "ts_epoch": now}}   # 0 is falsy -> ts_epoch
    assert hook._multi_signal_score(a, now) == pytest.approx(hook._multi_signal_score(b, now))
    assert hook._multi_signal_score(a, now) == pytest.approx(0.2 + 0.15 * 0.6 + 0.15 * 0.7)


@pytest.mark.parametrize("conf,trust", [("high", 1.0), ("HIGH", 1.0), ("medium", 0.7),
                                        ("low", 0.4), ("bogus", 0.6), ("", 0.6), (None, 0.6)])
def test_multi_signal_score_trust_tiers(hook, conf, trust):
    now = 1_000_000.0
    hit = {"score": 0.0, "importance": 0.0, "fused": 0.0, "payload": {"confidence": conf}}
    assert hook._multi_signal_score(hit, now) == pytest.approx(
        0.2 * 0.5 + 0.15 * trust + 0.15 * 0.7)


@pytest.mark.parametrize("rtype,w", [("observed", 1.0), ("inferred", 0.8), ("assumed", 0.6),
                                     ("gap", 0.4), ("weird", 0.7)])
def test_multi_signal_score_type_weights(hook, rtype, w):
    now = 1_000_000.0
    a = {"score": 0.0, "importance": 0.0, "fused": 0.0, "payload": {"record_type": rtype}}
    b = {"score": 0.0, "importance": 0.0, "fused": 0.0, "payload": {"type": rtype}}
    expect = 0.2 * 0.5 + 0.15 * 0.6 + 0.15 * w
    assert hook._multi_signal_score(a, now) == pytest.approx(expect)
    assert hook._multi_signal_score(b, now) == pytest.approx(expect)


def test_multi_signal_score_missing_payload_key_is_tolerated(hook):
    """Beam-fallback hits have no 'payload' key at all."""
    hit = {"score": 0.8, "importance": 0.5, "fused": 0.4}
    assert hook._multi_signal_score(hit, 1.0) == pytest.approx(0.495)


def test_multi_signal_score_requires_score_key_even_when_fused_present(hook):
    """BUG: the default expression `hit['score'] * ...` is evaluated eagerly,
    so a hit carrying 'fused' but no 'score' raises KeyError instead of using
    the fused value it already has."""
    with pytest.raises(KeyError):
        hook._multi_signal_score({"fused": 0.9, "importance": 0.5, "payload": {}}, 1.0)


def test_pheromone_boost_is_added_on_top(hook):
    now = 1_000_000.0
    plain = {"score": 0.0, "importance": 0.0, "fused": 0.0, "payload": {}}
    hot = {"score": 0.0, "importance": 0.0, "fused": 0.0,
           "payload": {"pheromone": 3.0}}   # no reinforced ts -> no evaporation
    assert (hook._multi_signal_score(hot, now) - hook._multi_signal_score(plain, now)
            == pytest.approx(0.08 * math.log1p(3.0)))


@pytest.mark.parametrize("payload,expect", [
    ({}, 0.0),
    ({"pheromone": 0.0}, 0.0),
    ({"pheromone": -5.0}, 0.0),
    ({"pheromone": None}, 0.0),
    ({"pheromone": 2.5}, 2.5),
    ({"pheromone": "3.0"}, 3.0),
])
def test_effective_pheromone_without_reinforcement(hook, payload, expect):
    assert hook._effective_pheromone(payload, 1_000_000.0) == pytest.approx(expect)


def test_effective_pheromone_evaporates_by_halflife(hook):
    now = 1_000_000.0
    p = {"pheromone": 8.0, "pheromone_reinforced_ts": now - 24 * 3600}
    assert hook._effective_pheromone(p, now) == pytest.approx(4.0)
    p2 = {"pheromone": 8.0, "pheromone_reinforced_ts": now - 48 * 3600}
    assert hook._effective_pheromone(p2, now) == pytest.approx(2.0)
    future = {"pheromone": 8.0, "pheromone_reinforced_ts": now + 9999}
    assert hook._effective_pheromone(future, now) == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# _pheromone_deposit
# ---------------------------------------------------------------------------

def test_pheromone_deposit_request_shape(hook):
    calls = fake_urlopen(hook, Resp({"result": True}))
    try:
        hook._pheromone_deposit("loci_memory", "pt-1", 2.0, 1_700_000_000.0)
    finally:
        calls.stop()
    c = calls[0]
    assert c["url"] == "http://qdrant.invalid:6333/collections/loci_memory/points/payload"
    assert c["body"] == {"points": ["pt-1"],
                         "payload": {"pheromone": 3.0,
                                     "pheromone_reinforced_ts": 1_700_000_000.0}}
    assert c["timeout"] == 0.5
    assert c["headers"]["Api-key"] == "test-key"


@pytest.mark.parametrize("pid", [None, "", 0])
def test_pheromone_deposit_skips_falsy_point_ids(hook, pid):
    """Note: point id 0 is a legal Qdrant id but is skipped here."""
    calls = fake_urlopen(hook, Resp({}))
    try:
        hook._pheromone_deposit("c", pid, 1.0, 1.0)
        assert calls == []
    finally:
        calls.stop()


def test_pheromone_deposit_is_fire_and_forget(hook):
    calls = fake_urlopen(hook, OSError("qdrant down"))
    try:
        assert hook._pheromone_deposit("c", "p", 1.0, 1.0) is None
    finally:
        calls.stop()


# ---------------------------------------------------------------------------
# _extract_intent / _clean_content
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expect", [
    ("  Can you fix the bug  ", "fix the bug"),
    ("PLEASE run the tests", "run the tests"),
    ("could you   check this", "check this"),
    ("I'd like you to refactor", "refactor"),
    ("I need you to look", "look"),
    ("would you mind", "mind"),
    ("help me debug", "debug"),
    ("I want you to stop", "stop"),
    ("fix the bug", "fix the bug"),
    ("we can you know just ship", "we can you know just ship"),  # only anchored at start
    ("please", "please"),                                        # needs trailing whitespace
    ("please please stop", "please stop"),                       # one substitution only
])
def test_extract_intent(hook, raw, expect):
    assert hook._extract_intent(raw) == expect


def test_extract_intent_truncates_to_200_chars(hook):
    out = hook._extract_intent("please " + "a" * 500)
    assert out == "a" * 200


@pytest.mark.parametrize("raw,expect", [
    ("  plain text  ", "plain text"),
    ("[not json but starts with bracket", "[not json but starts with bracket"),
    ("{\"a\": 1}", '{"a": 1}'),
])
def test_clean_content_passthrough(hook, raw, expect):
    assert hook._clean_content(raw) == expect


def test_clean_content_extracts_transcript_turns(hook):
    blob = json.dumps([
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "  what broke  "},
        {"role": "assistant", "content": "the parser"},
        {"role": "user", "content": "third turn is dropped"},
        "not a dict",
    ])
    assert hook._clean_content(blob) == "user: what broke | assistant: the parser"


def test_clean_content_truncates_each_turn_at_150(hook):
    blob = json.dumps([{"role": "user", "content": "z" * 400}])
    assert hook._clean_content(blob) == "user: " + "z" * 150


def test_clean_content_json_list_with_no_usable_turns_becomes_empty(hook):
    assert hook._clean_content(json.dumps([{"role": "system", "content": "x"}])) == ""
    assert hook._clean_content(json.dumps([{"role": "user", "content": "   "}])) == ""


def test_clean_content_drops_non_transcript_json_arrays(hook):
    """BUG: any payload that happens to start with '[{' or '["' is assumed to be
    a chat transcript. Structured non-transcript content is silently erased,
    which in _format_results means the hit is dropped from the output entirely."""
    assert hook._clean_content('[{"file": "a.py", "lines": 12}]') == ""
    assert hook._clean_content('["alpha", "beta"]') == ""


def test_clean_content_regex_fallback_for_broken_json(hook):
    broken = '[{"role": "user", "content": "the pipeline is failing", }'
    assert hook._clean_content(broken) == "user: the pipeline is failing"
    # <10 chars of content -> no match -> empty
    assert hook._clean_content('[{"role": "user", "content": "short", }') == ""


# ---------------------------------------------------------------------------
# _mmr_select
# ---------------------------------------------------------------------------

def _h(content, ms):
    return {"content": content, "_ms_score": ms}


def test_mmr_select_returns_input_untouched_when_within_top_k(hook):
    hits = [_h("a", 0.1), _h("b", 0.9)]
    out = hook._mmr_select(hits, 3)
    assert out is hits            # same object, NOT sorted by score
    assert [x["content"] for x in out] == ["a", "b"]


def test_mmr_select_prefers_diversity_over_raw_score(hook):
    a = _h("alpha beta gamma delta", 0.90)
    b = _h("alpha beta gamma delta", 0.85)   # near-duplicate of a
    c = _h("zeta eta theta iota", 0.80)
    with mock.patch("random.random", return_value=1.0):    # disable ε-exploration
        out = hook._mmr_select([a, b, c], 2)
    assert [x["_ms_score"] for x in out] == [0.90, 0.80]


def test_mmr_select_epsilon_exploration_fills_last_slot_randomly(hook):
    a, b, c = _h("aaa", 0.9), _h("bbb", 0.8), _h("ccc", 0.1)
    with mock.patch("random.random", return_value=0.0), \
            mock.patch("random.choice", side_effect=lambda seq: seq[-1]):
        out = hook._mmr_select([a, b, c], 2)
    assert [x["content"] for x in out] == ["aaa", "ccc"]   # lowest score, chosen at random


def test_mmr_select_empty_content_similarity_is_zero(hook):
    a, b, c = _h("", 0.9), _h("", 0.8), _h("", 0.7)
    with mock.patch("random.random", return_value=1.0):
        out = hook._mmr_select([a, b, c], 2)
    assert [x["_ms_score"] for x in out] == [0.9, 0.8]


def test_mmr_select_requires_ms_score(hook, monkeypatch):
    # Pin epsilon to 0: the random slot never reads ms_score, so unseeded this is flaky.
    monkeypatch.setattr(hook, "PHERO_EPSILON", 0.0)
    with pytest.raises(KeyError):
        hook._mmr_select([{"content": "a"}, {"content": "b"}], 1)


def test_mmr_select_epsilon_branch_does_not_need_ms_score(hook, monkeypatch):
    """The other side of the same coin: with epsilon at 1.0 the random slot is
    always taken, and that path legitimately never touches ms_score."""
    monkeypatch.setattr(hook, "PHERO_EPSILON", 1.0)
    out = hook._mmr_select([{"content": "a"}, {"content": "b"}], 1)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# _format_results
# ---------------------------------------------------------------------------

def test_format_results_line_shape_and_collection_dashes(hook):
    hits = [{"collection": "loci_memory", "score": 0.876, "content": "a finding"}]
    out = hook._format_results(hits, "why", False)
    assert out.splitlines()[0] == 'MEMORY MATCH (1 results from 3 Qdrant collections) for "why":'
    assert out.splitlines()[1] == "[loci-memory|0.88] a finding"
    assert out.splitlines()[2].startswith("Use mcp_mnemosyne_mnemosyne_recall")


def test_format_results_fallback_source_label(hook):
    hits = [{"collection": "mnemosyne_beam", "score": 0.5, "content": "x"}]
    assert "from BeamMemory fallback" in hook._format_results(hits, "q", True)


def test_format_results_source_label_counts_all_collections_even_when_unused(hook):
    """Note: the label reports len(COLLECTIONS), not how many were searched --
    the subagent path searches one collection but still claims three."""
    hits = [{"collection": "loci_memory", "score": 0.5, "content": "x"}]
    hook.COLLECTIONS = hook.COLLECTIONS + [("extra", "text", None, True)]
    assert "from 4 Qdrant collections" in hook._format_results(hits, "q", False)


def test_format_results_dedupes_on_first_80_chars(hook):
    a = "s" * 80 + " tail one"
    b = "s" * 80 + " tail two"
    hits = [{"collection": "c", "score": 0.9, "content": a},
            {"collection": "c", "score": 0.8, "content": b}]
    out = hook._format_results(hits, "q", False)
    assert out.count("[c|") == 1
    assert "(1 results" in out


def test_format_results_truncates_long_content_with_ellipsis(hook):
    hits = [{"collection": "c", "score": 0.9, "content": "y" * 250 + "   tail"}]
    line = hook._format_results(hits, "q", False).splitlines()[1]
    assert line == "[c|0.90] " + "y" * 200 + "..."


def test_format_results_caps_at_recall_top_k(hook):
    hits = [{"collection": "c", "score": 0.9, "content": f"item {i}"} for i in range(10)]
    out = hook._format_results(hits, "q", False)
    assert out.count("[c|") == 3
    assert "(3 results" in out


def test_format_results_skips_hits_cleaned_to_empty(hook):
    hits = [{"collection": "c", "score": 0.9, "content": '[{"role": "system", "content": "x"}]'},
            {"collection": "c", "score": 0.8, "content": "real content"}]
    out = hook._format_results(hits, "q", False)
    assert out.count("[c|") == 1 and "real content" in out


def test_format_results_header_states_real_coverage_on_a_partial_outage(hook):
    """The header is a coverage claim the model is asked to trust: on a partial
    outage it must not assert that all three collections answered."""
    hits = [{"collection": "loci_memory", "score": 0.5, "content": "x"}]
    out = hook._format_results(hits, "q", False, 2)
    assert out.splitlines()[0] == (
        'MEMORY MATCH (1 results from 1 of 3 Qdrant collections (2 unreachable)) for "q":'
    )


def test_format_results_returns_directive_when_nothing_survives(hook):
    assert hook._format_results([], "q", False) == hook.GROUNDING_DIRECTIVE
    only_empty = [{"collection": "c", "score": 0.9, "content": "   "}]
    assert hook._format_results(only_empty, "q", False) == hook.GROUNDING_DIRECTIVE


def test_grounding_directive_default_text_and_override():
    h = load_hook()
    assert h.GROUNDING_DIRECTIVE.startswith("[GROUNDING DIRECTIVE — active every turn]")
    assert "mnemosyne recall tool" in h.GROUNDING_DIRECTIVE
    assert load_hook({"GROUNDING_DIRECTIVE": "custom"}).GROUNDING_DIRECTIVE == "custom"


# ---------------------------------------------------------------------------
# main() -- early exits
# ---------------------------------------------------------------------------

def test_main_exits_quietly_on_unparseable_stdin(hook):
    code, out = run_main(hook, "not json at all")
    assert code == 0 and out == ""


def test_main_raises_on_non_object_json(hook):
    """BUG: the guard catches JSONDecodeError/OSError but not the AttributeError
    from calling .get() on a non-dict payload (e.g. a bare JSON string)."""
    with pytest.raises(AttributeError):
        run_main(hook, '"just a string"')


@pytest.mark.parametrize("event", ["", "PostToolUse", "Stop", "pre_tool_use", None])
def test_main_ignores_foreign_events(hook, event):
    payload = {"extra": {"user_message": "something meaty here"}}
    if event is not None:
        payload["hook_event_name"] = event
    code, out = run_main(hook, payload)
    assert code == 0 and out == ""


@pytest.mark.parametrize("event", ["pre_llm_call", "PreLlmCall", "UserPromptSubmit",
                                   "SubagentStart"])
def test_main_accepts_all_four_event_names(hook, event):
    hook._embed = lambda t: None
    hook._beam_fallback = lambda q: []
    hook._load_rules_summary = lambda: ""
    code, out = run_main(hook, {"hook_event_name": event,
                                "extra": {"user_message": "real question here"}})
    assert code is None
    assert "WARNING: memory grounding UNAVAILABLE" in context_of(out)


@pytest.mark.parametrize("msg", ["", "   ", "\n\t ", None, 123, {"a": 1}])
def test_main_exits_on_empty_or_non_string_message(hook, msg):
    code, out = run_main(hook, {"hook_event_name": "UserPromptSubmit",
                                "extra": {"user_message": msg}})
    assert code == 0 and out == ""


def test_main_exits_when_extra_missing_entirely(hook):
    code, out = run_main(hook, {"hook_event_name": "UserPromptSubmit", "extra": None})
    assert code == 0 and out == ""


@pytest.mark.parametrize("cmd", ["/help", "/clear", "/compact", "/cost", "/status",
                                 "/history", "/ide", "/doctor", "/login", "/logout"])
def test_main_skips_navigation_slash_commands(hook, cmd):
    for variant in (cmd, cmd.upper(), cmd + " with trailing args"):
        code, out = run_main(hook, {"hook_event_name": "UserPromptSubmit",
                                    "extra": {"user_message": variant}})
        assert (code, out) == (0, ""), variant


@pytest.mark.parametrize("cmd", ["/fix the thing", "/review", "/code-review",
                                 "/remember this", "/ultrareview", "/schedule",
                                 "/helper", "/statusline"])
def test_main_grounds_non_navigation_slash_commands(hook, cmd):
    """Prefix collisions matter: /helper and /statusline are NOT skipped."""
    hook._embed = lambda t: None
    hook._beam_fallback = lambda q: []
    hook._load_rules_summary = lambda: ""
    code, out = run_main(hook, {"hook_event_name": "UserPromptSubmit",
                                "extra": {"user_message": cmd}})
    assert code is None and out != ""


def test_main_grounds_very_short_messages(hook):
    """The v2 min-length guard is gone; MIN_PROMPT_LEN is now dead config."""
    seen = {}
    hook._embed = lambda t: seen.setdefault("q", t) and None
    hook._beam_fallback = lambda q: []
    hook._load_rules_summary = lambda: ""
    code, out = run_main(hook, {"hook_event_name": "UserPromptSubmit",
                                "extra": {"user_message": "why?"}})
    assert code is None and seen["q"] == "why?"


# ---------------------------------------------------------------------------
# main() -- degraded / fail-open paths
# ---------------------------------------------------------------------------

def _prompt(msg="what happened to the loader", **extra):
    e = {"user_message": msg}
    e.update(extra)
    return {"hook_event_name": "UserPromptSubmit", "extra": e}


def test_main_dual_failure_emits_explicit_warning(hook):
    hook._embed = lambda t: None
    hook._beam_fallback = lambda q: []
    hook._load_rules_summary = lambda: ""
    code, out = run_main(hook, _prompt())
    ctx = context_of(out)
    assert code is None
    assert ctx.startswith("[WARNING: memory grounding UNAVAILABLE this turn")
    assert "Ollama unreachable" in ctx and "BeamMemory fallback also failed" in ctx
    assert ctx.endswith(hook.GROUNDING_DIRECTIVE)


def test_main_dual_failure_still_appends_rules(hook):
    hook._embed = lambda t: None
    hook._beam_fallback = lambda q: []
    hook._load_rules_summary = lambda: "ACTIVE RULES FILES:\n[rules/x.md]: A"
    _, out = run_main(hook, _prompt())
    ctx = context_of(out)
    assert ctx.endswith("\n\nACTIVE RULES FILES:\n[rules/x.md]: A")


def test_main_beam_fallback_path_labels_source(hook):
    hook._embed = lambda t: None
    hook._beam_fallback = lambda q: [{"collection": "mnemosyne_beam", "score": 0.7,
                                      "importance": 0.5, "fused": 0.35,
                                      "content": "beam recalled this"}]
    hook._load_rules_summary = lambda: ""
    _, out = run_main(hook, _prompt())
    ctx = context_of(out)
    assert "from BeamMemory fallback" in ctx
    assert "[mnemosyne-beam|0.70] beam recalled this" in ctx


def test_main_empty_search_results_emit_bare_directive(hook):
    hook._embed = lambda t: [0.1] * 4
    hook._search_collection = lambda *a, **k: []
    hook._load_rules_summary = lambda: ""
    _, out = run_main(hook, _prompt())
    assert context_of(out) == hook.GROUNDING_DIRECTIVE


def test_main_intent_stripping_reaches_the_embedder_and_the_header(hook):
    seen = {}

    def _embed(t):
        seen["intent"] = t
        return [0.5]

    hook._embed = _embed
    hook._search_collection = lambda *a, **k: [
        {"collection": "mnemosyne", "point_id": "1", "score": 0.9, "importance": 0.9,
         "fused": 0.81, "content": "the loader was rewritten", "payload": {}}]
    hook._load_rules_summary = lambda: ""
    _, out = run_main(hook, _prompt("Can you tell me about the loader"))
    assert seen["intent"] == "tell me about the loader"
    assert 'for "tell me about the loader":' in context_of(out)


# ---------------------------------------------------------------------------
# main() -- full fan-out path
# ---------------------------------------------------------------------------

def test_main_fans_out_over_every_configured_collection(hook):
    calls = []

    def _search(col, vec, cf, impf, named, top_k):
        calls.append((col, tuple(vec), cf, impf, named, top_k))
        return []

    hook._embed = lambda t: [0.25]
    hook._search_collection = _search
    hook._load_rules_summary = lambda: ""
    run_main(hook, _prompt())
    assert sorted(calls) == sorted([
        ("mnemosyne", (0.25,), "content", "importance", True, 3),
        ("loci_sessions", (0.25,), "content_preview", None, True, 3),
        ("loci_memory", (0.25,), "text", "numeric_confidence", True, 3),
    ])


def test_main_one_collection_blowing_up_does_not_sink_the_turn(hook):
    def _search(col, vec, cf, impf, named, top_k):
        if col == "mnemosyne":
            raise RuntimeError("collection on fire")
        return [{"collection": col, "point_id": None, "score": 0.9, "importance": 0.9,
                 "fused": 0.81, "content": f"from {col}", "payload": {}}]

    hook._embed = lambda t: [0.1]
    hook._search_collection = _search
    hook._load_rules_summary = lambda: ""
    _, out = run_main(hook, _prompt())
    ctx = context_of(out)
    assert "from loci_sessions" in ctx and "from loci_memory" in ctx
    assert "from mnemosyne" not in ctx
    assert "(2 results" in ctx
    assert "2 of 3 Qdrant collections (1 unreachable)" in ctx


def test_main_partial_outage_with_no_hits_warns_instead_of_a_bare_directive(hook):
    """An empty result is what a genuine miss looks like too. When part of memory
    was dark, say so -- the total-outage path already does."""
    def _search(col, vec, cf, impf, named, top_k):
        if col == "mnemosyne":
            raise RuntimeError("collection on fire")
        return []

    hook._embed = lambda t: [0.1]
    hook._search_collection = _search
    hook._load_rules_summary = lambda: ""
    _, out = run_main(hook, _prompt())
    ctx = context_of(out)
    assert ctx.startswith("[WARNING: memory grounding INCOMPLETE this turn")
    assert "1 of 3 Qdrant collections were unreachable" in ctx
    assert ctx.endswith(hook.GROUNDING_DIRECTIVE)


def test_main_healthy_empty_search_stays_a_bare_directive(hook):
    hook._embed = lambda t: [0.1]
    hook._search_collection = lambda *a, **k: []
    hook._load_rules_summary = lambda: ""
    _, out = run_main(hook, _prompt())
    assert context_of(out) == hook.GROUNDING_DIRECTIVE


def test_main_filters_hits_below_min_importance(hook):
    hits = [{"collection": "loci_memory", "point_id": None, "score": 0.9,
             "importance": 0.19, "fused": 0.17, "content": "too unimportant", "payload": {}},
            {"collection": "loci_memory", "point_id": None, "score": 0.9,
             "importance": 0.2, "fused": 0.18, "content": "exactly at the floor",
             "payload": {}}]
    hook._embed = lambda t: [0.1]
    hook._search_collection = lambda col, *a, **k: hits if col == "loci_memory" else []
    hook._load_rules_summary = lambda: ""
    _, out = run_main(hook, _prompt())
    ctx = context_of(out)
    assert "exactly at the floor" in ctx and "too unimportant" not in ctx


def test_main_deposits_pheromone_only_on_loci_memory_hits(hook):
    deposits = []
    hook._embed = lambda t: [0.1]
    hook._pheromone_deposit = lambda *a: deposits.append(a)
    hook._load_rules_summary = lambda: ""

    def _search(col, *a, **k):
        if col == "loci_memory":
            return [{"collection": col, "point_id": "hm-1", "score": 0.9, "importance": 0.9,
                     "fused": 0.81, "content": "finding one",
                     "payload": {"pheromone": 4.0}},
                    {"collection": col, "point_id": None, "score": 0.9, "importance": 0.9,
                     "fused": 0.81, "content": "no id, no deposit", "payload": {}}]
        if col == "mnemosyne":
            return [{"collection": col, "point_id": "mn-1", "score": 0.9, "importance": 0.9,
                     "fused": 0.81, "content": "a memory", "payload": {}}]
        return []

    hook._search_collection = _search
    run_main(hook, _prompt())
    assert len(deposits) == 1
    col, pid, current, ts = deposits[0]
    assert (col, pid, current) == ("loci_memory", "hm-1", 4.0)
    assert abs(ts - time.time()) < 30


def test_main_orders_output_by_multi_signal_score_not_raw_score(hook):
    now = time.time()
    hook._embed = lambda t: [0.1]
    hook._load_rules_summary = lambda: ""
    hook._pheromone_deposit = lambda *a: None
    hook._search_collection = lambda col, *a, **k: ([
        # higher cosine, stale + low confidence
        {"collection": "loci_memory", "point_id": None, "score": 0.95, "importance": 0.9,
         "fused": 0.855, "content": "stale but similar",
         "payload": {"created_at_ts": now - 400 * 86400, "confidence": "low",
                     "record_type": "gap"}},
        # lower cosine, fresh + trusted + observed
        {"collection": "loci_memory", "point_id": None, "score": 0.70, "importance": 0.9,
         "fused": 0.63, "content": "fresh and trusted",
         "payload": {"created_at_ts": now, "confidence": "high",
                     "record_type": "observed"}},
    ] if col == "loci_memory" else [])
    with mock.patch("random.random", return_value=1.0):
        _, out = run_main(hook, _prompt())
    lines = [ln for ln in context_of(out).splitlines() if ln.startswith("[loci-memory")]
    assert lines[0].endswith("fresh and trusted")
    assert lines[1].endswith("stale but similar")


# ---------------------------------------------------------------------------
# main() -- spreading activation enrichment
# ---------------------------------------------------------------------------

def _sa_module(results, delay=0.0):
    calls = []

    def run_spreading_activation(db_path, seed_ids, seed_scores, max_results):
        calls.append({"db_path": db_path, "seed_ids": seed_ids,
                      "seed_scores": seed_scores, "max_results": max_results})
        if delay:
            time.sleep(delay)
        return results

    mod = types.SimpleNamespace(run_spreading_activation=run_spreading_activation)
    return mod, calls


def _sa_setup(hook, mnemosyne_hits, sa_results, delay=0.0):
    hook.SA_ENABLED = True
    hook._SA_MODULE, calls = _sa_module(sa_results, delay)
    hook._embed = lambda t: [0.1]
    hook._load_rules_summary = lambda: ""
    hook._pheromone_deposit = lambda *a: None
    hook._search_collection = lambda col, *a, **k: (
        mnemosyne_hits if col == "mnemosyne" else [])
    return calls


def _mn_hit(content, mid=None, score=0.9):
    p = {"mnemosyne_id": mid} if mid else {}
    return {"collection": "mnemosyne", "point_id": None, "score": score,
            "importance": 0.9, "fused": score * 0.9, "content": content, "payload": p}


def test_sa_seeds_come_only_from_mnemosyne_hits_with_ids(hook):
    calls = _sa_setup(hook, [_mn_hit("seeded", mid="m-1", score=0.91),
                             _mn_hit("no id here")],
                      [{"content": "linked memory", "activation": 0.4, "importance": 0.6}])
    with mock.patch.dict(os.environ, {"MNEMOSYNE_DB": "/tmp/db.sqlite"}), \
            mock.patch("random.random", return_value=1.0):
        _, out = run_main(hook, _prompt())
    assert calls[0]["seed_ids"] == ["m-1"]
    assert calls[0]["seed_scores"] == {"m-1": 0.91}
    assert calls[0]["max_results"] == 2
    assert calls[0]["db_path"] == "/tmp/db.sqlite"
    assert "[mnemosyne-sa|0.40] linked memory" in context_of(out)


def test_sa_skipped_when_no_seed_ids(hook):
    calls = _sa_setup(hook, [_mn_hit("no id")], [{"content": "x", "activation": 0.9}])
    with mock.patch("random.random", return_value=1.0):
        run_main(hook, _prompt())
    assert calls == []


def test_sa_disabled_by_flag(hook):
    calls = _sa_setup(hook, [_mn_hit("seeded", mid="m-1")],
                      [{"content": "x", "activation": 0.9}])
    hook.SA_ENABLED = False
    with mock.patch("random.random", return_value=1.0):
        _, out = run_main(hook, _prompt())
    assert calls == []
    assert "mnemosyne-sa" not in context_of(out)


def test_sa_results_discarded_when_it_overruns_the_budget(hook):
    """The budget is checked AFTER the call: slowness is paid for, then the
    work is thrown away. It bounds pollution, not latency."""
    _sa_setup(hook, [_mn_hit("seeded", mid="m-1")],
              [{"content": "linked memory", "activation": 0.4}], delay=0.06)
    with mock.patch("random.random", return_value=1.0):
        _, out = run_main(hook, _prompt())
    assert "linked memory" not in context_of(out)


def test_sa_exception_is_swallowed(hook):
    _sa_setup(hook, [_mn_hit("seeded", mid="m-1")], [])

    def boom(**kw):
        raise RuntimeError("sqlite gone")

    hook._SA_MODULE.run_spreading_activation = boom
    with mock.patch("random.random", return_value=1.0):
        _, out = run_main(hook, _prompt())
    assert "seeded" in context_of(out)


def test_sa_hits_are_appended_after_selection_so_a_full_pool_hides_them(hook):
    """BUG: SA results are appended after MMR has already chosen RECALL_TOP_K
    hits, and _format_results stops at RECALL_TOP_K lines. Whenever the vector
    search already returned K usable hits -- the normal case -- the spreading
    activation work is executed and then never shown."""
    pool = [_mn_hit("first finding", mid="m-1", score=0.9),
            _mn_hit("second finding", score=0.89),
            _mn_hit("third finding", score=0.88)]
    calls = _sa_setup(hook, pool,
                      [{"content": "associatively linked memory", "activation": 0.99}])
    with mock.patch("random.random", return_value=1.0):
        _, out = run_main(hook, _prompt())
    ctx = context_of(out)
    assert calls, "SA ran"
    assert "associatively linked memory" not in ctx
    assert ctx.count("[mnemosyne|") == 3


# ---------------------------------------------------------------------------
# main() -- subagent path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload_extra,session_id,env", [
    ({"task_id": "task-subagent-3"}, None, {}),
    ({"task_id": "TASK-SUBAGENT-3"}, None, {}),
    ({}, "sess-subagent-9", {}),
    ({}, "plain-session", {"HERMES_SUBAGENT": "1"}),
])
def test_subagent_detection_sources(hook, payload_extra, session_id, env):
    hook._embed = lambda t: [0.1]
    hook._load_rules_summary = lambda: ""
    seen = []
    hook._search_collection = lambda *a, **k: seen.append((a, k)) or []
    p = _prompt(**payload_extra)
    if session_id:
        p["session_id"] = session_id
    _, out = run_main(hook, p, env=env)
    # lightweight path: exactly one collection, capped at 2 results
    assert len(seen) == 1
    assert seen[0][0][0] == "loci_memory"
    assert seen[0][0][2:] == ("text", "numeric_confidence", True)
    assert seen[0][1] == {"top_k": 2}
    assert "grounding unavailable in subagent context" in context_of(out)


def test_main_session_path_when_task_id_is_not_a_subagent(hook):
    hook._embed = lambda t: [0.1]
    hook._load_rules_summary = lambda: ""
    seen = []
    hook._search_collection = lambda col, *a, **k: seen.append(col) or []
    run_main(hook, {"hook_event_name": "UserPromptSubmit",
                    "session_id": "abc-123",
                    "extra": {"user_message": "what happened", "task_id": "task-1"}})
    assert sorted(seen) == ["loci_memory", "loci_sessions", "mnemosyne"]


def test_subagent_warning_when_qdrant_url_unset_skips_embedding(hook):
    hook.QDRANT_URL = ""
    called = []
    hook._embed = lambda t: called.append(t)
    hook._beam_fallback = lambda q: called.append(q)
    hook._load_rules_summary = lambda: ""
    _, out = run_main(hook, _prompt(task_id="subagent-1"))
    ctx = context_of(out)
    assert called == []          # no embed, no beam fallback at all
    assert ctx.startswith("[WARNING: memory grounding unavailable in subagent context")
    assert "Qdrant unreachable" in ctx
    assert ctx.endswith(hook.GROUNDING_DIRECTIVE)


def test_subagent_empty_results_report_qdrant_unreachable(hook):
    """BUG: 'Qdrant unreachable' is emitted whenever the search returns nothing,
    including a perfectly healthy search with no matches above MIN_SCORE."""
    hook._embed = lambda t: [0.1]
    hook._search_collection = lambda *a, **k: []
    hook._load_rules_summary = lambda: ""
    _, out = run_main(hook, _prompt(task_id="subagent-1"))
    assert "Qdrant unreachable" in context_of(out)


def test_subagent_falls_back_to_beam_when_embedding_fails(hook):
    hook._embed = lambda t: None
    hook._beam_fallback = lambda q: [{"collection": "mnemosyne_beam", "score": 0.6,
                                      "importance": 0.5, "fused": 0.3,
                                      "content": "beam says hi"}]
    hook._load_rules_summary = lambda: ""
    _, out = run_main(hook, _prompt(task_id="subagent-1"))
    ctx = context_of(out)
    assert "[mnemosyne-beam|0.60] beam says hi" in ctx
    # used_fallback is hard-coded False on this path -> label is wrong
    assert "Qdrant collections" in ctx and "BeamMemory fallback" not in ctx


def test_subagent_filters_importance_and_caps_at_two(hook):
    hits = [{"collection": "loci_memory", "point_id": None, "score": 0.9 - i / 100,
             "importance": 0.9, "fused": 0.8 - i / 100, "content": f"finding {i}",
             "payload": {}} for i in range(4)]
    hits.append({"collection": "loci_memory", "point_id": None, "score": 0.99,
                 "importance": 0.1, "fused": 0.099, "content": "unimportant",
                 "payload": {}})
    hook._embed = lambda t: [0.1]
    hook._search_collection = lambda *a, **k: hits
    hook._load_rules_summary = lambda: ""
    _, out = run_main(hook, _prompt(task_id="subagent-1"))
    ctx = context_of(out)
    assert "unimportant" not in ctx
    assert ctx.count("[loci-memory|") == 2
    assert "finding 0" in ctx and "finding 1" in ctx


def test_subagent_appends_rules_summary(hook):
    hook._embed = lambda t: None
    hook._beam_fallback = lambda q: []
    hook._load_rules_summary = lambda: "ACTIVE RULES FILES:\n[rules/r.md]: H"
    _, out = run_main(hook, _prompt(task_id="subagent-1"))
    assert context_of(out).endswith("\n\nACTIVE RULES FILES:\n[rules/r.md]: H")


def test_subagent_does_not_deposit_pheromone_or_run_sa(hook):
    deposits, sa_calls = [], []
    hook._pheromone_deposit = lambda *a: deposits.append(a)
    hook.SA_ENABLED = True
    hook._SA_MODULE = types.SimpleNamespace(
        run_spreading_activation=lambda **kw: sa_calls.append(kw) or [])
    hook._embed = lambda t: [0.1]
    hook._search_collection = lambda *a, **k: [
        {"collection": "loci_memory", "point_id": "p1", "score": 0.9, "importance": 0.9,
         "fused": 0.81, "content": "x", "payload": {"mnemosyne_id": "m1"}}]
    hook._load_rules_summary = lambda: ""
    run_main(hook, _prompt(task_id="subagent-1"))
    assert deposits == [] and sa_calls == []


# ---------------------------------------------------------------------------
# output contract
# ---------------------------------------------------------------------------

def test_output_is_a_single_json_line_on_stdout(hook):
    hook._embed = lambda t: None
    hook._beam_fallback = lambda q: []
    hook._load_rules_summary = lambda: ""
    _, out = run_main(hook, _prompt())
    assert out.endswith("\n") and out.count("\n") == 1
    assert list(json.loads(out)) == ["context"]
    assert isinstance(json.loads(out)["context"], str)


# ---------------------------------------------------------------------------
# Claude Code payload shape
# Claude Code sends a top-level "prompt"; Hermes nested it under extra.user_message.
# ---------------------------------------------------------------------------

def _dark_hook(hook):
    hook._embed = lambda t: None
    hook._beam_fallback = lambda q: []
    hook._load_rules_summary = lambda: ""


@pytest.mark.parametrize("event", ["UserPromptSubmit", "SubagentStart"])
def test_main_grounds_claude_code_top_level_prompt(hook, event):
    _dark_hook(hook)
    code, out = run_main(hook, {"hook_event_name": event,
                                "session_id": "s1",
                                "prompt": "why did retention delete the index"})
    assert code is None, "hook exited without emitting on a real Claude Code payload"
    assert "WARNING: memory grounding UNAVAILABLE" in context_of(out)


def test_main_still_grounds_legacy_hermes_shape(hook):
    _dark_hook(hook)
    code, out = run_main(hook, {"hook_event_name": "pre_llm_call",
                                "extra": {"user_message": "a real question"}})
    assert code is None
    assert "WARNING: memory grounding UNAVAILABLE" in context_of(out)


def test_main_prefers_top_level_prompt_over_extra(hook):
    seen = []
    hook._embed = lambda t: seen.append(t) or None
    hook._beam_fallback = lambda q: []
    hook._load_rules_summary = lambda: ""
    run_main(hook, {"hook_event_name": "UserPromptSubmit",
                    "prompt": "the real prompt",
                    "extra": {"user_message": "stale hermes text"}})
    assert seen and "the real prompt" in seen[0]


def test_main_still_exits_when_no_text_anywhere(hook):
    _dark_hook(hook)
    code, out = run_main(hook, {"hook_event_name": "UserPromptSubmit"})
    assert code == 0 and out == ""


# ---------------------------------------------------------------------------
# _embed_base_url
# ---------------------------------------------------------------------------

def test_embed_base_url_appends_v1_to_bare_ollama_host(hook):
    with mock.patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://h:11434"}, clear=True):
        assert hook._embed_base_url() == "http://h:11434/v1"


def test_embed_base_url_falls_back_to_mnemosyne_var(hook):
    # What the Hermes profile actually sets — already a full /v1 endpoint.
    with mock.patch.dict(os.environ,
                         {"MNEMOSYNE_EMBEDDING_API_URL": "http://h:11434/v1"},
                         clear=True):
        assert hook._embed_base_url() == "http://h:11434/v1"


def test_embed_base_url_none_when_unset(hook):
    with mock.patch.dict(os.environ, {}, clear=True):
        assert hook._embed_base_url() is None


# ---------------------------------------------------------------------------
# fan-out degradation
# Swallowed per-collection failures report a clean "no matches"; total failure must degrade to Beam.
# ---------------------------------------------------------------------------

def test_fanout_falls_back_to_beam_when_every_collection_fails(hook):
    hook._embed = lambda t: [0.1]
    hook._load_rules_summary = lambda: ""
    hook._beam_fallback = lambda q: [{"collection": "mnemosyne_beam", "score": 0.9,
                                      "importance": 0.9, "fused": 0.81,
                                      "content": "beam recalled this"}]
    calls = fake_urlopen(hook, OSError("qdrant down"))
    try:
        code, out = run_main(hook, {"hook_event_name": "UserPromptSubmit",
                                    "prompt": "a real question about the index"})
    finally:
        calls.stop()
    assert code is None
    assert "beam recalled this" in context_of(out)


def test_fanout_keeps_hits_when_only_some_collections_fail(hook):
    hook._embed = lambda t: [0.1]
    hook._load_rules_summary = lambda: ""
    hook._beam_fallback = lambda q: [{"collection": "mnemosyne_beam", "score": 0.9,
                                      "importance": 0.9, "fused": 0.81,
                                      "content": "beam should NOT be used"}]
    good = {"collection": "loci_memory", "point_id": None, "score": 0.9,
            "importance": 0.9, "fused": 0.81, "content": "qdrant recalled this",
            "payload": {}}
    real = hook._search_collection
    hook._search_collection = (
        lambda col, *a, **k: [good] if col == "loci_memory" else _raise(hook, col))
    try:
        code, out = run_main(hook, {"hook_event_name": "UserPromptSubmit",
                                    "prompt": "a real question about the index"})
    finally:
        hook._search_collection = real
    ctx = context_of(out)
    assert "qdrant recalled this" in ctx
    assert "beam should NOT be used" not in ctx


def _raise(hook, col):
    raise hook._SearchFailed(col)

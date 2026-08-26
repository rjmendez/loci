"""Characterization tests for scripts/hooks/session_end_sync.py.

This hook fires at the end of every turn and pushes a single Qdrant point
describing the current session.  Its whole design is "fail open": every
outbound dependency (sqlite state.db, Ollama, Qdrant, the Loci findings
file) is allowed to be missing or broken, and the hook must still exit 0
without a traceback.

These tests pin the behaviour AS IT IS TODAY, including several things
that are arguably wrong.  Those are called out with `BUG:` comments and
are asserted *as-is* on purpose -- they are the safety net, not the spec.

Nothing here touches the network.  urllib.request.urlopen is patched, or
the module-level helper that calls it is patched.  The module reads ~10
constants from the environment at *import* time, so every test loads a
fresh copy of the module through `load_hook()` under a controlled
environment.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import pathlib
import sqlite3
import urllib.error
from unittest import mock

import pytest

HOOK_PATH = pathlib.Path(__file__).resolve().parent.parent / "session_end_sync.py"

# HOME points somewhere that does not exist so any ~ default the module falls
# back to is inert rather than touching the real user's dotfiles.
BASE_ENV = {
    "HOME": "/nonexistent-home-for-session-end-sync-tests",
    "QDRANT_URL": "http://qdrant.invalid:6333",
    "QDRANT_API_KEY": "test-key",
    "OLLAMA_BASE_URL": "http://ollama.invalid:11434",
    "MNEMOSYNE_EMBEDDING_DIM": "4",
    "HERMES_AGENT_ID": "agent-7",
    "HERMES_PROFILE": "prof-x",
}


def load_hook(env: dict | None = None):
    """Exec a fresh copy of the hook under a fully controlled environment."""
    e = {k: v for k, v in {**BASE_ENV, **(env or {})}.items() if v is not None}
    with mock.patch.dict(os.environ, e, clear=True):
        spec = importlib.util.spec_from_file_location("_session_end_sync_uut", HOOK_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def paths(tmp_path):
    return {
        "db": str(tmp_path / "state.db"),
        "cache": str(tmp_path / "synccache"),
        "loci": str(tmp_path / "investigations"),
    }


@pytest.fixture
def hook(paths):
    """Fresh module wired to tmp paths; tests mutate its globals freely."""
    return load_hook({
        "HERMES_STATE_DB": paths["db"],
        "HERMES_SYNC_CACHE": paths["cache"],
        "LOCI_INVESTIGATIONS_DIR": paths["loci"],
    })


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_db(path, session=None, messages=()):
    """Build a state.db with just the columns the hook's SQL selects."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT, "
        "started_at, source TEXT, model TEXT)"
    )
    conn.execute(
        "CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, timestamp)"
    )
    if session is not None:
        conn.execute(
            "INSERT INTO sessions (id, title, started_at, source, model) "
            "VALUES (?,?,?,?,?)",
            (session.get("id", "s1"), session.get("title"),
             session.get("started_at", 1700000000.0),
             session.get("source"), session.get("model")),
        )
    for m in messages:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            (m.get("session_id", "s1"), m["role"], m["content"], m.get("ts", 0)),
        )
    conn.commit()
    conn.close()
    return path


class Resp:
    """Minimal stand-in for what urlopen returns (used as a context manager)."""

    def __init__(self, obj=None, raw=None):
        self._raw = raw if raw is not None else json.dumps(obj).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def patch_urlopen(hook, responder):
    """Record every urlopen call; `responder` may return a Resp or an Exception."""
    calls = []

    def _open(req, timeout=None):
        calls.append({
            "url": req.full_url,
            "method": req.get_method(),
            "timeout": timeout,
            "body": json.loads(req.data.decode()) if req.data else None,
            "headers": dict(req.header_items()),
        })
        out = responder(req) if callable(responder) else responder
        if isinstance(out, Exception):
            raise out
        return out

    p = mock.patch.object(hook.urllib.request, "urlopen", _open)
    p.start()
    return calls, p


def run_main(hook, payload):
    """Drive main() with `payload` on stdin. Returns the SystemExit code or None."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    with mock.patch.object(hook.sys, "stdin", io.StringIO(raw)):
        try:
            hook.main()
        except SystemExit as e:
            return e.code
    return None


# ---------------------------------------------------------------------------
# import-time configuration
# ---------------------------------------------------------------------------

def test_ollama_url_is_none_when_base_unset():
    """OLLAMA is None -- not "" -- when OLLAMA_BASE_URL is absent or empty."""
    h = load_hook({"OLLAMA_BASE_URL": None})
    assert h.OLLAMA is None
    h2 = load_hook({"OLLAMA_BASE_URL": ""})
    assert h2.OLLAMA is None


def test_ollama_url_appends_openai_style_path():
    h = load_hook({"OLLAMA_BASE_URL": "http://x:11434"})
    assert h.OLLAMA == "http://x:11434/v1/embeddings"


def test_qdrant_is_none_when_unset_but_key_defaults_to_empty_string():
    h = load_hook({"QDRANT_URL": None, "QDRANT_API_KEY": None})
    assert h.QDRANT is None
    assert h.QDRANT_KEY == ""


def test_fixed_constants():
    h = load_hook()
    assert h.COLLECTION == "hermes_sessions"
    assert h.MAX_CHARS == 4000
    assert h.EMBED_DIM == 4          # from MNEMOSYNE_EMBEDDING_DIM in BASE_ENV
    assert load_hook({"MNEMOSYNE_EMBEDDING_DIM": None}).EMBED_DIM == 768
    assert load_hook({"MNEMOSYNE_EMBEDDING_MODEL": None}).EMBED_MODEL == "nomic-embed-text"


def test_state_db_and_cache_are_user_expanded():
    h = load_hook({"HERMES_STATE_DB": None, "HERMES_SYNC_CACHE": None})
    assert h.STATE_DB == "/nonexistent-home-for-session-end-sync-tests/.hermes/state.db"
    assert h.CACHE_DIR == (
        "/nonexistent-home-for-session-end-sync-tests/.hermes/.session_sync_cache"
    )


def test_agent_identity_constants_default_to_empty():
    h = load_hook({"HERMES_AGENT_ID": None, "HERMES_PROFILE": None,
                   "HERMES_ACTIVE_INVESTIGATION": None})
    assert (h.AGENT_ID, h.PROFILE, h.ACTIVE_INV) == ("", "", "")


# ---------------------------------------------------------------------------
# stable_id
# ---------------------------------------------------------------------------

def test_stable_id_is_first_15_hex_of_sha256(hook):
    s = "session-abc"
    expected = int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)
    assert hook.stable_id(s) == expected
    # 15 hex nibbles == 60 bits, so it always fits in Qdrant's u64 id space.
    assert 0 <= hook.stable_id(s) < 2 ** 60


def test_stable_id_is_deterministic_and_input_sensitive(hook):
    assert hook.stable_id("a") == hook.stable_id("a")
    assert hook.stable_id("a") != hook.stable_id("b")
    assert hook.stable_id("") == int(
        hashlib.sha256(b"").hexdigest()[:15], 16)


# ---------------------------------------------------------------------------
# read_stdin_session_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ('{"session_id": "abc"}', "abc"),
    ('{"session_id": "abc", "other": 1}', "abc"),
    ("", ""),                       # empty stdin
    ("   \n\t ", ""),               # whitespace only
    ("not json", ""),               # unparseable
    ('{"session_id": null}', ""),   # explicit null
    ('{"session_id": ""}', ""),     # empty string is falsy -> ""
    ("{}", ""),                     # key missing
    ('["session_id"]', ""),         # list has no .get -> swallowed
    ('"session_id"', ""),           # bare string has no .get -> swallowed
    ("null", ""),                   # None has no .get -> swallowed
])
def test_read_stdin_session_id(hook, raw, expected):
    with mock.patch.object(hook.sys, "stdin", io.StringIO(raw)):
        assert hook.read_stdin_session_id() == expected


def test_read_stdin_session_id_rejects_non_strings(hook):
    """A numeric session_id used to be passed through as an int and then used in
    a SQL lookup and a path join."""
    with mock.patch.object(hook.sys, "stdin", io.StringIO('{"session_id": 123}')):
        assert hook.read_stdin_session_id() == ""


def test_read_stdin_payload_returns_whole_object(hook):
    raw = '{"session_id": "s1", "transcript_path": "/tmp/t.jsonl"}'
    with mock.patch.object(hook.sys, "stdin", io.StringIO(raw)):
        got = hook.read_stdin_payload()
    assert got == {"session_id": "s1", "transcript_path": "/tmp/t.jsonl"}


@pytest.mark.parametrize("raw", ["", "   ", "not json", "[1,2]", '"a string"'])
def test_read_stdin_payload_returns_empty_dict_on_junk(hook, raw):
    with mock.patch.object(hook.sys, "stdin", io.StringIO(raw)):
        assert hook.read_stdin_payload() == {}


# ---------------------------------------------------------------------------
# get_session_content
# ---------------------------------------------------------------------------

LONG = "x" * 40  # comfortably over the length(content) > 20 filter


def test_get_session_content_missing_db_returns_none(hook):
    assert not os.path.exists(hook.STATE_DB)
    assert hook.get_session_content("s1") is None


def test_get_session_content_unknown_session_returns_none(hook):
    make_db(hook.STATE_DB, session={"id": "other"}, messages=[])
    assert hook.get_session_content("s1") is None


def test_get_session_content_session_without_messages_returns_none(hook):
    """A real session row with zero qualifying messages is indistinguishable
    from a missing session: both are None."""
    make_db(hook.STATE_DB, session={"id": "s1"}, messages=[])
    assert hook.get_session_content("s1") is None


def test_get_session_content_happy_path_shape(hook):
    make_db(hook.STATE_DB,
            session={"id": "s1", "title": "T", "started_at": 1700000000.0,
                     "source": "web", "model": "opus"},
            messages=[{"role": "user", "content": LONG, "ts": 1},
                      {"role": "assistant", "content": LONG, "ts": 2}])
    got = hook.get_session_content("s1")
    assert set(got) == {"title", "started_at", "source", "model",
                        "content", "msg_count"}
    assert got["title"] == "T"
    assert got["source"] == "web"
    assert got["model"] == "opus"
    assert got["msg_count"] == 2
    assert got["started_at"] == "2023-11-14T22:13:20+00:00"
    assert got["content"] == f"USER: {LONG}\n\nASSISTANT: {LONG}"


def test_get_session_content_orders_by_timestamp_not_insertion(hook):
    make_db(hook.STATE_DB, session={"id": "s1"},
            messages=[{"role": "user", "content": "B" * 40, "ts": 9},
                      {"role": "assistant", "content": "A" * 40, "ts": 1}])
    content = hook.get_session_content("s1")["content"]
    assert content.startswith("ASSISTANT: AAA")
    assert content.endswith("B" * 40)


@pytest.mark.parametrize("content,kept", [
    ("y" * 20, False),   # length 20 is NOT > 20 -> dropped
    ("y" * 21, True),    # 21 is the first kept length
])
def test_get_session_content_length_filter_boundary(hook, content, kept):
    make_db(hook.STATE_DB, session={"id": "s1"},
            messages=[{"role": "user", "content": content, "ts": 1}])
    got = hook.get_session_content("s1")
    if kept:
        assert got["msg_count"] == 1
    else:
        assert got is None


def test_get_session_content_filters_roles_and_nulls(hook):
    make_db(hook.STATE_DB, session={"id": "s1"}, messages=[
        {"role": "system", "content": LONG, "ts": 1},
        {"role": "tool", "content": LONG, "ts": 2},
        {"role": "user", "content": None, "ts": 3},
        {"role": "user", "content": "short", "ts": 4},
        {"role": "assistant", "content": LONG, "ts": 5},
    ])
    got = hook.get_session_content("s1")
    # msg_count reflects the *filtered* set, not the true message total.
    assert got["msg_count"] == 1
    assert got["content"] == f"ASSISTANT: {LONG}"


def test_get_session_content_ignores_other_sessions_messages(hook):
    make_db(hook.STATE_DB, session={"id": "s1"}, messages=[
        {"session_id": "s2", "role": "user", "content": LONG, "ts": 1},
        {"session_id": "s1", "role": "user", "content": "z" * 30, "ts": 2},
    ])
    got = hook.get_session_content("s1")
    assert got["msg_count"] == 1
    assert got["content"] == "USER: " + "z" * 30


def test_get_session_content_strips_message_whitespace_and_uppercases_role(hook):
    make_db(hook.STATE_DB, session={"id": "s1"},
            messages=[{"role": "user", "content": "  \n" + LONG + "  \n", "ts": 1}])
    assert hook.get_session_content("s1")["content"] == f"USER: {LONG}"


def test_get_session_content_null_columns_get_defaults(hook):
    make_db(hook.STATE_DB,
            session={"id": "s1", "title": None, "source": None, "model": None},
            messages=[{"role": "user", "content": LONG, "ts": 1}])
    got = hook.get_session_content("s1")
    assert got["title"] == ""
    assert got["model"] == ""
    assert got["source"] == "cli"          # source is the only one with a default


def test_get_session_content_empty_source_also_becomes_cli(hook):
    make_db(hook.STATE_DB, session={"id": "s1", "source": ""},
            messages=[{"role": "user", "content": LONG, "ts": 1}])
    assert hook.get_session_content("s1")["source"] == "cli"


@pytest.mark.parametrize("raw,expected", [
    (1700000000.0, "2023-11-14T22:13:20+00:00"),
    ("1700000000", "2023-11-14T22:13:20+00:00"),   # numeric strings coerce
    ("2023-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),  # non-numeric passes through
    (None, "None"),                                # str(None) -- not empty!
])
def test_get_session_content_started_at_coercion(hook, raw, expected):
    make_db(hook.STATE_DB, session={"id": "s1", "started_at": raw},
            messages=[{"role": "user", "content": LONG, "ts": 1}])
    assert hook.get_session_content("s1")["started_at"] == expected


def test_get_session_content_swallows_broken_db(hook):
    """A file that is not a database (or lacks the tables) yields None, not a raise."""
    pathlib.Path(hook.STATE_DB).write_text("this is not sqlite")
    assert hook.get_session_content("s1") is None


def test_get_session_content_swallows_missing_tables(hook):
    conn = sqlite3.connect(hook.STATE_DB)
    conn.execute("CREATE TABLE unrelated (x)")
    conn.commit()
    conn.close()
    assert hook.get_session_content("s1") is None


# ---- the rolling MAX_CHARS window --------------------------------------------

def test_content_window_keeps_the_tail_and_truncates_the_oldest_line(hook):
    """Window is built from the newest message backwards; the oldest surviving
    line is head-truncated to whatever budget is left."""
    hook.MAX_CHARS = 80
    make_db(hook.STATE_DB, session={"id": "s1"}, messages=[
        {"role": "user", "content": "A" * 24, "ts": 1},
        {"role": "user", "content": "B" * 24, "ts": 2},
        {"role": "user", "content": "C" * 24, "ts": 3},
    ])
    got = hook.get_session_content("s1")
    l1, l2, l3 = ("USER: " + c * 24 for c in "ABC")
    assert got["content"] == "\n\n".join([l1[:20], l2, l3])
    # msg_count still counts every qualifying message, even the truncated one.
    assert got["msg_count"] == 3


def test_content_window_can_exceed_max_chars_because_separators_are_uncounted(hook):
    """BUG: the "\\n\\n" joiners are not charged against MAX_CHARS, so the
    embedded text is up to 2*(n-1) chars longer than the stated budget."""
    hook.MAX_CHARS = 80
    make_db(hook.STATE_DB, session={"id": "s1"}, messages=[
        {"role": "user", "content": c * 24, "ts": i}
        for i, c in enumerate("ABC")
    ])
    assert len(hook.get_session_content("s1")["content"]) == 84


def test_content_window_emits_a_leading_empty_line_on_exact_fit(hook):
    """BUG: when the budget is exactly consumed, the next line is sliced to
    line[:0] == "" and still joined in, so content starts with "\\n\\n"."""
    hook.MAX_CHARS = 60          # exactly two 30-char lines
    make_db(hook.STATE_DB, session={"id": "s1"}, messages=[
        {"role": "user", "content": c * 24, "ts": i}
        for i, c in enumerate("ABC")
    ])
    content = hook.get_session_content("s1")["content"]
    assert content.startswith("\n\n")
    assert content == "\n\n" + "USER: " + "B" * 24 + "\n\n" + "USER: " + "C" * 24


def test_single_oversized_message_is_head_truncated_to_max_chars(hook):
    hook.MAX_CHARS = 50
    make_db(hook.STATE_DB, session={"id": "s1"},
            messages=[{"role": "user", "content": "Q" * 500, "ts": 1}])
    content = hook.get_session_content("s1")["content"]
    assert content == ("USER: " + "Q" * 500)[:50]
    assert len(content) == 50


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def test_cache_path_is_md5_prefix_under_cache_dir(hook):
    p = hook.cache_path("sess-1")
    assert os.path.dirname(p) == hook.CACHE_DIR
    assert os.path.basename(p) == hashlib.md5(b"sess-1").hexdigest()[:12]
    assert len(os.path.basename(p)) == 12


def test_cache_path_creates_the_directory_as_a_side_effect(hook):
    assert not os.path.isdir(hook.CACHE_DIR)
    hook.cache_path("sess-1")
    assert os.path.isdir(hook.CACHE_DIR)


def test_cached_msg_count_missing_is_minus_one_and_still_makes_the_dir(hook):
    assert hook.cached_msg_count("nope") == -1
    assert os.path.isdir(hook.CACHE_DIR)


def test_cache_roundtrip(hook):
    hook.write_cache("s1", 12)
    assert hook.cached_msg_count("s1") == 12
    hook.write_cache("s1", 13)
    assert hook.cached_msg_count("s1") == 13


def test_cached_msg_count_tolerates_surrounding_whitespace(hook):
    pathlib.Path(hook.cache_path("s1")).write_text("  7\n")
    assert hook.cached_msg_count("s1") == 7


@pytest.mark.parametrize("junk", ["", "abc", "1.5", "1 2"])
def test_cached_msg_count_garbage_is_minus_one(hook, junk):
    pathlib.Path(hook.cache_path("s1")).write_text(junk)
    assert hook.cached_msg_count("s1") == -1


def test_write_cache_swallows_an_unusable_cache_dir(hook, tmp_path):
    """write_cache() calls cache_path() *inside* its try, so it is fail-open."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")
    hook.CACHE_DIR = str(blocker)
    hook.write_cache("s1", 5)          # must not raise
    assert blocker.read_text() == "i am a file"


def test_cached_msg_count_raises_when_the_cache_dir_cannot_be_created(hook, tmp_path):
    """BUG: cache_path() is called *outside* cached_msg_count()'s try block, so
    an unusable HERMES_SYNC_CACHE turns the "return -1 on anything" contract
    into an uncaught exception."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")
    hook.CACHE_DIR = str(blocker)
    with pytest.raises(FileExistsError):
        hook.cached_msg_count("s1")


def test_cached_msg_count_raises_on_a_missing_parent_directory(hook):
    hook.CACHE_DIR = "/proc/no-such-parent/cache"
    with pytest.raises(OSError):
        hook.cached_msg_count("s1")


def test_cache_keys_are_per_session(hook):
    hook.write_cache("a", 1)
    hook.write_cache("b", 2)
    assert (hook.cached_msg_count("a"), hook.cached_msg_count("b")) == (1, 2)


# ---------------------------------------------------------------------------
# _embed_headers
# ---------------------------------------------------------------------------

def test_embed_headers_default_is_json_only():
    h = load_hook({"EMBED_API_KEY": None})
    assert h._embed_headers() == {"Content-Type": "application/json"}


def test_embed_headers_default_header_is_bearer_authorization():
    h = load_hook({"EMBED_API_KEY": "sk-1"})
    assert h._embed_headers() == {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-1",
    }


def test_embed_headers_authorization_match_is_case_insensitive_and_normalises():
    """Any casing of "authorization" matches, and the emitted key is always the
    canonical "Authorization" -- the configured spelling is discarded."""
    h = load_hook({"EMBED_API_KEY": "sk-1", "EMBED_API_KEY_HEADER": "AUTHORIZATION"})
    assert h._embed_headers() == {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-1",
    }


def test_embed_headers_custom_header_gets_the_raw_key():
    h = load_hook({"EMBED_API_KEY": "sk-1", "EMBED_API_KEY_HEADER": "X-Api-Key"})
    assert h._embed_headers()["X-Api-Key"] == "sk-1"   # no "Bearer " prefix


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

def test_embed_posts_openai_shape_and_returns_first_embedding(hook):
    calls, p = patch_urlopen(hook, Resp({"data": [{"embedding": [1.0, 2.0, 3.0, 4.0]}]}))
    try:
        assert hook.embed("hello") == [1.0, 2.0, 3.0, 4.0]
    finally:
        p.stop()
    assert len(calls) == 1
    c = calls[0]
    assert c["url"] == "http://ollama.invalid:11434/v1/embeddings"
    assert c["method"] == "POST"
    assert c["timeout"] == 8
    assert c["body"] == {"model": "nomic-embed-text", "input": "hello"}


def test_embed_propagates_transport_errors(hook):
    _, p = patch_urlopen(hook, OSError("connection refused"))
    try:
        with pytest.raises(OSError):
            hook.embed("hi")
    finally:
        p.stop()


def test_embed_propagates_malformed_response(hook):
    _, p = patch_urlopen(hook, Resp({"data": []}))
    try:
        with pytest.raises(IndexError):
            hook.embed("hi")
    finally:
        p.stop()


def test_embed_raises_when_ollama_base_url_is_unset(paths):
    """With OLLAMA None the Request constructor blows up -- main() catches it."""
    h = load_hook({"OLLAMA_BASE_URL": None, "HERMES_STATE_DB": paths["db"]})
    with pytest.raises(Exception):
        h.embed("hi")


# ---------------------------------------------------------------------------
# qdrant_upsert
# ---------------------------------------------------------------------------

def test_qdrant_upsert_request_shape(hook):
    calls, p = patch_urlopen(hook, Resp({"status": "ok", "result": {}}))
    try:
        assert hook.qdrant_upsert(42, [0.1, 0.2], {"a": 1}) is True
    finally:
        p.stop()
    c = calls[0]
    assert c["url"] == "http://qdrant.invalid:6333/collections/hermes_sessions/points"
    assert c["method"] == "PUT"
    assert c["timeout"] == 5
    assert c["body"] == {"points": [
        {"id": 42, "vector": {"dense": [0.1, 0.2]}, "payload": {"a": 1}}
    ]}
    # header_items() title-cases names
    assert c["headers"]["Api-key"] == "test-key"
    assert c["headers"]["Content-type"] == "application/json"


@pytest.mark.parametrize("status,expected", [
    ("ok", True),
    ("acknowledged", False),
    (None, False),
])
def test_qdrant_upsert_only_ok_counts_as_success(hook, status, expected):
    body = {"status": status} if status is not None else {"result": {}}
    _, p = patch_urlopen(hook, Resp(body))
    try:
        assert hook.qdrant_upsert(1, [0.0], {}) is expected
    finally:
        p.stop()


def test_qdrant_upsert_propagates_transport_errors(hook):
    _, p = patch_urlopen(hook, urllib.error.URLError("down"))
    try:
        with pytest.raises(urllib.error.URLError):
            hook.qdrant_upsert(1, [0.0], {})
    finally:
        p.stop()


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------

def http_error(code):
    return urllib.error.HTTPError("u", code, "msg", {}, None)


def test_ensure_collection_noop_without_qdrant_url(paths):
    h = load_hook({"QDRANT_URL": None, "HERMES_STATE_DB": paths["db"]})
    calls, p = patch_urlopen(h, Resp({}))
    try:
        assert h.ensure_collection() is None
    finally:
        p.stop()
    assert calls == []


def test_ensure_collection_creates_on_404(hook):
    def responder(req):
        return http_error(404) if req.get_method() == "GET" else Resp({})

    calls, p = patch_urlopen(hook, responder)
    try:
        hook.ensure_collection()
    finally:
        p.stop()
    assert [c["method"] for c in calls] == ["GET", "PUT"]
    assert calls[0]["url"] == "http://qdrant.invalid:6333/collections/hermes_sessions"
    assert calls[0]["timeout"] == 5
    assert calls[1]["timeout"] == 10
    assert calls[1]["body"] == {
        "vectors": {"dense": {"size": 4, "distance": "Cosine"}},
        "hnsw_config": {"m": 32, "ef_construct": 200, "on_disk": False},
        "quantization_config": {
            "scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}
        },
    }


def test_ensure_collection_patches_when_it_already_exists(hook):
    calls, p = patch_urlopen(hook, Resp({}))
    try:
        hook.ensure_collection()
    finally:
        p.stop()
    assert [c["method"] for c in calls] == ["GET", "PATCH"]
    # the PATCH body drops on_disk and never mentions vectors
    assert calls[1]["body"] == {
        "hnsw_config": {"m": 32, "ef_construct": 200},
        "quantization_config": {
            "scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}
        },
    }


def test_ensure_collection_treats_any_non_404_http_error_as_existing(hook):
    def responder(req):
        return http_error(500) if req.get_method() == "GET" else Resp({})

    calls, p = patch_urlopen(hook, responder)
    try:
        hook.ensure_collection()
    finally:
        p.stop()
    assert [c["method"] for c in calls] == ["GET", "PATCH"]


def test_ensure_collection_bails_out_when_qdrant_is_unreachable(hook):
    calls, p = patch_urlopen(hook, urllib.error.URLError("no route"))
    try:
        assert hook.ensure_collection() is None
    finally:
        p.stop()
    assert [c["method"] for c in calls] == ["GET"]   # no create, no patch


def test_ensure_collection_create_failure_is_logged_not_raised(hook, capsys):
    def responder(req):
        if req.get_method() == "GET":
            return http_error(404)
        raise OSError("boom")

    _, p = patch_urlopen(hook, responder)
    try:
        hook.ensure_collection()
    finally:
        p.stop()
    err = capsys.readouterr().err
    assert "[session_end_sync] create collection failed: boom" in err


def test_ensure_collection_patch_failure_is_completely_silent(hook, capsys):
    def responder(req):
        if req.get_method() == "GET":
            return Resp({})
        raise OSError("boom")

    _, p = patch_urlopen(hook, responder)
    try:
        hook.ensure_collection()
    finally:
        p.stop()
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err == ""


# ---------------------------------------------------------------------------
# _check_wiring_obligations
# ---------------------------------------------------------------------------

def write_findings(hook, inv, records):
    d = pathlib.Path(hook.os.environ["LOCI_INVESTIGATIONS_DIR"]) / inv
    d.mkdir(parents=True, exist_ok=True)
    f = d / "findings.jsonl"
    f.write_text("\n".join(
        r if isinstance(r, str) else json.dumps(r) for r in records
    ) + "\n")
    return f


@pytest.fixture
def inv_env(hook, paths, monkeypatch):
    monkeypatch.setenv("LOCI_INVESTIGATIONS_DIR", paths["loci"])
    return hook


def gap(fid, text="t", tags=("wiring_obligation",), rt="gap"):
    return {"id": fid, "text": text, "tags": list(tags), "record_type": rt}


def test_wiring_missing_findings_file_returns_empty_and_leaves_payload_alone(inv_env):
    payload = {"a": 1}
    assert inv_env._check_wiring_obligations("nope", payload) == ""
    assert payload == {"a": 1}


def test_wiring_counts_gaps_and_mutates_payload(inv_env):
    write_findings(inv_env, "inv1", [gap("f1", "alpha"), gap("f2", "beta")])
    payload = {}
    note = inv_env._check_wiring_obligations("inv1", payload)
    assert note == " | ⚠ UNRESOLVED WIRING OBLIGATIONS: 2"
    assert payload["unresolved_wiring_obligations"] == 2
    # samples are in reverse file order (newest first)
    assert payload["unresolved_wiring_obligation_samples"] == ["beta", "alpha"]


def test_wiring_requires_both_the_tag_and_record_type_gap(inv_env):
    write_findings(inv_env, "inv1", [
        gap("f1", tags=["other"]),                 # wrong tag
        gap("f2", rt="resolution"),                # wrong record_type
        gap("f3", tags=[]),                        # no tags
        {"id": "f4", "text": "x"},                 # no tags/record_type at all
    ])
    payload = {}
    assert inv_env._check_wiring_obligations("inv1", payload) == ""
    assert payload == {}


def test_wiring_dedups_by_id_last_record_in_file_wins(inv_env):
    """Later lines override earlier ones: a gap that was later resolved drops out."""
    write_findings(inv_env, "inv1", [
        gap("f1", "still open"),
        gap("f1", "now resolved", rt="resolution"),
    ])
    payload = {}
    assert inv_env._check_wiring_obligations("inv1", payload) == ""
    assert payload == {}


def test_wiring_dedup_keeps_only_one_entry_per_id(inv_env):
    write_findings(inv_env, "inv1", [gap("f1", "old"), gap("f1", "new")])
    payload = {}
    inv_env._check_wiring_obligations("inv1", payload)
    assert payload["unresolved_wiring_obligations"] == 1
    assert payload["unresolved_wiring_obligation_samples"] == ["new"]


def test_wiring_records_without_an_id_collapse_into_one(inv_env):
    """BUG: a missing id defaults to "", so the empty string is the dedup key
    and every id-less finding after the first is silently dropped."""
    write_findings(inv_env, "inv1", [
        {"text": "no-id one", "tags": ["wiring_obligation"], "record_type": "gap"},
        {"text": "no-id two", "tags": ["wiring_obligation"], "record_type": "gap"},
        {"text": "no-id three", "tags": ["wiring_obligation"], "record_type": "gap"},
    ])
    payload = {}
    inv_env._check_wiring_obligations("inv1", payload)
    assert payload["unresolved_wiring_obligations"] == 1
    assert payload["unresolved_wiring_obligation_samples"] == ["no-id three"]


def test_wiring_skips_unparseable_lines(inv_env):
    write_findings(inv_env, "inv1", ["{not json", "", gap("f1", "ok")])
    payload = {}
    assert inv_env._check_wiring_obligations("inv1", payload).endswith(": 1")
    assert payload["unresolved_wiring_obligation_samples"] == ["ok"]


def test_wiring_sample_text_is_truncated_to_120_chars(inv_env):
    write_findings(inv_env, "inv1", [gap("f1", "L" * 500)])
    payload = {}
    inv_env._check_wiring_obligations("inv1", payload)
    assert payload["unresolved_wiring_obligation_samples"] == ["L" * 120]


def test_wiring_falls_back_to_the_id_when_text_is_absent(inv_env):
    write_findings(inv_env, "inv1", [
        {"id": "finding-9", "tags": ["wiring_obligation"], "record_type": "gap"}
    ])
    payload = {}
    inv_env._check_wiring_obligations("inv1", payload)
    assert payload["unresolved_wiring_obligation_samples"] == ["finding-9"]


def test_wiring_samples_capped_at_three_but_count_is_total(inv_env):
    write_findings(inv_env, "inv1", [gap(f"f{i}", f"t{i}") for i in range(7)])
    payload = {}
    note = inv_env._check_wiring_obligations("inv1", payload)
    assert note.endswith(": 7")
    assert payload["unresolved_wiring_obligations"] == 7
    assert payload["unresolved_wiring_obligation_samples"] == ["t6", "t5", "t4"]


def test_wiring_explicit_null_text_aborts_the_whole_scan(inv_env):
    """BUG: `rec.get("text", fid)` returns None for `"text": null`; the
    resulting TypeError is swallowed by the outer handler, so *every*
    obligation in the file is discarded, not just the bad record."""
    write_findings(inv_env, "inv1", [
        gap("f1", "real obligation"),
        {"id": "f2", "text": None, "tags": ["wiring_obligation"], "record_type": "gap"},
    ])
    payload = {}
    assert inv_env._check_wiring_obligations("inv1", payload) == ""
    assert payload == {}


def test_wiring_reads_the_investigations_dir_at_call_time(hook, tmp_path, monkeypatch):
    """Unlike the module's other config, LOCI_INVESTIGATIONS_DIR is resolved
    per call, so it can be changed after import."""
    other = tmp_path / "elsewhere"
    (other / "inv1").mkdir(parents=True)
    (other / "inv1" / "findings.jsonl").write_text(json.dumps(gap("f1", "here")) + "\n")
    monkeypatch.setenv("LOCI_INVESTIGATIONS_DIR", str(other))
    payload = {}
    assert hook._check_wiring_obligations("inv1", payload).endswith(": 1")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

@pytest.fixture
def wired(hook):
    """main() with every outbound call replaced by a recorder."""
    rec = {"ensure": 0, "embed": [], "upsert": [], "embed_result": [0.1] * 4,
           "upsert_result": True}

    def _ensure():
        rec["ensure"] += 1

    def _embed(text):
        rec["embed"].append(text)
        r = rec["embed_result"]
        if isinstance(r, Exception):
            raise r
        return r

    def _upsert(pid, vec, payload):
        rec["upsert"].append({"id": pid, "vector": vec, "payload": payload})
        r = rec["upsert_result"]
        if isinstance(r, Exception):
            raise r
        return r

    hook.ensure_collection = _ensure
    hook.embed = _embed
    hook.qdrant_upsert = _upsert
    hook._rec = rec
    return hook


def seed_session(hook, n=2, **session):
    make_db(hook.STATE_DB, session={"id": "s1", "title": "T", "source": "web",
                                    "model": "opus", **session},
            messages=[{"role": "user", "content": f"{i}" + "m" * 40, "ts": i}
                      for i in range(n)])


def test_main_exits_zero_without_a_session_id(wired, capsys):
    assert run_main(wired, {}) == 0
    assert wired._rec["ensure"] == 0
    assert capsys.readouterr().out == ""


def test_main_exits_zero_when_the_session_is_not_in_the_db(wired):
    seed_session(wired)
    assert run_main(wired, {"session_id": "unknown"}) == 0
    assert wired._rec["ensure"] == 0          # bails before touching Qdrant


def test_main_happy_path(wired, capsys):
    seed_session(wired, n=2)
    assert run_main(wired, {"session_id": "s1"}) is None   # no SystemExit on success
    assert wired._rec["ensure"] == 1
    assert wired._rec["embed"] == ["USER: 0" + "m" * 40 + "\n\nUSER: 1" + "m" * 40]

    up = wired._rec["upsert"][0]
    assert up["id"] == wired.stable_id("s1")
    assert up["vector"] == [0.1] * 4
    assert list(up["payload"]) == [
        "session_id", "title", "started_at", "source", "model",
        "msg_count", "content_preview", "last_synced", "agent_id", "profile",
    ]
    assert up["payload"]["session_id"] == "s1"
    assert up["payload"]["msg_count"] == 2
    assert up["payload"]["agent_id"] == "agent-7"
    assert up["payload"]["profile"] == "prof-x"
    assert up["payload"]["content_preview"] == wired._rec["embed"][0][:500]
    assert up["payload"]["last_synced"].endswith("+00:00")
    # success is recorded in the cache and announced on stdout
    assert wired.cached_msg_count("s1") == 2
    out = capsys.readouterr().out
    assert out.startswith("[session_end_sync] synced s1 (2 msgs) in ")
    assert out.rstrip().endswith("s")


def test_main_content_preview_is_capped_at_500_chars(wired):
    make_db(wired.STATE_DB, session={"id": "s1"},
            messages=[{"role": "user", "content": "z" * 2000, "ts": 1}])
    run_main(wired, {"session_id": "s1"})
    payload = wired._rec["upsert"][0]["payload"]
    assert len(payload["content_preview"]) == 500
    assert len(wired._rec["embed"][0]) == 2006   # the full 4000-budget content


def test_main_session_id_is_truncated_to_20_chars_in_the_log_line(wired, capsys):
    sid = "s" * 60
    make_db(wired.STATE_DB, session={"id": sid},
            messages=[{"session_id": sid, "role": "user", "content": LONG, "ts": 1}])
    run_main(wired, {"session_id": sid})
    assert "synced " + "s" * 20 + " (1 msgs)" in capsys.readouterr().out


def test_main_fast_path_skips_embed_when_msg_count_is_unchanged(wired, capsys):
    seed_session(wired, n=2)
    wired.write_cache("s1", 2)
    assert run_main(wired, {"session_id": "s1"}) == 0
    assert wired._rec["embed"] == []
    assert wired._rec["upsert"] == []
    assert capsys.readouterr().out == ""


def test_fast_path_still_calls_ensure_collection_first(wired):
    """BUG (latency): the docstring promises an immediate exit when nothing
    changed, but ensure_collection() -- one or two Qdrant round-trips -- runs
    *before* the cache comparison."""
    seed_session(wired, n=2)
    wired.write_cache("s1", 2)
    run_main(wired, {"session_id": "s1"})
    assert wired._rec["ensure"] == 1


def test_main_crashes_if_the_cache_dir_is_unusable(wired, tmp_path):
    """BUG (fail-open violated): every other degraded path exits 0, but an
    unusable HERMES_SYNC_CACHE lets cache_path()'s makedirs error escape
    main(), so the hook dies with a traceback and a non-zero status."""
    seed_session(wired)
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    wired.CACHE_DIR = str(blocker)
    with pytest.raises(FileExistsError):
        run_main(wired, {"session_id": "s1"})
    assert wired._rec["embed"] == []       # it dies before embedding


def test_main_resyncs_when_the_message_count_changed(wired):
    seed_session(wired, n=2)
    wired.write_cache("s1", 1)
    run_main(wired, {"session_id": "s1"})
    assert len(wired._rec["upsert"]) == 1
    assert wired.cached_msg_count("s1") == 2


def test_main_resyncs_when_the_count_went_down(wired):
    """The guard is equality, not "greater than" -- a shrunken session re-syncs."""
    seed_session(wired, n=2)
    wired.write_cache("s1", 99)
    run_main(wired, {"session_id": "s1"})
    assert len(wired._rec["upsert"]) == 1


def test_main_embed_failure_is_non_fatal(wired, capsys):
    seed_session(wired)
    wired._rec["embed_result"] = RuntimeError("ollama down")
    assert run_main(wired, {"session_id": "s1"}) == 0
    cap = capsys.readouterr()
    assert cap.err.strip() == "[session_end_sync] embed error: ollama down"
    assert cap.out == ""
    assert wired._rec["upsert"] == []
    assert wired.cached_msg_count("s1") == -1     # cache untouched -> retried later


def test_main_rejects_a_wrong_length_vector(wired, capsys):
    seed_session(wired)
    wired._rec["embed_result"] = [0.1, 0.2]       # EMBED_DIM is 4 here
    assert run_main(wired, {"session_id": "s1"}) == 0
    assert "unexpected vector dim 2, expected 4" in capsys.readouterr().err
    assert wired._rec["upsert"] == []


def test_main_upsert_exception_is_non_fatal(wired, capsys):
    seed_session(wired)
    wired._rec["upsert_result"] = urllib.error.URLError("qdrant down")
    assert run_main(wired, {"session_id": "s1"}) == 0
    cap = capsys.readouterr()
    assert "[session_end_sync] upsert error:" in cap.err
    assert cap.out == ""
    assert wired.cached_msg_count("s1") == -1


def test_main_upsert_returning_false_is_silent(wired, capsys):
    """A non-"ok" Qdrant status produces no log line at all and leaves the
    cache alone; the only observable difference from success is the silence."""
    seed_session(wired)
    wired._rec["upsert_result"] = False
    assert run_main(wired, {"session_id": "s1"}) is None
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err == ""
    assert wired.cached_msg_count("s1") == -1


def test_main_appends_the_wiring_note_when_an_investigation_is_active(wired, paths,
                                                                     capsys):
    wired.ACTIVE_INV = "inv1"
    d = pathlib.Path(paths["loci"]) / "inv1"
    d.mkdir(parents=True)
    (d / "findings.jsonl").write_text(json.dumps(gap("f1", "wire me")) + "\n")
    seed_session(wired)
    with mock.patch.dict(os.environ, {"LOCI_INVESTIGATIONS_DIR": paths["loci"]}):
        run_main(wired, {"session_id": "s1"})
    assert "| ⚠ UNRESOLVED WIRING OBLIGATIONS: 1" in capsys.readouterr().out


def test_wiring_payload_fields_never_reach_qdrant(wired, paths, capsys):
    """BUG: _check_wiring_obligations() mutates `payload`, but it is called
    *after* qdrant_upsert() has already serialised and sent it.  The
    unresolved_wiring_obligations fields are therefore never persisted --
    they only ever show up in the local stdout line."""
    wired.ACTIVE_INV = "inv1"
    d = pathlib.Path(paths["loci"]) / "inv1"
    d.mkdir(parents=True)
    (d / "findings.jsonl").write_text(json.dumps(gap("f1", "wire me")) + "\n")
    seed_session(wired)

    sent = {}

    orig = wired.qdrant_upsert

    def spy(pid, vec, payload):
        sent.update(payload)          # snapshot at send time
        return orig(pid, vec, payload)

    wired.qdrant_upsert = spy
    with mock.patch.dict(os.environ, {"LOCI_INVESTIGATIONS_DIR": paths["loci"]}):
        run_main(wired, {"session_id": "s1"})

    assert "unresolved_wiring_obligations" not in sent
    # ...yet the very same dict object has been mutated after the fact
    assert wired._rec["upsert"][0]["payload"]["unresolved_wiring_obligations"] == 1


def test_main_skips_the_wiring_check_without_an_active_investigation(wired, capsys):
    assert wired.ACTIVE_INV == ""
    seed_session(wired)
    run_main(wired, {"session_id": "s1"})
    out = capsys.readouterr().out
    assert "UNRESOLVED" not in out
    assert "unresolved_wiring_obligations" not in wired._rec["upsert"][0]["payload"]


def test_main_wiring_check_failure_is_swallowed(wired, capsys):
    wired.ACTIVE_INV = "inv1"
    wired._check_wiring_obligations = mock.Mock(side_effect=RuntimeError("nope"))
    seed_session(wired)
    assert run_main(wired, {"session_id": "s1"}) is None
    assert "[session_end_sync] synced s1 (2 msgs)" in capsys.readouterr().out


def test_main_end_to_end_over_a_patched_urlopen(hook, capsys):
    """No function-level stubs: exercise ensure_collection + embed + upsert
    through the real request-building code."""
    seed_session(hook, n=1)

    def responder(req):
        url, method = req.full_url, req.get_method()
        if url.endswith("/v1/embeddings"):
            return Resp({"data": [{"embedding": [0.0, 1.0, 2.0, 3.0]}]})
        if url.endswith("/collections/hermes_sessions") and method == "GET":
            raise http_error(404)
        if url.endswith("/collections/hermes_sessions") and method == "PUT":
            return Resp({"result": True, "status": "ok"})
        if url.endswith("/points"):
            return Resp({"result": {}, "status": "ok"})
        raise AssertionError(f"unexpected {method} {url}")

    calls, p = patch_urlopen(hook, responder)
    try:
        assert run_main(hook, {"session_id": "s1"}) is None
    finally:
        p.stop()

    urls = [(c["method"], c["url"]) for c in calls]
    assert urls == [
        ("GET", "http://qdrant.invalid:6333/collections/hermes_sessions"),
        ("PUT", "http://qdrant.invalid:6333/collections/hermes_sessions"),
        ("POST", "http://ollama.invalid:11434/v1/embeddings"),
        ("PUT", "http://qdrant.invalid:6333/collections/hermes_sessions/points"),
    ]
    assert calls[-1]["body"]["points"][0]["id"] == hook.stable_id("s1")
    assert calls[-1]["body"]["points"][0]["vector"] == {"dense": [0.0, 1.0, 2.0, 3.0]}
    assert hook.cached_msg_count("s1") == 1
    assert "[session_end_sync] synced s1 (1 msgs)" in capsys.readouterr().out


def test_main_survives_a_missing_state_db_entirely(wired, capsys):
    assert not os.path.exists(wired.STATE_DB)
    assert run_main(wired, {"session_id": "s1"}) == 0
    assert capsys.readouterr().out == ""


def test_main_ignores_extra_stdin_keys_and_bad_json(wired):
    assert run_main(wired, "garbage not json") == 0
    assert run_main(wired, {"sessionId": "s1"}) == 0     # camelCase is not read


# ---------------------------------------------------------------------------
# _embeddings_url
#
# The module read only OLLAMA_BASE_URL. The Hermes profile sets
# MNEMOSYNE_EMBEDDING_API_URL (already a full /v1 endpoint), so the sync had no
# embeddings endpoint at all under the profile's own environment.
# ---------------------------------------------------------------------------

def test_embeddings_url_from_bare_ollama_host():
    mod = load_hook({"OLLAMA_BASE_URL": "http://h:11434"})
    assert mod.OLLAMA == "http://h:11434/v1/embeddings"


def test_embeddings_url_falls_back_to_mnemosyne_var():
    mod = load_hook({"OLLAMA_BASE_URL": None,
                     "MNEMOSYNE_EMBEDDING_API_URL": "http://h:11434/v1"})
    assert mod.OLLAMA == "http://h:11434/v1/embeddings"


def test_embeddings_url_prefers_ollama_base_url():
    mod = load_hook({"OLLAMA_BASE_URL": "http://a:1",
                     "MNEMOSYNE_EMBEDDING_API_URL": "http://b:2/v1"})
    assert mod.OLLAMA == "http://a:1/v1/embeddings"


def test_embeddings_url_none_when_neither_set():
    mod = load_hook({"OLLAMA_BASE_URL": None})
    assert mod.OLLAMA is None


# ---------------------------------------------------------------------------
# transcript_session_content
#
# STATE_DB is Hermes' session store: 765 rows, all Hermes-format ids, none
# written since 2026-06-19. A Claude Code session id never resolves there, so
# get_session_content returned None and the Stop hook exited 0 without ever
# syncing a single Claude Code session. The Stop payload carries
# transcript_path, which is the record that actually exists.
# ---------------------------------------------------------------------------

def _write_transcript(tmp_path, records):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(p)


def _msg(kind, text, ts="2026-08-26T12:00:00Z", model=None):
    if kind == "user":
        body = {"role": "user", "content": text}
    else:
        body = {"role": "assistant", "content": [{"type": "text", "text": text}]}
        if model:
            body["model"] = model
    return {"type": kind, "timestamp": ts, "message": body}


def test_transcript_content_builds_a_session(hook, tmp_path):
    path = _write_transcript(tmp_path, [
        {"type": "ai-title", "aiTitle": "Public Loci code"},
        _msg("user", "a" * 40, ts="2026-08-24T17:40:37.466Z"),
        _msg("assistant", "b" * 40, model="claude-opus-5"),
    ])
    got = hook.transcript_session_content(path, "sess-uuid")
    assert got["title"] == "Public Loci code"
    assert got["model"] == "claude-opus-5"
    assert got["source"] == "claude-code"
    assert got["started_at"] == "2026-08-24T17:40:37.466Z"
    assert got["msg_count"] == 2
    assert "USER: " + "a" * 40 in got["content"]
    assert "ASSISTANT: " + "b" * 40 in got["content"]


def test_transcript_content_extracts_only_text_blocks(hook, tmp_path):
    rec = {"type": "assistant", "timestamp": "t", "message": {
        "role": "assistant", "model": "m", "content": [
            {"type": "thinking", "thinking": "hidden reasoning here padded out"},
            {"type": "text", "text": "visible answer padded out to pass length"},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        ]}}
    got = hook.transcript_session_content(_write_transcript(tmp_path, [rec]), "s")
    assert "visible answer" in got["content"]
    assert "hidden reasoning" not in got["content"]
    assert "tool_use" not in got["content"]


def test_transcript_content_skips_unparseable_lines(hook, tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("{not json\n" + json.dumps(_msg("user", "c" * 40)) + "\n\n")
    got = hook.transcript_session_content(str(p), "s")
    assert got["msg_count"] == 1


def test_transcript_content_caps_at_max_chars(hook, tmp_path):
    hook.MAX_CHARS = 200
    records = [_msg("user", str(i) * 120) for i in range(40)]
    got = hook.transcript_session_content(_write_transcript(tmp_path, records), "s")
    assert got["msg_count"] == 40
    assert len(got["content"]) <= 200
    # The tail is what gets embedded, so the newest message must be present.
    assert "39" in got["content"]


def test_transcript_content_none_for_missing_or_empty(hook, tmp_path):
    assert hook.transcript_session_content("", "s") is None
    assert hook.transcript_session_content(str(tmp_path / "nope.jsonl"), "s") is None
    assert hook.transcript_session_content(_write_transcript(tmp_path, []), "s") is None
    # Records with no usable text are not a session.
    short = _write_transcript(tmp_path, [_msg("user", "hi")])
    assert hook.transcript_session_content(short, "s") is None

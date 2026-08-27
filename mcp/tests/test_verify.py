"""Tests for verify — adversarial finding verification. Generation is stubbed; no live Ollama."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import verify as V  # noqa: E402


# --- stub gen_fn factories: match the shared contract gen_fn(prompt, *, fmt, max_tokens) ---

def _ok(text):
    def _fn(prompt, *, fmt=None, max_tokens=256):
        assert fmt == "json"                # verify_finding must request JSON format
        assert isinstance(prompt, str) and "CLAIM:" in prompt
        return {"text": text, "ok": True}
    return _fn


def _not_ok(prompt, *, fmt=None, max_tokens=256):
    return {"text": "irrelevant", "ok": False}   # caller should fall back


def _raises(prompt, *, fmt=None, max_tokens=256):
    raise RuntimeError("boom")


_REFUTED = ('{"verdict": "refuted", "refutation": "The base already sends 1005 at 0.1Hz, '
            'so the claim is false.", "confidence": 0.9}')

_CONFIRMED = ('{"verdict": "confirmed", "refutation": "Tried to break it; the log lines '
              'directly support the claim.", "confidence": 0.8}')

_CONFIRMED_PROSE = (
    "Sure, here's my analysis:\n"
    "```json\n"
    '{"verdict": "confirmed", "refutation": "cannot refute", "confidence": 0.7}\n'
    "```\n"
    "Hope that helps."
)

_CONFIRMED_REASONING = (
    '{"verdict": "confirmed", "reasoning": "Line 2 assigns x=1 and returns it, so the claim '
    'that it returns 1 holds.", "refutation": "cannot refute", "confidence": 0.8}'
)


def test_refutation_yields_refuted():
    r = V.verify_finding("The base omits RTCM 1005", gen_fn=_ok(_REFUTED))
    assert r["verdict"] == "refuted"
    assert r["degraded"] is False
    assert "1005" in r["refutation"]
    assert r["confidence"] == 0.9


def test_confirmation_yields_confirmed():
    r = V.verify_finding("The log shows a decode", context="AcGg rx_ok=3", gen_fn=_ok(_CONFIRMED))
    assert r["verdict"] == "confirmed"
    assert r["degraded"] is False
    assert 0.0 <= r["confidence"] <= 1.0


def test_confirmed_embedded_in_prose_with_fences():
    r = V.verify_finding("claim", gen_fn=_ok(_CONFIRMED_PROSE))
    assert r["verdict"] == "confirmed"
    assert r["degraded"] is False


def test_gen_not_ok_fails_open_to_uncertain():
    r = V.verify_finding("some claim", gen_fn=_not_ok)
    assert r["verdict"] == "uncertain"
    assert r["degraded"] is True
    assert r["confidence"] == 0.0


def test_gen_error_fails_open_to_uncertain():
    r = V.verify_finding("some claim", gen_fn=_raises)
    assert r["verdict"] == "uncertain"
    assert r["degraded"] is True


def test_malformed_json_fails_open_to_uncertain():
    r = V.verify_finding("some claim", gen_fn=_ok('{"verdict": "refuted", oops'))
    assert r["verdict"] == "uncertain"
    assert r["degraded"] is True


def test_garbage_no_braces_fails_open():
    r = V.verify_finding("some claim", gen_fn=_ok("no json here at all"))
    assert r["verdict"] == "uncertain"
    assert r["degraded"] is True


def test_unknown_verdict_coerced_to_uncertain():
    # Model returns valid JSON but an out-of-set verdict -> skeptical default.
    r = V.verify_finding("c", gen_fn=_ok('{"verdict": "maybe", "confidence": 0.5}'))
    assert r["verdict"] == "uncertain"
    assert r["degraded"] is False        # parsed fine; just not a keep-worthy verdict


def test_missing_refutation_and_confidence_are_coerced():
    r = V.verify_finding("c", gen_fn=_ok('{"verdict": "confirmed"}'))
    assert r["verdict"] == "confirmed"
    assert r["refutation"] == ""         # missing -> empty string, not a crash
    assert r["confidence"] == 0.0        # missing -> cautious 0.0


def test_out_of_range_confidence_clamped():
    r = V.verify_finding("c", gen_fn=_ok('{"verdict": "refuted", "confidence": 5}'))
    assert r["confidence"] == 1.0        # clamped into [0,1]


def test_nonstring_confidence_defaults_to_zero():
    r = V.verify_finding("c", gen_fn=_ok('{"verdict": "confirmed", "confidence": "high"}'))
    assert r["confidence"] == 0.0


def test_empty_claim_short_circuits():
    r = V.verify_finding("   ", gen_fn=_ok(_CONFIRMED))
    assert r["verdict"] == "uncertain"
    assert r["degraded"] is True


def test_investigation_id_pulls_rag_context_fail_open():
    # rag_fn is injectable; a returned context should be woven into the prompt.
    captured = {}

    def _rag(query, *, limit=5):
        return {"context": "GROUNDING: the base emits 1005 every 10s"}

    def _gen(prompt, *, fmt=None, max_tokens=256):
        captured["prompt"] = prompt
        return {"text": _REFUTED, "ok": True}

    r = V.verify_finding("claim without context", investigation_id="inv-1",
                         gen_fn=_gen, rag_fn=_rag)
    assert r["verdict"] == "refuted"
    assert "GROUNDING: the base emits 1005" in captured["prompt"]


def test_rag_error_does_not_break_verification():
    def _rag(query, *, limit=5):
        raise RuntimeError("qdrant down")

    r = V.verify_finding("claim", investigation_id="inv-2",
                         gen_fn=_ok(_CONFIRMED), rag_fn=_rag)
    # RAG blew up but verification still proceeds ungrounded.
    assert r["verdict"] == "confirmed"


def test_explicit_context_skips_rag():
    def _rag(query, *, limit=5):
        raise AssertionError("rag_fn must not be called when context is provided")

    r = V.verify_finding("claim", context="explicit evidence here",
                         investigation_id="inv-3", gen_fn=_ok(_REFUTED), rag_fn=_rag)
    assert r["verdict"] == "refuted"


def test_code_ref_fetches_source_into_prompt():
    # A stubbed reader returns file text; the cited lines must land in the skeptic's prompt.
    captured = {}
    file_text = "def f():\n    x = 1\n    return x\n"

    def _reader(path):
        captured["path"] = path
        return file_text

    def _gen(prompt, *, fmt=None, max_tokens=256):
        captured["prompt"] = prompt
        return {"text": _CONFIRMED, "ok": True}

    r = V.verify_finding("f() returns 1", code_refs=["mymod.py:2-3"],
                         gen_fn=_gen, reader=_reader)
    assert r["verdict"] == "confirmed"
    assert captured["path"] == "mymod.py"
    # The fetched, line-numbered source is present (not just the prose claim).
    assert "2: " in captured["prompt"] and "x = 1" in captured["prompt"]
    assert "return x" in captured["prompt"]
    assert "mymod.py:2-3" in captured["prompt"]


def test_file_ref_in_context_is_auto_fetched():
    # A ref embedded in the context string is picked up without an explicit code_refs arg.
    captured = {}

    def _reader(path):
        return "alpha\nbeta\ngamma\n"

    def _gen(prompt, *, fmt=None, max_tokens=256):
        captured["prompt"] = prompt
        return {"text": _REFUTED, "ok": True}

    r = V.verify_finding("the second line is beta", context="see src/data.py:2",
                         gen_fn=_gen, reader=_reader)
    assert r["verdict"] == "refuted"
    assert "beta" in captured["prompt"]


def test_unreadable_ref_fails_open_and_still_verifies():
    def _reader(path):
        raise FileNotFoundError(path)

    r = V.verify_finding("claim", code_refs=["nope.py:1-3"],
                         gen_fn=_ok(_CONFIRMED), reader=_reader)
    # Reader blew up but verification proceeds ungrounded.
    assert r["verdict"] == "confirmed"


def _capture_fetch(refs, reader):
    """Exercise _fetch_code directly (bypasses ref-parsing) so we can assert the block."""
    return V._fetch_code(refs, reader)


def test_zero_line_ref_normalizes_to_line_one():
    # file.py:0 clamps to line 1 and the header must show the displayed line, not the raw 0.
    def _reader(path):
        return "alpha\nbeta\ngamma\n"

    block = _capture_fetch([("m.py", 0, 0)], _reader)
    assert block, "0-line ref should still emit a non-empty block"
    assert "--- m.py:1 ---" in block          # header from actual span, not ':0'
    assert "1: alpha" in block
    assert ":0" not in block                   # raw 0 never surfaces


def test_truncated_ref_header_reflects_displayed_span():
    # An oversized end must report the real clamped span, not the requested one.
    body = "".join(f"line{i}\n" for i in range(1, 201))   # 200 lines
    block = _capture_fetch([("big.py", 1, 999)], lambda p: body)
    cap = V._MAX_LINES_PER_REF
    assert f"--- big.py:1-{cap} ---" in block            # clamped to the cap, not 1-999
    assert "1-999" not in block
    # Exactly _MAX_LINES_PER_REF numbered lines are present.
    assert f"{cap}: line{cap}" in block
    assert f"{cap + 1}: line{cap + 1}" not in block


def test_ref_clamped_to_eof_header_reflects_last_line():
    # end past EOF (but within the line cap) clamps the header to the last real line.
    block = _capture_fetch([("m.py", 2, 50)], lambda p: "a\nb\nc\n")   # only 3 lines
    assert "--- m.py:2-3 ---" in block
    assert "3: c" in block


def test_reasoning_field_is_surfaced():
    r = V.verify_finding("f() returns 1", gen_fn=_ok(_CONFIRMED_REASONING))
    assert r["verdict"] == "confirmed"
    assert "Line 2 assigns x=1" in r["reasoning"]


def test_reasoning_falls_back_to_raw_text_when_absent():
    # No explicit reasoning field -> caller still gets the model's raw output to judge.
    r = V.verify_finding("c", gen_fn=_ok('{"verdict": "confirmed", "confidence": 0.6}'))
    assert r["verdict"] == "confirmed"
    assert '"verdict": "confirmed"' in r["reasoning"]


def test_degraded_coerces_nonstring_text_to_string():
    # _degraded is the single normalization point that keeps the all-strings return shape.
    def _not_ok_none_text(prompt, *, fmt=None, max_tokens=256):
        return {"ok": False, "text": None}

    r = V.verify_finding("some claim", gen_fn=_not_ok_none_text)
    assert r["verdict"] == "uncertain"
    assert r["degraded"] is True
    assert isinstance(r["reasoning"], str) and r["reasoning"] == ""
    assert isinstance(r["refutation"], str)


def test_degraded_direct_nonstring_args_coerced():
    d = V._degraded(refutation=None, reasoning=123)
    assert d["refutation"] == "" and d["reasoning"] == "123"
    assert isinstance(d["refutation"], str) and isinstance(d["reasoning"], str)


def test_claim_only_still_works_without_refs():
    # No code_refs, no file:line anywhere -> reader must never be consulted; behavior unchanged.
    def _reader(path):
        raise AssertionError("reader must not be called when there are no refs")

    r = V.verify_finding("The base omits RTCM 1005", gen_fn=_ok(_REFUTED), reader=_reader)
    assert r["verdict"] == "refuted"
    assert r["confidence"] == 0.9


# --- SECURITY: file refs from free-form text must not read arbitrary files ---

def test_parse_refs_drops_absolute_and_traversal_paths():
    # Absolute paths and '..' traversal parsed from free-form text must never become refs.
    refs = V._parse_refs("see /etc/passwd:1 and ../../secret.txt:2 but src/ok.py:3 is fine")
    paths = [p for (p, _s, _e) in refs]
    assert "/etc/passwd" not in paths
    assert not any(".." in p.split("/") for p in paths)
    assert "src/ok.py" in paths        # the legitimate repo-relative ref survives


def test_default_reader_rejects_absolute_path():
    # The default FS reader must refuse to read an absolute path (local file disclosure).
    assert V._lazy_read_file("/etc/passwd") == ""


def test_default_reader_rejects_traversal():
    assert V._lazy_read_file("../../../../../../etc/passwd") == ""


def test_default_reader_reads_repo_relative_file():
    # A legitimate repo-relative path resolves under the repo root and reads.
    text = V._lazy_read_file("mcp/verify.py")
    assert "_safe_resolve" in text


def test_default_reader_size_cap(monkeypatch):
    # An oversized read is capped, not slurped whole.
    monkeypatch.setattr(V, "_MAX_FILE_BYTES", 16)
    text = V._lazy_read_file("mcp/verify.py")
    assert 0 < len(text) <= 16


def test_default_reader_size_cap_is_byte_accurate(monkeypatch, tmp_path):
    # The cap is BYTES, not characters: text-mode f.read(n) caps characters and overruns.
    p = tmp_path / "multibyte.txt"
    p.write_text("é" * 100, encoding="utf-8")  # each 'é' is 2 UTF-8 bytes
    monkeypatch.setattr(V, "_MAX_FILE_BYTES", 10)
    # Patch the layer the reader actually calls: _ground_ref picks the checkout, the reader
    # only caps and decodes.
    monkeypatch.setattr(V, "_ground_ref", lambda path, want_hash=None: (str(p), ""))
    text = V._lazy_read_file("multibyte.txt")
    # Never decode more than the byte cap allowed (10 bytes -> at most 5 'é' chars).
    assert len(text.encode("utf-8")) <= 10
    assert 0 < len(text) <= 5


def test_absolute_code_ref_reads_nothing_end_to_end():
    # Passing an absolute path via code_refs with the default reader leaks no file content.
    fetched = {}

    def _gen(prompt, *, fmt=None, max_tokens=256):
        fetched["prompt"] = prompt
        return {"text": _CONFIRMED, "ok": True}

    r = V.verify_finding("claim", code_refs=["/etc/passwd:1-3"], gen_fn=_gen)
    assert r["verdict"] == "confirmed"          # still verifies (fail-open)
    assert "root:" not in fetched["prompt"]     # no /etc/passwd content in the prompt
    assert "(none)" in fetched["prompt"]        # code block was empty


# --- code_refs coercion: accept list or single string, ignore other types ---

def test_code_refs_single_string_is_accepted():
    # A bare string (not a list) must be treated as one ref, not split into characters.
    captured = {}

    def _reader(path):
        captured["path"] = path
        return "def f():\n    x = 1\n    return x\n"

    def _gen(prompt, *, fmt=None, max_tokens=256):
        captured["prompt"] = prompt
        return {"text": _CONFIRMED, "ok": True}

    r = V.verify_finding("f() returns 1", code_refs="mymod.py:2-3",
                         gen_fn=_gen, reader=_reader)
    assert r["verdict"] == "confirmed"
    assert captured["path"] == "mymod.py"       # whole path, not a single char
    assert "x = 1" in captured["prompt"]


def test_code_refs_nonlist_type_is_ignored_and_autodetect_survives():
    # A junk code_refs type must be ignored without killing the surrounding auto-detection.
    def _reader(path):
        return "alpha\nbeta\ngamma\n"

    captured = {}

    def _gen(prompt, *, fmt=None, max_tokens=256):
        captured["prompt"] = prompt
        return {"text": _REFUTED, "ok": True}

    r = V.verify_finding("second line is beta", context="see src/data.py:2",
                         code_refs=123, gen_fn=_gen, reader=_reader)
    assert r["verdict"] == "refuted"
    assert "beta" in captured["prompt"]         # auto-detected ref still fetched


def test_default_gen_fn_is_lazy_and_fails_open(monkeypatch):
    # With no llm_local importable, the lazy default must fail-open, not raise.
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "llm_local":
            raise ImportError("llm_local not importable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    r = V.verify_finding("some claim")   # no gen_fn -> lazy import path
    assert r["verdict"] == "uncertain"
    assert r["degraded"] is True


# --- MULTI-REPO code grounding: many checkouts of many repos, one right revision ---
#
# hermes_memory is a multi-repo corpus. Sandboxing every ref to _repo_root() rejected 100%
# of its stored code_refs (measured: 62 refs, 0 resolvable) even though the files exist on
# the host. The hazard in widening is that the host also carries many WORKTREES of the same
# repo, so the same relative path exists in several of them; picking the first would ground
# the skeptic on an arbitrary revision. These tests pin the rule that makes it safe:
# hash-match wins, unanimous content is fine, disagreement is REFUSED.

import hashlib  # noqa: E402
import pytest   # noqa: E402


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _mk_checkout(parent, name, files):
    root = parent / name
    (root / ".git").mkdir(parents=True)
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return root


@pytest.fixture
def hostlike(tmp_path, monkeypatch):
    """A fake host: two worktrees that DISAGREE on pkg/mod.py, two that AGREE on pkg/same.py,
    and a non-git directory that must never become a root."""
    parent = tmp_path / "home"
    parent.mkdir()
    _mk_checkout(parent, "repo-wt-a", {"pkg/mod.py": "A_ONE\nA_TWO\n", "pkg/same.py": "SAME\n"})
    _mk_checkout(parent, "repo-wt-b", {"pkg/mod.py": "B_ONE\nB_TWO\n", "pkg/same.py": "SAME\n"})
    _mk_checkout(parent, "repo-wt-c", {"pkg/only.py": "ONLY\n", "pkg/.env": "SECRET=1\n"})
    plain = parent / "not-a-repo"
    (plain / "pkg").mkdir(parents=True)
    (plain / "pkg" / "loose.py").write_text("LOOSE\n")
    # An empty primary repo root, so these tests never depend on this repo's own contents.
    empty = tmp_path / "primary"
    empty.mkdir()
    monkeypatch.setattr(V, "_repo_root", lambda: str(empty))
    monkeypatch.setenv("LOCI_CODE_ROOT_PARENTS", str(parent))
    monkeypatch.delenv("LOCI_CODE_ROOTS", raising=False)
    V._search_roots.cache_clear()
    yield parent
    V._search_roots.cache_clear()


def test_stored_hash_selects_the_matching_checkout(hostlike):
    # The stamped hash is the disambiguator: it names the exact revision, so the worktree
    # holding those bytes is read and the other is not.
    full, label = V._ground_ref("pkg/mod.py", _sha("B_ONE\nB_TWO\n"))
    assert full is not None
    assert full.startswith(os.path.realpath(str(hostlike / "repo-wt-b")) + os.sep)
    assert "exact revision" in label


def test_disagreeing_worktrees_are_refused_not_guessed(hostlike):
    # THE CORE SAFETY PROPERTY. Same relative path, two different contents, nothing to
    # choose between them -> refuse. A confident answer off the wrong revision is worse
    # than no grounding.
    assert V._ground_ref("pkg/mod.py") == (None, "")
    # A hash that matches NEITHER checkout is just as ambiguous.
    assert V._ground_ref("pkg/mod.py", _sha("something else\n")) == (None, "")


def test_unanimous_content_needs_no_hash(hostlike):
    # When every candidate holds byte-identical content the revision question has one
    # answer, so there is nothing to guess and the ref grounds.
    full, label = V._ground_ref("pkg/same.py")
    assert full is not None
    with open(full) as fh:
        assert fh.read() == "SAME\n"
    assert "identical in 2 checkouts" in label


def test_stale_finding_still_grounds_but_says_so(hostlike):
    # Unanimous content whose hash no longer matches: read it, but label it CHANGED so the
    # block never presents drifted source as the revision the finding was recorded against.
    full, label = V._ground_ref("pkg/same.py", _sha("OLD\n"))
    assert full is not None
    assert "CHANGED since the finding was recorded" in label


def test_non_git_directories_are_not_roots(hostlike):
    # Requiring a .git marker is what keeps this an allow-list rather than "all of $HOME".
    assert V._ground_ref("pkg/loose.py") == (None, "")


def test_search_roots_are_git_worktrees_only(hostlike):
    roots = V._search_roots()
    names = {os.path.basename(r) for r in roots}
    assert {"repo-wt-a", "repo-wt-b", "repo-wt-c"} <= names
    assert "not-a-repo" not in names


def test_explicit_roots_env_is_honored(tmp_path, monkeypatch):
    # An operator can name a checkout that is not a child of a scanned parent.
    root = _mk_checkout(tmp_path, "elsewhere", {"pkg/x.py": "X\n"})
    empty = tmp_path / "primary2"
    empty.mkdir()
    monkeypatch.setattr(V, "_repo_root", lambda: str(empty))
    monkeypatch.setenv("LOCI_CODE_ROOT_PARENTS", str(tmp_path / "nothing-here"))
    monkeypatch.setenv("LOCI_CODE_ROOTS", str(root))
    V._search_roots.cache_clear()
    try:
        full, _label = V._ground_ref("pkg/x.py")
        assert full == os.path.realpath(str(root / "pkg" / "x.py"))
    finally:
        V._search_roots.cache_clear()


# --- SECURITY: widening the roots must not widen what a ref may name ---

def test_multi_root_search_still_refuses_absolute_and_traversal(hostlike):
    assert V._ground_ref("/etc/passwd") == (None, "")
    assert V._ground_ref("../../../../etc/passwd") == (None, "")
    assert V._ground_ref("pkg/../../repo-wt-b/pkg/mod.py") == (None, "")
    assert V._lazy_read_file("/etc/passwd") == ""
    assert V._lazy_read_file("../../../../etc/passwd") == ""


def test_dot_prefixed_components_are_refused(hostlike):
    # 83 checkouts' worth of .env / .ssh / .git must not become readable just because the
    # search widened. Measured cost on the live corpus: 6 refs of 1836, all CI/config.
    assert (hostlike / "repo-wt-c" / "pkg" / ".env").is_file()   # it IS there
    assert V._ground_ref("pkg/.env") == (None, "")               # and still refused
    assert V._safe_resolve(".git/config") is None


def test_symlink_out_of_a_root_is_refused(hostlike, tmp_path):
    secret = tmp_path / "outside.txt"
    secret.write_text("TOP SECRET\n")
    link = hostlike / "repo-wt-c" / "pkg" / "escape.py"
    os.symlink(str(secret), str(link))
    V._search_roots.cache_clear()
    assert V._ground_ref("pkg/escape.py") == (None, "")
    assert V._lazy_read_file("pkg/escape.py") == ""


# --- end to end: a stored {path, hash} ref becomes real source in the skeptic's prompt ---

def test_stored_code_ref_dict_reaches_the_prompt(hostlike):
    captured = {}

    def _gen(prompt, *, fmt=None, max_tokens=256):
        captured["prompt"] = prompt
        return {"text": _REFUTED, "ok": True}

    r = V.verify_finding(
        "pkg/mod.py starts with B_ONE",
        code_refs=[{"path": "pkg/mod.py", "hash": _sha("B_ONE\nB_TWO\n")}],
        gen_fn=_gen,
    )
    assert r["verdict"] == "refuted"
    assert "B_ONE" in captured["prompt"]           # the right worktree's source
    assert "A_ONE" not in captured["prompt"]       # not the other one's
    assert "exact revision" in captured["prompt"]  # header names the basis


def test_ambiguous_stored_ref_contributes_no_code(hostlike):
    captured = {}

    def _gen(prompt, *, fmt=None, max_tokens=256):
        captured["prompt"] = prompt
        return {"text": _REFUTED, "ok": True}

    V.verify_finding("pkg/mod.py does something",
                     code_refs=[{"path": "pkg/mod.py"}], gen_fn=_gen)
    assert "A_ONE" not in captured["prompt"]
    assert "B_ONE" not in captured["prompt"]
    assert "(none)" in captured["prompt"]


def test_coerce_code_ref_hashes_ignores_junk():
    assert V._coerce_code_ref_hashes([{"path": "a.py", "hash": "h"}]) == {"a.py": "h"}
    assert V._coerce_code_ref_hashes([{"path": "a.py"}, "b.py:1", 7, None]) == {}
    assert V._coerce_code_ref_hashes("nope") == {}


def test_safe_resolve_rejects_before_containment_not_only_by_it(tmp_path):
    # Defense-in-depth: the abs / '..' / dot-component rejects must fire on their OWN.
    # The final containment check happens to catch the OBVIOUS forms (/etc/passwd lands
    # outside any narrow root), which masks their removal. These forms all resolve back
    # INSIDE the root, so only the explicit rejects can stop them.
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "in.py").write_text("IN\n")
    (root / ".env").write_text("SECRET=1\n")

    # Absolute, but pointing at a file that IS under the root: os.path.join drops the root
    # prefix entirely, so containment cannot see anything wrong.
    assert V._safe_resolve(str(root / "sub" / "in.py"), root=str(root)) is None
    # '..' that walks out and back in again: realpath lands under the root.
    assert V._safe_resolve("sub/../sub/in.py", root=str(root)) is None
    # A dot-component file that really is inside the root.
    assert V._safe_resolve(".env", root=str(root)) is None
    assert V._safe_resolve("", root=str(root)) is None
    assert V._safe_resolve(None, root=str(root)) is None
    # The legitimate ref still resolves, so the rejects are not just "refuse everything".
    assert V._safe_resolve("sub/in.py", root=str(root)) == os.path.realpath(
        str(root / "sub" / "in.py"))

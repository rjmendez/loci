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


def test_reasoning_field_is_surfaced():
    r = V.verify_finding("f() returns 1", gen_fn=_ok(_CONFIRMED_REASONING))
    assert r["verdict"] == "confirmed"
    assert "Line 2 assigns x=1" in r["reasoning"]


def test_reasoning_falls_back_to_raw_text_when_absent():
    # No explicit reasoning field -> caller still gets the model's raw output to judge.
    r = V.verify_finding("c", gen_fn=_ok('{"verdict": "confirmed", "confidence": 0.6}'))
    assert r["verdict"] == "confirmed"
    assert '"verdict": "confirmed"' in r["reasoning"]


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
    # A junk type (e.g. int) must be ignored without raising, and auto-detection from the
    # context must still work (the broad try must not be killed by list(code_refs)).
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

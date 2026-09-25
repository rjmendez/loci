"""Tests for adversarial — red-team / gap review over a finding set. Generation is stubbed; no live Ollama."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import adversarial as A  # noqa: E402


# --- stub gen_fn factories: match the shared contract gen_fn(prompt, *, fmt, max_tokens) ---

def _ok(text):
    """Stub that answers ``text`` and records every call in ``_fn.calls``.

    It never raises: adversarial wraps gen_fn in ``except Exception`` and would turn
    a raised assertion into a quiet degraded result. A call without fmt="json" is
    answered as a failed generation instead, so the result comes back degraded.
    """
    def _fn(prompt, *, fmt=None, max_tokens=256):
        _fn.calls.append({"prompt": prompt, "fmt": fmt, "max_tokens": max_tokens})
        if fmt != "json" or not isinstance(prompt, str):
            return {"text": "", "ok": False}
        return {"text": text, "ok": True}
    _fn.calls = []
    return _fn


def _not_ok(prompt, *, fmt=None, max_tokens=256):
    return {"text": "irrelevant", "ok": False}     # caller should treat as degraded


def _raises(prompt, *, fmt=None, max_tokens=256):
    raise RuntimeError("boom")


_RT = ('{"exploitable": true, "attack": "replay the cached JWT against /api/v1/plates", '
       '"preconditions": "read access to persist.img", "impact": "fleet-wide plate query", '
       '"confirm": "decode the JWT aud/scope and hit the endpoint"}')

_RT_SAFE = '{"exploitable": false, "attack": "", "preconditions": "", "impact": "", "confirm": ""}'

_GAPS = ('{"gaps": ["persist.img never parsed as a filesystem", "APKs not decompiled"], '
         '"next_probes": ["jadx the Flock APK", "test client_credentials grant"], '
         '"summary": "The analysis stopped at strings; the real surface is untouched."}')


def test_redteam_happy_path():
    out = A.adversarial_review(["cached bearer JWT in persist.img"], mode="redteam",
                              gen_fn=_ok(_RT))
    assert out["mode"] == "redteam"
    assert out["degraded"] is False
    assert len(out["results"]) == 1
    r = out["results"][0]
    assert r["exploitable"] is True
    assert "plates" in r["attack"]
    assert r["degraded"] is False


def test_redteam_multiple_and_partial_failure():
    # one finding parses, one backend-fails -> batch survives, per-item degraded flags set
    calls = {"n": 0}

    def _fn(prompt, *, fmt=None, max_tokens=256):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"text": _RT, "ok": True}
        return {"text": "", "ok": False}

    out = A.adversarial_review(["finding a", "finding b"], mode="redteam", gen_fn=_fn)
    assert len(out["results"]) == 2
    assert out["results"][0]["degraded"] is False
    assert out["results"][1]["degraded"] is True
    assert out["degraded"] is False          # at least one succeeded


def test_redteam_not_exploitable():
    out = A.adversarial_review(["cosmetic string in a log"], mode="redteam", gen_fn=_ok(_RT_SAFE))
    assert out["results"][0]["exploitable"] is False


def test_gaps_happy_path():
    out = A.adversarial_review(["found build.prop", "found an APK"], mode="gaps", gen_fn=_ok(_GAPS))
    assert out["mode"] == "gaps"
    assert out["degraded"] is False
    assert "persist.img never parsed as a filesystem" in out["gaps"]
    assert len(out["next_probes"]) == 2
    assert out["summary"].startswith("The analysis stopped")


def test_not_ok_is_degraded():
    rt = A.adversarial_review(["x"], mode="redteam", gen_fn=_not_ok)
    assert rt["degraded"] is True
    assert rt["results"][0]["degraded"] is True
    gp = A.adversarial_review(["x"], mode="gaps", gen_fn=_not_ok)
    assert gp["degraded"] is True
    assert gp["gaps"] == []


def test_gen_raises_is_fail_open():
    out = A.adversarial_review(["x"], mode="redteam", gen_fn=_raises)
    assert out["degraded"] is True
    assert out["results"][0]["degraded"] is True


def test_unparseable_output_is_degraded():
    out = A.adversarial_review(["x"], mode="redteam", gen_fn=_ok("not json at all"))
    assert out["results"][0]["degraded"] is True
    assert out["degraded"] is True


def test_gaps_salvages_narrated_text():
    # a chatty model that narrates instead of emitting JSON -> degraded, but the critique
    # is preserved in summary rather than discarded.
    prose = "The analysis never parsed persist.img and skipped the APK decompile entirely."
    out = A.adversarial_review(["a", "b"], mode="gaps", gen_fn=_ok(prose))
    assert out["degraded"] is True
    assert out["gaps"] == []
    assert "persist.img" in out["summary"]


def test_json_in_prose_is_extracted():
    fenced = "Sure:\n```json\n" + _GAPS + "\n```\nhope that helps"
    out = A.adversarial_review(["a"], mode="gaps", gen_fn=_ok(fenced))
    assert out["degraded"] is False
    assert len(out["gaps"]) == 2


def test_empty_findings_degraded_not_raised():
    assert A.adversarial_review([], mode="redteam")["degraded"] is True
    assert A.adversarial_review([], mode="gaps")["degraded"] is True
    assert A.adversarial_review(None, mode="redteam")["results"] == []


def test_unknown_mode_falls_back_to_redteam():
    fn = _ok(_RT)
    out = A.adversarial_review(["x"], mode="bogus", gen_fn=fn)
    assert (out["mode"], out["degraded"]) == ("redteam", False)
    assert [(r["finding"], r["exploitable"], r["degraded"]) for r in out["results"]] == [("x", True, False)]
    # one red-team prompt per finding, requesting JSON
    assert [(c["fmt"], c["max_tokens"]) for c in fn.calls] == [("json", 400)]


def test_findings_coercion():
    # dicts flatten to k=v text; a bare string is accepted; a huge list is capped
    fn = _ok(_RT)
    out = A.adversarial_review({"kind": "secret", "where": "persist", "n": 3, "skip": None},
                               mode="redteam", gen_fn=fn)
    assert [r["finding"] for r in out["results"]] == ["kind=secret; where=persist; n=3"]
    assert out["degraded"] is False
    assert "kind=secret; where=persist; n=3" in fn.calls[0]["prompt"]
    assert [r["finding"] for r in A.adversarial_review("  bare  ", gen_fn=_ok(_RT))["results"]] == ["bare"]
    big = A.adversarial_review([f"f{i}" for i in range(A._MAX_FINDINGS + 20)],
                               mode="redteam", gen_fn=_ok(_RT))
    assert [r["finding"] for r in big["results"]] == [f"f{i}" for i in range(A._MAX_FINDINGS)]

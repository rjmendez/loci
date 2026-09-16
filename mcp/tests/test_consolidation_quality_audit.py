"""Tests for consolidation_quality_audit — advisory Mnemosyne merge spot-checks."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import consolidation_quality_audit as C  # noqa: E402


_SOURCES = [
    {
        "id": "wm-1",
        "source": "notes",
        "timestamp": "2026-09-16T00:00:00Z",
        "content": "Alice rotated the AWS key after the incident.",
    },
    {
        "id": "wm-2",
        "source": "notes",
        "timestamp": "2026-09-16T00:05:00Z",
        "content": "Bob disabled the compromised CI runner pending rebuild.",
    },
]


def _ok(text):
    def _fn(prompt, *, fmt=None, max_tokens=256):
        assert fmt == "json"
        assert "ORIGINAL ENTRIES:" in prompt and "MERGED RESULT:" in prompt
        return {"text": text, "ok": True}
    return _fn


def _not_ok(prompt, *, fmt=None, max_tokens=256):
    return {"text": "", "ok": False, "why": "no Ollama endpoint resolved"}


def test_bad_merge_is_flagged_when_model_detects_lost_information():
    result = C.audit_merge_quality(
        "The incident was handled.",
        _SOURCES,
        gen_fn=_ok(
            '{"verdict":"lost_or_conflated","concern":"The merged result omits both the AWS key rotation and the CI runner shutdown.","confidence":0.92}'
        ),
    )
    assert result["available"] is True
    assert result["verdict"] == "lost_or_conflated"
    assert "AWS key rotation" in result["concern"]
    assert result["degraded"] is False


def test_clean_merge_passes_when_model_says_distinct_details_are_preserved():
    result = C.audit_merge_quality(
        "Alice rotated the AWS key after the incident, and Bob disabled the compromised CI runner pending rebuild.",
        _SOURCES,
        gen_fn=_ok(
            '{"verdict":"preserved","concern":"","confidence":0.81}'
        ),
    )
    assert result["available"] is True
    assert result["verdict"] == "preserved"
    assert result["concern"] == ""
    assert result["degraded"] is False


def test_model_unavailable_fails_open():
    result = C.audit_merge_quality(
        "summary",
        _SOURCES,
        gen_fn=_not_ok,
    )
    assert result["available"] is False
    assert result["verdict"] is None
    assert result["degraded"] is True
    assert result["error"] == "no Ollama endpoint resolved"

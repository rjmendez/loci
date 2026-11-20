"""Content extraction for ``call_litellm_proxy`` across proxy response shapes.

Guards the fix for the content:null defect: the LiteLLM proxy billed tokens
(``usage.completion_tokens`` > 0) but the tool surfaced no text because the
generated text lived in a non-string field (Anthropic-via-LiteLLM content
blocks, or ``reasoning_content``) rather than ``choices[0].message.content``.

Only the network dependency (``urllib.request.urlopen``) is faked; the unit
under test (the extraction logic) runs for real.
"""
import json
import urllib.request

import pytest

from loci.litellm_client import call_litellm_proxy


class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


@pytest.fixture
def fake_proxy(monkeypatch):
    holder = {"payload": None}

    def _fake_urlopen(req, timeout=None):  # noqa: ARG001 - signature match
        return _FakeResp(holder["payload"])

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    return holder


def _call(fake_proxy, payload):
    fake_proxy["payload"] = payload
    return call_litellm_proxy("medium", [{"role": "user", "content": "hi"}])


def test_string_content(fake_proxy):
    r = _call(fake_proxy, {"choices": [{"message": {"content": "plain-string-answer"}}],
                           "usage": {"completion_tokens": 3}})
    assert r.content == "plain-string-answer"


def test_anthropic_content_blocks(fake_proxy):
    # The real defect shape: text is a list of blocks; the content string is absent.
    r = _call(fake_proxy, {"choices": [{"message": {"content": [
        {"type": "text", "text": "block-one "},
        {"type": "text", "text": "block-two"},
    ]}}], "usage": {"completion_tokens": 9}})
    assert r.content == "block-one block-two"


def test_content_blocks_skip_non_text(fake_proxy):
    r = _call(fake_proxy, {"choices": [{"message": {"content": [
        {"type": "thinking", "text": "SHOULD-NOT-APPEAR"},
        {"type": "text", "text": "visible"},
    ]}}], "usage": {"completion_tokens": 4}})
    assert r.content == "visible"


def test_reasoning_content(fake_proxy):
    r = _call(fake_proxy, {"choices": [{"message": {
        "content": None, "reasoning_content": "from-reasoning-field"}}],
        "usage": {"completion_tokens": 5}})
    assert r.content == "from-reasoning-field"


def test_legacy_text_field(fake_proxy):
    r = _call(fake_proxy, {"choices": [{"text": "legacy-completion-text"}],
                           "usage": {"completion_tokens": 2}})
    assert r.content == "legacy-completion-text"


def test_no_content_anywhere_returns_empty(fake_proxy):
    # content null, tokens billed, no other field -> empty string (degraded case),
    # but usage is still surfaced so callers can detect "billed but empty".
    r = _call(fake_proxy, {"choices": [{"message": {"content": None}}],
                           "usage": {"completion_tokens": 7}})
    assert r.content == ""
    assert r.completion_tokens == 7

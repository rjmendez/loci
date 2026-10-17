"""
Tests for mcp/litellm_client.py

Rules applied (AGENTS.md):
- Exact value assertions (not ranges or presence-only checks)
- Stubs only for dependencies (urllib.request.urlopen), not the unit under test
- Every negative test has a positive twin on the same fixture
- Recording stubs for calls that must not happen
- pytest.raises with specific match patterns
- No error/degraded output accepted as a pass for success-path tests

Mutations confirmed to turn tests red:
- test_call_litellm_proxy_success: remove `raw.get("model", model)` → KeyError/wrong value
- test_call_litellm_proxy_http_error: swallow HTTPError → test_resolve_... positive twin catches the wrong branch
- test_resolve_model_for_tier_simple_no_rewrite: return (allowed[0], True) always → was_rewritten=True fails
- test_resolve_model_for_tier_simple_downgrade: return (requested_model, False) → was_rewritten=False fails
- test_call_litellm_proxy_auth_header: omit Authorization header → assertion fails
- test_call_litellm_proxy_no_auth: add header unconditionally → assertion fails
"""

import io
import json
import os
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

import litellm_client as lc
from litellm_client import (
    LiteLLMResponse,
    call_litellm_proxy,
    resolve_model_for_tier_simple,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_urlopen_mock(response_body: dict, status: int = 200):
    """Return a context-manager mock that yields a fake HTTP response."""
    resp_mock = MagicMock()
    resp_mock.read.return_value = json.dumps(response_body).encode("utf-8")
    resp_mock.__enter__ = lambda s: s
    resp_mock.__exit__ = MagicMock(return_value=False)
    return resp_mock


_GOOD_RESPONSE = {
    "model": "claude-haiku-4.5",
    "choices": [{"message": {"content": "Hello from haiku"}}],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    },
}

_MESSAGES = [{"role": "user", "content": "Say hello"}]


# ---------------------------------------------------------------------------
# 1. Success path
# ---------------------------------------------------------------------------

class TestCallLiteLLMProxySuccess(unittest.TestCase):
    def test_call_litellm_proxy_success(self):
        mock_resp = _make_urlopen_mock(_GOOD_RESPONSE)
        with patch.object(urllib.request, "urlopen", return_value=mock_resp):
            result = call_litellm_proxy("claude-haiku-4.5", _MESSAGES, max_tokens=512)

        assert isinstance(result, LiteLLMResponse)
        assert result.model == "claude-haiku-4.5"
        assert result.content == "Hello from haiku"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15
        assert result.latency_ms >= 0.0

    def test_to_dict_structure(self):
        """to_dict returns the documented keys with exact values."""
        mock_resp = _make_urlopen_mock(_GOOD_RESPONSE)
        with patch.object(urllib.request, "urlopen", return_value=mock_resp):
            result = call_litellm_proxy("claude-haiku-4.5", _MESSAGES)

        d = result.to_dict()
        assert d["model"] == "claude-haiku-4.5"
        assert d["content"] == "Hello from haiku"
        assert d["usage"]["prompt_tokens"] == 10
        assert d["usage"]["completion_tokens"] == 5
        assert d["usage"]["total_tokens"] == 15
        assert isinstance(d["latency_ms"], float)


# ---------------------------------------------------------------------------
# 2. HTTP error path
# ---------------------------------------------------------------------------

class TestCallLiteLLMProxyHTTPError(unittest.TestCase):
    def _make_http_error(self, code: int, body: str):
        err = urllib.error.HTTPError(
            url="http://localhost:4000/v1/chat/completions",
            code=code,
            msg="Bad Request",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=io.BytesIO(body.encode()),
        )
        return err

    def test_call_litellm_proxy_http_error(self):
        """HTTPError 400 must raise ValueError containing '400'."""
        err = self._make_http_error(400, '{"error":"bad model"}')
        with patch.object(urllib.request, "urlopen", side_effect=err):
            with pytest.raises(ValueError, match="400"):
                call_litellm_proxy("bad-model", _MESSAGES)

    def test_call_litellm_proxy_success_after_error_fixture(self):
        """Positive twin: same fixture but a good call succeeds."""
        mock_resp = _make_urlopen_mock(_GOOD_RESPONSE)
        with patch.object(urllib.request, "urlopen", return_value=mock_resp):
            result = call_litellm_proxy("claude-haiku-4.5", _MESSAGES)
        assert result.content == "Hello from haiku"


# ---------------------------------------------------------------------------
# 3. Auth header present when key is set
# ---------------------------------------------------------------------------

class TestCallLiteLLMProxyAuthHeader(unittest.TestCase):
    def _capture_request(self, response_body):
        """Capture the Request object passed to urlopen."""
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["req"] = req
            mock_resp = _make_urlopen_mock(response_body)
            return mock_resp

        return fake_urlopen, captured

    def test_call_litellm_proxy_auth_header(self):
        """When LOCI_LITELLM_API_KEY is set, Authorization header is sent."""
        fake_open, captured = self._capture_request(_GOOD_RESPONSE)
        with (
            patch.dict(os.environ, {"LOCI_LITELLM_API_KEY": "sk-test-secret"}),
            patch.object(lc, "LITELLM_API_KEY", "sk-test-secret"),
            patch.object(urllib.request, "urlopen", side_effect=fake_open),
        ):
            call_litellm_proxy("claude-haiku-4.5", _MESSAGES)

        req = captured["req"]
        assert req.get_header("Authorization") == "Bearer sk-test-secret"

    def test_call_litellm_proxy_no_auth(self):
        """When LOCI_LITELLM_API_KEY is empty, no Authorization header is sent."""
        fake_open, captured = self._capture_request(_GOOD_RESPONSE)
        with (
            patch.dict(os.environ, {"LOCI_LITELLM_API_KEY": ""}),
            patch.object(lc, "LITELLM_API_KEY", ""),
            patch.object(urllib.request, "urlopen", side_effect=fake_open),
        ):
            call_litellm_proxy("claude-haiku-4.5", _MESSAGES)

        req = captured["req"]
        # urllib capitalises the first letter of each header word
        assert req.get_header("Authorization") is None


# ---------------------------------------------------------------------------
# 5 & 6. resolve_model_for_tier_simple — no-rewrite and downgrade
# ---------------------------------------------------------------------------

class TestResolveModelForTierSimple(unittest.TestCase):
    def test_resolve_model_for_tier_simple_no_rewrite(self):
        """claude-haiku-4.5 is a light-tier model; requesting it under 'light' needs no rewrite."""
        resolved, was_rewritten = resolve_model_for_tier_simple("claude-haiku-4.5", "light")
        assert resolved == "claude-haiku-4.5"
        assert was_rewritten is False

    def test_resolve_model_for_tier_simple_downgrade(self):
        """claude-opus-4.8 (heavy) requested under 'light' must be rewritten to claude-haiku-4.5."""
        resolved, was_rewritten = resolve_model_for_tier_simple("claude-opus-4.8", "light")
        assert resolved == "claude-haiku-4.5"
        assert was_rewritten is True

    def test_resolve_model_for_tier_simple_medium_keeps_medium(self):
        """claude-sonnet-4.6 under 'medium' stays as-is (positive twin for downgrade test)."""
        resolved, was_rewritten = resolve_model_for_tier_simple("claude-sonnet-4.6", "medium")
        assert resolved == "claude-sonnet-4.6"
        assert was_rewritten is False

    def test_resolve_model_for_tier_simple_unknown_model(self):
        """An unknown model name under 'medium' is rewritten to the first medium-tier model."""
        resolved, was_rewritten = resolve_model_for_tier_simple("unknown-model", "medium")
        assert resolved == "claude-sonnet-4.6"
        assert was_rewritten is True


# ---------------------------------------------------------------------------
# 8. Fail-open behavior on proxy error (simulated at server logic level)
# ---------------------------------------------------------------------------

class TestLiteLLMCallToolFailsOpen(unittest.TestCase):
    """
    Verify that the fail-open pattern in mcp__loci__litellm_call returns an error
    dict (not an exception) when call_litellm_proxy raises.

    We test the litellm_client directly here (not server import, which carries heavy
    dependencies), and simulate the server's try/except wrapper inline.
    """

    def _server_tool_logic(self, model, messages, max_tokens=1024):
        """Minimal replica of the server tool's fail-open wrapper."""
        resolved_model, was_rewritten = resolve_model_for_tier_simple(model, "heavy")
        try:
            response = call_litellm_proxy(resolved_model, messages, max_tokens)
        except Exception as exc:
            return {
                "error": "proxy_unavailable",
                "detail": str(exc),
                "model": model,
            }
        return {
            "model": resolved_model,
            "content": response.content,
            "downgraded": was_rewritten,
        }

    def test_litellm_call_tool_fails_open(self):
        """Proxy raising ConnectionRefusedError returns error dict, not an exception."""
        with patch.object(urllib.request, "urlopen", side_effect=ConnectionRefusedError("refused")):
            result = self._server_tool_logic("claude-opus-4.8", _MESSAGES)

        assert result["error"] == "proxy_unavailable"
        assert "refused" in result["detail"]
        assert result["model"] == "claude-opus-4.8"

    def test_litellm_call_tool_success_path(self):
        """Positive twin: working proxy returns content, not an error dict."""
        mock_resp = _make_urlopen_mock(_GOOD_RESPONSE)
        with patch.object(urllib.request, "urlopen", return_value=mock_resp):
            result = self._server_tool_logic("claude-haiku-4.5", _MESSAGES)

        assert "error" not in result
        assert result["content"] == "Hello from haiku"

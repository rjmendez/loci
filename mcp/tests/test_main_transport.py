"""main()'s transport dispatch.

FastMCP.run() accepts only (transport, mount_path) — passing host/port raises
TypeError, so every non-stdio transport died at startup. The bind address has to
go on mcp.settings instead.
"""
import os
import unittest
from unittest import mock

import server


def _run_main(env):
    """stdio path: FastMCP.run() is still the entry point."""
    with mock.patch.dict(os.environ, env, clear=True), \
         mock.patch.object(server.mcp, "run") as run, \
         mock.patch.object(server, "logger"):
        server.main()
    return run


def _run_http(env):
    """HTTP path: served through uvicorn so a bearer gate can wrap the app.

    FastMCP.run() builds its own app internally with no injection point for
    middleware, and the SDK's token_verifier hook requires AuthSettings with an
    issuer_url — the full OAuth resource-server model. Serving the app directly
    is what makes a shared-secret gate possible without pretending to be OAuth.
    """
    with mock.patch.dict(os.environ, env, clear=True), \
         mock.patch.object(server.mcp, "streamable_http_app") as sapp, \
         mock.patch.object(server.mcp, "sse_app") as eapp, \
         mock.patch.object(server, "logger"), \
         mock.patch("uvicorn.run") as urun:
        sapp.return_value = mock.sentinel.http_app
        eapp.return_value = mock.sentinel.sse_app
        server.main()
    return urun


def test_stdio_is_the_default():
    run = _run_main({})
    run.assert_called_once_with(transport="stdio")


def test_sse_binds_via_settings():
    urun = _run_http({
        "HERMES_MCP_TRANSPORT": "sse",
        "HERMES_MCP_HOST": "127.0.0.1",
        "HERMES_MCP_PORT": "9001",
    })
    assert server.mcp.settings.host == "127.0.0.1"
    assert server.mcp.settings.port == 9001
    assert urun.call_args.kwargs["host"] == "127.0.0.1"
    assert urun.call_args.kwargs["port"] == 9001


def test_streamable_http_defaults_to_loopback():
    """An unauthenticated server must not reach the network by default.

    This asserted 0.0.0.0 and was describing the behaviour rather than defending
    it. A wide bind is a decision; not making one should not produce the exposed
    outcome. docker-compose sets HERMES_MCP_HOST=0.0.0.0 explicitly and publishes
    on 127.0.0.1, so the containerised path is unaffected.
    """
    urun = _run_http({"HERMES_MCP_TRANSPORT": "streamable-http"})
    assert server.mcp.settings.host == "127.0.0.1"
    assert server.mcp.settings.port == 8000
    assert urun.call_args.kwargs["host"] == "127.0.0.1"


def test_a_wide_bind_is_still_available_explicitly():
    """…but only with a token. See TestHttpTokenAuth for the refusal."""
    urun = _run_http({"HERMES_MCP_TRANSPORT": "streamable-http",
                      "HERMES_MCP_HOST": "0.0.0.0",
                      "HERMES_MCP_TOKEN": "s3cret"})
    assert server.mcp.settings.host == "0.0.0.0"
    assert urun.call_args.kwargs["host"] == "0.0.0.0"


def test_unknown_transport_falls_back_to_stdio():
    run = _run_main({"HERMES_MCP_TRANSPORT": "carrier-pigeon"})
    run.assert_called_once_with(transport="stdio")


class TestHttpTokenAuth(unittest.TestCase):
    """#78: the HTTP transports expose every tool. A wide bind needs a secret."""

    def test_wide_bind_without_a_token_refuses_to_start(self):
        """The property that matters: it must not serve, not merely warn."""
        with self.assertRaises(SystemExit) as ctx:
            _run_http({"HERMES_MCP_TRANSPORT": "streamable-http",
                       "HERMES_MCP_HOST": "0.0.0.0"})
        msg = str(ctx.exception)
        self.assertIn("HERMES_MCP_TOKEN", msg)
        self.assertIn("0.0.0.0", msg)

    def test_loopback_without_a_token_still_serves(self):
        """Local development must not need a secret."""
        urun = _run_http({"HERMES_MCP_TRANSPORT": "streamable-http"})
        self.assertEqual(urun.call_args.kwargs["host"], "127.0.0.1")

    def test_a_token_wraps_the_app_in_the_auth_gate(self):
        urun = _run_http({"HERMES_MCP_TRANSPORT": "streamable-http",
                          "HERMES_MCP_TOKEN": "s3cret"})
        self.assertIsInstance(urun.call_args.args[0], server._BearerAuthMiddleware)

    def test_no_token_leaves_the_app_unwrapped(self):
        urun = _run_http({"HERMES_MCP_TRANSPORT": "streamable-http"})
        self.assertIs(urun.call_args.args[0], mock.sentinel.http_app)

    def test_ipv6_loopback_is_recognised(self):
        urun = _run_http({"HERMES_MCP_TRANSPORT": "sse", "HERMES_MCP_HOST": "::1"})
        self.assertEqual(urun.call_args.kwargs["host"], "::1")


class TestBearerMiddleware(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.inner = mock.AsyncMock()
        self.mw = server._BearerAuthMiddleware(self.inner, "s3cret")

    async def _call(self, headers, path="/mcp"):
        sent = []
        scope = {"type": "http", "path": path, "headers": headers}
        await self.mw(scope, mock.AsyncMock(), lambda m: sent.append(m) or _noop())
        return sent

    async def test_correct_token_passes_through(self):
        sent = await self._call({b"authorization": b"Bearer s3cret"})
        self.inner.assert_awaited_once()
        self.assertEqual(sent, [])

    async def test_wrong_token_is_401(self):
        sent = await self._call({b"authorization": b"Bearer wrong"})
        self.inner.assert_not_awaited()
        self.assertEqual(sent[0]["status"], 401)

    async def test_missing_header_is_401(self):
        sent = await self._call({})
        self.inner.assert_not_awaited()
        self.assertEqual(sent[0]["status"], 401)

    async def test_health_is_exempt(self):
        """Liveness probes must work without the secret; /health discloses nothing."""
        await self._call({}, path="/health")
        self.inner.assert_awaited_once()

    async def test_non_http_scope_passes_through(self):
        await self.mw({"type": "lifespan"}, mock.AsyncMock(), mock.AsyncMock())
        self.inner.assert_awaited_once()


async def _noop():
    return None

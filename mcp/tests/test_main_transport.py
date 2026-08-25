"""main()'s transport dispatch.

FastMCP.run() accepts only (transport, mount_path) — passing host/port raises
TypeError, so every non-stdio transport died at startup. The bind address has to
go on mcp.settings instead.
"""
import os
from unittest import mock

import server


def _run_main(env):
    with mock.patch.dict(os.environ, env, clear=True), \
         mock.patch.object(server.mcp, "run") as run, \
         mock.patch.object(server, "logger"):
        server.main()
    return run


def test_stdio_is_the_default():
    run = _run_main({})
    run.assert_called_once_with(transport="stdio")


def test_sse_binds_via_settings():
    run = _run_main({
        "HERMES_MCP_TRANSPORT": "sse",
        "HERMES_MCP_HOST": "127.0.0.1",
        "HERMES_MCP_PORT": "9001",
    })
    run.assert_called_once_with(transport="sse")
    assert server.mcp.settings.host == "127.0.0.1"
    assert server.mcp.settings.port == 9001


def test_streamable_http_defaults_to_loopback():
    """An unauthenticated server must not reach the network by default.

    This asserted 0.0.0.0 and was describing the behaviour rather than defending
    it. A wide bind is a decision; not making one should not produce the exposed
    outcome. docker-compose sets HERMES_MCP_HOST=0.0.0.0 explicitly and publishes
    on 127.0.0.1, so the containerised path is unaffected.
    """
    run = _run_main({"HERMES_MCP_TRANSPORT": "streamable-http"})
    run.assert_called_once_with(transport="streamable-http")
    assert server.mcp.settings.host == "127.0.0.1"
    assert server.mcp.settings.port == 8000


def test_a_wide_bind_is_still_available_explicitly():
    run = _run_main({"HERMES_MCP_TRANSPORT": "streamable-http",
                     "HERMES_MCP_HOST": "0.0.0.0"})
    run.assert_called_once_with(transport="streamable-http")
    assert server.mcp.settings.host == "0.0.0.0"


def test_unknown_transport_falls_back_to_stdio():
    run = _run_main({"HERMES_MCP_TRANSPORT": "carrier-pigeon"})
    run.assert_called_once_with(transport="stdio")

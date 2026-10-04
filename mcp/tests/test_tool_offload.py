"""Sync MCP tools must not run on the event loop.

Regression for 2026-09-24: FastMCP called sync tools inline, so one slow Ollama
call froze /health and every client's MCP handshake until it returned.
"""
from __future__ import annotations

import asyncio
import contextvars
import threading
import time

import pytest
from mcp.server.fastmcp import FastMCP

import tool_offload


def _run(coro):
    return asyncio.run(coro)


def _offloaded_server(name: str = "t") -> FastMCP:
    server = FastMCP(name)
    tool_offload.install(server, tool_offload.make_executor(1))
    return server


def test_blocked_tool_leaves_the_event_loop_free():
    server = _offloaded_server()
    release = threading.Event()
    entered = threading.Event()

    @server.tool()
    def slow() -> str:
        entered.set()
        # Bounded so a regression (tool on the loop thread) fails instead of hanging:
        # inline, the loop cannot run the code that sets release, so this times out.
        return "released" if release.wait(timeout=2.0) else "timed out"

    async def scenario():
        call = asyncio.create_task(server.call_tool("slow", {}))
        ticks = 0
        deadline = time.monotonic() + 1.0
        while not entered.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert entered.is_set(), "tool never started"
        # The tool is now blocked in release.wait(); the loop must keep running.
        for _ in range(20):
            await asyncio.sleep(0.005)
            ticks += 1
        release.set()
        result = await call
        return ticks, result

    ticks, result = _run(scenario())
    assert ticks == 20
    content = result[0] if isinstance(result, tuple) else result
    assert [c.text for c in content] == ["released"]


def test_published_schema_is_identical_to_the_unwrapped_function():
    def search(query: str, limit: int = 10, include_retracted: bool = False) -> str:
        """Search findings by similarity."""
        return query

    plain = FastMCP("plain")
    plain.tool()(search)
    wrapped = _offloaded_server("wrapped")
    wrapped.tool()(search)

    (p,) = _run(plain.list_tools())
    (w,) = _run(wrapped.list_tools())
    assert (w.name, w.description, w.inputSchema) == (p.name, p.description, p.inputSchema)


def test_decorator_returns_the_original_sync_function():
    server = _offloaded_server()

    def add(a: int, b: int) -> int:
        return a + b

    returned = server.tool()(add)
    assert returned is add
    assert add(2, 3) == 5


def test_caller_context_reaches_the_worker_thread():
    var: contextvars.ContextVar[str] = contextvars.ContextVar("caller", default="unset")
    server = _offloaded_server()

    @server.tool()
    def who() -> str:
        return var.get()

    async def scenario():
        var.set("agent-alice")
        return await server.call_tool("who", {})

    result = _run(scenario())
    content = result[0] if isinstance(result, tuple) else result
    assert [c.text for c in content] == ["agent-alice"]


def test_default_single_worker_keeps_tools_serial_and_on_one_thread():
    server = _offloaded_server()
    events: list[tuple[str, int, int]] = []
    lock = threading.Lock()

    @server.tool()
    def step(n: int) -> int:
        with lock:
            events.append(("enter", n, threading.get_ident()))
        time.sleep(0.05)
        with lock:
            events.append(("exit", n, threading.get_ident()))
        return n

    async def scenario():
        return await asyncio.gather(*(server.call_tool("step", {"n": i}) for i in range(3)))

    _run(scenario())
    kinds = [e[0] for e in events]
    assert kinds == ["enter", "exit"] * 3
    thread_ids = {e[2] for e in events}
    assert len(thread_ids) == 1
    assert thread_ids != {threading.get_ident()}


def test_async_tools_are_registered_unchanged():
    server = _offloaded_server()

    async def ping() -> str:
        return "pong"

    assert tool_offload.offload(ping, tool_offload.make_executor(1)) is ping
    server.tool()(ping)
    result = _run(server.call_tool("ping", {}))
    content = result[0] if isinstance(result, tuple) else result
    assert [c.text for c in content] == ["pong"]


@pytest.mark.parametrize("raw, expected", [("", 1), ("4", 4), ("0", 1), ("-3", 1), ("x", 1)])
def test_worker_count_env(monkeypatch, raw, expected):
    monkeypatch.setenv("LOCI_TOOL_WORKERS", raw)
    assert tool_offload.worker_count() == expected


def test_every_loci_server_tool_is_offloaded_with_an_unchanged_schema():
    import server as loci_server

    tools = loci_server.mcp._tool_manager.list_tools()
    assert len(tools) >= 40
    not_async = [t.name for t in tools if not t.is_async]
    assert not_async == []

    reference = FastMCP("reference")
    for t in tools:
        original = getattr(t.fn, "__wrapped__", t.fn)
        reference.tool()(original)
    ref = {t.name: t for t in _run(reference.list_tools())}
    live = {t.name: t for t in _run(loci_server.mcp.list_tools())}
    assert set(live) == set(ref)
    mismatched = [n for n in live if (live[n].description, live[n].inputSchema)
                  != (ref[n].description, ref[n].inputSchema)]
    assert mismatched == []

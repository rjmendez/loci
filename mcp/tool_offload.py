"""Run synchronous MCP tools off the event loop.

FastMCP (mcp 1.28) calls a sync ``@mcp.tool()`` function inline on the event
loop: ``fn(**kwargs)``, no executor. Every Loci tool is sync, and several block
for minutes on network I/O (``llm_local.generate`` waits up to 120 s per attempt
on Ollama). While one of those calls runs, the whole server stops: it cannot
accept connections, answer ``/health``, or complete the MCP handshake for any
other client. On 2026-09-24 that looked like a hung server and a client
"connection timed out after 30000ms", while the process was just waiting on a
GPU-starved Ollama.

:func:`install` wraps ``mcp.tool`` so every sync tool is registered as an async
tool that runs the original function in a dedicated executor. The event loop
stays free for the transport.

Design choices:

* **One worker by default** (``LOCI_TOOL_WORKERS``). Tools still run one at a
  time, exactly as they did on the loop thread, and always on the same thread.
  Tool code was written under that serialization and keeps some per-thread and
  unlocked module state; raising the worker count is an explicit opt-in.
* **The caller's context is copied** into the worker (``contextvars``), so the
  MCP ``request_ctx`` that :mod:`caller_identity` reads to bind ACL identity is
  the calling request's, not whatever the worker saw last.
* **The decorator returns the original function**, so module-level callers and
  tests that call tools directly still get a plain sync function.
* Async tools are registered unchanged.
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_LOG = logging.getLogger("loci-mcp.tool_offload")

_ATTR = "__loci_offloaded__"


def worker_count() -> int:
    raw = os.environ.get("LOCI_TOOL_WORKERS", "").strip()
    if not raw:
        return 1
    try:
        n = int(raw)
    except ValueError:
        _LOG.warning("LOCI_TOOL_WORKERS=%r is not an integer; using 1", raw)
        return 1
    return max(1, n)


def make_executor(workers: int | None = None) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=workers or worker_count(),
                              thread_name_prefix="loci-tool")


def offload(fn: Callable[..., Any], executor: ThreadPoolExecutor) -> Callable[..., Any]:
    """Return an async callable that runs sync ``fn`` in ``executor``.

    ``functools.wraps`` keeps ``__wrapped__``, ``__name__``, ``__doc__`` and
    ``__annotations__``, which is what FastMCP reads (via ``inspect.signature``)
    to build the tool's name, description and input schema, so the published
    schema is identical to the unwrapped function's.
    """
    if inspect.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def _offloaded(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        call = functools.partial(ctx.run, fn, *args, **kwargs)
        return await loop.run_in_executor(executor, call)

    setattr(_offloaded, _ATTR, True)
    return _offloaded


def install(mcp: Any, executor: ThreadPoolExecutor | None = None) -> ThreadPoolExecutor:
    """Patch ``mcp.tool`` so sync tools registered from now on run off the loop.

    Must be called before any tool is registered. Returns the executor.
    """
    if getattr(mcp, _ATTR, False):
        return getattr(mcp, "_loci_tool_executor")
    ex = executor or make_executor()
    original_tool = mcp.tool

    def tool(*dargs: Any, **dkwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        register = original_tool(*dargs, **dkwargs)

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            register(offload(fn, ex))
            return fn

        return decorator

    mcp.tool = tool
    setattr(mcp, _ATTR, True)
    mcp._loci_tool_executor = ex
    _LOG.info("sync MCP tools run off the event loop on %d worker thread(s)",
              ex._max_workers)
    return ex

"""Fixture: the @mcp.tool()-style registration decorator shape, plus a
FastAPI-style route decorator, a wrapping decorator, and one decorator no
rule should recognize (used to assert `classify_decorator` returns
"unknown" rather than silently mis-tagging it as registering)."""
import functools


class _FakeMCP:
    def tool(self):
        def deco(fn):
            return fn
        return deco


mcp = _FakeMCP()


@mcp.tool()
def registered_tool(x: int, y: str = "default") -> dict:
    """First line of the docstring."""
    return {"x": x, "y": y}


@functools.lru_cache
def cached_helper(n: int) -> int:
    return n * 2


@some_unrecognized_decorator
def mystery(z):
    return z

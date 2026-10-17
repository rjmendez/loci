"""Fixture: MAN-LOOP — a manifest tuple whose loop body registers each
element via `<obj>.tool()(fn)` — and, right next to it, a loop over an
equally bare-name tuple whose body just calls a method ON each element
(backends.py's real `_reset_cache` shape). The second loop must NOT be
classified MAN-LOOP: only the registering-call shape qualifies."""


class _FakeMCP:
    def tool(self):
        def deco(fn):
            return fn
        return deco


def tool_a():
    return "a"


def tool_b():
    return "b"


def register(mcp):
    for fn in (tool_a, tool_b):
        mcp.tool()(fn)


def _config():
    return {}


def _reset_cache():
    for fn in (_config,):
        fn.cache_clear()

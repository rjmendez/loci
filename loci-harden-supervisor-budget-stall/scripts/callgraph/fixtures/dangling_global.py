"""Fixture: mirrors BUG C (mcp/graph_tools.py's `_symbol_index_cache`) — a
function declares a name `global`, assigns it, but the module never binds
that name at module level anywhere. The real slot lives in a different
module (see reexport_source.py's `_INTERNAL_STATE` for a contrast: a name
that IS bound at module level, so it must NOT be flagged global-only)."""


def invalidate_cache():
    global _symbol_index_cache, _symbol_index_count
    _symbol_index_cache = {}
    _symbol_index_count = 0


def bump(n):
    global _real_counter
    _real_counter += n


_real_counter = 0  # module-level binding: this one must NOT be global-only

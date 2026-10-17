"""Fixture: the module dangling_global.py's `_symbol_index_cache` /
`_symbol_index_count` OUGHT to have been reading/writing — mirrors
mcp/ladybug_ops.py:106-107 relative to mcp/graph_tools.py's dangling
`global` in real BUG C. Module-level-bound (not global-only), so
`cg writes-dead`'s dangling-global check must name THIS module as the real
slot when diagnosing dangling_global.py's finding."""
_symbol_index_cache = None
_symbol_index_count = -1


def rebuild(count):
    global _symbol_index_cache, _symbol_index_count
    _symbol_index_cache = {}
    _symbol_index_count = count

"""REGRESSION FIXTURE -- BUG C's shape AFTER the real fix (mutation check).

Mirrors the accepted fix in c1c40a9: the invalidation is delegated to the
module that OWNS the cache instead of being written through a `global` in a
module that never binds those names. The point of this file is negative --
`dangling_globals()` must find nothing here. A detector that fires on the
fix as loudly as on the bug teaches people to ignore it.

The `global _relink_count` below is deliberate: it is a REAL module-level
slot, declared global and bound at module level, so it exercises the
"legitimate global write" path that must NOT be reported.
"""
import json
import logging

logger = logging.getLogger(__name__)

_relink_count = 0


def _get_ladybug():
    return None


def code_memory_relink() -> str:
    ks = _get_ladybug()
    if not ks:
        return json.dumps({"error": "LadybugDB graph store unavailable."})
    try:
        from graph import linker
        result = linker.relink_all(ks)
        # Must go through ladybug_ops -- assigning the names here would bind
        # this module's own globals and leave the real cache stale.
        import ladybug_ops
        ladybug_ops.invalidate_symbol_index()
        global _relink_count
        _relink_count += 1
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.debug("code_memory_relink failed: %r", exc)
        return json.dumps({"error": f"code_memory_relink failed: {exc!r}"})


def relink_count() -> int:
    return _relink_count

"""Fixture: REG-FN's injector side — a `register()` that assigns one
parameter straight to a `global`-declared slot, and pulls a second slot out
of a `deps["key"]` dict parameter. Mirrors graph_tools.py/
investigation_tools.py's real shape exactly."""
_get_thing = None
_helper_a = None


def register(get_thing, deps):
    global _get_thing, _helper_a
    _get_thing = get_thing
    _helper_a = deps["helper_a"]


def use_thing():
    """Mirrors graph_tools.py's tool functions: a bare call to the
    module-global slot `register()` fills in, resolvable only once
    extract/dispatch.py's rung 4 (name-via-injected-global) has run."""
    return _get_thing()


def call_use_thing():
    """A PROVEN (name-def-local) hop onto use_thing's own PROBABLE
    (name-via-injected-global) hop — fixture for the path-confidence
    combinator: the two-hop path from here to real_thing must report
    PROBABLE overall, the weaker of the two links, not PROVEN."""
    return use_thing()

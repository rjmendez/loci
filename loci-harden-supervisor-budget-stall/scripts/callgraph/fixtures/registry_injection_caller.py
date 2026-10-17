"""Fixture: REG-FN's call-site side — mirrors
`graph_tools.register(mcp, _get_ladybug)` at mcp/server.py: a module-level
call passing a bare function reference for the direct param and a dict
literal (bare-name values) for the deps param."""
import registry_injection


def real_thing():
    return 1


def real_helper_a():
    return 2


registry_injection.register(real_thing, {"helper_a": real_helper_a})

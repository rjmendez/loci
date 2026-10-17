"""Fixture: the module a re-export pulls from (mirrors mcp/graph_tools.py's
role relative to mcp/server.py)."""


def symbol_impact(name: str) -> dict:
    return {"name": name}


def impact_report(name: str) -> list:
    return []


_INTERNAL_STATE = None

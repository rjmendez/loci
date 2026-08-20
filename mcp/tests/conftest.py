"""Shared pytest configuration for mcp/tests/.

Inserts the mcp/ directory at the front of sys.path so that every test in
this package can do ``import server`` and resolve the MCP server module —
regardless of which directory pytest is invoked from or whether a2a_server
tests are collected in the same session.
"""
import sys
from pathlib import Path

_MCP_DIR = str(Path(__file__).resolve().parent.parent)


def pytest_configure(config):  # noqa: ARG001
    if _MCP_DIR not in sys.path:
        sys.path.insert(0, _MCP_DIR)

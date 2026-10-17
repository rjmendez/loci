"""Fixture: re-exports names from reexport_source.py, mirroring
mcp/server.py:7401's `from graph_tools import (symbol_impact, ...)`, plus a
bare module-level `A = B` alias."""
from reexport_source import symbol_impact, impact_report

# Bare module-level aliasing: `runner` should ALIAS to symbol_impact's
# FUNCTION node, not create a second, disconnected identity for it.
runner = symbol_impact

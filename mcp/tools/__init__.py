"""
Tool registry: import each tool module and call its register() function.
"""
import importlib

_TOOL_SUBMODULES = [
    "policy_tools",
    "drpg_tools",
    "litellm_tools",
    "memory_route_tools",
    "investigation_tools",
    "analytics_tools",
]

def register_all(mcp_server):
    """Import every tool module and register its tools on the given FastMCP server."""
    # Determine the package prefix: "mcp.tools" when installed, "tools" when running from mcp/
    pkg = __name__  # either "mcp.tools" or "tools"
    for sub in _TOOL_SUBMODULES:
        mod = importlib.import_module(f"{pkg}.{sub}")
        mod.register(mcp_server)

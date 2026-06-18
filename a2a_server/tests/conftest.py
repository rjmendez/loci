import os
import sys


def pytest_configure(config):
    # When both mcp/tests/ and a2a_server/tests/ run in the same pytest session,
    # mcp/tests/test_reflection_loop.py imports `server` first (the MCP server).
    # Clear the cache so this directory's `import server` loads a2a_server/server.py.
    sys.modules.pop("server", None)
    a2a_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    if a2a_dir not in sys.path:
        sys.path.insert(0, a2a_dir)

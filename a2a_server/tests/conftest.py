"""a2a_server/tests setup: live-store isolation, then the `server` module path.

server.py defaults MNEMOSYNE_DATA_DIR to ~/.hermes/mnemosyne/data, reads
~/.hermes/.env, and the memory skills query QDRANT_URL. On a machine running
Loci, an unisolated run therefore read the live Mnemosyne DB and reached the
live Qdrant; one test timed out against it. testsupport/loci_hermetic.py
explains the mechanism; it must be installed here, at conftest import, before
any test module loads server.py. Opt out only for a deliberate live smoke test
with LOCI_TESTS_LIVE=1.
"""
import importlib.util
import os
import sys
from pathlib import Path


def _load_hermetic():
    if "loci_hermetic" in sys.modules:
        return sys.modules["loci_hermetic"]
    path = Path(__file__).resolve().parents[2] / "testsupport" / "loci_hermetic.py"
    spec = importlib.util.spec_from_file_location("loci_hermetic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["loci_hermetic"] = module
    spec.loader.exec_module(module)
    return module


loci_hermetic = _load_hermetic()
loci_hermetic.install()
globals().update(loci_hermetic.pytest_hooks())


def pytest_configure(config):
    # When both mcp/tests/ and a2a_server/tests/ run in the same pytest session,
    # mcp/tests/test_reflection_loop.py imports `server` first (the MCP server).
    # Clear the cache so this directory's `import server` loads a2a_server/server.py.
    sys.modules.pop("server", None)
    a2a_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    if a2a_dir not in sys.path:
        sys.path.insert(0, a2a_dir)

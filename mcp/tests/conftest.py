"""Shared pytest configuration for mcp/tests/.

Inserts the mcp/ directory at the front of sys.path so that every test in
this package can do ``import server`` and resolve the MCP server module —
regardless of which directory pytest is invoked from or whether a2a_server
tests are collected in the same session.
"""
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

_MCP_DIR = str(Path(__file__).resolve().parent.parent)
_RUN_MEMORY_DIR = Path(_MCP_DIR) / ".pytest_cache" / "loci-memory" / uuid.uuid4().hex


def _configure_memory_isolation() -> None:
    _RUN_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    isolated = str(_RUN_MEMORY_DIR)
    os.environ["LOCI_MEMORY_DIR"] = isolated
    os.environ["HERMES_MEMORY_DIR"] = isolated
    os.environ.setdefault("TMPDIR", str(Path(_MCP_DIR) / ".pytest_cache" / "tmp"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = os.environ["TMPDIR"]


_configure_memory_isolation()


def pytest_configure(config):  # noqa: ARG001
    if _MCP_DIR not in sys.path:
        sys.path.insert(0, _MCP_DIR)
    _configure_memory_isolation()


def pytest_unconfigure(config):  # noqa: ARG001
    shutil.rmtree(_RUN_MEMORY_DIR, ignore_errors=True)

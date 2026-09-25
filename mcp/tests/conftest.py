"""Shared pytest configuration for mcp/tests/.

Inserts the mcp/ directory at the front of sys.path so that every test in
this package can do ``import server`` and resolve the MCP server module —
regardless of which directory pytest is invoked from or whether a2a_server
tests are collected in the same session.

Also isolates the whole session from the live Loci stores before anything
imports ``server``: temp HOME / LOCI_MEMORY_DIR / MNEMOSYNE_DATA_DIR, no
backends.toml, unreachable Qdrant/Ollama/vLLM, the repo .env files not loaded,
and any access under the real ~/.loci or ~/.hermes refused and failed. See
testsupport/loci_hermetic.py. Opt out only for a deliberate live smoke test:
LOCI_TESTS_LIVE=1.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent


def _load_hermetic():
    """Import testsupport/loci_hermetic.py by path (it is not on sys.path)."""
    if "loci_hermetic" in sys.modules:
        return sys.modules["loci_hermetic"]
    path = _MCP_DIR.parent / "testsupport" / "loci_hermetic.py"
    spec = importlib.util.spec_from_file_location("loci_hermetic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["loci_hermetic"] = module
    spec.loader.exec_module(module)
    return module


# Must run at conftest import: pytest imports this file before any test module,
# and server/backends/qdrant_ops read the environment at import time. A fixture,
# even a session-scoped one, runs after collection has already imported them.
loci_hermetic = _load_hermetic()
loci_hermetic.install()
# Autouse fixture failing any test that reached ~/.loci or ~/.hermes, plus the
# report header and the session-level check. Opt-out: LOCI_TESTS_LIVE=1.
globals().update(loci_hermetic.pytest_hooks())


def _configure_pytest_temp_root() -> None:
    """Pin pytest temp roots inside the repository when unset.

    Some machines deny access to the default OS temp root used by pytest's
    tmp_path fixture (for example, when `%TEMP%\\pytest-of-<user>` is blocked).
    PYTEST_DEBUG_TEMPROOT is the official override and keeps tmp_path behavior
    unchanged while avoiding machine-specific absolute paths.
    """

    if os.environ.get("PYTEST_DEBUG_TEMPROOT"):
        return
    temp_root = (_MCP_DIR / ".tmp").resolve(strict=False)
    temp_root.mkdir(parents=True, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(temp_root)


_configure_pytest_temp_root()


def pytest_configure(config):  # noqa: ARG001
    mcp_dir = str(_MCP_DIR)
    if mcp_dir not in sys.path:
        sys.path.insert(0, mcp_dir)


@pytest.fixture(autouse=True)
def _isolate_the_audit_log(tmp_path, monkeypatch):
    """No test may touch ~/.hermes/memcheck-audit.jsonl.

    memcheck's audit log defaults to the operator's real home, and
    test_daemon.py drives process_action -> _append_audit_line with no
    isolation. That was harmless while the writer only appended.

    On 2026-09-01 a change adding size-based rotation to that writer turned it
    destructive: running mcp/tests once took the real log from 9,588,755 bytes
    and 49,412 records spanning 2026-06-22 to 2,102,181 bytes and 10,733
    records from 2026-08-23. About 38,700 records over 62 days, with no backup.
    A clean CI runner has no such file, so the rotation hits FileNotFoundError
    and returns -- green on the runner, destructive on the machine in daily use.

    autouse and unconditional: an opt-in fixture is one a new test file forgets.
    """
    monkeypatch.setenv("MEMCHECK_AUDIT_LOG", str(tmp_path / "memcheck-audit.jsonl"))


@pytest.fixture(autouse=True)
def _isolate_offload_audit(tmp_path, monkeypatch):
    """offload_tool_loop writes per-run JSONL under MEMORY_DIR/../audit/offload by default.

    Same rationale as _isolate_the_audit_log: no test may write into the operator's home.
    """
    monkeypatch.setenv("LOCI_OFFLOAD_AUDIT_DIR", str(tmp_path / "offload-audit"))


"""Standalone scripts must still accept the legacy HERMES_* spelling.

The LOCI_* rename wired mcp/legacy_env.py into the servers and the hooks, but
not into scripts that run on their own from cron or systemd. Those read the new
name only, so an existing deployment exporting the old one silently lost its
setting — the opposite of the in-place upgrade the rename promised. Found in
review, not by these tests, which is why they exist now.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys
from unittest import mock

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "mcp"))


def _fresh(mod: str):
    sys.modules.pop(mod, None)
    return importlib.import_module(mod)


@pytest.mark.parametrize("module, attr, legacy, current", [
    ("state_db_qdrant_sync", "STATE_DB", "HERMES_STATE_DB", "LOCI_STATE_DB"),
    ("mnemosyne_qdrant_sync", "_ENV_FILE", "HERMES_ENV_FILE", "LOCI_ENV_FILE"),
])
def test_the_legacy_spelling_is_still_honoured(module, attr, legacy, current):
    with mock.patch.dict(os.environ, {legacy: "/tmp/from-legacy"}, clear=True):
        assert "/tmp/from-legacy" in str(getattr(_fresh(module), attr))


@pytest.mark.parametrize("module, attr, legacy, current", [
    ("state_db_qdrant_sync", "STATE_DB", "HERMES_STATE_DB", "LOCI_STATE_DB"),
    ("mnemosyne_qdrant_sync", "_ENV_FILE", "HERMES_ENV_FILE", "LOCI_ENV_FILE"),
])
def test_the_current_spelling_wins(module, attr, legacy, current):
    env = {legacy: "/tmp/from-legacy", current: "/tmp/from-current"}
    with mock.patch.dict(os.environ, env, clear=True):
        assert "/tmp/from-current" in str(getattr(_fresh(module), attr))


def test_backends_memory_dir_accepts_the_legacy_variable():
    import backends
    with mock.patch.dict(os.environ, {"HERMES_MEMORY_DIR": "/tmp/legacy-mem"}, clear=True):
        assert "/tmp/legacy-mem" in str(backends.memory_dir())


def test_the_settings_json_legacy_registration_name_is_hermes_memory():
    """It names files that already exist on disk, so it cannot be renamed."""
    for f in ("ebbinghaus_consolidation.py", "reembed_daemon.py",
              "qdrant_payload_indexes.py", "memgas_hierarchy.py"):
        src = (REPO / "scripts" / f).read_text()
        if "mcpServers" not in src and "servers" not in src:
            continue
        assert "loci_memory" not in src or "hermes_memory" in src, (
            f"{f} looks up a legacy MCP registration name that never existed")


@pytest.mark.parametrize("attr, legacy", [
    ("LOCAL_A2A_URL",       "HERMES_A2A_URL"),
    ("LOCAL_A2A_TOKEN",     "HERMES_A2A_TOKEN"),
    ("LOCAL_A2A_TOTP_SEED", "HERMES_A2A_TOTP_SEED"),
])
def test_the_context_bridge_honours_the_legacy_a2a_names(attr, legacy, tmp_path):
    """scripts/systemd/mrpink-context-bridge.service points EnvironmentFile= at a
    profile that supplies only HERMES_A2A_*. Reading the new name alone left the
    bridge with an empty token and no TOTP header, so every send 401'd."""
    home = tmp_path / "no-hermes-here"          # keep the module's .env loader a no-op
    env = {"HERMES_HOME": str(home), legacy: "from-legacy"}
    with mock.patch.dict(os.environ, env, clear=True):
        assert getattr(_fresh("a2a_context_bridge"), attr) == "from-legacy"


@pytest.mark.parametrize("attr, legacy, current", [
    ("LOCAL_A2A_URL",       "HERMES_A2A_URL",       "LOCI_A2A_URL"),
    ("LOCAL_A2A_TOKEN",     "HERMES_A2A_TOKEN",     "LOCI_A2A_TOKEN"),
    ("LOCAL_A2A_TOTP_SEED", "HERMES_A2A_TOTP_SEED", "LOCI_A2A_TOTP_SEED"),
])
def test_the_context_bridge_prefers_the_current_a2a_names(attr, legacy, current, tmp_path):
    home = tmp_path / "no-hermes-here"
    env = {"HERMES_HOME": str(home), legacy: "from-legacy", current: "from-current"}
    with mock.patch.dict(os.environ, env, clear=True):
        assert getattr(_fresh("a2a_context_bridge"), attr) == "from-current"


def test_the_contract_hook_falls_back_too():
    src = (REPO / "scripts" / "hooks" / "post-commit-contract-extract.sh").read_text()
    assert "HERMES_ACTIVE_INVESTIGATION" in src

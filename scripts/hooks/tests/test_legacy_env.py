"""The legacy-name map, and the fact that its two copies must stay identical.

The hooks are deployed standalone into ~/.claude/hooks, so they cannot import
from mcp/. The map therefore exists twice. It diverged within an hour of being
created — memory_dir() was added to the mcp copy only — so the sameness is
asserted rather than remembered.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
from unittest import mock

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
MCP_COPY = REPO / "mcp" / "legacy_env.py"
HOOK_COPY = REPO / "scripts" / "hooks" / "legacy_env.py"


def _load(path):
    spec = importlib.util.spec_from_file_location(f"_legacy_env_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_two_copies_are_byte_identical():
    assert MCP_COPY.read_bytes() == HOOK_COPY.read_bytes(), (
        "mcp/legacy_env.py and scripts/hooks/legacy_env.py have diverged; "
        "the hooks ship their own copy and it must match"
    )


INSTALL_SH = REPO / "scripts" / "hooks" / "install.sh"


def _install_hooks_array() -> list[str]:
    """The words of install.sh's HOOKS=( ... ) array -- what it actually copies.
    Comments inside the array are dropped, as bash drops them."""
    m = re.search(r"^HOOKS=\((.*?)\)", INSTALL_SH.read_text(), re.M | re.S)
    assert m, "install.sh no longer declares a HOOKS=( ... ) array"
    words = []
    for line in m.group(1).splitlines():
        words.extend(line.split("#", 1)[0].split())
    return words


def test_install_deploys_the_map_with_the_hooks():
    # Without it a deployed hook reading LOCI_* gets nothing from a wrapper that
    # still exports HERMES_*, and the session sync silently stops working.
    assert "legacy_env.py" in _install_hooks_array()


def test_install_ships_every_hook_that_imports_the_map():
    hooks_dir = REPO / "scripts" / "hooks"
    importers = {p.name for p in hooks_dir.glob("*.py")
                 if "from legacy_env import" in p.read_text()}
    assert importers >= {"pre_llm_grounding.py", "pre_tool_grounding.py",
                         "session_end_sync.py"}
    assert importers <= set(_install_hooks_array())


@pytest.mark.skipif(shutil.which("bash") is None, reason="install.sh is a bash script")
def test_install_run_copies_the_map_next_to_the_hooks(tmp_path):
    dest = tmp_path / "hooks"
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
           "CLAUDE_HOOKS_DIR": str(dest),
           "CLAUDE_SETTINGS": str(tmp_path / "settings.json")}
    proc = subprocess.run(["bash", str(INSTALL_SH)], env=env, capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "installed legacy_env.py" in proc.stdout.splitlines()
    assert (dest / "legacy_env.py").read_bytes() == HOOK_COPY.read_bytes()
    assert sorted(p.name for p in dest.iterdir()) == sorted(_install_hooks_array())


def test_legacy_name_is_mapped_onto_the_current_one():
    mod = _load(MCP_COPY)
    env = {"HERMES_A2A_TOKEN": "from-legacy"}
    assert mod.apply(env) == ["HERMES_A2A_TOKEN"]
    assert env["LOCI_A2A_TOKEN"] == "from-legacy"


def test_the_current_name_wins_when_both_are_set():
    mod = _load(MCP_COPY)
    env = {"HERMES_A2A_TOKEN": "old", "LOCI_A2A_TOKEN": "new"}
    assert mod.apply(env) == []
    assert env["LOCI_A2A_TOKEN"] == "new"


def test_an_empty_legacy_value_is_not_propagated():
    mod = _load(MCP_COPY)
    env = {"HERMES_A2A_TOKEN": ""}
    mod.apply(env)
    assert "LOCI_A2A_TOKEN" not in env


def test_hermes_owned_names_are_never_mapped():
    """These name the Hermes installation Loci runs inside, not Loci."""
    mod = _load(MCP_COPY)
    for name in ("HERMES_PROFILE", "HERMES_HOME", "HERMES_VENV_SITE",
                 "HERMES_SUBAGENT", "HERMES_AGENT_ID"):
        assert name not in mod.RENAMED, f"{name} belongs to Hermes and must not be renamed"


def test_every_mapping_only_changes_the_prefix():
    mod = _load(MCP_COPY)
    for old, new in mod.RENAMED.items():
        assert old.startswith("HERMES_") and new.startswith("LOCI_")
        assert old[len("HERMES_"):] == new[len("LOCI_"):], (old, new)


def test_memory_dir_prefers_an_explicit_setting():
    mod = _load(MCP_COPY)
    with mock.patch.dict(os.environ, {"LOCI_MEMORY_DIR": "/tmp/explicit"}, clear=True):
        assert str(mod.memory_dir()) == "/tmp/explicit"


def test_memory_dir_accepts_the_legacy_variable(tmp_path):
    mod = _load(MCP_COPY)
    with mock.patch.dict(os.environ, {"HERMES_MEMORY_DIR": str(tmp_path)}, clear=True):
        assert str(mod.memory_dir()) == str(tmp_path)


def test_memory_dir_falls_back_to_an_existing_legacy_directory(tmp_path):
    mod = _load(MCP_COPY)
    home = tmp_path / "home"
    (home / ".hermes" / "memory-sessions").mkdir(parents=True)
    with mock.patch.dict(os.environ, {"HOME": str(home)}, clear=True):
        with mock.patch.object(pathlib.Path, "home", staticmethod(lambda: home)):
            assert mod.memory_dir() == home / ".hermes" / "memory-sessions"


def test_memory_dir_prefers_the_new_location_when_it_exists(tmp_path):
    mod = _load(MCP_COPY)
    home = tmp_path / "home"
    (home / ".hermes" / "memory-sessions").mkdir(parents=True)
    (home / ".loci" / "memory-sessions").mkdir(parents=True)
    with mock.patch.dict(os.environ, {"HOME": str(home)}, clear=True):
        with mock.patch.object(pathlib.Path, "home", staticmethod(lambda: home)):
            assert mod.memory_dir() == home / ".loci" / "memory-sessions"


def test_a_fresh_install_gets_the_new_location(tmp_path):
    """Neither directory exists: nothing to inherit, so use the current name."""
    mod = _load(MCP_COPY)
    home = tmp_path / "home"
    home.mkdir()
    with mock.patch.dict(os.environ, {"HOME": str(home)}, clear=True):
        with mock.patch.object(pathlib.Path, "home", staticmethod(lambda: home)):
            assert mod.memory_dir() == home / ".loci" / "memory-sessions"


def test_validate_hermes_endpoint_policy_allows_loopback():
    mod = _load(MCP_COPY)
    env = {"QDRANT_URL": "http://127.0.0.1:6333", "OLLAMA_BASE_URL": "http://localhost:11434"}
    assert mod.validate_hermes_endpoint_policy(env, files=[]) == []


def test_validate_hermes_endpoint_policy_rejects_private_bridge():
    mod = _load(MCP_COPY)
    env = {"QDRANT_URL": "http://172.21.171.198:6333"}
    issues = mod.validate_hermes_endpoint_policy(env, files=[])
    assert issues and "bridge/private" in issues[0]

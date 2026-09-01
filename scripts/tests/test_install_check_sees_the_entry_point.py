"""`install.sh --check` graded the files this repo ships, not the ones Claude Code runs.

~/.claude/settings.json points the Stop event at ~/.claude/hooks/session_end_sync.sh,
a wrapper that exports QDRANT_URL, the embedding endpoint and five other variables
before exec'ing session_end_sync.py. That wrapper exists in no commit, so the HOOKS
array does not name it, so --check never looked at it -- and printed "hooks in sync"
over the one file the Stop hook actually executes. The drift detector was blind to
the entry point.

--check must not claim sync over a hook it did not examine.
"""
import json
import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
INSTALL = REPO / "scripts" / "hooks" / "install.sh"
MANAGED = "pre_llm_grounding.py"      # named in HOOKS
UNMANAGED = "session_end_sync.sh"     # invoked by settings.json, in no commit


def _hooks_named_by_install() -> list:
    """The HOOKS array, read out of the script itself rather than duplicated here."""
    src = INSTALL.read_text()
    body = src.split("HOOKS=(", 1)[1].split(")", 1)[0]
    return body.split()


def _stage(tmp_path, settings_hooks):
    """A deployed hooks dir with no file drift, plus a settings.json invoking it."""
    dest = tmp_path / "hooks"
    dest.mkdir()
    for name in _hooks_named_by_install():
        shutil.copy(REPO / "scripts" / "hooks" / name, dest / name)
    (dest / UNMANAGED).write_text("#!/usr/bin/env bash\nexec true\n")

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"hooks": {
        "Stop": [{"hooks": [{"type": "command", "command": str(dest / h)}
                            for h in settings_hooks]}],
    }}))
    return dest, settings


def _check(dest, settings):
    return subprocess.run(
        ["bash", str(INSTALL), "--check"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(dest.parent),
             "CLAUDE_HOOKS_DIR": str(dest), "CLAUDE_SETTINGS": str(settings)},
    )


def test_check_names_a_hook_settings_json_runs_that_the_repo_does_not_ship(tmp_path):
    dest, settings = _stage(tmp_path, [UNMANAGED])
    run = _check(dest, settings)
    assert f"UNMANAGED {UNMANAGED}" in run.stdout, (
        f"--check examined none of settings.json's own hook commands:\n{run.stdout}")
    assert "hooks in sync\n" not in run.stdout, (
        "claimed a clean bill over a hook it never compared:\n" + run.stdout)


def test_check_still_says_in_sync_when_settings_runs_only_managed_hooks(tmp_path):
    dest, settings = _stage(tmp_path, [MANAGED])
    run = _check(dest, settings)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "hooks in sync" in run.stdout
    assert "UNMANAGED" not in run.stdout


@pytest.mark.parametrize("missing", ["absent settings.json"])
def test_check_survives_a_settings_file_that_is_not_there(tmp_path, missing):
    """A fresh machine has no settings.json yet; --check must still grade the files."""
    dest, settings = _stage(tmp_path, [UNMANAGED])
    settings.unlink()
    run = _check(dest, settings)
    assert run.returncode == 0, f"{missing}: {run.stdout}{run.stderr}"
    assert "hooks in sync" in run.stdout


def test_unmanaged_hooks_do_not_change_the_exit_status(tmp_path):
    """Exit 1 means file drift. An unmanaged entry point is reported, not graded --
    resolving it is a decision about what this repo ships, not a sync action."""
    dest, settings = _stage(tmp_path, [UNMANAGED])
    assert _check(dest, settings).returncode == 0
    (dest / MANAGED).write_text("# hand-edited on the box\n")
    drifted = _check(dest, settings)
    assert drifted.returncode == 1
    assert f"DRIFTED  {MANAGED}" in drifted.stdout

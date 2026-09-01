"""A hook that cannot be executed fails the way a hook is designed to fail.

.git/hooks/post-commit invokes scripts/hooks/post-commit-contract-extract.sh by
path. That file was committed 100644, so every commit printed

    .git/hooks/post-commit: line 36: .../post-commit-contract-extract.sh: Permission denied

and the caller swallowed it with `|| true`. Contract extraction has therefore
never run on any commit. The hook fired, exited 0, and did nothing.

The mode is a property of the index, not the working tree -- a fresh clone gets
whatever was committed -- so this reads `git ls-files -s`.
"""
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


def _indexed_modes(subdir: str) -> dict:
    out = subprocess.run(
        ["git", "ls-files", "-s", "--", subdir],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    modes = {}
    for line in out:
        meta, path = line.split("\t", 1)
        modes[path] = meta.split()[0]
    return modes


def _has_shebang(rel: str) -> bool:
    blob = subprocess.run(
        ["git", "show", f":{rel}"], cwd=REPO, capture_output=True, check=True,
    ).stdout
    return blob.startswith(b"#!")


def _is_script(rel: str) -> bool:
    """Shell scripts anywhere, plus anything under scripts/hooks -- git hooks
    carry no extension (scripts/hooks/post-commit) and are still executed."""
    return rel.endswith((".sh", ".bash")) or rel.startswith("scripts/hooks/")


@pytest.mark.parametrize("subdir", ["scripts/hooks", "scripts"])
def test_every_shebanged_script_is_committed_executable(subdir):
    modes = _indexed_modes(subdir)
    bad = [
        rel for rel, mode in sorted(modes.items())
        if mode == "100644" and _is_script(rel) and _has_shebang(rel)
    ]
    assert not bad, (
        "committed non-executable, so anything invoking them by path gets "
        f"Permission denied: {bad}"
    )


def test_the_contract_extract_hook_is_runnable():
    """Named on its own: this is the one that was broken, and the one whose
    failure is invisible because the caller ends in `|| true`."""
    rel = "scripts/hooks/post-commit-contract-extract.sh"
    assert _indexed_modes("scripts/hooks")[rel] == "100755"

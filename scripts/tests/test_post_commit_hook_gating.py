"""scripts/hooks/post-commit must only spawn the background ingest for commits on
main in the primary worktree.

Before this gate, every .py commit anywhere -- linked worktrees, feature branches,
detached clones -- ran ``claude -p --dangerously-skip-permissions
/loci-codebase-ingest`` in the background, so unmerged code was ingested into the
shared loci-codebase investigation. In linked worktrees the log path
``$repo_root/.git/loci-hook-logs`` also broke, because ``.git`` is a file there.

These tests run the real hook in throwaway repos with a fake ``claude`` on PATH
that only records that it was called.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "scripts" / "hooks" / "post-commit"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _env(tmp: Path, marker: Path, **extra: str) -> dict[str, str]:
    bindir = tmp / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "claude"
    fake.write_text(f'#!/usr/bin/env bash\necho "$PWD" >> "{marker}"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    env = {
        "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(tmp / "home"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    (tmp / "home").mkdir(exist_ok=True)
    env.update(extra)
    return env


def _git(cwd: Path, env: dict[str, str], *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout


def _init(repo: Path, env: dict[str, str]) -> None:
    repo.mkdir()
    _git(repo, env, "init", "-q", "-b", "main")
    _git(repo, env, "config", "commit.gpgsign", "false")
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    shutil.copy(HOOK, hooks / "post-commit")
    (hooks / "post-commit").chmod(0o755)


def _commit_py(wt: Path, env: dict[str, str], name: str) -> str:
    (wt / f"{name}.py").write_text("x = 1\n")
    _git(wt, env, "add", f"{name}.py")
    # git sends hook output to stderr; collect both streams.
    proc = subprocess.run(
        ["git", "commit", "-q", "-m", name], cwd=wt, env=env, check=True,
        capture_output=True, text=True,
    )
    return proc.stdout + proc.stderr


_SPAWN_LINE = "triggering loci-codebase re-ingest"


def _assert_not_spawned(out: str, common_git_dir: Path, marker: Path) -> None:
    """Deterministic: the hook prints the spawn line and creates the log dir
    synchronously, before it backgrounds `claude`, and git commit returns only
    after the hook exits. A 0.5 s poll of the marker instead missed a spawn that
    happened a little later (switch removed + `sleep 1` passed)."""
    assert _SPAWN_LINE not in out, out
    assert not (common_git_dir / "loci-hook-logs").exists()
    assert not (marker.exists() and marker.read_text().strip())


def _called(marker: Path, wait: float) -> bool:
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if marker.exists() and marker.read_text().strip():
            return True
        time.sleep(0.05)
    return marker.exists() and bool(marker.read_text().strip())


def test_main_in_primary_worktree_ingests_and_logs_under_common_dir(tmp_path):
    marker = tmp_path / "calls"
    env = _env(tmp_path, marker)
    repo = tmp_path / "repo"
    _init(repo, env)
    out = _commit_py(repo, env, "a")
    assert _SPAWN_LINE in out
    assert _called(marker, 5.0)
    assert (repo / ".git" / "loci-hook-logs" / "loci-post-commit-ingest.log").exists()


def test_feature_branch_skips(tmp_path):
    marker = tmp_path / "calls"
    env = _env(tmp_path, marker)
    repo = tmp_path / "repo"
    _init(repo, env)
    _git(repo, env, "checkout", "-q", "-b", "feature/x")
    out = _commit_py(repo, env, "a")
    assert "[post-commit] branch 'feature/x' is not 'main'; skipping ingest." in out
    _assert_not_spawned(out, repo / ".git", marker)


def test_linked_worktree_skips_even_on_ingest_branch(tmp_path):
    marker = tmp_path / "calls"
    repo = tmp_path / "repo"
    env = _env(tmp_path, marker, LOCI_HOOK_INGEST="0")
    _init(repo, env)
    _commit_py(repo, env, "seed")
    wt = tmp_path / "wt"
    _git(repo, env, "worktree", "add", "-q", "-b", "wtbranch", str(wt))
    # Make the worktree's branch the ingest branch so only the worktree check
    # can stop it.
    env = _env(tmp_path, marker, LOCI_HOOK_INGEST_BRANCH="wtbranch")
    out = _commit_py(wt, env, "b")
    assert "linked worktree" in out
    _assert_not_spawned(out, repo / ".git", marker)


def test_detached_clone_skips(tmp_path):
    marker = tmp_path / "calls"
    repo = tmp_path / "repo"
    env = _env(tmp_path, marker, LOCI_HOOK_INGEST="0")
    _init(repo, env)
    _commit_py(repo, env, "seed")
    clone = tmp_path / "clone"
    _git(tmp_path, env, "clone", "-q", str(repo), str(clone))
    shutil.copy(HOOK, clone / ".git" / "hooks" / "post-commit")
    (clone / ".git" / "hooks" / "post-commit").chmod(0o755)
    _git(clone, env, "checkout", "-q", "--detach")
    env = _env(tmp_path, marker)
    out = _commit_py(clone, env, "b")
    assert "detached" in out
    _assert_not_spawned(out, clone / ".git", marker)


def test_disable_switch(tmp_path):
    marker = tmp_path / "calls"
    env = _env(tmp_path, marker, LOCI_HOOK_INGEST="0")
    repo = tmp_path / "repo"
    _init(repo, env)
    out = _commit_py(repo, env, "a")
    assert "[post-commit] LOCI_HOOK_INGEST=0; skipping ingest." in out
    _assert_not_spawned(out, repo / ".git", marker)

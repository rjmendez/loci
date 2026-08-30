from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
POST_COMMIT = REPO / "scripts" / "hooks" / "post-commit"


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "hooks@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Hook Tests"], cwd=path, check=True)
    hook_dir = path / ".git" / "hooks"
    shutil.copy2(POST_COMMIT, hook_dir / "post-commit")
    (hook_dir / "post-commit").chmod(0o755)
    dependency = path / "scripts" / "hooks" / "post-commit-contract-extract.sh"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("#!/usr/bin/env bash\nexit 0\n")
    dependency.chmod(0o755)


def _commit(path: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    subprocess.run(["git", "add", "."], cwd=path, check=True, env=env)
    return subprocess.run(
        ["git", "commit", "-m", "root"],
        cwd=path,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_post_commit_ingest_fires_on_root_commit(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "app.py").write_text("print('hello')\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "claude.log"
    claude = fake_bin / "claude"
    claude.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s|%s\\n' \"$PWD\" \"$*\" >> \"$CLAUDE_INVOCATION_LOG\"\n"
        "exit 0\n"
    )
    claude.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "CLAUDE_INVOCATION_LOG": str(log),
    }

    result = _commit(repo, env)

    assert result.returncode == 0, result.stderr
    for _ in range(50):
        if log.exists():
            break
        time.sleep(0.05)
    assert log.exists()
    entry = log.read_text()
    assert str(repo) in entry
    assert "-p --dangerously-skip-permissions /loci-codebase-ingest" in entry


def test_post_commit_missing_claude_never_blocks_commit(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "app.py").write_text("print('hello')\n")

    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    result = _commit(repo, env)

    assert result.returncode == 0, result.stderr


def test_post_commit_failing_claude_never_blocks_commit(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "app.py").write_text("print('hello')\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    claude = fake_bin / "claude"
    claude.write_text("#!/usr/bin/env bash\nexit 86\n")
    claude.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin"}

    result = _commit(repo, env)

    assert result.returncode == 0, result.stderr

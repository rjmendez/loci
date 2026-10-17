from __future__ import annotations

import json
import os
import pathlib
import subprocess
import textwrap

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "workflow_rel,args_obj,expected",
    [
        (
            "deep_think_loci/workflows/contract-sync.js",
            {"root": "/repo", "loci_investigation": "undefined"},
            "invalid loci_investigation",
        ),
        (
            "deep_think_loci/workflows/performance-audit.js",
            {"root": "/repo", "loci_investigation": "undefined"},
            "invalid loci_investigation",
        ),
        (
            "deep_think_loci/workflows/deep-think-v4.js",
            {"run_id": "undefined", "targets": [{"name": "x", "files": [], "focus": "y"}]},
            "invalid run_id",
        ),
    ],
)
def test_workflows_reject_stringified_missing_investigation_ids(workflow_rel, args_obj, expected):
    module_path = (REPO / workflow_rel).resolve()
    script = textwrap.dedent(
        f"""
        import fs from 'node:fs';
        globalThis.args = {json.dumps(args_obj)};
        globalThis.log = () => {{}};
        globalThis.phase = () => {{}};
        globalThis.agent = async () => {{ throw new Error('agent should not run'); }};
        globalThis.parallel = async () => {{ throw new Error('parallel should not run'); }};
        const src = fs.readFileSync({json.dumps(str(module_path))}, 'utf8')
          .replace(/^export const meta =/m, 'const meta =');
        const AsyncFunction = Object.getPrototypeOf(async function () {{}}).constructor;
        const run = new AsyncFunction(src);
        try {{
          await run();
          console.error('workflow unexpectedly completed');
          process.exit(1);
        }} catch (err) {{
          const msg = String(err && (err.stack || err.message || err));
          if (!msg.includes({json.dumps(expected)})) {{
            console.error(msg);
            process.exit(1);
          }}
          console.log(msg);
        }}
        """
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert expected in result.stdout


def test_post_commit_hook_rejects_invalid_active_investigation_id(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    (repo / "README.md").write_text("seed\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=repo, check=True)
    (repo / "changed.py").write_text("print('hello')\n")
    subprocess.run(["git", "add", "changed.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "feature"], cwd=repo, check=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_claude = fake_bin / "claude"
    fake_claude.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_claude.chmod(0o755)

    hook = REPO / "scripts" / "hooks" / "post-commit-contract-extract.sh"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["LOCI_ACTIVE_INVESTIGATION"] = "undefined"

    result = subprocess.run(
        ["bash", str(hook)],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 1
    assert "invalid active investigation id: undefined" in result.stderr

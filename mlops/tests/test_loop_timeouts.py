"""Every child process the nightly loop spawns must be bounded.

The loop shells out ten times — dataset rebuild, train, canary, SFT bake,
drift, active-learn. None carried a timeout, so a stalled child hung the whole
run: a first real run reached "dataset pairs: 5418 | retrain=True" and then sat
in subprocess.run().communicate() until it was killed. Unattended, that is a
nightly that never finishes and never reports why.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import subprocess
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
LOOP = REPO / "mlops" / "loop.py"


def _load():
    spec = importlib.util.spec_from_file_location("_loop_uut", LOOP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("path", ["mlops/loop.py", "mlops/finetune/train_lora.py"])
def test_no_unbounded_subprocess_run(path):
    tree = ast.parse((REPO / path).read_text())
    bad = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "run"
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess"
        and not any(k.arg == "timeout" for k in n.keywords)
    ]
    assert not bad, f"{path} has subprocess.run with no timeout at lines {bad}"


def test_a_hung_child_is_killed_and_reported_as_failure():
    m = _load()
    started = time.time()
    r = m._run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=2)
    elapsed = time.time() - started
    assert elapsed < 10, f"took {elapsed:.1f}s — the bound did not fire"
    assert r.returncode != 0, "a timeout must not read as success"
    assert "timed out" in r.stderr


def test_a_normal_child_is_untouched():
    m = _load()
    r = m._run([sys.executable, "-c", "print('fine')"], timeout=30)
    assert r.returncode == 0 and r.stdout.strip() == "fine"


def test_the_bound_is_configurable():
    m = _load()
    assert isinstance(m.STEP_TIMEOUT_S, int) and m.STEP_TIMEOUT_S > 0


def test_the_timeout_result_keeps_the_completedprocess_shape():
    """Call sites read .returncode/.stdout/.stderr — a timeout must not change that."""
    m = _load()
    r = m._run([sys.executable, "-c", "import time; time.sleep(20)"], timeout=1)
    assert isinstance(r, subprocess.CompletedProcess)
    assert isinstance(r.stdout, str) and isinstance(r.stderr, str)

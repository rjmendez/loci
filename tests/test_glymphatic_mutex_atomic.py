"""glymphatic_sweep._Mutex must acquire atomically, not check-then-act.

The old __enter__ did `if os.path.exists(...)` (with a stale-pid check that
may remove the file) and only then `open(path, "w")` to claim ownership --
two separate steps with no OS-level exclusion between them. Two processes
racing on a *stale* lock (owner pid dead) both pass the exists()/pid-alive
check, both remove it, and both then unconditionally write the file and
believe they hold the mutex.

This test reproduces exactly that: a lock file naming a dead pid, and two
threads entering _Mutex concurrently. With the atomic O_CREAT|O_EXCL fix,
exactly one of them may end up holding the lock at a time.
"""

import importlib.util
import os
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_sweep():
    spec = importlib.util.spec_from_file_location(
        "glymphatic_sweep_under_test_mutex", REPO / "scripts" / "glymphatic_sweep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sweep():
    return _load_sweep()


def _dead_pid():
    """A pid that is guaranteed not to exist right now."""
    candidate = 2**30
    while os.path.exists(f"/proc/{candidate}"):
        candidate -= 1
    return candidate


def test_stale_lock_race_never_double_acquires(sweep, tmp_path):
    lock_path = str(tmp_path / "glymphatic.lock")
    with open(lock_path, "w") as f:
        f.write(f"pid={_dead_pid()} ts={int(time.time())}")

    holders_inside = []
    errors = []
    barrier = threading.Barrier(2)

    def worker():
        try:
            barrier.wait()
            with sweep._Mutex(lock_path):
                holders_inside.append(1)
                # Hold the lock briefly so a second, concurrent winner
                # would overlap with this one if the acquire were racy.
                time.sleep(0.05)
                holders_inside.pop()
        except RuntimeError:
            errors.append("lock_busy")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    max_concurrent = []

    # Patch in a watcher: sample holders_inside length while threads run.
    def watcher():
        for _ in range(200):
            max_concurrent.append(len(holders_inside))
            time.sleep(0.002)

    w = threading.Thread(target=watcher, daemon=True)
    for t in threads:
        t.start()
    w.start()
    for t in threads:
        t.join()

    assert max(max_concurrent) <= 1, (
        "two threads held the glymphatic mutex at the same time -- "
        "the stale-lock check-then-act race let both acquire"
    )
    # Lock file must not be left behind (both __exit__ paths clean up).
    assert not os.path.exists(lock_path)

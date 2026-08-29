"""A nightly that shows nothing until a step finishes cannot be watched.

train.py announces which model it is fitting, but loop.py collected that with
capture_output and printed a tail once the child had exited — so the log cron
mails you was silent for the 17 minutes the step actually took, which is
indistinguishable from a hang. These run real children, because the defect was
in the plumbing, not in a caller.
"""
import pathlib
import subprocess
import sys
import threading
import textwrap
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from mlops import loop  # noqa: E402


def _child(body: str):
    return [sys.executable, "-c", textwrap.dedent(body)]


class _Recorder:
    """Thread-safe stand-in for sys.stdout so the main thread can watch what the
    drain thread has written so far."""

    def __init__(self):
        self._lock = threading.Lock()
        self._buf = []

    def write(self, text):
        with self._lock:
            self._buf.append(text)
        return len(text)

    def flush(self):
        pass

    def seen(self):
        with self._lock:
            return "".join(self._buf)


def test_output_appears_while_the_child_is_still_running(monkeypatch):
    """The whole point. capfd cannot tell live output from a dump at exit, so
    watch the parent's stdout from another thread while the child still sleeps."""
    rec = _Recorder()
    monkeypatch.setattr(sys, "stdout", rec)
    result = {}

    def go():
        result["r"] = loop._run(_child("""
            import time
            print("first line")
            time.sleep(4.0)
            print("second line")
        """), timeout=30)

    t = threading.Thread(target=go, daemon=True)
    started = time.monotonic()
    t.start()

    deadline = started + 3.0          # comfortably inside the child's 4s sleep
    while time.monotonic() < deadline and "first line" not in rec.seen():
        time.sleep(0.05)

    saw_at = time.monotonic() - started
    assert "first line" in rec.seen(), (
        f"nothing after {saw_at:.1f}s while the child was still running — "
        "output is being buffered until exit, which is the bug"
    )
    assert "second line" not in rec.seen(), "the child cannot have finished yet"

    t.join(timeout=30)
    assert result["r"].returncode == 0
    assert "first line" in result["r"].stdout, "streaming must not stop capture"
    assert "second line" in result["r"].stdout


def test_a_child_writing_to_a_pipe_is_not_block_buffered(capfd):
    """Without PYTHONUNBUFFERED the child buffers 8KB before the first flush, and
    the streaming is a no-op for exactly the small progress lines it exists for."""
    result = loop._run(_child('''
        print("tiny")
    '''), timeout=30)
    assert "tiny" in capfd.readouterr().out
    assert "tiny" in result.stdout


def test_stdout_is_still_captured_for_callers_that_parse_it(capfd):
    """_run_active_learn and the drift step read result.stdout as data."""
    result = loop._run(_child('''
        import json
        print(json.dumps({"drift": 0.4}))
    '''), timeout=30)
    capfd.readouterr()
    assert '"drift"' in result.stdout
    assert result.returncode == 0


def test_stderr_is_captured_but_not_echoed(capfd):
    """Failures print one extracted line via _fail; echoing the whole traceback
    live would bury it."""
    result = loop._run(_child('''
        import sys
        sys.stderr.write("boom\\n")
        sys.exit(3)
    '''), timeout=30)
    assert "boom" not in capfd.readouterr().out
    assert "boom" in result.stderr
    assert result.returncode == 3


def test_a_hung_child_is_killed_and_reported_as_124(capfd):
    result = loop._run(_child('''
        import time
        print("started")
        time.sleep(60)
    '''), timeout=2)
    assert result.returncode == 124
    assert "timed out after 2s" in result.stderr
    assert "started" in capfd.readouterr().out, "output before the kill is kept"


def test_the_label_names_the_script_not_the_interpreter():
    assert loop._child_label([sys.executable, "/a/b/train.py", "--x"]) == "train"
    assert loop._child_label([sys.executable, "/a/build_grounding_dataset.py"]) \
        == "build_grounding_dataset"


@pytest.mark.parametrize("stream", [True, False])
def test_both_modes_return_the_same_completedprocess_shape(stream, capfd):
    result = loop._run(_child('print("x")'), timeout=30, stream=stream)
    capfd.readouterr()
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0
    assert "x" in result.stdout

"""The diagnostic has to survive the trip through mlops/loop.py, not just capsys.

The in-process tests assert these lines with capsys, which reads the string
wherever it is written. mlops/loop.py runs this script as a CHILD: _run drains
its stdout with echo=True and its stderr with echo=False, and _run_active_learn
returns only the exit code. A diagnostic on stderr is green in every capsys test
and absent from the one log that reads it, so the nightly prints

    [active_learn] boundary=0 hard_negatives=0 total=0 -> .../cands.jsonl

and the operator sees an ordinary zero. That is the defect this file exists to
catch, so this drives the real subprocess through the real _run.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = REPO / "mlops" / "grounding" / "active_learn.py"

joblib = pytest.importorskip("joblib")
np = pytest.importorskip("numpy")
sklearn = pytest.importorskip("sklearn")


def _loop_module():
    spec = importlib.util.spec_from_file_location("loop_under_test", REPO / "mlops" / "loop.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "mlops"))
    spec.loader.exec_module(mod)
    return mod


def _unscorable_corpus(tmp_path):
    """Rows with no (claim, evidence) pair the sampler can build features from."""
    ds = tmp_path / "grounding_dataset.jsonl"
    ds.write_text("".join(
        json.dumps({"id": f"r{i}", "text": "too short"}) + "\n" for i in range(3)
    ))
    return ds


def _live_model(tmp_path):
    from sklearn.linear_model import LogisticRegression
    sys.path.insert(0, str(REPO / "deep_think_loci" / "grounding"))
    import features as F
    clf = LogisticRegression()
    clf.fit(np.zeros((2, F.CURRENT_DIM), dtype=np.float32), [0, 1])
    path = tmp_path / "clf.joblib"
    joblib.dump(clf, path)
    return path


def test_the_reason_reaches_the_nightly_log(tmp_path, capfd):
    """Run the sampler the way the loop runs it and read what the loop printed."""
    loop = _loop_module()
    out = tmp_path / "cands.jsonl"
    result = loop._run([
        sys.executable, str(SCRIPT),
        "--model", str(_live_model(tmp_path)),
        "--dataset", str(_unscorable_corpus(tmp_path)),
        "--out", str(out),
        "--ollama", "http://127.0.0.1:1",
    ])
    printed = capfd.readouterr().out

    assert result.returncode == 0, "the sampler is supposed to exit clean on an empty result"
    assert "0 carried a (claim, evidence) pair" in printed, (
        "the loop printed no reason for the empty result:\n" + printed
    )
    assert "boundary=0" in printed, printed


def test_the_diagnostic_is_not_written_where_the_loop_discards_it():
    """Pins the mechanism rather than the wording: _run echoes stdout and drops
    stderr, so a diagnostic on stderr is invisible however well it reads."""
    src = SCRIPT.read_text()
    assert "file=sys.stderr" not in src, (
        "a diagnostic on stderr never reaches the nightly log; use _say()"
    )


def test_a_crashing_sampler_is_reported_rather_than_skipped(tmp_path):
    """The other half: when it exits non-zero, _run_active_learn must say so."""
    loop = _loop_module()
    result = loop._run([sys.executable, "-c", "import sys; sys.exit(3)"])
    assert result.returncode == 3
    assert "_fail(\"active_learn\"" in (REPO / "mlops" / "loop.py").read_text(), (
        "a non-zero exit from the sampler leaves the previous candidates file in "
        "place; the step has to report it like dataset-rebuild and train.py do"
    )

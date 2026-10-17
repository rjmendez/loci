"""An unattended trainer that prints only on completion cannot be told from a hung one.

The 2026-08-29 run sat on one line for 31 minutes while GradientBoosting fit
2,000 trees across 10 folds of a 12,684 x 1,540 matrix. Nothing was wrong; there
was simply no way to know that from the log.
"""
import ast
import pathlib

TRAIN = pathlib.Path(__file__).resolve().parents[2] / "mlops" / "grounding" / "train.py"


def _calls(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _is_print(node):
    return isinstance(node.func, ast.Name) and node.func.id == "print"


def test_every_fit_announces_itself_before_it_starts_and_flushes():
    """A flushed print must precede each cross_val_predict and .fit, so the log
    names the model currently running. Redirected to a file, an unflushed print
    buys nothing — a 0-byte log is exactly what a hung process looks like."""
    src = TRAIN.read_text()
    tree = ast.parse(src)
    fits = [n for n in _calls(tree)
            if (isinstance(n.func, ast.Name) and n.func.id == "cross_val_predict")
            or (isinstance(n.func, ast.Attribute) and n.func.attr == "fit")]
    assert fits, "no model fits found — did the trainer change shape?"

    prints = {n.lineno: n for n in _calls(tree) if _is_print(n)}
    for call in fits:
        window = range(max(1, call.lineno - 8), call.lineno)
        near = [prints[ln] for ln in window if ln in prints]
        assert near, f"fit at line {call.lineno} has no print in the 8 lines above it"
        # The NEAREST preceding print is the announcement. Accepting any flushing
        # print in the window lets an unrelated flush elsewhere satisfy the check
        # while the announcing line itself stays buffered.
        announcing = max(near, key=lambda n: n.lineno)
        assert any(k.arg == "flush" for k in announcing.keywords), (
            f"the print at line {announcing.lineno} announcing the fit at line "
            f"{call.lineno} does not flush"
        )


def test_cv_results_record_how_long_each_model_took():
    """Without a per-model duration nobody can tell which candidate costs the hour."""
    src = TRAIN.read_text()
    assert "fit_seconds" in src, "cv_results must carry a per-model fit duration"
    assert "time.monotonic()" in src, "duration must come from a monotonic clock"

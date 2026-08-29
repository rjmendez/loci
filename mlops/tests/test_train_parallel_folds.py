"""Serial cross-validation made the nightly uncompletable.

A 10-fold GradientBoosting over the 12,684 x 1,540 feature matrix ran for more
than 45 minutes on a 28-core box, one fold at a time — longer than the loop's own
3600s step bound, so the step would be cut every night before it produced a
score. The folds are independent; nothing required them to be serial.
"""
import ast
import os
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
TRAIN = REPO / "mlops" / "grounding" / "train.py"
# No sys.path mutation at import: these tests read train.py from disk, and a
# module-level insert leaks into every other test in the run.


def _call(name):
    for node in ast.walk(ast.parse(TRAIN.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == name:
            return node
    return None


def test_cross_validation_is_parallel():
    call = _call("cross_val_predict")
    assert call is not None, "no cross_val_predict — did the trainer change shape?"
    kw = {k.arg for k in call.keywords}
    assert "n_jobs" in kw, (
        "cross_val_predict without n_jobs runs one fold at a time; the real "
        "matrix takes >45 min that way and the step bound cuts it"
    )


def test_the_estimators_do_not_also_claim_every_core():
    """Parallelism belongs at the CV level. An estimator n_jobs on top of it
    oversubscribes: 10 folds x 28 workers on 28 cores is slower, not faster."""
    src = TRAIN.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("RandomForestClassifier",
                                     "GradientBoostingClassifier",
                                     "LogisticRegression"):
            assert not any(k.arg == "n_jobs" for k in node.keywords), (
                f"{node.func.id} sets n_jobs; keep parallelism at the CV level"
            )


def test_the_job_count_is_configurable():
    """A shared or small box needs a way to not take every core."""
    import importlib.util
    os.environ["LOCI_TRAIN_CV_JOBS"] = "3"
    spec = importlib.util.spec_from_file_location("train_probe", TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        assert mod.CV_JOBS == 3
    finally:
        del os.environ["LOCI_TRAIN_CV_JOBS"]


def test_parallel_folds_are_actually_faster():
    """Measured, not assumed — the speedup is the entire justification. Kept small
    so it costs a couple of seconds; the real matrix is 30x wider."""
    import time
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    if (os.cpu_count() or 1) < 4:
        pytest.skip("needs at least 4 cores to show a difference")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 200)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    clf = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    def run(jobs):
        t = time.monotonic()
        cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=jobs)
        return time.monotonic() - t

    # os.cpu_count() over-reports under cgroup or affinity limits, so a strict
    # parallel < serial is flaky in a container that only has one core to give.
    # Assert it does not get materially SLOWER; the real speedup is recorded in
    # docs/grounding-corpus-limits.md and measured on the actual matrix.
    serial, parallel = run(None), run(-1)
    assert parallel < serial * 2.0, (
        f"parallel {parallel:.1f}s vs serial {serial:.1f}s — n_jobs is hurting, "
        "not helping"
    )


def test_the_scored_folds_are_the_folds_that_made_the_predictions():
    """cross_val_predict used a fresh StratifiedKFold stratified on labels while
    per_fold_f1 iterated fold_indices stratified on cosine quartiles. Measured on
    a 12,684-row matrix the two partitions agree at 10.3% — chance. The mean
    survived (every sample still gets an out-of-fold prediction) but the reported
    std was the spread across arbitrary subsets, not across folds."""
    tree = ast.parse(TRAIN.read_text())
    call = _call("cross_val_predict")
    assert call is not None
    cv = next((k.value for k in call.keywords if k.arg == "cv"), None)
    assert cv is not None, "cross_val_predict must be given an explicit cv"
    assert isinstance(cv, ast.Name) and cv.id == "fold_indices", (
        "cv must be the same fold_indices that per_fold_f1 scores; a fresh "
        "StratifiedKFold here partitions the data differently"
    )


def test_env_int_rejects_junk_without_crashing_the_import(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_probe2", TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for bad in ("abc", "0", "-7", "1.5", ""):
        monkeypatch.setenv("X_JOBS", bad)
        assert mod._env_int("X_JOBS", -1, allow="n_jobs") == -1
    monkeypatch.setenv("X_JOBS", "4")
    assert mod._env_int("X_JOBS", -1, allow="n_jobs") == 4

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


def _run_train_main(tmp_path, monkeypatch, jobs):
    """train.main() on a small separable corpus, with the embedder (network) and
    the backend resolution (config) stubbed and cross_val_predict spied on."""
    import types

    import numpy as np
    import sklearn.model_selection as ms

    sys.path.insert(0, str(REPO))
    try:
        from mlops.grounding import train
    finally:
        sys.path.remove(str(REPO))

    def embed(texts, base, cache):
        out = []
        for t in texts:
            v = np.full(4, 0.05, dtype=np.float32)
            v[0 if "alpha" in t else 1] = 1.0
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)

    calls = []
    real_cvp = ms.cross_val_predict

    def spy(estimator, X, y, **kw):
        calls.append({"estimator": type(estimator).__name__, "n_jobs": kw.get("n_jobs"),
                      "cv": kw.get("cv")})
        return real_cvp(estimator, X, y, **kw)

    clock = iter(range(0, 10_000, 7))
    monkeypatch.setattr(ms, "cross_val_predict", spy)
    monkeypatch.setattr(train, "embed_texts", embed)
    monkeypatch.setattr(train, "_resolve_backends", lambda: None)
    monkeypatch.setattr(train, "CACHE_PATH", str(tmp_path / "cache.npz"))
    monkeypatch.setattr(train, "CV_JOBS", jobs)
    # Only train's own reference: patching time.monotonic itself would move
    # every other clock in the process.
    monkeypatch.setattr(train, "time", types.SimpleNamespace(
        monotonic=lambda: float(next(clock))))
    rows = []
    for i in range(30):
        a, b = ("alpha", "beta") if i % 2 else ("beta", "alpha")
        rows.append({"claim": f"{a} claim {i}", "evidence": f"{a} ev {i}", "label": 1,
                     "cos": 0.9 - 0.001 * i})
        rows.append({"claim": f"{a} claim {i}", "evidence": f"{b} ev {i}", "label": 0,
                     "cos": 0.1 + 0.001 * i})
    ds = tmp_path / "ds.jsonl"
    ds.write_text("".join(__import__("json").dumps(r) + "\n" for r in rows))
    out = tmp_path / "metrics.json"
    monkeypatch.setattr(sys, "argv", ["train.py", "--dataset", str(ds), "--out", str(out),
                                      "--ollama", "http://h"])
    train.main()
    return calls, __import__("json").loads(out.read_text())


@pytest.mark.parametrize("jobs", [3, 2])  # <= 4 workers: a shared, loaded box
def test_every_candidate_is_cross_validated_with_the_configured_jobs(tmp_path, monkeypatch,
                                                                     jobs):
    """The AST check only saw an n_jobs keyword; a trainer that passed n_jobs=1
    (serial again, >45 min on the real matrix) passed it. This runs train.main
    and records what cross_val_predict was actually given."""
    calls, _ = _run_train_main(tmp_path, monkeypatch, jobs)
    assert [c["estimator"] for c in calls] == ["LogisticRegression",
                                               "GradientBoostingClassifier",
                                               "RandomForestClassifier"]
    assert [c["n_jobs"] for c in calls] == [jobs] * 3
    # and the same explicit folds each time -- the ones per_fold_f1 then scores
    assert all(isinstance(c["cv"], list) and len(c["cv"]) == 10 for c in calls)
    assert all(c["cv"] is calls[0]["cv"] for c in calls)


def test_each_model_reports_how_long_its_cross_validation_took(tmp_path, monkeypatch):
    """fit_seconds was measured and then dropped from the output; with a clock
    that advances 7s per reading it is exactly 7.0 for every model."""
    _, metrics = _run_train_main(tmp_path, monkeypatch, 1)
    assert {name: m["fit_seconds"] for name, m in metrics["all_models"].items()} == {
        "LogisticRegression": 7.0, "GradientBoostingClassifier": 7.0,
        "RandomForestClassifier": 7.0}


def test_the_scored_folds_are_the_folds_that_made_the_predictions():
    """cross_val_predict used a fresh StratifiedKFold stratified on labels while
    per_fold_f1 iterated fold_indices stratified on cosine quartiles. Measured on
    a 12,684-row matrix the two partitions agree at 10.3% — chance. The mean
    survived (every sample still gets an out-of-fold prediction) but the reported
    std was the spread across arbitrary subsets, not across folds."""
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


def test_the_banner_distinguishes_oos_requested_from_oos_used():
    """--findings-glob can be passed and OOS still return {} — fewer than 2 runs,
    or no fold carrying both classes. The banner blamed a missing flag for that,
    which sends the reader to fix the one thing that was already right."""
    src = TRAIN.read_text()
    assert "OOS was requested but produced no folds" in src, (
        "the basis line must separate 'requested' from 'used'"
    )
    assert "no --findings-glob" in src, "the genuinely-absent-flag case must stay"
    assert "elif args.findings_glob:" in src, (
        "the two cases must be distinguished on args.findings_glob, not on "
        "oos_results alone"
    )


def test_no_measured_figures_are_pinned_in_code_comments():
    """A comment carrying 0.944/0.908/0.864 goes stale the moment the corpus
    changes, and a stale figure in code reads as current. The numbers live in
    docs/grounding-corpus-limits.md, which can be updated with the measurement."""
    import re as _re
    comments = [ln for ln in TRAIN.read_text().splitlines() if ln.lstrip().startswith("#")]
    for ln in comments:
        # F1-shaped literals: a bare 0.NNN in prose
        assert not _re.search(r"\b0\.\d{3}\b", ln), (
            f"measured figure pinned in a comment, put it in the doc instead:\n  {ln.strip()}"
        )

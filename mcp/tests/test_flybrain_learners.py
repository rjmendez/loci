"""Offline tests for flybrain_learners (tiny synthetic fixtures only)."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_training as fbct  # noqa: E402
import flybrain_learners as fl  # noqa: E402


def _synthetic(n=480, n_groups=48, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.asarray([f"g{i % n_groups}" for i in range(n)], dtype=object)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = np.asarray(["red", "green", "blue"], dtype=object)[rng.integers(0, 3, size=n)]
    score = x1 + 0.8 * (cat == "red") - 0.8 * (cat == "blue") + 0.2 * rng.normal(size=n)
    labels = np.where(score > 0.6, "high", np.where(score < -0.6, "low", "mid")).astype(object)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "colour": cat})
    frame.loc[::17, "x2"] = np.nan
    frame[fl.TEXT_COLUMN] = [f"colour {c} bucket {'pos' if a > 0 else 'neg'}" for c, a in zip(cat, x1)]
    return frame, labels, groups


def _split(groups):
    test = np.asarray([int(g[1:]) % 4 == 0 for g in groups])
    return np.flatnonzero(~test), np.flatnonzero(test)


@pytest.mark.parametrize("backend", ["nb", "logreg", "hgb"])
def test_backends_fit_predict_and_beat_majority(backend):
    frame, y, groups = _synthetic()
    tr, te = _split(groups)
    learner = fl.make_learner(backend, seed=3)
    learner.fit(frame.iloc[tr], y[tr], groups=groups[tr])
    proba = learner.predict_proba(frame.iloc[te])
    assert proba.shape == (len(te), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert learner.classes_ == ("high", "low", "mid")
    acc = np.mean(np.asarray(learner.predict(frame.iloc[te])) == y[te])
    majority = max(np.mean(y[te] == c) for c in set(y[tr]))
    assert acc > majority + 0.05
    desc = learner.describe()
    assert desc["backend"] == backend and desc["training_fingerprint"]["data_sha256"]


@pytest.mark.parametrize("backend", ["logreg", "hgb", "nb"])
def test_backends_are_deterministic(backend):
    frame, y, groups = _synthetic()
    a = fl.make_learner(backend, seed=1).fit(frame, y, groups=groups).predict_proba(frame)
    b = fl.make_learner(backend, seed=1).fit(frame, y, groups=groups).predict_proba(frame)
    assert np.array_equal(a, b)


def test_nb_matches_legacy_predictor():
    frame, y, _ = _synthetic(n=120)
    learner = fl.make_learner("nb").fit(frame, y)
    proba = learner.predict_proba(frame)
    for i, text in enumerate(frame[fl.TEXT_COLUMN]):
        label, conf = fbct._predict_multinomial_nb(learner.model, text)
        assert learner.classes_[int(np.argmax(proba[i]))] == label
        assert proba[i].max() == pytest.approx(conf)


def test_hgb_uses_grouped_inner_validation():
    frame, y, groups = _synthetic()
    learner = fl.make_learner("hgb", seed=0, max_iter=200).fit(frame, y, groups=groups)
    assert learner.fit_info["early_stopping"] == "grouped_inner_val"
    assert 1 <= learner.fit_info["n_iter"] <= 200


def test_inner_group_split_never_straddles_groups():
    _, y, groups = _synthetic()
    tr, va = fl._inner_group_split(groups, y, fraction=0.2, seed=5)
    assert not set(groups[tr]) & set(groups[va])


def test_mlp_backend_roundtrip(tmp_path):
    frame, y, groups = _synthetic(n=240, n_groups=24)
    learner = fl.make_learner("mlp", seed=0, epochs=30, hidden=16, layers=1, batch_size=64).fit(frame, y, groups=groups)
    proba = learner.predict_proba(frame)
    assert np.allclose(proba.sum(axis=1), 1.0)
    fl.save_learner(learner, tmp_path / "mlp")
    assert (tmp_path / "mlp" / "model.pt").is_file()
    loaded = fl.load_learner(tmp_path / "mlp", trusted_root=tmp_path)
    assert np.allclose(loaded.predict_proba(frame), proba, atol=1e-6)


def test_unknown_backend_and_params_fail():
    with pytest.raises(ValueError, match="unknown learner backend"):
        fl.make_learner("xgboost")
    with pytest.raises(ValueError, match="unknown params"):
        fl.make_learner("logreg", depth=3)


def test_single_class_fit_fails_closed():
    frame, _, _ = _synthetic(n=50)
    with pytest.raises(ValueError, match="two classes"):
        fl.make_learner("logreg").fit(frame, ["a"] * 50)


def test_schema_prepare_fails_on_missing_column():
    frame, y, _ = _synthetic(n=60)
    learner = fl.make_learner("logreg").fit(frame, y)
    with pytest.raises(ValueError, match="missing schema feature columns"):
        learner.predict_proba(frame.drop(columns=["x1"]))


def test_unseen_category_is_handled():
    frame, y, _ = _synthetic(n=200)
    for backend in ("logreg", "hgb"):
        learner = fl.make_learner(backend).fit(frame, y)
        probe = frame.head(3).copy()
        probe["colour"] = "ultraviolet"
        assert np.allclose(learner.predict_proba(probe).sum(axis=1), 1.0)


# ----------------------------------------------------------------- calibration


def test_ece_and_reliability_table():
    classes = ("a", "b")
    proba = np.asarray([[0.9, 0.1]] * 10 + [[0.6, 0.4]] * 10)
    y = ["a"] * 9 + ["b"] + ["a"] * 6 + ["b"] * 4
    assert fl.expected_calibration_error(proba, y, classes, n_bins=10) == pytest.approx(0.0)
    table = fl.reliability_table(proba, y, classes, n_bins=10)
    assert sum(r["count"] for r in table) == 20
    overconfident = np.asarray([[0.99, 0.01]] * 10)
    assert fl.expected_calibration_error(overconfident, ["a"] * 5 + ["b"] * 5, classes) == pytest.approx(0.49)


@pytest.mark.parametrize("method", fl.CALIBRATION_METHODS)
def test_calibrated_learner_grouped_cv(method, tmp_path):
    frame, y, groups = _synthetic()
    cal = fl.CalibratedLearner(fl.make_learner("hgb", seed=0), method=method, cv=4).fit(frame, y, groups=groups)
    proba = cal.predict_proba(frame)
    assert proba.shape == (len(y), 3) and np.allclose(proba.sum(axis=1), 1.0)
    assert cal.fit_info["calibration_source"] == "grouped_cv_oof"
    fl.save_learner(cal, tmp_path / "cal")
    loaded = fl.load_learner(tmp_path / "cal", trusted_root=tmp_path)
    assert isinstance(loaded, fl.CalibratedLearner)
    assert np.allclose(loaded.predict_proba(frame), proba)


def test_calibration_needs_groups_and_prefit_works():
    frame, y, groups = _synthetic()
    with pytest.raises(ValueError, match="groups"):
        fl.CalibratedLearner(fl.make_learner("logreg")).fit(frame, y)
    tr, te = _split(groups)
    base = fl.make_learner("logreg").fit(frame.iloc[tr], y[tr])
    cal = fl.CalibratedLearner(base, method="temperature").fit_prefit(frame.iloc[te], y[te])
    assert cal.calibrator["temperature"] > 0


def test_isotonic_calibration_reduces_ece_of_overconfident_model():
    rng = np.random.default_rng(0)
    n = 2000
    y = np.where(rng.random(n) < 0.7, "a", "b")
    proba = np.where((y == "a")[:, None], [[0.99, 0.01]], [[0.01, 0.99]])
    flip = rng.random(n) < 0.3
    proba[flip] = proba[flip][:, ::-1]
    state = fl.fit_calibrator(proba, y, ("a", "b"), "isotonic")
    after = fl.apply_calibrator(state, proba)
    assert fl.expected_calibration_error(after, y, ("a", "b")) < fl.expected_calibration_error(proba, y, ("a", "b")) - 0.1


def test_grouped_folds_keep_groups_together():
    groups = [f"g{i % 13}" for i in range(130)]
    folds = fl.grouped_folds(groups, n_splits=5, seed=1)
    assert len(folds) == 5
    arr = np.asarray(groups)
    for train_idx, test_idx in folds:
        assert not set(arr[train_idx]) & set(arr[test_idx])
    with pytest.raises(ValueError):
        fl.grouped_folds(["one"] * 10, n_splits=5, seed=0)


# ----------------------------------------------------------------- artifacts


def _saved(tmp_path, backend="logreg"):
    frame, y, groups = _synthetic(n=200)
    learner = fl.make_learner(backend, seed=0).fit(frame, y, groups=groups)
    manifest = fl.save_learner(learner, tmp_path / "root" / "model", metrics={"val_accuracy": 0.9})
    return frame, learner, manifest


def test_artifact_roundtrip_and_manifest(tmp_path):
    frame, learner, manifest = _saved(tmp_path)
    directory = tmp_path / "root" / "model"
    for name in ("manifest.json", "meta.json", "schema.json", "labels.json", "fingerprint.json", "metrics.json", "model.joblib"):
        assert (directory / name).is_file()
    assert set(manifest["files"]) == {"meta.json", "schema.json", "labels.json", "fingerprint.json", "metrics.json", "model.joblib"}
    loaded = fl.load_learner(directory, trusted_root=tmp_path / "root", expected_manifest_sha256=manifest["manifest_sha256"])
    assert np.array_equal(loaded.predict_proba(frame), learner.predict_proba(frame))
    assert json.loads((directory / "metrics.json").read_text())["val_accuracy"] == 0.9


def test_artifact_tamper_detected_before_unpickle(tmp_path, monkeypatch):
    _, _, _ = _saved(tmp_path)
    directory = tmp_path / "root" / "model"
    with open(directory / "model.joblib", "ab") as handle:
        handle.write(b"x")
    import joblib

    monkeypatch.setattr(joblib, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("unpickled!")))
    with pytest.raises(ValueError, match="hash mismatch"):
        fl.load_learner(directory, trusted_root=tmp_path)


def test_artifact_unlisted_file_and_pin_and_root(tmp_path):
    _, _, manifest = _saved(tmp_path)
    directory = tmp_path / "root" / "model"
    with pytest.raises(ValueError, match="pinned"):
        fl.load_learner(directory, trusted_root=tmp_path, expected_manifest_sha256="0" * 64)
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(ValueError, match="outside the trusted root"):
        fl.load_learner(directory, trusted_root=other)
    (directory / "evil.pkl").write_bytes(b"not listed")
    with pytest.raises(ValueError, match="not in the manifest"):
        fl.load_learner(directory, trusted_root=tmp_path)


def test_artifact_manifest_edit_is_detected(tmp_path):
    _, _, _ = _saved(tmp_path)
    directory = tmp_path / "root" / "model"
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["files"]["model.joblib"] = "0" * 64
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        fl.load_learner(directory, trusted_root=tmp_path)


def test_save_rejects_missing_declared_model_files(tmp_path):
    frame, y, groups = _synthetic(n=100)
    learner = fl.make_learner("logreg").fit(frame, y, groups=groups)
    learner._save_model = lambda directory: None
    with pytest.raises(RuntimeError, match="were not written"):
        fl.save_learner(learner, tmp_path / "broken")


def test_save_refuses_existing_non_artifact_dir(tmp_path):
    frame, y, _ = _synthetic(n=100)
    learner = fl.make_learner("logreg").fit(frame, y)
    target = tmp_path / "x"
    target.mkdir()
    (target / "keep.txt").write_text("user data")
    with pytest.raises(ValueError, match="already exists"):
        fl.save_learner(learner, target)
    with pytest.raises(ValueError, match="not a learner artifact"):
        fl.save_learner(learner, target, overwrite=True)
    assert (target / "keep.txt").read_text() == "user data"


def test_nb_artifact_is_json_only(tmp_path):
    frame, y, _ = _synthetic(n=100)
    learner = fl.make_learner("nb").fit(frame, y)
    manifest = fl.save_learner(learner, tmp_path / "nb")
    assert "model.json" in manifest["files"] and "model.joblib" not in manifest["files"]
    loaded = fl.load_learner(tmp_path / "nb", trusted_root=tmp_path)
    assert np.allclose(loaded.predict_proba(frame), learner.predict_proba(frame))


# ----------------------------------------------------------------- samples


def test_training_sample_features_keep_legacy_digest():
    kwargs = dict(sample_id="s1", region_id="r", input_text="a 1", expected_label="x", expected_confidence=0.5,
                  provenance_refs=("p",))
    legacy = fbct.TrainingSample(**kwargs)
    assert "features" not in legacy.as_dict()
    with_features = fbct.TrainingSample(**kwargs, features={"deg": 1.5, "cls": "a", "flag": True, "gap": None})
    with_features.validate()
    assert with_features.as_dict()["features"]["deg"] == 1.5
    with pytest.raises(ValueError, match="invalid feature name"):
        fbct.TrainingSample(**kwargs, features={"bad name": 1}).validate()
    with pytest.raises(ValueError, match="finite"):
        fbct.TrainingSample(**kwargs, features={"x": float("nan")}).validate()


def test_features_frame_from_samples():
    samples = [{"features": {"b": 1.0, "a": "x"}, "input_text": "t1"}, {"features": {"b": 2.0}, "input_text": "t2"}]
    frame = fl.features_frame(samples)
    assert list(frame.columns) == ["a", "b", fl.TEXT_COLUMN]
    assert frame[fl.TEXT_COLUMN].tolist() == ["t1", "t2"]

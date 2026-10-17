"""Characterization tests for mlops/routing/ (mlops/routing/critic.py).

`mlops/routing/__init__.py` is empty, so `RetrievalCritic` in critic.py is the
entire public surface of the package.

These tests pin the CURRENT behaviour of the module, bugs included. They are a
safety net for a later refactor, not a statement of what the critic *should*
do. Where a test pins something that is arguably wrong the docstring says so,
and the finding is reported separately.

No external services are used. scikit-learn is a real, local, CPU-only
dependency and is exercised for real in a couple of end-to-end tests; every
other training test swaps in a fake estimator so the exact feature matrix
handed to the model can be asserted. Both module-level path constants
(`_LABELS_PATH`, `_MODEL_PATH`) point *inside the package directory*, so an
autouse fixture redirects them into tmp_path — without it, running this suite
would write critic_labels.jsonl / critic_model.pkl into the repo.
"""

import json
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import sklearn.linear_model  # noqa: E402

from mlops.routing.critic import RetrievalCritic  # noqa: E402
from mlops.routing import critic as critic_mod  # noqa: E402


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def sandbox_paths(tmp_path, monkeypatch):
    """Redirect the two module-level paths that would otherwise write into the repo."""
    labels = tmp_path / "critic_labels.jsonl"
    model = tmp_path / "critic_model.pkl"
    monkeypatch.setattr(critic_mod, "_LABELS_PATH", labels)
    monkeypatch.setattr(critic_mod, "_MODEL_PATH", model)
    return {"labels": labels, "model": model}


def write_labels(sandbox, rows):
    """Write `rows` (dicts, or raw strings for malformed lines) to the labels file."""
    lines = []
    for r in rows:
        lines.append(r if isinstance(r, str) else json.dumps(r))
    sandbox["labels"].write_text("\n".join(lines) + "\n")


def read_labels(sandbox):
    return [l for l in sandbox["labels"].read_text().split("\n") if l != ""]


# ── stub estimators (module level so pickle can find them) ────────────────────

class StubClf:
    """Returns a fixed positive-class probability and records what it was called with."""

    def __init__(self, proba=0.9):
        self.proba = proba
        self.calls = []

    def predict_proba(self, X):
        self.calls.append(X)
        return [[1.0 - self.proba, self.proba]]


class RaisingClf:
    def predict_proba(self, X):
        raise RuntimeError("model blew up")


class OneColumnClf:
    """A single-class estimator: predict_proba yields one column, so [0][1] IndexErrors."""

    def predict_proba(self, X):
        return [[1.0]]


class FakeLR:
    """Stand-in for LogisticRegression that captures the training matrix."""

    captured = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = list(X)
        self.y = list(y)
        FakeLR.captured = self
        return self

    def predict_proba(self, X):
        return [[0.25, 0.75]]


class UnpicklableLR(FakeLR):
    def __reduce__(self):
        raise TypeError("nope")


@pytest.fixture
def fake_lr(monkeypatch):
    FakeLR.captured = None
    monkeypatch.setattr(sklearn.linear_model, "LogisticRegression", FakeLR)
    return FakeLR


# ── module constants ──────────────────────────────────────────────────────────

def test_module_constants_are_the_documented_thresholds():
    assert critic_mod.MIN_SAMPLES_TRAIN == 50
    assert critic_mod.TOP_SCORE_THRESHOLD == 0.50
    assert critic_mod.MEAN_SCORE_THRESHOLD == 0.45


def test_state_paths_live_inside_the_package_directory():
    """Both persisted files are written next to critic.py; there is no env override.

    Pinning the real (unpatched) values, not the sandboxed ones: a refactor that
    moves this state out of the source tree changes deployment behaviour and
    should have to update this test deliberately.
    """
    pkg_dir = Path(critic_mod.__file__).parent
    # importlib.reload is not needed: read the defaults off a fresh import of the source.
    src = Path(critic_mod.__file__).read_text()
    assert 'Path(__file__).parent / "critic_labels.jsonl"' in src
    assert 'Path(__file__).parent / "critic_model.pkl"' in src
    assert pkg_dir.name == "routing"


# ── _feature_vector ───────────────────────────────────────────────────────────

def test_feature_vector_order_and_values():
    """Exact feature order: [top, mean, min, spread, n_chunks, n_query_words]."""
    c = RetrievalCritic()
    feats = c._feature_vector("alpha beta gamma", ["c1", "c2"], [0.8, 0.2])
    assert feats == [0.8, 0.5, 0.2, 0.6000000000000001, 2.0, 3.0]
    assert len(feats) == 6


def test_feature_vector_empty_scores_is_all_zeros():
    c = RetrievalCritic()
    assert c._feature_vector("", [], []) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_feature_vector_n_comes_from_chunks_not_scores():
    """The count feature is len(chunks); mismatched chunk/score lists are not validated."""
    c = RetrievalCritic()
    feats = c._feature_vector("q", [1, 2, 3], [0.9, 0.1])
    assert feats[4] == 3.0  # three chunks
    assert feats[0] == 0.9 and feats[2] == 0.1  # ...but only two scores


def test_feature_vector_query_length_is_whitespace_token_count():
    c = RetrievalCritic()
    assert c._feature_vector("   ", [], [])[5] == 0.0
    assert c._feature_vector("a-b c", [], [])[5] == 2.0
    assert c._feature_vector("a\tb\nc  d", [], [])[5] == 4.0


def test_feature_vector_does_not_coerce_score_derived_features_to_float():
    """Only n and the query length are float()-ed; int scores stay ints."""
    c = RetrievalCritic()
    feats = c._feature_vector("q", ["a", "b"], [1, 0])
    assert feats == [1, 0.5, 0, 1, 2.0, 1.0]
    assert isinstance(feats[0], int)
    assert isinstance(feats[4], float)


# ── route(): heuristic backend ────────────────────────────────────────────────

def test_route_return_shape_is_exactly_these_five_keys():
    c = RetrievalCritic()
    v = c.route("q", ["a"], [0.9])
    assert set(v) == {"pass", "reason", "confidence", "backend", "scores"}
    assert set(v["scores"]) == {"top", "mean"}
    assert isinstance(v["pass"], bool)


def test_route_passes_when_both_thresholds_are_exactly_met():
    """Both comparisons are >=, so the threshold values themselves pass."""
    c = RetrievalCritic()
    v = c.route("q", ["a", "b"], [0.50, 0.40])  # top 0.50, mean 0.45
    assert v["pass"] is True
    assert v["reason"] == "heuristic ok"
    assert v["backend"] == "heuristic"
    assert v["confidence"] == 0.50
    assert v["scores"] == {"top": 0.50, "mean": 0.45}


def test_route_fails_just_below_the_top_threshold():
    c = RetrievalCritic()
    v = c.route("q", ["a"], [0.4999])
    assert v["pass"] is False
    assert v["reason"] == "top_score=0.500<0.5"  # mean 0.4999 still clears 0.45


def test_route_reason_rounding_can_contradict_the_verdict():
    """The reason renders scores to 3dp, so a near-threshold failure reads as
    "top_score=0.500<0.5" — a pass-looking number attached to a fail. Pinned as-is."""
    c = RetrievalCritic()
    v = c.route("q", ["a"], [0.4999])
    assert "0.500<0.5" in v["reason"]
    assert v["pass"] is False


def test_route_reports_only_the_failing_threshold():
    c = RetrievalCritic()
    # top clears 0.50, mean (0.3) does not clear 0.45
    v = c.route("q", ["a", "b", "c"], [0.9, 0.0, 0.0])
    assert v["pass"] is False
    assert v["reason"] == "mean_score=0.300<0.45"


def test_route_reports_both_failing_thresholds_separated_by_semicolon():
    c = RetrievalCritic()
    v = c.route("q", ["a"], [0.1, 0.2])
    assert v["reason"] == "top_score=0.200<0.5; mean_score=0.150<0.45"


def test_route_with_empty_scores_fails_closed():
    """No scores at all => top/mean of 0.0 => a normal failing verdict, not an error."""
    c = RetrievalCritic()
    v = c.route("q", [], [])
    assert v == {
        "pass": False,
        "reason": "top_score=0.000<0.5; mean_score=0.000<0.45",
        "confidence": 0.0,
        "backend": "heuristic",
        "scores": {"top": 0.0, "mean": 0.0},
    }


def test_route_ignores_chunk_contents_entirely():
    """Only the scores (and, for the classifier, counts) matter — chunk text is never read."""
    c = RetrievalCritic()
    good = [0.9, 0.9]
    assert c.route("q", ["", ""], good)["pass"] is True
    assert c.route("q", [None, None], good)["pass"] is True
    assert c.route("q", [object(), object()], good)["pass"] is True


def test_route_does_not_require_chunks_and_scores_to_be_the_same_length():
    """An empty chunk list with strong scores still passes: no consistency check."""
    c = RetrievalCritic()
    v = c.route("q", [], [0.9, 0.9])
    assert v["pass"] is True
    assert v["scores"] == {"top": 0.9, "mean": 0.9}


def test_route_mean_is_raw_float_division_with_no_rounding():
    c = RetrievalCritic()
    v = c.route("q", ["a", "b"], [0.1, 0.2])
    assert v["scores"]["mean"] == 0.15000000000000002


def test_route_heuristic_confidence_is_the_raw_top_score_not_a_probability():
    """confidence is unbounded on the heuristic path — it can exceed 1.0 and be an int.

    Callers that treat confidence uniformly across backends will be wrong here.
    """
    c = RetrievalCritic()
    v = c.route("q", ["a"], [3])
    assert v["confidence"] == 3
    assert isinstance(v["confidence"], int)
    assert v["pass"] is True

    fail = c.route("q", ["a", "b"], [0.44, 0.44])
    assert fail["pass"] is False
    assert fail["confidence"] == 0.44  # confidence is reported even when the verdict fails


def test_route_ignores_query_content_on_the_heuristic_path():
    c = RetrievalCritic()
    a = c.route("", ["x"], [0.9])
    b = c.route("a very long query with many words indeed", ["x"], [0.9])
    assert a == b


def test_route_negative_scores_fail():
    c = RetrievalCritic()
    v = c.route("q", ["a"], [-1.0, -2.0])
    assert v["pass"] is False
    assert v["scores"] == {"top": -1.0, "mean": -1.5}


# ── route(): classifier backend ───────────────────────────────────────────────

def test_route_uses_classifier_when_loaded():
    c = RetrievalCritic()
    c._clf = StubClf(proba=0.9)
    v = c.route("two words", ["a"], [0.01])  # scores that the heuristic would reject
    assert v["pass"] is True
    assert v["backend"] == "classifier"
    assert v["reason"] == "classifier proba=0.900"
    assert v["confidence"] == 0.9
    # ...but the reported scores are still the heuristic's raw stats
    assert v["scores"] == {"top": 0.01, "mean": 0.01}


def test_classifier_receives_the_feature_vector_wrapped_in_a_list():
    c = RetrievalCritic()
    stub = StubClf()
    c._clf = stub
    c.route("two words", ["a", "b"], [0.8, 0.2])
    assert stub.calls == [[[0.8, 0.5, 0.2, 0.6000000000000001, 2.0, 2.0]]]


def test_classifier_threshold_is_half_inclusive():
    c = RetrievalCritic()
    c._clf = StubClf(proba=0.5)
    assert c.route("q", ["a"], [0.9])["pass"] is True
    c._clf = StubClf(proba=0.49999)
    assert c.route("q", ["a"], [0.9])["pass"] is False


def test_classifier_can_reject_what_the_heuristic_would_pass():
    c = RetrievalCritic()
    c._clf = StubClf(proba=0.1)
    v = c.route("q", ["a"], [0.99, 0.99])
    assert v["pass"] is False
    assert v["backend"] == "classifier"
    assert v["confidence"] == 0.1


def test_classifier_confidence_is_coerced_to_float():
    class IntProbaClf:
        def predict_proba(self, X):
            return [[0, 1]]

    c = RetrievalCritic()
    c._clf = IntProbaClf()
    v = c.route("q", ["a"], [0.9])
    assert v["confidence"] == 1.0
    assert isinstance(v["confidence"], float)


def test_route_falls_back_to_heuristic_silently_when_the_classifier_raises():
    """Any classifier exception degrades to the heuristic and reports backend="heuristic".

    The caller gets no signal that a trained model exists and is failing.
    """
    c = RetrievalCritic()
    c._clf = RaisingClf()
    v = c.route("q", ["a", "b"], [0.9, 0.9])
    assert v["backend"] == "heuristic"
    assert v["pass"] is True
    assert v["reason"] == "heuristic ok"
    assert c._clf is not None  # the broken model is kept, so every call re-fails


def test_single_class_classifier_degrades_to_heuristic_via_indexerror():
    c = RetrievalCritic()
    c._clf = OneColumnClf()
    v = c.route("q", ["a"], [0.1])
    assert v["backend"] == "heuristic"
    assert v["pass"] is False


# ── classifier loading ────────────────────────────────────────────────────────

def test_no_model_file_means_no_classifier():
    c = RetrievalCritic()
    assert c._clf is None
    assert c.route("q", ["a"], [0.9])["backend"] == "heuristic"


def test_corrupt_model_file_is_swallowed_and_leaves_clf_none(sandbox_paths):
    sandbox_paths["model"].write_bytes(b"this is not a pickle")
    c = RetrievalCritic()
    assert c._clf is None
    assert c.route("q", ["a"], [0.9])["backend"] == "heuristic"


def test_empty_model_file_is_swallowed(sandbox_paths):
    sandbox_paths["model"].write_bytes(b"")
    assert RetrievalCritic()._clf is None


def test_model_file_is_loaded_with_bare_pickle_at_construction(sandbox_paths):
    """Whatever the pickle contains is adopted as the classifier — no type check.

    This is arbitrary-code-execution-by-file-drop; pinned because it is current behaviour.
    """
    sandbox_paths["model"].write_bytes(pickle.dumps(StubClf(proba=0.8)))
    c = RetrievalCritic()
    assert isinstance(c._clf, StubClf)
    assert c.route("q", ["a"], [0.0])["reason"] == "classifier proba=0.800"


def test_model_is_loaded_once_at_construction_not_per_call(sandbox_paths):
    c = RetrievalCritic()
    assert c._clf is None
    sandbox_paths["model"].write_bytes(pickle.dumps(StubClf(proba=0.8)))
    # the already-constructed critic keeps using the heuristic
    assert c.route("q", ["a"], [0.9])["backend"] == "heuristic"
    assert RetrievalCritic().route("q", ["a"], [0.9])["backend"] == "classifier"


# ── record_label ──────────────────────────────────────────────────────────────

def test_record_label_writes_one_json_line_with_four_keys(sandbox_paths):
    c = RetrievalCritic()
    c.record_label("hello there", ["chunk one", "chunk two"], [0.9, 0.8], 1)
    lines = read_labels(sandbox_paths)
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "query": "hello there",
        "n_chunks": 2,
        "scores": [0.9, 0.8],
        "label": 1,
    }


def test_record_label_discards_chunk_text(sandbox_paths):
    """Only the chunk *count* is persisted, so labels can never be re-featurised
    against the chunk bodies later."""
    c = RetrievalCritic()
    c.record_label("q", ["secret body text"], [0.9], 1)
    row = json.loads(read_labels(sandbox_paths)[0])
    assert "chunks" not in row
    assert row["n_chunks"] == 1
    assert "secret" not in read_labels(sandbox_paths)[0]


def test_record_label_appends(sandbox_paths):
    c = RetrievalCritic()
    c.record_label("a", [], [0.1], 0)
    c.record_label("b", ["x"], [0.9], 1)
    RetrievalCritic().record_label("c", ["x", "y"], [0.5], 1)
    rows = [json.loads(l) for l in read_labels(sandbox_paths)]
    assert [r["query"] for r in rows] == ["a", "b", "c"]


def test_record_label_does_not_validate_the_label(sandbox_paths):
    """Anything JSON-serialisable is accepted as a label, including non-binary values."""
    c = RetrievalCritic()
    c.record_label("q", [], [], 7)
    c.record_label("q", [], [], "yes")
    c.record_label("q", [], [], None)
    labels = [json.loads(l)["label"] for l in read_labels(sandbox_paths)]
    assert labels == [7, "yes", None]


def test_record_label_returns_none_and_does_not_retrain(sandbox_paths):
    c = RetrievalCritic()
    assert c.record_label("q", [], [0.9], 1) is None
    assert c._clf is None


def test_record_label_raises_if_the_parent_directory_is_missing(sandbox_paths, monkeypatch):
    monkeypatch.setattr(
        critic_mod, "_LABELS_PATH", sandbox_paths["labels"].parent / "nope" / "l.jsonl"
    )
    with pytest.raises(FileNotFoundError):
        RetrievalCritic().record_label("q", [], [0.9], 1)


def test_record_label_propagates_json_serialisation_errors(sandbox_paths):
    """A non-serialisable score type raises out of record_label (after the file is opened)."""
    with pytest.raises(TypeError):
        RetrievalCritic().record_label("q", [], [object()], 1)


# ── train(): refusal paths ────────────────────────────────────────────────────

def test_train_without_a_labels_file():
    assert RetrievalCritic().train() == {"trained": False, "reason": "no labels file"}


def test_train_below_min_samples_reports_the_counts(sandbox_paths):
    write_labels(sandbox_paths, [{"query": "q", "scores": [0.9], "label": 1}] * 3)
    assert RetrievalCritic().train() == {
        "trained": False,
        "reason": "need 50 samples, have 3",
    }


def test_train_min_samples_is_overridable_with_no_floor(sandbox_paths, fake_lr):
    """min_samples has no lower bound: a 2-sample model trains and is deployed."""
    write_labels(
        sandbox_paths,
        [
            {"query": "q", "n_chunks": 1, "scores": [0.9], "label": 1},
            {"query": "q", "n_chunks": 1, "scores": [0.1], "label": 0},
        ],
    )
    c = RetrievalCritic()
    assert c.train(min_samples=2) == {"trained": True, "n_samples": 2}
    assert isinstance(c._clf, FakeLR)


def test_train_skips_malformed_json_lines_and_does_not_count_them(sandbox_paths):
    write_labels(
        sandbox_paths,
        [
            {"query": "q", "scores": [0.9], "label": 1},
            "not json at all",
            "",
            "{broken",
            {"query": "q", "scores": [0.1], "label": 0},
        ],
    )
    assert RetrievalCritic().train(min_samples=3) == {
        "trained": False,
        "reason": "need 3 samples, have 2",
    }


def test_train_crashes_on_valid_json_that_is_not_an_object(sandbox_paths):
    """`null`/numbers/arrays parse fine, count toward the quota, then AttributeError
    out of the whole training run — one bad line poisons the batch."""
    write_labels(sandbox_paths, ["null", "null"])
    assert RetrievalCritic().train(min_samples=2) == {
        "trained": False,
        "reason": "'NoneType' object has no attribute 'get'",
    }

    write_labels(sandbox_paths, ["123", "456"])
    assert RetrievalCritic().train(min_samples=2) == {
        "trained": False,
        "reason": "'int' object has no attribute 'get'",
    }


def test_train_reports_missing_sklearn(sandbox_paths, monkeypatch):
    write_labels(sandbox_paths, [{"query": "q", "scores": [0.9], "label": i % 2} for i in range(4)])
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", None)
    assert RetrievalCritic().train(min_samples=2) == {
        "trained": False,
        "reason": "scikit-learn not installed",
    }


def test_train_surfaces_sklearn_errors_as_the_reason_string(sandbox_paths):
    """A single-class label set is a real, reachable failure (all-positive feedback)."""
    write_labels(sandbox_paths, [{"query": "q", "scores": [0.9], "label": 1}] * 4)
    out = RetrievalCritic().train(min_samples=2)
    assert out["trained"] is False
    assert "at least 2 classes" in out["reason"]


def test_train_with_zero_min_samples_and_an_empty_file(sandbox_paths):
    """min_samples=0 defeats the guard entirely and hands sklearn an empty matrix."""
    sandbox_paths["labels"].write_text("")
    out = RetrievalCritic().train(min_samples=0)
    assert out["trained"] is False
    assert "Expected 2D array" in out["reason"]


def test_failed_training_leaves_the_existing_classifier_in_place(sandbox_paths):
    write_labels(sandbox_paths, [{"query": "q", "scores": [0.9], "label": 1}] * 4)
    c = RetrievalCritic()
    stub = StubClf(proba=0.7)
    c._clf = stub
    assert c.train(min_samples=2)["trained"] is False
    assert c._clf is stub
    assert c.route("q", ["a"], [0.0])["backend"] == "classifier"


def test_pickle_failure_is_reported_and_the_model_is_not_adopted(sandbox_paths, monkeypatch):
    write_labels(
        sandbox_paths,
        [
            {"query": "q", "n_chunks": 1, "scores": [0.9], "label": 1},
            {"query": "q", "n_chunks": 1, "scores": [0.1], "label": 0},
        ],
    )
    monkeypatch.setattr(sklearn.linear_model, "LogisticRegression", UnpicklableLR)
    c = RetrievalCritic()
    assert c.train(min_samples=2) == {"trained": False, "reason": "nope"}
    assert c._clf is None  # self._clf is assigned only after a successful dump


def test_pickle_failure_truncates_a_previously_good_model_file(sandbox_paths, monkeypatch):
    """The output file is opened "wb" before dumping, so a failed retrain destroys
    the model on disk: the next process starts with no classifier at all."""
    sandbox_paths["model"].write_bytes(pickle.dumps(StubClf(proba=0.8)))
    assert RetrievalCritic()._clf is not None

    write_labels(
        sandbox_paths,
        [
            {"query": "q", "n_chunks": 1, "scores": [0.9], "label": 1},
            {"query": "q", "n_chunks": 1, "scores": [0.1], "label": 0},
        ],
    )
    monkeypatch.setattr(sklearn.linear_model, "LogisticRegression", UnpicklableLR)
    assert RetrievalCritic().train(min_samples=2)["trained"] is False

    assert sandbox_paths["model"].stat().st_size == 0
    assert RetrievalCritic()._clf is None


# ── train(): feature construction ─────────────────────────────────────────────

def test_train_feature_matrix_matches_the_serving_feature_order(sandbox_paths, fake_lr):
    write_labels(
        sandbox_paths,
        [
            {"query": "a b", "n_chunks": 2, "scores": [0.9, 0.8], "label": 1},
            {"query": "c", "n_chunks": 1, "scores": [0.1], "label": 0},
        ],
    )
    c = RetrievalCritic()
    assert c.train(min_samples=2)["trained"] is True
    fitted = fake_lr.captured
    assert fitted.X == [
        [0.9, 0.8500000000000001, 0.8, 0.09999999999999998, 2.0, 2.0],
        [0.1, 0.1, 0.1, 0.0, 1.0, 1.0],
    ]
    assert fitted.y == [1, 0]
    # and the serving path builds the identical row for the same inputs
    assert c._feature_vector("a b", ["x", "y"], [0.9, 0.8]) == fitted.X[0]


def test_train_falls_back_to_len_scores_when_n_chunks_is_absent(sandbox_paths, fake_lr):
    write_labels(
        sandbox_paths,
        [
            {"query": "a", "scores": [0.9, 0.8, 0.7], "label": 1},
            {"query": "b", "scores": [0.1], "label": 0},
        ],
    )
    assert RetrievalCritic().train(min_samples=2)["trained"] is True
    assert [row[4] for row in fake_lr.captured.X] == [3.0, 1.0]


def test_train_treats_a_row_with_no_scores_as_all_zero_features(sandbox_paths, fake_lr):
    write_labels(
        sandbox_paths,
        [
            {"query": "", "scores": [], "label": 1},
            {"query": "b", "n_chunks": 1, "scores": [0.1], "label": 0},
        ],
    )
    assert RetrievalCritic().train(min_samples=2)["trained"] is True
    assert fake_lr.captured.X[0] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_train_defaults_a_missing_label_to_negative(sandbox_paths, fake_lr):
    """A record written without a label silently becomes a *negative* training example."""
    write_labels(
        sandbox_paths,
        [
            {"query": "a", "n_chunks": 1, "scores": [0.9]},
            {"query": "b", "n_chunks": 1, "scores": [0.1], "label": 1},
        ],
    )
    assert RetrievalCritic().train(min_samples=2)["trained"] is True
    assert fake_lr.captured.y == [0, 1]


def test_train_coerces_string_labels_via_int(sandbox_paths, fake_lr):
    write_labels(
        sandbox_paths,
        [
            {"query": "a", "n_chunks": 1, "scores": [0.9], "label": "1"},
            {"query": "b", "n_chunks": 1, "scores": [0.1], "label": False},
        ],
    )
    assert RetrievalCritic().train(min_samples=2)["trained"] is True
    assert fake_lr.captured.y == [1, 0]


def test_train_is_constructed_with_max_iter_500(sandbox_paths, fake_lr):
    write_labels(
        sandbox_paths,
        [
            {"query": "a", "n_chunks": 1, "scores": [0.9], "label": 1},
            {"query": "b", "n_chunks": 1, "scores": [0.1], "label": 0},
        ],
    )
    RetrievalCritic().train(min_samples=2)
    assert fake_lr.captured.kwargs == {"max_iter": 500}


def test_train_n_samples_counts_parsed_rows_not_fitted_rows(sandbox_paths, fake_lr):
    write_labels(
        sandbox_paths,
        [{"query": "q", "n_chunks": 1, "scores": [0.9], "label": i % 2} for i in range(6)]
        + ["garbage"],
    )
    assert RetrievalCritic().train(min_samples=2) == {"trained": True, "n_samples": 6}


# ── end-to-end with real scikit-learn (local, CPU-only) ───────────────────────

def test_end_to_end_record_train_and_route(sandbox_paths):
    c = RetrievalCritic()
    for _ in range(10):
        c.record_label("a good query here", ["x", "y"], [0.95, 0.90], 1)
        c.record_label("a bad query here", ["x", "y"], [0.05, 0.02], 0)

    out = c.train(min_samples=20)
    assert out == {"trained": True, "n_samples": 20}
    assert sandbox_paths["model"].stat().st_size > 0

    good = c.route("a good query here", ["x", "y"], [0.95, 0.90])
    bad = c.route("a bad query here", ["x", "y"], [0.05, 0.02])
    assert good["backend"] == bad["backend"] == "classifier"
    assert good["pass"] is True
    assert bad["pass"] is False
    assert 0.0 <= good["confidence"] <= 1.0
    assert good["reason"] == "classifier proba={:.3f}".format(good["confidence"])

    # a fresh critic picks the persisted model up and agrees exactly
    reloaded = RetrievalCritic()
    assert reloaded._clf is not None
    assert reloaded.route("a good query here", ["x", "y"], [0.95, 0.90]) == good


def test_trained_classifier_overrides_the_heuristic_thresholds(sandbox_paths):
    """Once trained on labels that call low scores good, the critic passes inputs the
    heuristic would have rejected — the thresholds are not a floor."""
    c = RetrievalCritic()
    for _ in range(10):
        c.record_label("q", ["x"], [0.10], 1)   # low score labelled GOOD
        c.record_label("q", ["x"], [0.90], 0)   # high score labelled BAD
    assert c.train(min_samples=20)["trained"] is True

    v = c.route("q", ["x"], [0.10])
    assert v["backend"] == "classifier"
    assert v["pass"] is True
    assert RetrievalCritic().route("q", ["x"], [0.10])["pass"] is True
    # the pure heuristic would have failed this input
    assert RetrievalCritic()._feature_vector("q", ["x"], [0.10])[0] < critic_mod.TOP_SCORE_THRESHOLD

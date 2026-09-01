"""Characterization tests for the mlops/grounding/ package.

Covers mlops/grounding/canary.py, train.py and active_learn.py.

These tests pin the CURRENT behaviour of the package, bugs included. They are a
safety net for a later refactor, not a specification of what the grounding
MLOps code *should* do. Where a test pins something that is arguably wrong, the
docstring says so and the finding is reported separately.

No external services are used: Ollama HTTP calls are replaced with in-process
fakes, and every module-level path constant that would otherwise write into the
repository (canary_history.jsonl, promotions.jsonl, monitor_history.jsonl,
.emb_cache.npz) is redirected into a tmp dir by an autouse fixture.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import joblib  # noqa: E402
from mlops.grounding import active_learn as al  # noqa: E402
from mlops.grounding import canary, train  # noqa: E402


# ── shared fakes ──────────────────────────────────────────────────────────────

def _prefix_vec(text):
    """Deterministic 2-D unit embedding keyed on the first whitespace token.

    "alpha ..." -> [1, 0], "beta ..." -> [0, 1], anything else -> [0.6, 0.8].
    Gives exactly predictable cosines: 1.0 for a matching prefix, 0.0 otherwise.
    """
    head = text.split()[0] if text.split() else ""
    if head == "alpha":
        return [1.0, 0.0]
    if head == "beta":
        return [0.0, 1.0]
    return [0.6, 0.8]


def fake_embed(texts, ollama_url):
    """Drop-in for canary._embed — never touches the network."""
    return np.array([_prefix_vec(t) for t in texts], dtype=np.float32)


class CosClf:
    """Stub classifier whose P(class=1) is just the cosine feature (last column)."""

    def __init__(self):
        self.seen = []

    def predict_proba(self, X):
        X = np.asarray(X)
        self.seen.append(X)
        p1 = X[:, -1].astype(np.float64)
        return np.stack([1.0 - p1, p1], axis=1)


class ConstClf:
    """Stub classifier returning a constant P(class=1)."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        return np.stack([np.full(n, 1.0 - self.p), np.full(n, self.p)], axis=1)


@pytest.fixture(autouse=True)
def isolate_module_paths(tmp_path, monkeypatch):
    """Redirect every module-level write target away from the repo."""
    monkeypatch.setattr(canary, "_HISTORY_PATH", tmp_path / "canary_history.jsonl")
    monkeypatch.setattr(canary, "_PROMOTIONS_PATH", tmp_path / "promotions.jsonl")
    monkeypatch.setattr(canary, "_MONITOR_PATH", tmp_path / "monitor_history.jsonl")
    monkeypatch.setattr(train, "CACHE_PATH", str(tmp_path / "emb_cache.npz"))
    yield


@pytest.fixture
def no_network_embed(monkeypatch):
    monkeypatch.setattr(canary, "_embed", fake_embed)


@pytest.fixture
def stub_joblib_load(monkeypatch):
    """Make joblib.load return a registered stub instead of unpickling."""
    registry = {}

    def _load(path):
        key = str(path)
        if key not in registry:
            raise AssertionError(f"unexpected joblib.load({key!r})")
        return registry[key]

    monkeypatch.setattr(joblib, "load", _load)

    def register(path, obj):
        Path(path).write_bytes(b"stub")  # so os.path.exists() checks pass
        registry[str(path)] = obj
        return str(path)

    return register


def write_findings(dir_path, records):
    dir_path.mkdir(parents=True, exist_ok=True)
    p = dir_path / "findings.jsonl"
    with open(p, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


def standard_run_records():
    """12 findings over 2 targets, the last one deliberately mislabelled.

    Findings 0-5:  "alpha ..." tagged dt_target:alpha   (cos 1.0 vs alpha)
    Findings 6-10: "beta ..."  tagged dt_target:beta    (cos 1.0 vs beta)
    Finding 11:    "beta ..."  tagged dt_target:alpha   (the traitor)

    Under the fake embedding and the cosine gate this yields, at threshold 0.5,
    TP=11 FP=1 FN=1 TN=11 -> precision = recall = f1 = 11/12.
    """
    recs = [{"text": f"alpha finding {i}", "tags": ["dt_target:alpha"]} for i in range(6)]
    recs += [{"text": f"beta finding {i}", "tags": ["dt_target:beta"]} for i in range(5)]
    recs += [{"text": "beta traitor finding", "tags": ["dt_target:alpha"]}]
    return recs


@pytest.fixture
def one_run_glob(tmp_path):
    write_findings(tmp_path / "runs" / "dt-loci-1", standard_run_records())
    return str(tmp_path / "runs" / "*" / "findings.jsonl")


# ══════════════════════════════════════════════════════════════════════════════
# canary.py — module constants
# ══════════════════════════════════════════════════════════════════════════════

def test_canary_constants():
    """Tuning knobs other code (loop.py, CI) depends on."""
    assert canary.DEFAULT_MIN_MARGIN == 0.02
    assert canary.HISTORY_WINDOW == 10
    assert canary.MIN_FINDINGS_PER_RUN == 10
    assert canary.MONITOR_ROLLBACK_WINDOW == 2
    assert canary.DEFAULT_THRESHOLD == float(os.environ.get("DTL_GROUND_THRESHOLD", "0.59"))
    assert canary.DEFAULT_OLLAMA == (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    assert canary.DEFAULT_FINDINGS_GLOB.endswith("/.hermes/memory-sessions/dt-loci-*/findings.jsonl")


# ══════════════════════════════════════════════════════════════════════════════
# canary._dt_target
# ══════════════════════════════════════════════════════════════════════════════

def test_dt_target_extracts_tag_value():
    assert canary._dt_target({"tags": ["dt_phase:recon", "dt_target:alpha"]}) == "alpha"


def test_dt_target_keeps_everything_after_the_first_colon():
    """Only the first colon separates key from value."""
    assert canary._dt_target({"tags": ["dt_target:a:b:c"]}) == "a:b:c"


def test_dt_target_last_duplicate_tag_wins():
    assert canary._dt_target({"tags": ["dt_target:first", "dt_target:second"]}) == "second"


def test_dt_target_empty_value_is_empty_string_not_none():
    """"dt_target:" yields "" — falsy, so _load_runs drops the finding."""
    assert canary._dt_target({"tags": ["dt_target:"]}) == ""


@pytest.mark.parametrize("finding", [
    {},
    {"tags": None},
    {"tags": []},
    {"tags": ["nocolon"]},
    {"tags": ["dt_phase:recon"]},
])
def test_dt_target_returns_none_when_absent(finding):
    assert canary._dt_target(finding) is None


# ══════════════════════════════════════════════════════════════════════════════
# canary._metrics
# ══════════════════════════════════════════════════════════════════════════════

def test_metrics_basic_confusion_matrix():
    m = canary._metrics([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2], 0.5)
    assert m == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "accuracy": 0.5}


def test_metrics_threshold_is_inclusive():
    """keep = score >= threshold, so a score exactly on the threshold is kept."""
    assert canary._metrics([1], [0.5], 0.5)["recall"] == 1.0
    assert canary._metrics([1], [0.499999], 0.5)["recall"] == 0.0


def test_metrics_returns_zeros_rather_than_dividing_by_zero():
    m = canary._metrics([], [], 0.5)
    assert m == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}


def test_metrics_all_negative_labels_gives_zero_f1_but_real_accuracy():
    m = canary._metrics([0, 0], [0.9, 0.1], 0.5)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert m["accuracy"] == 0.5


def test_metrics_keys_and_types():
    m = canary._metrics([1, 0], [0.9, 0.1], 0.5)
    assert set(m) == {"precision", "recall", "f1", "accuracy"}
    assert all(isinstance(v, float) for v in m.values())


# ══════════════════════════════════════════════════════════════════════════════
# canary._load_runs
# ══════════════════════════════════════════════════════════════════════════════

def test_load_runs_keys_on_parent_directory_name(tmp_path):
    write_findings(tmp_path / "dt-loci-abc", standard_run_records())
    runs = canary._load_runs(str(tmp_path / "*" / "findings.jsonl"))
    assert list(runs) == ["dt-loci-abc"]
    assert len(runs["dt-loci-abc"]) == 12
    assert runs["dt-loci-abc"][0] == {"text": "alpha finding 0", "target": "alpha"}


def test_load_runs_drops_runs_below_min_findings(tmp_path):
    """MIN_FINDINGS_PER_RUN is a >= boundary, counted AFTER tag filtering."""
    write_findings(tmp_path / "small", [{"text": f"alpha {i}", "tags": ["dt_target:alpha"]} for i in range(9)])
    write_findings(tmp_path / "exact", [{"text": f"alpha {i}", "tags": ["dt_target:alpha"]} for i in range(10)])
    runs = canary._load_runs(str(tmp_path / "*" / "findings.jsonl"))
    assert list(runs) == ["exact"]


def test_load_runs_untagged_findings_do_not_count_towards_the_minimum(tmp_path):
    recs = [{"text": f"alpha {i}", "tags": ["dt_target:alpha"]} for i in range(9)]
    recs += [{"text": "untagged"} for _ in range(50)]
    write_findings(tmp_path / "run", recs)
    assert canary._load_runs(str(tmp_path / "*" / "findings.jsonl")) == {}


def test_load_runs_skips_blank_and_malformed_lines(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    with open(d / "findings.jsonl", "w") as fh:
        for i in range(10):
            fh.write(json.dumps({"text": f"alpha {i}", "tags": ["dt_target:alpha"]}) + "\n")
        fh.write("\n")
        fh.write("   \n")
        fh.write("{not json\n")
        fh.write(json.dumps({"text": "no tags here"}) + "\n")
    runs = canary._load_runs(str(tmp_path / "*" / "findings.jsonl"))
    assert len(runs["run"]) == 10


def test_load_runs_truncates_text_to_2000_chars_and_nulls_become_empty(tmp_path):
    recs = [{"text": "alpha " + "x" * 5000, "tags": ["dt_target:alpha"]}]
    recs += [{"text": None, "tags": ["dt_target:alpha"]}]
    recs += [{"text": f"alpha {i}", "tags": ["dt_target:alpha"]} for i in range(8)]
    write_findings(tmp_path / "run", recs)
    findings = canary._load_runs(str(tmp_path / "*" / "findings.jsonl"))["run"]
    assert len(findings[0]["text"]) == 2000
    assert findings[1]["text"] == ""


def test_load_runs_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_findings(tmp_path / "sessions" / "dt-loci-9", standard_run_records())
    runs = canary._load_runs("~/sessions/dt-loci-*/findings.jsonl")
    assert list(runs) == ["dt-loci-9"]


def test_load_runs_no_matches_is_empty_dict(tmp_path):
    assert canary._load_runs(str(tmp_path / "nothing" / "*.jsonl")) == {}


# ══════════════════════════════════════════════════════════════════════════════
# canary._eval_run
# ══════════════════════════════════════════════════════════════════════════════

def test_eval_run_enumerates_every_target_x_finding_pair(no_network_embed):
    findings = [{"text": r["text"], "target": canary._dt_target(r)} for r in standard_run_records()]
    out = canary._eval_run(findings, 0.5, None, "http://unused")
    assert out["n_pairs"] == 2 * 12
    assert out["model"] is None
    assert out["cosine"]["f1"] == pytest.approx(11 / 12)
    assert out["cosine"]["precision"] == pytest.approx(11 / 12)
    assert out["cosine"]["accuracy"] == pytest.approx(22 / 24)


def test_eval_run_feature_layout_is_absdiff_prod_cos(no_network_embed):
    """A model that does not say what it wants is fed the legacy 2*d+1 layout:
    [|f-q|, f*q, cos]. That is the shipped classifier's contract."""
    findings = [{"text": "alpha one", "target": "alpha"}, {"text": "beta two", "target": "beta"}]
    clf = CosClf()
    canary._eval_run(findings, 0.5, clf, "http://unused")
    X = clf.seen[0]
    assert X.shape == (4, 2 * 2 + 1)
    # order: target "alpha" x [alpha one, beta two], then target "beta" x ...
    # q=[1,0], f=[1,0]
    assert list(X[0]) == pytest.approx([0.0, 0.0, 1.0, 0.0, 1.0])
    # q=[1,0], f=[0,1]
    assert list(X[1]) == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0])
    # q=[0,1], f=[1,0]
    assert list(X[2]) == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0])
    # q=[0,1], f=[0,1]
    assert list(X[3]) == pytest.approx([0.0, 0.0, 0.0, 1.0, 1.0])


def test_eval_run_model_metrics_use_a_hardcoded_half_threshold(no_network_embed):
    """The candidate model is always cut at 0.5, never at `threshold`."""
    findings = [{"text": r["text"], "target": canary._dt_target(r)} for r in standard_run_records()]
    out = canary._eval_run(findings, 0.0, CosClf(), "http://unused")
    # cosine at threshold 0.0 keeps everything
    assert out["cosine"]["recall"] == 1.0
    assert out["cosine"]["precision"] == pytest.approx(0.5)
    # the model sees the same cosines but is cut at 0.5
    assert out["model"]["f1"] == pytest.approx(11 / 12)


def test_eval_run_returns_none_for_empty_findings(no_network_embed):
    assert canary._eval_run([], 0.5, None, "http://unused") is None


def test_eval_run_deduplicates_texts_but_not_findings(no_network_embed):
    """Repeated text is embedded once yet still contributes one pair per finding."""
    findings = [{"text": "alpha same", "target": "alpha"}] * 3
    out = canary._eval_run(findings, 0.5, None, "http://unused")
    assert out["n_pairs"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# canary.evaluate_gate
# ══════════════════════════════════════════════════════════════════════════════

def test_evaluate_gate_no_runs_returns_nan_shaped_hold():
    out = canary.evaluate_gate(findings_glob="/nonexistent/*/findings.jsonl")
    assert out["n_runs"] == 0 and out["n_findings"] == 0
    assert out["decision"] == "HOLD" and out["beat_baseline"] is False
    assert np.isnan(out["delta_f1"])
    for side in ("cosine", "model"):
        assert np.isnan(out[side]["mean_f1"]) and np.isnan(out[side]["std_f1"])
        assert out[side]["per_run"] == []
    assert out["min_beat_margin"] == canary.DEFAULT_MIN_MARGIN
    assert set(out) == {"n_runs", "n_findings", "cosine", "model", "beat_baseline",
                        "delta_f1", "min_beat_margin", "decision"}


def test_evaluate_gate_without_candidate_always_holds(no_network_embed, one_run_glob):
    out = canary.evaluate_gate(findings_glob=one_run_glob, candidate_model_path=None, threshold=0.5)
    assert out["n_runs"] == 1 and out["n_findings"] == 12
    assert out["cosine"]["mean_f1"] == pytest.approx(11 / 12)
    assert out["cosine"]["std_f1"] == 0.0
    assert np.isnan(out["model"]["mean_f1"])
    assert out["model"]["per_run"] == []
    assert out["decision"] == "HOLD" and out["beat_baseline"] is False
    assert np.isnan(out["delta_f1"])


def test_evaluate_gate_per_run_entries_carry_run_id_and_metrics(no_network_embed, one_run_glob):
    out = canary.evaluate_gate(findings_glob=one_run_glob, threshold=0.5)
    (entry,) = out["cosine"]["per_run"]
    assert entry["run_id"] == "dt-loci-1"
    assert set(entry) == {"run_id", "precision", "recall", "f1", "accuracy"}


def test_evaluate_gate_promotes_when_model_beats_cosine(no_network_embed, one_run_glob,
                                                        stub_joblib_load, tmp_path):
    """Cosine is crippled with threshold 0.0; the model (cut at 0.5) wins."""
    path = stub_joblib_load(tmp_path / "cand.joblib", CosClf())
    out = canary.evaluate_gate(findings_glob=one_run_glob, candidate_model_path=path, threshold=0.0)
    assert out["cosine"]["mean_f1"] == pytest.approx(2 / 3)
    assert out["model"]["mean_f1"] == pytest.approx(11 / 12)
    assert out["delta_f1"] == pytest.approx(11 / 12 - 2 / 3)
    assert out["beat_baseline"] is True
    assert out["decision"] == "PROMOTE"


def test_evaluate_gate_holds_on_a_tie(no_network_embed, one_run_glob, stub_joblib_load, tmp_path):
    path = stub_joblib_load(tmp_path / "cand.joblib", CosClf())
    out = canary.evaluate_gate(findings_glob=one_run_glob, candidate_model_path=path, threshold=0.5)
    assert out["delta_f1"] == pytest.approx(0.0)
    assert out["decision"] == "HOLD"


def test_evaluate_gate_margin_boundary_is_inclusive(no_network_embed, one_run_glob,
                                                    stub_joblib_load, tmp_path):
    """delta >= min_beat_margin promotes, so margin 0.0 promotes on an exact tie."""
    path = stub_joblib_load(tmp_path / "cand.joblib", CosClf())
    out = canary.evaluate_gate(findings_glob=one_run_glob, candidate_model_path=path,
                               threshold=0.5, min_beat_margin=0.0)
    assert out["decision"] == "PROMOTE"


def test_evaluate_gate_holds_when_model_is_worse(no_network_embed, one_run_glob,
                                                 stub_joblib_load, tmp_path):
    path = stub_joblib_load(tmp_path / "cand.joblib", ConstClf(1.0))
    out = canary.evaluate_gate(findings_glob=one_run_glob, candidate_model_path=path, threshold=0.5)
    assert out["model"]["mean_f1"] == pytest.approx(2 / 3)  # predicts everything positive
    assert out["delta_f1"] < 0
    assert out["decision"] == "HOLD"


def test_evaluate_gate_averages_across_runs(no_network_embed, tmp_path):
    write_findings(tmp_path / "runs" / "dt-loci-1", standard_run_records())
    clean = [{"text": f"alpha finding {i}", "tags": ["dt_target:alpha"]} for i in range(6)]
    clean += [{"text": f"beta finding {i}", "tags": ["dt_target:beta"]} for i in range(6)]
    write_findings(tmp_path / "runs" / "dt-loci-2", clean)
    out = canary.evaluate_gate(findings_glob=str(tmp_path / "runs" / "*" / "findings.jsonl"),
                               threshold=0.5)
    assert out["n_runs"] == 2 and out["n_findings"] == 24
    assert [e["run_id"] for e in out["cosine"]["per_run"]] == ["dt-loci-1", "dt-loci-2"]
    assert out["cosine"]["mean_f1"] == pytest.approx((11 / 12 + 1.0) / 2)
    assert out["cosine"]["std_f1"] == pytest.approx(abs(11 / 12 - 1.0) / 2)


def _fit_lr(n_features, seed=0):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(seed)
    X = rng.rand(40, n_features)
    y = (X[:, 0] > 0.5).astype(int)
    y[0], y[1] = 0, 1
    return LogisticRegression(max_iter=500).fit(X, y)


def test_evaluate_gate_scores_a_train_shaped_model(no_network_embed, one_run_glob,
                                                   stub_joblib_load, tmp_path):
    """A candidate from train.py (the wider 2*d+4 layout) must be scoreable.

    canary used to build a hard-coded 2*d+1 vector, so sklearn raised and the
    exception escaped evaluate_gate uncaught — the promotion gate could never
    pass, and loop.py read the crash as 'canary drift detected'.
    """
    path = stub_joblib_load(tmp_path / "cand.joblib", _fit_lr(2 * 2 + 4))
    out = canary.evaluate_gate(findings_glob=one_run_glob, candidate_model_path=path,
                               threshold=0.5)
    assert out["decision"] in ("PROMOTE", "HOLD")
    assert out["model"]["mean_f1"] == out["model"]["mean_f1"]  # scored, not NaN


def test_evaluate_gate_still_scores_the_legacy_shipped_model(no_network_embed, one_run_glob,
                                                             stub_joblib_load, tmp_path):
    """The joblib in production is the 2*d+1 one; it must keep working."""
    path = stub_joblib_load(tmp_path / "cand.joblib", _fit_lr(2 * 2 + 1))
    out = canary.evaluate_gate(findings_glob=one_run_glob, candidate_model_path=path,
                               threshold=0.5)
    assert out["model"]["mean_f1"] == out["model"]["mean_f1"]


def test_evaluate_gate_refuses_a_model_of_unknown_width(no_network_embed, one_run_glob,
                                                        stub_joblib_load, tmp_path):
    """Neither contract produces 2*d+2 columns — say so rather than guess."""
    path = stub_joblib_load(tmp_path / "cand.joblib", _fit_lr(2 * 2 + 2))
    with pytest.raises(ValueError, match="no grounding feature contract produces 6"):
        canary.evaluate_gate(findings_glob=one_run_glob, candidate_model_path=path, threshold=0.5)


def test_canary_has_no_second_copy_of_the_feature_layout(tmp_path):
    """A private copy is how canary drifted away from the trainer in the first place."""
    src = (Path(canary.__file__)).read_text()
    assert "_feat.make_features" in src
    assert "np.abs(fv - qv), fv * qv, [cos]" not in src


# ══════════════════════════════════════════════════════════════════════════════
# canary.promote / _append_history
# ══════════════════════════════════════════════════════════════════════════════

def test_promote_copies_candidate_and_records_previous_sha(tmp_path):
    import hashlib
    cand = tmp_path / "cand.joblib"
    cand.write_bytes(b"NEW")
    target = tmp_path / "live.joblib"
    target.write_bytes(b"OLD")

    canary.promote(cand, target, {"f1": 0.9})

    assert target.read_bytes() == b"NEW"
    rec = json.loads(canary._PROMOTIONS_PATH.read_text().strip())
    assert rec["model_path"] == str(target)
    assert rec["metrics"] == {"f1": 0.9}
    assert rec["previous_sha256"] == hashlib.sha256(b"OLD").hexdigest()
    assert rec["promoted_at"].endswith("+00:00")
    assert set(rec) == {"promoted_at", "model_path", "metrics", "previous_sha256"}


def test_promote_records_null_sha_for_a_fresh_target(tmp_path):
    cand = tmp_path / "cand.joblib"
    cand.write_bytes(b"NEW")
    canary.promote(cand, tmp_path / "brand-new.joblib", {})
    assert json.loads(canary._PROMOTIONS_PATH.read_text().strip())["previous_sha256"] is None


def test_promote_appends_one_line_per_call(tmp_path):
    cand = tmp_path / "cand.joblib"
    cand.write_bytes(b"NEW")
    for _ in range(3):
        canary.promote(cand, tmp_path / "live.joblib", {})
    assert len(canary._PROMOTIONS_PATH.read_text().strip().split("\n")) == 3


def test_promote_raises_when_candidate_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        canary.promote(tmp_path / "nope.joblib", tmp_path / "live.joblib", {})


def test_append_history_writes_jsonl(tmp_path):
    canary._append_history({"a": 1})
    canary._append_history({"a": 2})
    lines = canary._HISTORY_PATH.read_text().strip().split("\n")
    assert [json.loads(x)["a"] for x in lines] == [1, 2]


# ══════════════════════════════════════════════════════════════════════════════
# canary.zscore_drift_check
# ══════════════════════════════════════════════════════════════════════════════

def _write_history(path, records):
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_zscore_missing_history_file_is_nan_and_no_drift(tmp_path):
    out = canary.zscore_drift_check({"cosine_f1": 0.8, "model_f1": 0.8},
                                    history_path=tmp_path / "absent.jsonl")
    assert np.isnan(out["cosine_z"]) and np.isnan(out["model_z"])
    assert out["drift"] is False and out["reason"] == "ok"
    assert set(out) == {"cosine_z", "model_z", "drift", "reason"}


def test_zscore_needs_at_least_two_historical_points(tmp_path):
    hp = tmp_path / "h.jsonl"
    _write_history(hp, [{"cosine_f1": 0.8}])
    out = canary.zscore_drift_check({"cosine_f1": 0.1}, history_path=hp)
    assert np.isnan(out["cosine_z"]) and out["drift"] is False


def test_zscore_flags_drift_below_minus_two_sigma(tmp_path):
    hp = tmp_path / "h.jsonl"
    _write_history(hp, [{"cosine_f1": v, "model_f1": v} for v in (0.80, 0.81, 0.79, 0.80, 0.82)])
    out = canary.zscore_drift_check({"cosine_f1": 0.10, "model_f1": 0.80}, history_path=hp)
    assert out["drift"] is True
    assert out["cosine_z"] < -2.0
    assert out["reason"].startswith("cosine_f1 dropped")
    assert "model_f1" not in out["reason"]


def test_zscore_reports_both_sides_joined_by_semicolon(tmp_path):
    hp = tmp_path / "h.jsonl"
    _write_history(hp, [{"cosine_f1": v, "model_f1": v} for v in (0.80, 0.81, 0.79, 0.80, 0.82)])
    out = canary.zscore_drift_check({"cosine_f1": 0.10, "model_f1": 0.10}, history_path=hp)
    assert out["reason"].count(";") == 1
    assert out["reason"].startswith("cosine_f1 dropped")
    assert "model_f1 dropped" in out["reason"]


def test_zscore_is_one_sided_improvement_never_drifts(tmp_path):
    """A huge positive Z is not drift — only degradation trips the gate."""
    hp = tmp_path / "h.jsonl"
    _write_history(hp, [{"cosine_f1": v, "model_f1": v} for v in (0.10, 0.11, 0.09, 0.10, 0.12)])
    out = canary.zscore_drift_check({"cosine_f1": 0.99, "model_f1": 0.99}, history_path=hp)
    assert out["cosine_z"] > 2.0
    assert out["drift"] is False and out["reason"] == "ok"


def test_zscore_zero_variance_history_returns_zero(tmp_path):
    hp = tmp_path / "h.jsonl"
    _write_history(hp, [{"cosine_f1": 0.5} for _ in range(3)])
    out = canary.zscore_drift_check({"cosine_f1": 0.0}, history_path=hp)
    assert out["cosine_z"] == 0.0
    assert out["drift"] is False


def test_zscore_uses_a_ten_record_rolling_window(tmp_path):
    """Only the last HISTORY_WINDOW records count; older ones are ignored."""
    hp = tmp_path / "h.jsonl"
    old = [{"cosine_f1": 0.0} for _ in range(20)]
    recent = [{"cosine_f1": v} for v in (0.90, 0.91, 0.89, 0.90, 0.92, 0.90, 0.91, 0.89, 0.90, 0.91)]
    _write_history(hp, old + recent)
    out = canary.zscore_drift_check({"cosine_f1": 0.50}, history_path=hp)
    assert out["drift"] is True  # would be a positive Z if the zeros were included
    assert out["cosine_z"] < -2.0


def test_zscore_uses_sample_stddev_ddof_one(tmp_path):
    hp = tmp_path / "h.jsonl"
    vals = [0.1, 0.2, 0.3, 0.4]
    _write_history(hp, [{"cosine_f1": v} for v in vals])
    out = canary.zscore_drift_check({"cosine_f1": 0.0}, history_path=hp)
    expected = (0.0 - np.mean(vals)) / np.std(vals, ddof=1)
    assert out["cosine_z"] == pytest.approx(expected)


def test_zscore_ignores_history_rows_missing_the_metric(tmp_path):
    hp = tmp_path / "h.jsonl"
    _write_history(hp, [{"other": 1}, {"other": 2}, {"cosine_f1": 0.5}])
    out = canary.zscore_drift_check({"cosine_f1": 0.1}, history_path=hp)
    assert np.isnan(out["cosine_z"])  # only one usable point survives the nan filter


def test_zscore_skips_malformed_history_lines(tmp_path):
    hp = tmp_path / "h.jsonl"
    with open(hp, "w") as fh:
        fh.write('{"cosine_f1": 0.8}\n\nnot json\n{"cosine_f1": 0.9}\n')
    out = canary.zscore_drift_check({"cosine_f1": 0.85}, history_path=hp)
    assert out["cosine_z"] == pytest.approx((0.85 - 0.85) / np.std([0.8, 0.9], ddof=1))


def test_zscore_nan_current_value_is_nan_z(tmp_path):
    hp = tmp_path / "h.jsonl"
    _write_history(hp, [{"cosine_f1": v} for v in (0.8, 0.9, 0.85)])
    out = canary.zscore_drift_check({"cosine_f1": float("nan")}, history_path=hp)
    assert np.isnan(out["cosine_z"]) and out["drift"] is False


def test_zscore_missing_current_key_defaults_to_nan(tmp_path):
    hp = tmp_path / "h.jsonl"
    _write_history(hp, [{"cosine_f1": v, "model_f1": v} for v in (0.8, 0.9, 0.85)])
    out = canary.zscore_drift_check({}, history_path=hp)
    assert np.isnan(out["cosine_z"]) and np.isnan(out["model_z"])


def test_zscore_json_null_in_history_raises(tmp_path):
    """BUG PIN: a null metric in the history file crashes the drift check."""
    hp = tmp_path / "h.jsonl"
    _write_history(hp, [{"cosine_f1": None}, {"cosine_f1": 0.5}, {"cosine_f1": 0.6}])
    with pytest.raises(TypeError):
        canary.zscore_drift_check({"cosine_f1": 0.1}, history_path=hp)


# ══════════════════════════════════════════════════════════════════════════════
# canary.monitor_live
# ══════════════════════════════════════════════════════════════════════════════

def test_monitor_live_missing_model_returns_a_short_dict(tmp_path):
    """BUG PIN: the degraded dict omits cosine_f1/model_f1 that main() formats."""
    out = canary.monitor_live(tmp_path / "absent.joblib", findings_glob=str(tmp_path / "*.jsonl"))
    assert out == {"drift": False, "rollback_recommended": False, "reason": "no live model found"}
    assert "cosine_f1" not in out


def test_monitor_live_no_findings_returns_a_short_dict(tmp_path, stub_joblib_load):
    path = stub_joblib_load(tmp_path / "live.joblib", CosClf())
    out = canary.monitor_live(path, findings_glob=str(tmp_path / "nothing" / "*.jsonl"))
    assert out == {"drift": False, "rollback_recommended": False, "reason": "no findings to evaluate"}
    assert "model_f1" not in out


def test_monitor_live_happy_path_shape(no_network_embed, one_run_glob, stub_joblib_load, tmp_path):
    path = stub_joblib_load(tmp_path / "live.joblib", CosClf())
    out = canary.monitor_live(path, findings_glob=one_run_glob, threshold=0.5)
    assert set(out) == {"drift", "rollback_recommended", "cosine_f1", "model_f1", "reason"}
    assert out["cosine_f1"] == pytest.approx(11 / 12)
    assert out["model_f1"] == pytest.approx(11 / 12)
    assert out["drift"] is False and out["rollback_recommended"] is False
    assert out["reason"] == "ok"


def test_monitor_live_dry_run_writes_nothing(no_network_embed, one_run_glob, stub_joblib_load, tmp_path):
    path = stub_joblib_load(tmp_path / "live.joblib", CosClf())
    canary.monitor_live(path, findings_glob=one_run_glob, threshold=0.5, dry_run=True)
    assert not canary._MONITOR_PATH.exists()


def test_monitor_live_appends_a_monitor_record(no_network_embed, one_run_glob, stub_joblib_load, tmp_path):
    path = stub_joblib_load(tmp_path / "live.joblib", CosClf())
    canary.monitor_live(path, findings_glob=one_run_glob, threshold=0.5)
    rec = json.loads(canary._MONITOR_PATH.read_text().strip())
    assert rec["mode"] == "monitor_live"
    assert rec["n_runs"] == 1
    assert rec["drift"] is False and rec["reason"] == "ok"
    assert set(rec) == {"run_at", "mode", "n_runs", "cosine_f1", "model_f1", "drift", "reason"}


def test_monitor_live_flags_drift_against_monitor_history(no_network_embed, one_run_glob,
                                                          stub_joblib_load, tmp_path):
    path = stub_joblib_load(tmp_path / "live.joblib", CosClf())
    _write_history(canary._MONITOR_PATH,
                   [{"cosine_f1": v, "model_f1": v, "drift": False}
                    for v in (0.990, 0.991, 0.989, 0.990, 0.992)])
    out = canary.monitor_live(path, findings_glob=one_run_glob, threshold=0.5)
    assert out["drift"] is True
    assert out["rollback_recommended"] is False  # only one consecutive drift so far
    assert "dropped" in out["reason"]


def test_monitor_live_recommends_rollback_after_two_consecutive_drifts(
        no_network_embed, one_run_glob, stub_joblib_load, tmp_path):
    path = stub_joblib_load(tmp_path / "live.joblib", CosClf())
    hist = [{"cosine_f1": v, "model_f1": v, "drift": False} for v in (0.990, 0.991, 0.989, 0.990)]
    hist.append({"cosine_f1": 0.992, "model_f1": 0.992, "drift": True})
    _write_history(canary._MONITOR_PATH, hist)
    out = canary.monitor_live(path, findings_glob=one_run_glob, threshold=0.5)
    assert out["drift"] is True
    assert out["rollback_recommended"] is True


def test_monitor_live_dry_run_never_recommends_rollback(no_network_embed, one_run_glob,
                                                        stub_joblib_load, tmp_path):
    """Rollback detection lives inside the `if not dry_run` branch."""
    path = stub_joblib_load(tmp_path / "live.joblib", CosClf())
    hist = [{"cosine_f1": v, "model_f1": v, "drift": True} for v in (0.990, 0.991, 0.989, 0.990, 0.992)]
    _write_history(canary._MONITOR_PATH, hist)
    out = canary.monitor_live(path, findings_glob=one_run_glob, threshold=0.5, dry_run=True)
    assert out["drift"] is True
    assert out["rollback_recommended"] is False


# ══════════════════════════════════════════════════════════════════════════════
# train.py — pure helpers
# ══════════════════════════════════════════════════════════════════════════════

def test_sha_is_sha256_of_utf8():
    import hashlib
    assert train._sha("héllo") == hashlib.sha256("héllo".encode()).hexdigest()


def test_token_overlap_is_whitespace_jaccard():
    assert train._token_overlap("a b c", "b c d") == pytest.approx(2 / 4)
    assert train._token_overlap("a", "a") == 1.0
    assert train._token_overlap("a", "b") == 0.0


def test_token_overlap_is_case_sensitive_and_dedupes_tokens():
    assert train._token_overlap("A", "a") == 0.0
    assert train._token_overlap("a a a", "a") == 1.0


def test_token_overlap_two_empty_strings_is_one():
    """Quirk: two empty strings are 'identical', but empty vs non-empty is 0."""
    assert train._token_overlap("", "") == 1.0
    assert train._token_overlap("", "a") == 0.0
    assert train._token_overlap("   ", "") == 1.0


def test_len_ratio_is_symmetric_with_a_plus_one_denominator():
    assert train._len_ratio("ab", "abcd") == pytest.approx(2 / 5)
    assert train._len_ratio("abcd", "ab") == pytest.approx(2 / 5)
    assert train._len_ratio("abcd", "abcd") == pytest.approx(4 / 5)


def test_len_ratio_of_two_empty_strings_is_zero():
    """Inconsistent with _token_overlap, which calls the same pair 1.0."""
    assert train._len_ratio("", "") == 0.0


def test_make_features_layout_and_dimension():
    ec = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    ee = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    X = train.make_features(["hello world", "aa bb cc"], ["hello world", "bb cc dd"], ec, ee)
    assert X.shape == (2, 2 * 2 + 4)
    # row 0: diff, prod, cos, cos^2, len_ratio, jaccard
    assert list(X[0]) == pytest.approx([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 11 / 12, 1.0])
    assert list(X[1]) == pytest.approx([0.6, 0.2, 0.0, 0.8, 0.8, 0.64, 8 / 9, 0.5], abs=1e-6)


def test_make_features_dimension_is_what_canary_now_builds_for_such_a_model():
    """The trainer's default is 2*d+4, and canary builds that width when the
    candidate says it wants it — the mismatch the two used to have."""
    d = 5
    ec = np.zeros((1, d), dtype=np.float32)
    ee = np.zeros((1, d), dtype=np.float32)
    assert train.make_features(["a"], ["b"], ec, ee).shape[1] == 2 * d + 4


def test_cosine_f1_on_folds_picks_the_best_threshold_per_validation_fold():
    """NOTE: the threshold is tuned ON the validation fold — optimistic by design."""
    cos = np.array([0.9, 0.85, 0.1, 0.2, 0.95, 0.05])
    lab = np.array([1, 1, 0, 0, 1, 0])
    folds = [(np.array([0, 1, 2]), np.array([3, 4, 5])),
             (np.array([3, 4, 5]), np.array([0, 1, 2]))]
    assert train.cosine_f1_on_folds(cos, lab, folds) == (1.0, 0.0)


def test_cosine_f1_on_folds_all_negative_fold_is_zero():
    cos = np.array([0.1, 0.2, 0.3])
    lab = np.array([0, 0, 0])
    mean, std = train.cosine_f1_on_folds(cos, lab, [(np.array([0]), np.array([0, 1, 2]))])
    assert (mean, std) == (0.0, 0.0)


def test_cosine_f1_on_folds_ignores_the_train_index():
    cos = np.array([0.9, 0.1])
    lab = np.array([1, 0])
    a = train.cosine_f1_on_folds(cos, lab, [(np.array([0, 1]), np.array([0, 1]))])
    b = train.cosine_f1_on_folds(cos, lab, [(np.array([]), np.array([0, 1]))])
    assert a == b == (1.0, 0.0)


def test_measured_cosines_keeps_absence_distinct_from_zero():
    rows = [
        {"cos": 0.8, "label": 1},     # measured
        {"cos": 0.0, "label": 0},     # measured, and genuinely 0.0
        {"label": 1, "signal": "lineage"},   # never computed
        {"cos": None, "label": 1},    # written, but not a number
    ]
    scores, measured = train.measured_cosines(rows)
    assert list(measured) == [True, True, False, False]
    assert scores[0] == np.float32(0.8)
    assert scores[1] == np.float32(0.0), "a real 0.0 must survive as a measurement"
    assert np.isnan(scores[2]) and np.isnan(scores[3])


def test_cosine_f1_on_folds_excludes_rows_with_no_measured_cosine():
    """A row whose cosine was never computed is not a row with cosine 0.0.

    The grounding dataset's unmeasured rows are all positives (the `lineage`
    pairs, which the builder writes without a `cos` field). Scoring them 0.0
    makes them false negatives at every threshold in the sweep, which can only
    push the cosine baseline DOWN — and that baseline is what a candidate model
    must beat before train.py overwrites the deployed classifier.
    """
    # Two clean positives, two clean negatives, plus two positives whose cosine
    # was never measured (the nan rows).
    cos = np.array([0.9, 0.85, 0.1, 0.2, np.nan, np.nan])
    lab = np.array([1, 1, 0, 0, 1, 1])
    measured = np.array([True, True, True, True, False, False])
    folds = [(np.array([0]), np.array([0, 1, 2, 3, 4, 5]))]

    honest = train.cosine_f1_on_folds(cos, lab, folds, measured=measured)
    assert honest == (1.0, 0.0), (
        "the four rows with a measured cosine separate perfectly; the unmeasured "
        f"rows must not drag the baseline down, got {honest}"
    )

    # Same fold, the shipped behaviour: absence spelled 0.0.
    as_zero = train.cosine_f1_on_folds(np.nan_to_num(cos, nan=0.0), lab, folds)
    assert as_zero[0] < honest[0], (
        "sanity: spelling the missing cosines 0.0 is what depressed the baseline"
    )


def test_cosine_f1_on_folds_returns_nan_when_no_fold_is_measurable():
    """An unmeasurable baseline must not read as 0.0 — 0.0 is beatable by anything."""
    cos = np.array([np.nan, np.nan])
    lab = np.array([1, 0])
    mean, std = train.cosine_f1_on_folds(cos, lab, [(np.array([0]), np.array([0, 1]))],
                                         measured=np.array([False, False]))
    assert np.isnan(mean) and np.isnan(std)
    # And NaN cannot be beaten, so the promotion gate holds.
    assert not (0.99 > mean)


def test_cosine_f1_on_folds_skips_a_fold_left_single_class_by_the_mask():
    cos = np.array([0.9, 0.8, 0.1])
    lab = np.array([1, 1, 0])
    measured = np.array([True, True, False])
    # Masking row 2 away leaves only positives in the validation set.
    mean, std = train.cosine_f1_on_folds(cos, lab, [(np.array([0]), np.array([0, 1, 2]))],
                                         measured=measured)
    assert np.isnan(mean) and np.isnan(std)


# ══════════════════════════════════════════════════════════════════════════════
# train._extract_topic
# ══════════════════════════════════════════════════════════════════════════════

def test_extract_topic_prefers_dt_target():
    assert train._extract_topic({"tags": ["dt_target:alpha", "dt_phase:bench"]}) == "alpha"


def test_extract_topic_bench_uses_the_leading_colon_prefix_lowercased():
    rec = {"tags": ["dt_phase:bench"], "text": "  Latency Numbers: 3ms"}
    assert train._extract_topic(rec) == "bench:latency numbers"


def test_extract_topic_bench_falls_back_to_misc():
    assert train._extract_topic({"tags": ["dt_phase:bench"], "text": "no colon here"}) == "bench:misc"
    assert train._extract_topic({"tags": ["dt_phase:bench"]}) == "bench:misc"


def test_extract_topic_bench_prefix_must_be_3_to_40_chars():
    assert train._extract_topic({"tags": ["dt_phase:bench"], "text": "ab: x"}) == "bench:misc"
    assert train._extract_topic({"tags": ["dt_phase:bench"], "text": "x" * 41 + ": y"}) == "bench:misc"
    assert train._extract_topic({"tags": ["dt_phase:bench"], "text": "abc: y"}) == "bench:abc"


@pytest.mark.parametrize("phase", ["final", "adversarial"])
def test_extract_topic_synthesis_phases(phase):
    assert train._extract_topic({"tags": [f"dt_phase:{phase}"]}) == f"synthesis:{phase}"


def test_extract_topic_other_phase_passes_through():
    assert train._extract_topic({"tags": ["dt_phase:recon"]}) == "recon"


def test_extract_topic_returns_empty_string_when_nothing_matches():
    assert train._extract_topic({}) == ""
    assert train._extract_topic({"tags": ["plainstring"]}) == ""
    assert train._extract_topic({"tags": None}) == ""


def test_extract_topic_bench_with_null_text_raises():
    """BUG PIN: `.get("text", "")` returns None for an explicit JSON null."""
    with pytest.raises(TypeError):
        train._extract_topic({"tags": ["dt_phase:bench"], "text": None})


# ══════════════════════════════════════════════════════════════════════════════
# train — embedding cache
# ══════════════════════════════════════════════════════════════════════════════

def test_load_cache_missing_file_is_empty_dict():
    assert train._load_cache() == {}


def test_cache_round_trips_float32_arrays():
    key = train._sha("hello")
    train._save_cache({key: np.array([1.0, 2.0, 3.0], dtype=np.float32)})
    back = train._load_cache()
    assert list(back) == [key]
    assert back[key].dtype == np.float32
    assert list(back[key]) == [1.0, 2.0, 3.0]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_ollama(monkeypatch):
    calls = []

    def _urlopen(req, timeout=None):
        body = json.loads(req.data.decode())
        calls.append({"url": req.full_url, "body": body, "timeout": timeout})
        data = [{"embedding": [float(len(t)), 1.0]} for t in body["input"]]
        return _FakeResponse(json.dumps({"data": data}).encode())

    monkeypatch.setattr(train.urllib.request, "urlopen", _urlopen)
    return calls


def test_embed_texts_batches_by_sixteen_and_hits_the_v1_endpoint(fake_ollama):
    cache = {}
    out = train.embed_texts([f"text {i}" for i in range(20)], "http://host:11434/", cache)
    assert [len(c["body"]["input"]) for c in fake_ollama] == [16, 4]
    assert fake_ollama[0]["url"] == "http://host:11434/v1/embeddings"
    assert fake_ollama[0]["body"]["model"] == train._emb_model() == "nomic-embed-text"
    assert fake_ollama[0]["timeout"] == 60
    assert out.shape == (20, 2)


def test_embed_texts_returns_l2_normalised_rows_but_caches_raw(fake_ollama):
    cache = {}
    out = train.embed_texts(["abcd"], "http://h", cache)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert list(cache[train._sha("abcd")]) == [4.0, 1.0]  # raw, un-normalised


def test_embed_texts_mutates_the_cache_and_skips_a_second_call(fake_ollama):
    cache = {}
    first = train.embed_texts(["one", "two"], "http://h", cache)
    assert len(fake_ollama) == 1 and len(cache) == 2
    fake_ollama.clear()
    second = train.embed_texts(["one", "two"], "http://h", cache)
    assert fake_ollama == []
    assert np.allclose(first, second)


def test_embed_texts_truncates_input_to_2000_chars(fake_ollama):
    train.embed_texts(["x" * 3000], "http://h", {})
    assert len(fake_ollama[0]["body"]["input"][0]) == 2000


def test_embed_texts_empty_input_raises(fake_ollama):
    """BUG PIN: an empty text list makes np.array([]) 1-D, so normalising blows up."""
    with pytest.raises(np.exceptions.AxisError):
        train.embed_texts([], "http://h", {})
    assert fake_ollama == []


# ══════════════════════════════════════════════════════════════════════════════
# train.oos_from_findings
# ══════════════════════════════════════════════════════════════════════════════

def _write_oos_runs(tmp_path, spec):
    for run, targets in spec.items():
        recs = []
        for target, texts in targets.items():
            for t in texts:
                recs.append({"text": t, "tags": [f"dt_target:{target}"]})
        write_findings(tmp_path / run, recs)
    return str(tmp_path / "*" / "findings.jsonl")


def test_oos_from_findings_needs_at_least_two_runs(tmp_path, capsys):
    glob_pat = _write_oos_runs(tmp_path, {"run-a": {"alpha": ["one", "two"]}})
    assert train.oos_from_findings(glob_pat, {}, "http://h", {"lr": object()}) == {}
    assert "need >=2" in capsys.readouterr().out


def test_oos_from_findings_zero_runs_is_empty(tmp_path):
    assert train.oos_from_findings(str(tmp_path / "*" / "f.jsonl"), {}, "http://h", {}) == {}


def test_oos_from_findings_returns_empty_when_no_fold_has_both_classes(tmp_path, monkeypatch):
    """Every training fold is single-label, so each held-out run is skipped."""
    monkeypatch.setattr(train, "embed_texts",
                        lambda texts, base, cache: np.eye(len(texts), 4, dtype=np.float32))
    glob_pat = _write_oos_runs(tmp_path, {
        "run-a": {"alpha": ["a one", "a two", "a three"]},
        "run-b": {"alpha": ["b one", "b two", "b three"]},
    })
    assert train.oos_from_findings(glob_pat, {}, "http://h", {"lr": object()}) == {}


def test_oos_from_findings_crashes_on_dead_topic_loop(tmp_path, monkeypatch):
    """BUG PIN: dead code indexes a (text, topic) TUPLE with a string key.

    Any fold whose training set contains at least one record of a known topic —
    i.e. every realistic invocation — raises TypeError, so the whole OOS branch
    of train.main() is unreachable in practice.
    """
    from sklearn.linear_model import LogisticRegression

    def _fake(texts, base, cache):
        rng = np.random.RandomState(0)
        v = rng.rand(len(texts), 4).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(train, "embed_texts", _fake)
    glob_pat = _write_oos_runs(tmp_path, {
        "run-a": {"alpha": ["a one", "a two"], "beta": ["a three", "a four"]},
        "run-b": {"alpha": ["b one", "b two"], "beta": ["b three", "b four"]},
    })
    with pytest.raises(TypeError, match="tuple indices"):
        train.oos_from_findings(glob_pat, {}, "http://h",
                                {"lr": LogisticRegression(max_iter=100)})


def test_oos_from_findings_drops_records_without_a_topic(tmp_path, monkeypatch, capsys):
    """Untagged findings are filtered out before the >=2-run check on files."""
    monkeypatch.setattr(train, "embed_texts",
                        lambda texts, base, cache: np.eye(max(len(texts), 1), 4, dtype=np.float32)[:len(texts)])
    write_findings(tmp_path / "run-a", [{"text": "no tags"}, {"text": "still none"}])
    write_findings(tmp_path / "run-b", [{"text": "b one", "tags": ["dt_target:alpha"]}])
    out = train.oos_from_findings(str(tmp_path / "*" / "findings.jsonl"), {}, "http://h", {})
    assert out == {}
    assert "embedding 1 unique texts across 1 runs" in capsys.readouterr().out


# ══════════════════════════════════════════════════════════════════════════════
# active_learn.py
# ══════════════════════════════════════════════════════════════════════════════

def test_active_learn_constants():
    assert al.DEFAULT_N_BOUNDARY == 100
    assert al.DEFAULT_N_HARD_NEG == 50
    assert al.DEFAULT_BAND == 0.2
    assert al.DEFAULT_EMB_MODEL == os.environ.get("EMBED_MODEL", "nomic-embed-text")


def test_load_dataset_missing_file_is_empty():
    assert al._load_dataset("/definitely/not/here.jsonl") == []


def test_load_dataset_skips_blank_and_malformed_lines(tmp_path):
    p = tmp_path / "ds.jsonl"
    p.write_text('{"a": 1}\nnot json\n\n   \n{"b": 2}\n')
    assert al._load_dataset(str(p)) == [{"a": 1}, {"b": 2}]


def test_embed_posts_to_the_native_ollama_api(monkeypatch):
    """active_learn talks to /api/embeddings; canary and train use /v1/embeddings."""
    seen = {}

    def _urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        seen["timeout"] = timeout
        return _FakeResponse(json.dumps({"embedding": [1.0, 2.0]}).encode())

    import urllib.request as urlreq
    monkeypatch.setattr(urlreq, "urlopen", _urlopen)
    out = al._embed("some text", "http://h:11434", "nomic-embed-text")
    assert out == [1.0, 2.0]
    assert seen["url"] == "http://h:11434/api/embeddings"
    assert seen["body"] == {"model": "nomic-embed-text", "prompt": "some text"}
    assert seen["timeout"] == 30


# ── boundary_samples ──────────────────────────────────────────────────────────

# The sampler scores (claim, evidence) pairs through the shared grounding feature
# contract, so a fake model has to advertise one of the widths that contract
# builds at the production embedding size.
_EMBED_DIM = 768
_LEGACY_DIM = 2 * _EMBED_DIM + 1


def _stub_vec():
    return [1.0] + [0.0] * (_EMBED_DIM - 1)


@pytest.fixture
def boundary_dataset(tmp_path):
    texts = [f"candidate text number {i} padded out to length" for i in range(6)]
    p = tmp_path / "ds.jsonl"
    with open(p, "w") as fh:
        for i, t in enumerate(texts):
            fh.write(json.dumps({"text": t, "evidence": f"evidence {i}", "id": i}) + "\n")
        fh.write(json.dumps({"text": "short", "evidence": "e", "id": 99}) + "\n")
    return str(p), texts


@pytest.fixture
def graded_model(tmp_path, monkeypatch):
    """A model whose P(1) walks away from 0.5 as the record's row position grows."""
    monkeypatch.setattr(al, "_embed", lambda text, url, model: _stub_vec())

    class Graded:
        n_features_in_ = _LEGACY_DIM

        def predict_proba(self, feats):
            return np.array([[0.5 - 0.05 * i, 0.5 + 0.05 * i] for i in range(len(feats))])

    path = tmp_path / "m.joblib"
    path.write_bytes(b"stub")
    monkeypatch.setattr(joblib, "load", lambda p: Graded())
    return str(path)


def test_boundary_samples_missing_model_returns_empty(tmp_path, boundary_dataset):
    ds, _ = boundary_dataset
    assert al.boundary_samples(str(tmp_path / "absent.joblib"), ds) == []


def test_boundary_samples_unloadable_model_returns_empty(tmp_path, boundary_dataset, capsys):
    ds, _ = boundary_dataset
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"garbage")
    assert al.boundary_samples(str(bad), ds) == []
    assert "could not load model" in capsys.readouterr().out


def test_boundary_samples_empty_dataset_returns_empty(graded_model, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert al.boundary_samples(graded_model, str(empty)) == []


def test_boundary_samples_sorts_by_uncertainty_and_annotates(graded_model, boundary_dataset):
    ds, _ = boundary_dataset
    out = al.boundary_samples(graded_model, ds, uncertainty_band=2.0)
    assert [c["id"] for c in out] == [0, 1, 2, 3, 4, 5]  # already ascending in |p-0.5|
    assert out[0]["candidate_type"] == "boundary"
    assert out[0]["proba"] == pytest.approx(0.5)
    assert out[0]["uncertainty"] == pytest.approx(0.0)
    assert out[3]["uncertainty"] == pytest.approx(0.15)
    assert set(out[0]) == {"text", "evidence", "id", "candidate_type", "proba", "uncertainty"}


def test_boundary_samples_band_is_halved(graded_model, boundary_dataset):
    """A record is kept when |proba - 0.5| <= band / 2."""
    ds, _ = boundary_dataset
    assert [c["id"] for c in al.boundary_samples(graded_model, ds, uncertainty_band=0.2)] == [0, 1, 2]
    assert [c["id"] for c in al.boundary_samples(graded_model, ds, uncertainty_band=0.0)] == [0]


def test_boundary_samples_respects_n(graded_model, boundary_dataset):
    ds, _ = boundary_dataset
    assert len(al.boundary_samples(graded_model, ds, n=2, uncertainty_band=2.0)) == 2


def test_boundary_samples_skips_records_shorter_than_twenty_chars(graded_model, boundary_dataset):
    ds, _ = boundary_dataset
    assert 99 not in [c["id"] for c in al.boundary_samples(graded_model, ds, uncertainty_band=2.0)]


def test_boundary_samples_reports_a_scoring_failure_instead_of_returning_a_bare_empty(
        tmp_path, monkeypatch, boundary_dataset, capsys):
    """A model that raises at predict_proba must say so once, not be indistinguishable
    from a corpus with no uncertain samples."""
    from sklearn.linear_model import LogisticRegression
    ds, _ = boundary_dataset
    rng = np.random.RandomState(0)
    X = rng.rand(40, _LEGACY_DIM)
    y = (X[:, 0] > 0.5).astype(int)
    model = LogisticRegression(max_iter=50).fit(X, y)
    model.n_features_in_ = _LEGACY_DIM
    path = tmp_path / "m.joblib"
    path.write_bytes(b"stub")
    monkeypatch.setattr(joblib, "load", lambda p: model)
    monkeypatch.setattr(al, "_embed", lambda text, url, m: [0.1, 0.2, 0.3, 0.4])
    assert al.boundary_samples(str(path), ds) == []
    assert "scoring 6 rows failed" in capsys.readouterr().out


def test_boundary_samples_uses_text_claim_content_query_in_that_order(tmp_path, monkeypatch,
                                                                     graded_model):
    """`claim` sits second in the chain: it is the key the live corpus actually uses,
    and adding it must not displace the three that were already read."""
    seen = []
    real = al._embed
    monkeypatch.setattr(al, "_embed", lambda text, url, m: seen.append(text) or real(text, url, m))
    ds = tmp_path / "ds.jsonl"
    with open(ds, "w") as fh:
        fh.write(json.dumps({"text": "from the text field padded out", "claim": "ignored",
                             "evidence": "e0"}) + "\n")
        fh.write(json.dumps({"claim": "from the claim field padded ok", "content": "ignored",
                             "evidence": "e1"}) + "\n")
        fh.write(json.dumps({"content": "from the content field padded", "evidence": "e2"}) + "\n")
        fh.write(json.dumps({"query": "from the query field padded ok", "evidence": "e3"}) + "\n")
        fh.write(json.dumps({"other": "invisible field, no text at all", "evidence": "e4"}) + "\n")
    al.boundary_samples(graded_model, str(ds), uncertainty_band=2.0)
    assert [t for t in seen if not t.startswith("e")] == [
        "from the text field padded out",
        "from the claim field padded ok",
        "from the content field padded",
        "from the query field padded ok",
    ]


def test_boundary_samples_scores_the_claim_evidence_schema_the_corpus_uses(tmp_path,
                                                                          graded_model):
    """The live grounding_dataset.jsonl rows carry only {claim, evidence, label, ...}.

    Before the fix the field chain read text/content/query, so `len(text) < 20`
    fired on all 5418 rows and boundary_samples() returned [] having consulted
    nothing — while the loop logged 'boundary=0' as an ordinary result.
    """
    ds = tmp_path / "corpus.jsonl"
    with open(ds, "w") as fh:
        for i in range(4):
            fh.write(json.dumps({"claim": f"a claim long enough to survive the gate {i}",
                                 "evidence": f"supporting evidence {i}",
                                 "label": i % 2, "signal": "topical"}) + "\n")
    out = al.boundary_samples(graded_model, str(ds), uncertainty_band=2.0)
    assert len(out) == 4
    assert [c["claim"][-1] for c in out] == ["0", "1", "2", "3"]
    assert out[0]["candidate_type"] == "boundary"
    assert "signal" in out[0]  # the source row is carried through intact


def test_boundary_samples_feeds_the_model_pair_features_at_its_own_width(tmp_path, monkeypatch,
                                                                        boundary_dataset):
    """It used to hand predict_proba one raw 768-d embedding per record. The live
    classifier declares n_features_in_ = 1537, so that raised for every record and
    the per-record `except: continue` swallowed it."""
    ds, _ = boundary_dataset
    monkeypatch.setattr(al, "_embed", lambda text, url, m: _stub_vec())
    shapes = []

    class Recorder:
        n_features_in_ = _LEGACY_DIM

        def predict_proba(self, feats):
            shapes.append(np.asarray(feats).shape)
            return np.array([[0.5, 0.5]] * len(feats))

    path = tmp_path / "m.joblib"
    path.write_bytes(b"stub")
    monkeypatch.setattr(joblib, "load", lambda p: Recorder())
    al.boundary_samples(str(path), ds)
    assert shapes == [(6, _LEGACY_DIM)]


def test_boundary_samples_embeds_each_distinct_string_once(tmp_path, monkeypatch, graded_model):
    """The corpus repeats its claims and evidences (5418 rows over 143 of each), so
    a naive two-embeds-per-row loop would be ~38x the Ollama traffic it needs."""
    calls = []
    monkeypatch.setattr(al, "_embed", lambda text, url, m: calls.append(text) or _stub_vec())
    ds = tmp_path / "dupes.jsonl"
    with open(ds, "w") as fh:
        for _ in range(5):
            fh.write(json.dumps({"claim": "one claim repeated across every row here",
                                 "evidence": "one evidence repeated too"}) + "\n")
    al.boundary_samples(graded_model, str(ds), uncertainty_band=2.0)
    assert sorted(calls) == ["one claim repeated across every row here",
                             "one evidence repeated too"]


def test_boundary_samples_says_so_when_no_row_is_scorable(tmp_path, graded_model, capsys):
    """An all-dropped run must be distinguishable from a genuine 'no uncertain
    samples', which is exactly what 'boundary=0' hid for the whole corpus."""
    ds = tmp_path / "unscorable.jsonl"
    with open(ds, "w") as fh:
        for i in range(3):
            fh.write(json.dumps({"summary": f"no field this sampler reads {i}"}) + "\n")
    assert al.boundary_samples(graded_model, str(ds)) == []
    assert "examined 3 rows, 0 carried a (claim, evidence) pair" in capsys.readouterr().out


def test_boundary_samples_refuses_a_model_of_unknown_feature_width(tmp_path, monkeypatch,
                                                                  boundary_dataset, capsys):
    ds, _ = boundary_dataset
    monkeypatch.setattr(al, "_embed", lambda text, url, m: _stub_vec())

    class Wrong:
        n_features_in_ = 12

        def predict_proba(self, feats):  # pragma: no cover - must never be reached
            raise AssertionError("scored with an unknown feature contract")

    path = tmp_path / "m.joblib"
    path.write_bytes(b"stub")
    monkeypatch.setattr(joblib, "load", lambda p: Wrong())
    assert al.boundary_samples(str(path), ds) == []
    assert "expects 12 features" in capsys.readouterr().out


# ── hard_negatives ────────────────────────────────────────────────────────────

def test_hard_negatives_needs_two_positives(tmp_path):
    p = tmp_path / "ds.jsonl"
    p.write_text(json.dumps({"text": "the quick brown fox", "label": 1}) + "\n")
    assert al.hard_negatives(str(p)) == []
    assert al.hard_negatives("/definitely/not/here.jsonl") == []


def test_hard_negatives_pairs_positives_with_three_plus_shared_tokens(tmp_path):
    p = tmp_path / "ds.jsonl"
    recs = [
        {"text": "negative record ignored entirely", "label": 0},
        {"text": "the quick brown fox jumps", "label": 1},
        {"text": "the quick brown dog sleeps", "label": 1},
        {"text": "totally different vocabulary here", "label": 1},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = al.hard_negatives(str(p))
    assert len(out) == 1
    assert out[0] == {
        "text": "the quick brown fox jumps",
        "context": "the quick brown dog sleeps",
        "label": 0,
        "candidate_type": "hard_negative",
        "vocab_overlap": 3,
        # NOTE: indices into the POSITIVES list, not the dataset
        "source_indices": [0, 1],
    }


def test_hard_negatives_overlap_is_case_insensitive(tmp_path):
    p = tmp_path / "ds.jsonl"
    recs = [
        {"text": "THE QUICK BROWN fox", "label": 1},
        {"text": "the quick brown dog", "label": 1},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = al.hard_negatives(str(p))
    assert len(out) == 1 and out[0]["vocab_overlap"] == 3


def test_hard_negatives_requires_strictly_three_shared_tokens(tmp_path):
    p = tmp_path / "ds.jsonl"
    recs = [
        {"text": "alpha beta gamma", "label": 1},
        {"text": "alpha beta delta", "label": 1},  # overlap 2 -> dropped
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert al.hard_negatives(str(p)) == []


def test_hard_negatives_emits_each_unordered_pair_once_lowest_index_first(tmp_path):
    p = tmp_path / "ds.jsonl"
    recs = [{"text": f"the quick brown token{i}", "label": 1} for i in range(3)]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = al.hard_negatives(str(p))
    assert [c["source_indices"] for c in out] == [[0, 1], [0, 2], [1, 2]]


def test_hard_negatives_sorts_by_overlap_desc_and_truncates_to_n(tmp_path):
    p = tmp_path / "ds.jsonl"
    recs = [
        {"text": "one two three four five", "label": 1},
        {"text": "one two three nine ten", "label": 1},     # overlap 3 with #0
        {"text": "one two three four eleven", "label": 1},  # overlap 4 with #0
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = al.hard_negatives(str(p), n=2)
    assert [c["vocab_overlap"] for c in out] == [4, 3]
    assert len(out) == 2


def test_hard_negatives_ignores_the_query_field(tmp_path):
    """Unlike boundary_samples, hard_negatives has no `query` fallback."""
    p = tmp_path / "ds.jsonl"
    recs = [
        {"query": "the quick brown fox", "label": 1},
        {"query": "the quick brown dog", "label": 1},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert al.hard_negatives(str(p)) == []


def test_hard_negatives_defaults_missing_label_to_zero(tmp_path):
    p = tmp_path / "ds.jsonl"
    recs = [{"text": "the quick brown fox"}, {"text": "the quick brown dog"}]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert al.hard_negatives(str(p)) == []


def test_embed_model_is_read_at_call_time_not_import_time(monkeypatch):
    """A module constant is fixed before _resolve_backends() has read the config
    file, so an EMBED_MODEL set there would never reach the request body."""
    monkeypatch.setenv("EMBED_MODEL", "some-other-embedder")
    assert train._emb_model() == "some-other-embedder"
    monkeypatch.delenv("EMBED_MODEL")
    assert train._emb_model() == "nomic-embed-text"


def test_boundary_samples_reports_embed_failures_once_not_once_per_string(
        tmp_path, monkeypatch, graded_model, capsys):
    """With Ollama down this is every distinct text in the corpus, and the nightly
    log is the only reader."""
    monkeypatch.setattr(al, "_embed", lambda text, url, m: (_ for _ in ()).throw(
        RuntimeError("connection refused")))
    ds = tmp_path / "ds.jsonl"
    with open(ds, "w") as fh:
        for i in range(4):
            fh.write(json.dumps({"claim": f"a claim long enough to survive it {i}",
                                 "evidence": f"evidence {i}"}) + "\n")
    assert al.boundary_samples(graded_model, str(ds)) == []
    out = capsys.readouterr().out
    assert out.count("embeds failed") == 1
    assert "8/8 embeds failed, first: connection refused" in out
    assert "4 scorable rows, 0 embedded" in out

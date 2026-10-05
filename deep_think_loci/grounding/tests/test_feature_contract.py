"""A promoted model must be one the live gate can feed.

build_grounding_dataset.py and ground_gate.py emit 1537 columns; the MLOps
trainer emits 1540 because it adds cos**2, a length ratio and a token overlap.
Nothing caught it: the loop has never promoted, so the mismatched model never
reached the gate. The first promotion would have raised inside the live
grounding path, on real traffic.
"""
import pathlib
import sys

import numpy as np
import pytest

GROUNDING = pathlib.Path(__file__).resolve().parents[1]
REPO = GROUNDING.parents[1]
sys.path.insert(0, str(GROUNDING))
import features as F  # noqa: E402


def _emb(n=5):
    rng = np.random.default_rng(0)
    return rng.random((n, F.EMBED_DIM)).astype(np.float32)


def test_both_contract_versions_have_the_widths_the_models_expect():
    a, b = _emb(), _emb()
    txt = ["alpha beta"] * 5
    assert F.make_features(txt, txt, a, b, dim=F.LEGACY_DIM).shape[1] == 1537
    assert F.make_features(txt, txt, a, b).shape[1] == 1540
    assert F.supported_dims() == (1537, 1540)


def test_the_column_layout_is_pinned_value_by_value():
    """A trained model reads columns by position, so the ORDER is the contract, not
    just the width. Hand-computed at d=2: a=(1,0), b=(0.6,0.8), cos = 0.6;
    "a b" vs "b c": length ratio 3/(3+1), token overlap |{b}|/|{a,b,c}| = 1/3."""
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.6, 0.8]], dtype=np.float32)
    legacy = F.make_features(["a b"], ["b c"], a, b, dim=5)[0]
    current = F.make_features(["a b"], ["b c"], a, b, dim=8)[0]
    #                    |a-b|        a*b      cos
    np.testing.assert_allclose(legacy, [0.4, 0.8, 0.6, 0.0, 0.6], atol=1e-6)
    #                    |a-b|        a*b      cos  cos^2  len-ratio  overlap
    np.testing.assert_allclose(current, [0.4, 0.8, 0.6, 0.0, 0.6, 0.36, 0.75, 1 / 3], atol=1e-6)


def test_the_contract_is_a_shape_rule_not_two_magic_numbers():
    """The two layouts are 2d+1 and 2d+4 at whatever width the embeddings came
    in. Pinning them to 768 made the shared definition unusable to the callers
    that had to keep their own copy — which is how they drifted."""
    assert F.dims_for(F.EMBED_DIM) == (F.LEGACY_DIM, F.CURRENT_DIM) == (1537, 1540)
    assert F.dims_for(2) == (5, 8)
    a, b = _emb(3), _emb(3)
    txt = ["alpha beta"] * 3
    small = np.ones((3, 2), dtype=np.float32)
    assert F.make_features(txt, txt, small, small, dim=5).shape[1] == 5
    assert F.make_features(txt, txt, small, small, dim=8).shape[1] == 8
    # no dim = the current layout at that width, not the 768-sized constant
    assert F.make_features(txt, txt, small, small).shape[1] == 8
    assert F.make_features(txt, txt, a, b).shape[1] == 1540


def test_an_unknown_width_is_refused_rather_than_guessed():
    a, b = _emb(), _emb()
    with pytest.raises(ValueError, match="no grounding feature contract"):
        F.make_features(["x"] * 5, ["y"] * 5, a, b, dim=1539)


def test_the_trainer_builds_exactly_the_shared_features():
    """A second copy is how the two drifted apart in the first place. The OOS pass is
    pinned against the contract in mlops/grounding/tests/test_grounding.py
    (test_oos_from_findings_trains_on_the_shared_feature_contract)."""
    sys.path.insert(0, str(REPO))
    train = pytest.importorskip("mlops.grounding.train")
    a, b = _emb(4), _emb(4)[::-1].copy()
    claims = ["a b c", "b c", "c", "d e f g"]
    evid = ["b c d", "x", "c c", "d e"]
    np.testing.assert_array_equal(train.make_features(claims, evid, a, b),
                                  F.make_features(claims, evid, a, b))
    assert train._token_overlap is F.token_overlap and train._len_ratio is F.len_ratio


class _Model:
    """A fitted-model stand-in: declares its width like sklearn, scores a column."""

    def __init__(self, n_features, column):
        self.n_features_in_ = n_features
        self.column = column
        self.seen = []

    def predict_proba(self, X):
        X = np.asarray(X)
        assert X.shape[1] == self.n_features_in_, "gate built the wrong width"
        self.seen.append(X.copy())
        p = X[:, self.column]
        return np.stack([1 - p, p], axis=1)


def _gate_with_model(monkeypatch, tmp_path, model):
    import ground_gate
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "clf.joblib"
    joblib.dump(model, path)
    def unit(x, y):                     # a 768-wide vector in the e0/e1 plane
        v = np.zeros(F.EMBED_DIM, dtype=np.float32)
        v[0], v[1] = x, y
        return v
    vecs = {"q": unit(1.0, 0.0), "on": unit(0.8, 0.6), "off": unit(0.3, 0.9539392)}
    monkeypatch.setattr(ground_gate, "embed", lambda texts: np.stack([vecs[t] for t in texts]))
    return ground_gate.gate("q", [{"id": "a", "text": "on"}, {"id": "b", "text": "off"}],
                            threshold=0.99, model=str(path))


@pytest.mark.parametrize("dim", [F.LEGACY_DIM, F.CURRENT_DIM])
def test_the_gate_builds_for_the_model_it_loaded_not_a_fixed_width(monkeypatch, tmp_path, dim):
    # The model scores the cos column (index 2d = 1536): 0.8 keeps, 0.3 drops. The
    # cosine threshold (0.99) would drop both, so a kept "a" proves the model decided.
    r = _gate_with_model(monkeypatch, tmp_path, _Model(dim, column=2 * F.EMBED_DIM))
    assert r["mode"] == f"model:clf.joblib:{dim}f"
    assert [k["id"] for k in r["kept"]] == ["a"] and [d["id"] for d in r["dropped"]] == ["b"]
    assert [k["score"] for k in r["kept"]] == [0.8]


def test_the_gate_refuses_a_model_of_unknown_width(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="expects 1539 features"):
        _gate_with_model(monkeypatch, tmp_path, _Model(1539, column=0))


def test_the_shipped_model_is_still_loadable_under_the_contract():
    """The live joblib is 1537. The point of keeping LEGACY_DIM is that this
    change must not break the model currently in production."""
    import warnings
    joblib = pytest.importorskip("joblib")
    p = GROUNDING / "grounding_bleed_clf.joblib"
    if not p.exists():
        pytest.skip("no shipped model in this checkout")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = joblib.load(p)
    dim = int(getattr(clf, "n_features_in_", 0))
    assert dim in F.supported_dims(), f"shipped model wants {dim}, contract offers {F.supported_dims()}"
    feats = F.make_features(["a b"] * 3, ["b c"] * 3, _emb(3), _emb(3), dim=dim)
    assert clf.predict_proba(feats).shape == (3, 2)

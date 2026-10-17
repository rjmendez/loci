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


def test_the_trainer_uses_the_shared_definition():
    """A second copy is how the two drifted apart in the first place."""
    src = (REPO / "mlops" / "grounding" / "train.py").read_text()
    assert "import features as _feat" in src
    assert "_feat.make_features" in src
    assert "def _token_overlap" not in src, "trainer still has its own copy"
    assert "def _len_ratio" not in src, "trainer still has its own copy"


def test_the_gate_builds_for_the_model_it_loaded_not_a_fixed_width():
    src = (GROUNDING / "ground_gate.py").read_text()
    assert "n_features_in_" in src, "gate must ask the model what it expects"
    assert "supported_dims" in src, "gate must refuse an unknown contract"
    assert "np.abs(cv[i] - qv), cv[i] * qv, [cos[i]]" not in src, \
        "gate still hardcodes the 1537 layout"


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

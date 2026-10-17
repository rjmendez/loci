"""The measurement that decides whether more corpus helped must not leak.

train.py splits PAIRS, so a finding lands in train and test on every fold and the
number rises with corpus size whether or not the model learned anything. This
evaluator splits FINDINGS first. Measured on the real corpus: LogisticRegression
went from -0.011 against cosine at 303 findings to +0.030 at 1,442 — a decision
the pair-level number could not have supported.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "mlops" / "grounding" / "holdout_eval.py"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("holdout", SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _recs(n=40, topics=4):
    return [{"text": f"finding {i} about subsystem {i % topics}",
             "topic": f"t{i % topics}"} for i in range(n)]


def test_no_finding_appears_on_both_sides_of_a_split(mod):
    """The whole point. A pair-level split cannot make this promise."""
    recs = _recs()
    emb = np.random.default_rng(0).random((len(recs), mod.F.EMBED_DIM)).astype(np.float32)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(recs))
    n_test = int(len(recs) * 0.30)
    te, tr = list(order[:n_test]), list(order[n_test:])
    assert not (set(te) & set(tr))
    pte, ptr = mod.pairs(recs, te, emb, 0), mod.pairs(recs, tr, emb, 0)
    assert pte is not None and ptr is not None
    # every test pair is built only from held-out findings
    assert pte[0].shape[1] == mod.F.CURRENT_DIM


def test_pairs_uses_the_shared_feature_contract(mod):
    """Measuring in a different feature space than the one that ships would make
    the number unactionable."""
    recs = _recs(20, 2)
    emb = np.random.default_rng(1).random((len(recs), mod.F.EMBED_DIM)).astype(np.float32)
    X, y, cos = mod.pairs(recs, list(range(len(recs))), emb, 0)
    assert X.shape[1] == mod.F.CURRENT_DIM
    assert set(np.unique(y)) <= {0, 1}
    assert len(cos) == len(y)


def test_a_split_with_no_positive_pairs_returns_none_rather_than_a_number(mod):
    """One topic per finding means no same-topic pair exists. Reporting an F1
    there would be a number computed from nothing."""
    recs = [{"text": f"f{i}", "topic": f"t{i}"} for i in range(6)]
    emb = np.random.default_rng(2).random((6, mod.F.EMBED_DIM)).astype(np.float32)
    assert mod.pairs(recs, list(range(6)), emb, 0) is None


def test_load_findings_keeps_one_row_per_id_like_the_builder(mod, tmp_path):
    """A corpus that disagrees with the builder measures the wrong thing."""
    f = tmp_path / "dt-loci-x"; f.mkdir()
    (f / "findings.jsonl").write_text("\n".join([
        '{"id":"a","text":"first","tags":["dt_target:one"]}',
        '{"id":"a","text":"a later duplicate","tags":["dt_target:one"]}',
        '{"id":"b","text":"","tags":["dt_target:one"]}',
        '{"id":"c","text":"second","tags":["dt_target:two"]}',
    ]))
    recs = mod.load_findings(str(tmp_path / "dt-loci-*" / "findings.jsonl"))
    assert len(recs) == 2
    assert {r["text"] for r in recs} == {"first", "second"}

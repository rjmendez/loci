"""The measurement that decides whether more corpus helped must not leak.

train.py splits PAIRS, so a finding lands in train and test on every fold and the
number rises with corpus size whether or not the model learned anything. This
evaluator splits FINDINGS first. Measured on the real corpus: LogisticRegression
went from -0.011 against cosine at 303 findings to +0.030 at 1,442 — a decision
the pair-level number could not have supported.
"""
import importlib.util
import json
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


def _corpus(tmp_path, n=40, topics=4):
    d = tmp_path / "dt-loci-x"
    d.mkdir()
    (d / "findings.jsonl").write_text("".join(
        json.dumps({"id": f"f{i}", "text": f"finding {i} about subsystem {i % topics}",
                    "tags": [f"dt_target:t{i % topics}"]}) + "\n" for i in range(n)))
    return str(tmp_path / "dt-loci-*" / "findings.jsonl")


def test_no_finding_appears_on_both_sides_of_a_split(mod, tmp_path, monkeypatch):
    """The whole point. A pair-level split cannot make this promise.

    The old version re-implemented the split inside the test and checked its own
    copy, so main() could leak test findings into train and it still passed.
    This runs the real main() and records the index sets it hands to pairs()."""
    seen = []
    real_pairs = mod.pairs

    def spy(recs, idxs, emb, seed, **kw):
        seen.append((seed, list(idxs)))
        return real_pairs(recs, idxs, emb, seed, **kw)

    def embed(texts, base, cache):
        rng = np.random.default_rng(len(texts))
        v = rng.random((len(texts), 8)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(mod, "pairs", spy)
    monkeypatch.setattr(mod.T, "embed_texts", embed)
    monkeypatch.setattr(mod.T, "CACHE_PATH", str(tmp_path / "cache.npz"))
    out = tmp_path / "holdout.json"
    monkeypatch.setattr(sys, "argv", ["holdout_eval.py", "--findings", _corpus(tmp_path),
                                      "--seeds", "3", "--model", "lr", "--out", str(out)])
    assert mod.main() == 0

    # two calls per seed: (held-out side, training side)
    assert [s for s, _ in seen] == [0, 0, 1, 1, 2, 2]
    for k in range(0, len(seen), 2):
        te, tr = set(seen[k][1]), set(seen[k + 1][1])
        assert not te & tr, "a finding is on both sides of the split"
        assert te | tr == set(range(40)), "the split dropped findings"
        assert len(te) == 12  # int(40 * 0.30)
    assert seen[0][1] != seen[2][1], "every seed made the same split"
    assert json.loads(out.read_text())["seeds"] == 3


def test_pairs_are_built_only_from_the_findings_it_is_given(mod):
    """Every pair must come from `idxs`. Here those 5 findings share one topic, so
    all C(5,2) = 10 pairs are positive and there are no negatives to sample; a
    pairs() that ignored idxs would build from all 10 findings (45 pairs)."""
    recs = [{"text": f"f{i}", "topic": "same" if i < 5 else f"other{i}"} for i in range(10)]
    emb = np.eye(10, mod.F.EMBED_DIM, dtype=np.float32)
    X, y, cos = mod.pairs(recs, [0, 1, 2, 3, 4], emb, 0)
    assert X.shape == (10, mod.F.CURRENT_DIM)
    assert y.tolist() == [1] * 10
    assert cos.tolist() == [0.0] * 10


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

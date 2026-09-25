"""Hermetic tests for mcp/sparse_expansion_index.py (offline design study).

No network, no Ollama, no Qdrant: everything is synthetic numpy data and
spec files written under tmp_path.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

import sparse_expansion_index as sei

MCP_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- specs


def test_idealised_spec_is_uniform_with_fixed_claws():
    s = sei.idealised_spec(10, 3)
    assert s.n_glomeruli == 10
    np.testing.assert_allclose(s.glomerulus_weights, 0.1)
    assert s.claw_pmf.tolist() == [0, 0, 0, 1]
    assert s.mean_claws == 3
    with pytest.raises(ValueError):
        sei.idealised_spec(4, 5)
    with pytest.raises(ValueError):
        sei.idealised_spec(4, 0)


def _toy_matrix():
    # 4 glomeruli x 6 KCs; glomerulus 3 has no claw and must be dropped
    return np.array([
        [1, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)


def test_spec_from_matrix_derives_claws_and_weights():
    s = sei.spec_from_matrix(_toy_matrix(), name="toy")
    assert s.n_glomeruli == 3
    # claws per KC: 2,1,3,1,1,0 -> pmf over 0..3
    np.testing.assert_allclose(s.claw_pmf, [1 / 6, 3 / 6, 1 / 6, 1 / 6])
    np.testing.assert_allclose(s.glomerulus_weights, [4 / 8, 3 / 8, 1 / 8])
    assert s.mean_claws == pytest.approx((1 * 3 + 2 * 1 + 3 * 1) / 5)
    with pytest.raises(sei.SpecIntegrityError):
        sei.spec_from_matrix(np.zeros((3, 3)), name="empty")


def test_spec_variants_change_only_what_they_claim():
    s = sei.spec_from_matrix(_toy_matrix(), name="toy")
    f = s.with_fixed_claws(2)
    assert f.claw_pmf.tolist() == [0, 0, 1] and f.matrix is None
    np.testing.assert_allclose(f.glomerulus_weights, s.glomerulus_weights)
    u = s.with_uniform_glomeruli()
    np.testing.assert_allclose(u.glomerulus_weights, 1 / 3)
    np.testing.assert_allclose(u.claw_pmf, s.claw_pmf)


# ---------------------------------------------------------------- connectivity


def test_sample_connectivity_shapes_claws_and_determinism():
    s = sei.idealised_spec(20, 4)
    c1 = sei.build_connectivity(s, 300, seed=7)
    c2 = sei.build_connectivity(s, 300, seed=7)
    c3 = sei.build_connectivity(s, 300, seed=8)
    assert c1.shape == (300, 20)
    assert set(np.unique(c1)) <= {0, 1}
    assert (c1.sum(axis=1) == 4).all()
    np.testing.assert_array_equal(c1, c2)
    assert not np.array_equal(c1, c3)


def test_sample_connectivity_follows_measured_weights_and_skips_zero_claws():
    w = np.array([0.7, 0.1, 0.1, 0.1])
    pmf = np.array([0.5, 0.5])  # half the KCs would have no claw
    s = sei.ExpansionSpec(name="w", n_glomeruli=4, claw_pmf=pmf, glomerulus_weights=w)
    c = sei.build_connectivity(s, 4000, seed=0)
    assert (c.sum(axis=1) == 1).all()  # zero-claw KCs are resampled away
    freq = c.sum(axis=0) / c.sum()
    assert freq[0] == pytest.approx(0.7, abs=0.03)
    assert freq[1] == pytest.approx(0.1, abs=0.03)


def test_matrix_mode_subsamples_measured_columns():
    m = _toy_matrix()
    s = sei.spec_from_matrix(m, name="toy")
    cols = {tuple(col) for col in s.matrix.T if col.sum() > 0}
    conn = sei.build_connectivity(s, 4, mode="matrix", seed=3)
    assert conn.shape == (4, 3)
    assert all(tuple(row) in cols for row in conn)
    assert conn.sum(axis=1).min() >= 1  # zero-claw KCs are never drawn


def test_matrix_mode_with_k_equal_n_uses_each_measured_kc_once():
    rng = np.random.default_rng(2)
    m = (rng.random((9, 30)) < 0.4).astype(np.uint8)
    m[:, :3] = 0  # three KCs without claws
    m[0, 3:] = 1
    s = sei.spec_from_matrix(m, name="r")
    n = int((s.matrix.sum(axis=0) > 0).sum())
    assert n == 27
    conn = sei.build_connectivity(s, n, mode="matrix", seed=5)
    np.testing.assert_array_equal(conn, s.matrix.T[3:])  # without replacement, index order


def test_matrix_mode_beyond_n_preserves_degrees_and_breaks_duplicates():
    rng = np.random.default_rng(0)
    m = (rng.random((12, 40)) < 0.3).astype(np.uint8)
    m[:, m.sum(axis=0) == 0] = 0
    m[0, m.sum(axis=0) == 0] = 1
    s = sei.spec_from_matrix(m, name="rand")
    conn = sei.build_connectivity(s, 200, mode="matrix", seed=1)
    base = s.matrix.T
    assert conn.shape == (200, s.n_glomeruli)
    np.testing.assert_array_equal(conn[:40], base)  # every measured KC used once, in order
    extra = conn[40:]
    # claw counts of the extra KCs come from the measured distribution
    assert set(extra.sum(axis=1)) <= set(base.sum(axis=1))
    # without the swap step extra rows would all be exact copies of measured columns
    measured = {tuple(r) for r in base}
    novel = sum(tuple(r) not in measured for r in extra)
    assert novel > 80


def test_degree_preserving_shuffle_keeps_row_and_column_sums():
    rng = np.random.default_rng(5)
    block = (rng.random((30, 10)) < 0.4).astype(np.uint8)
    out = sei._degree_preserving_shuffle(block, np.random.default_rng(9))
    np.testing.assert_array_equal(out.sum(axis=0), block.sum(axis=0))
    np.testing.assert_array_equal(out.sum(axis=1), block.sum(axis=1))
    assert not np.array_equal(out, block)


def test_matrix_mode_requires_matrix_and_valid_args():
    with pytest.raises(ValueError):
        sei.build_connectivity(sei.idealised_spec(5, 2), 10, mode="matrix")
    with pytest.raises(ValueError):
        sei.build_connectivity(sei.idealised_spec(5, 2), 10, mode="bogus")
    with pytest.raises(ValueError):
        sei.build_connectivity(sei.idealised_spec(5, 2), 0)


# ---------------------------------------------------------------- codes


def test_k_winners_exact_count_and_tie_break():
    a = np.array([[0.1, 0.9, 0.5, 0.9], [3.0, 1.0, 2.0, 0.0]])
    m = sei.k_winners(a, 2)
    assert m.tolist() == [[False, True, False, True], [True, False, True, False]]
    t = sei.k_winners(np.zeros((1, 5)), 2)
    assert t.tolist() == [[True, True, False, False, False]]
    with pytest.raises(ValueError):
        sei.k_winners(a, 0)
    with pytest.raises(ValueError):
        sei.k_winners(a, 5)


def test_pack_and_hamming_match_naive():
    rng = np.random.default_rng(1)
    bits = rng.random((7, 130)) < 0.3
    packed = sei.pack_codes(bits)
    assert packed.shape == (7, 3) and packed.dtype == np.uint64
    naive = (bits[:, None, :] != bits[None, :, :]).sum(axis=-1)
    np.testing.assert_array_equal(sei.hamming_distances(packed, packed, block=3), naive)
    inter = (bits[:, None, :] & bits[None, :, :]).sum(-1)
    union = (bits[:, None, :] | bits[None, :, :]).sum(-1)
    np.testing.assert_allclose(sei.jaccard_similarity(packed, packed, block=2), inter / union)


def _clusters(n_per=20, n_clusters=6, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_clusters, dim)) * 3
    x = np.vstack([c + 0.3 * rng.standard_normal((n_per, dim)) for c in centres])
    labels = np.repeat(np.arange(n_clusters), n_per)
    return x.astype(np.float32), labels


@pytest.mark.parametrize("projection", ["pca", "random"])
def test_encode_shape_and_exact_sparsity(projection):
    x, _ = _clusters()
    idx = sei.SparseExpansionIndex(sei.idealised_spec(8, 3), 200, wta_fraction=0.05, projection=projection, seed=2).fit(x)
    bits = idx.encode_bits(x)
    assert bits.shape == (len(x), 200)
    assert idx.k_active == 10
    assert (bits.sum(axis=1) == 10).all()
    assert idx.encode(x).shape == (len(x), 4)


def test_pca_projection_whitening_and_centring():
    x, _ = _clusters()
    plain = sei.SparseExpansionIndex(sei.idealised_spec(8, 3), 64, seed=0).fit(x).glomeruli(x)
    white = sei.SparseExpansionIndex(sei.idealised_spec(8, 3), 64, whiten=True, seed=0).fit(x).glomeruli(x)
    np.testing.assert_allclose(plain.mean(axis=0), 0, atol=1e-3)
    np.testing.assert_allclose(white.std(axis=0, ddof=1), 1.0, rtol=1e-3)
    assert plain.std(axis=0).max() / plain.std(axis=0).min() > 2  # un-whitened keeps PC variances
    with pytest.raises(ValueError):
        sei.SparseExpansionIndex(sei.idealised_spec(40, 3), 64).fit(x)  # G > dim


def test_precomputed_pca_basis_matches_and_is_checked():
    x, _ = _clusters()
    spec = sei.idealised_spec(8, 3)
    a = sei.SparseExpansionIndex(spec, 128, seed=3).fit(x).encode(x)
    b = sei.SparseExpansionIndex(spec, 128, seed=3).fit(x, basis=sei.pca_basis(x)).encode(x)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="different data"):
        sei.SparseExpansionIndex(spec, 128).fit(x, basis=sei.pca_basis(x + 1.0))


def test_identity_projection_requires_matching_dim():
    x, _ = _clusters(dim=8)
    sei.SparseExpansionIndex(sei.idealised_spec(8, 2), 64, projection="identity").fit(x)
    with pytest.raises(ValueError):
        sei.SparseExpansionIndex(sei.idealised_spec(6, 2), 64, projection="identity").fit(x)


def test_index_is_deterministic_per_seed():
    x, _ = _clusters()
    spec = sei.idealised_spec(8, 3)
    a = sei.SparseExpansionIndex(spec, 256, seed=11).fit(x).encode(x)
    b = sei.SparseExpansionIndex(spec, 256, seed=11).fit(x).encode(x)
    c = sei.SparseExpansionIndex(spec, 256, seed=12).fit(x).encode(x)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    # the PCA-component -> glomerulus assignment is itself seeded
    ga = sei.SparseExpansionIndex(spec, 256, seed=11).fit(x).glomeruli(x)
    gc = sei.SparseExpansionIndex(spec, 256, seed=12).fit(x).glomeruli(x)
    assert not np.allclose(ga, gc)
    np.testing.assert_allclose(np.sort(np.abs(ga).sum(axis=0)), np.sort(np.abs(gc).sum(axis=0)), rtol=1e-4)


@pytest.mark.parametrize("metric", ["hamming", "jaccard"])
def test_search_finds_self_and_cluster_mates(metric):
    x, labels = _clusters()
    idx = sei.SparseExpansionIndex(sei.idealised_spec(12, 4), 600, wta_fraction=0.05, metric=metric, seed=4).fit(x)
    idx.add(x)
    assert len(idx) == len(x)
    ids, scores = idx.search(x, k=5)
    codes = idx.encode(x)
    # the top hit is the item itself or an item with an identical code
    np.testing.assert_array_equal(codes[ids[:, 0]], codes)
    best = 0 if metric == "hamming" else 1.0
    assert (scores[:, 0] == best).all()
    unique = [i for i in range(len(x)) if (codes == codes[i]).all(axis=1).sum() == 1]
    assert unique and (ids[unique, 0] == np.array(unique)).all()
    purity = (labels[ids] == labels[:, None]).mean()
    assert purity > 0.95
    if metric == "hamming":
        assert (np.diff(scores, axis=1) >= 0).all()
    else:
        assert (np.diff(scores, axis=1) <= 0).all()


def test_top_k_from_scores_order_and_ties():
    s = np.array([[3, 1, 1, 0, 5]])
    assert sei.top_k_from_scores(s, 3, larger_is_better=False).tolist() == [[3, 1, 2]]
    assert sei.top_k_from_scores(s, 2, larger_is_better=True).tolist() == [[4, 0]]
    assert sei.top_k_from_scores(s, 10, larger_is_better=True).shape == (1, 5)


def test_novelty_zero_for_stored_and_higher_for_unseen_cluster():
    x, labels = _clusters(n_clusters=5)
    seen = labels < 4
    idx = sei.SparseExpansionIndex(sei.idealised_spec(10, 3), 400, seed=0).fit(x[seen])
    idx.add(x[seen])
    stored = idx.novelty(x[seen][:10])
    np.testing.assert_array_equal(stored, 0.0)
    unseen = idx.novelty(x[~seen])
    assert (unseen >= 0).all() and (unseen <= 1).all()
    held_in = x[seen][:5] + 0.05
    assert unseen.mean() > idx.novelty(held_in).mean() + 0.1


def test_empty_index_and_unfitted_fail_loudly():
    idx = sei.SparseExpansionIndex(sei.idealised_spec(4, 2), 32)
    with pytest.raises(RuntimeError):
        idx.encode(np.zeros((1, 4)))
    idx.fit(np.random.default_rng(0).standard_normal((20, 4)))
    with pytest.raises(RuntimeError):
        idx.search(np.zeros((1, 4)))
    with pytest.raises(RuntimeError):
        idx.novelty(np.zeros((1, 4)))
    with pytest.raises(ValueError):
        sei.SparseExpansionIndex(sei.idealised_spec(4, 2), 32, wta_fraction=0)
    with pytest.raises(ValueError):
        sei.SparseExpansionIndex(sei.idealised_spec(4, 2), 32, metric="cosine")


# ---------------------------------------------------------------- spec file loading


def _canon_digest(payload):
    c = json.loads(json.dumps(payload))
    c["params_sha256"] = ""
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_spec(tmp_path: Path, *, matrix=None, npz_name="m.npz", schema=sei.SPEC_SCHEMA_VERSION):
    matrix = _toy_matrix() if matrix is None else matrix
    buf = io.BytesIO()
    np.savez_compressed(buf, **{"fw_right__binary": matrix, "glomeruli": np.array(["A", "B", "C", "D"])})
    (tmp_path / "m.npz").write_bytes(buf.getvalue())
    payload = {
        "schema_version": schema,
        "primary": "fw_right",
        "glomeruli": ["A", "B", "C", "D"],
        "matrices": {"fw_right": {"binary_key": "fw_right__binary", "shape": list(matrix.shape)}},
        "matrix_file": {"name": npz_name, "sha256": hashlib.sha256(buf.getvalue()).hexdigest()},
        "threshold_synapses": 5,
        "provenance": {"fw": {"dataset_symbol": "fw", "version_id": "flywire783", "citation": "c", "licence": {"spdx_id": "CC-BY-4.0"}}},
        "params_sha256": "",
    }
    payload["params_sha256"] = _canon_digest(payload)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload))
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), payload


def test_load_spec_roundtrip(tmp_path):
    path, sha, _ = _write_spec(tmp_path)
    s = sei.load_spec(path, expected_sha256=sha)
    assert s.n_glomeruli == 3
    np.testing.assert_allclose(s.glomerulus_weights, [0.5, 0.375, 0.125])
    assert s.provenance["spec_sha256"] == sha
    assert s.provenance["source"]["fw"]["version_id"] == "flywire783"


def test_load_spec_sha_mismatch_fails_closed(tmp_path):
    path, sha, _ = _write_spec(tmp_path)
    wrong = ("0" if sha[0] != "0" else "1") + sha[1:]
    with pytest.raises(sei.SpecIntegrityError, match="sha256 mismatch"):
        sei.load_spec(path, expected_sha256=wrong)
    with pytest.raises(sei.SpecIntegrityError):
        sei.load_spec(path, expected_sha256="")
    with pytest.raises(sei.SpecIntegrityError):
        sei.load_spec(tmp_path / "missing.json", expected_sha256=sha)


def test_load_spec_rejects_tampered_matrix(tmp_path):
    path, sha, _ = _write_spec(tmp_path)
    data = bytearray((tmp_path / "m.npz").read_bytes())
    data[-10] ^= 0xFF
    (tmp_path / "m.npz").write_bytes(bytes(data))
    with pytest.raises(sei.SpecIntegrityError, match="matrix file sha256"):
        sei.load_spec(path, expected_sha256=sha)


def test_load_spec_rejects_self_digest_tamper_even_with_pinned_sha(tmp_path):
    path, _, payload = _write_spec(tmp_path)
    payload["threshold_synapses"] = 1  # edited without recomputing params_sha256
    path.write_text(json.dumps(payload))
    with pytest.raises(sei.SpecIntegrityError, match="self-digest"):
        sei.load_spec(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("npz_name", ["../m.npz", "sub/m.npz", ""])
def test_load_spec_rejects_non_sibling_matrix_path(tmp_path, npz_name):
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    path, sha, _ = _write_spec(spec_dir, npz_name=npz_name)
    # plant a valid matrix at the escaped location so only the path rule can refuse it
    (spec_dir / "sub").mkdir()
    for target in (tmp_path / "m.npz", spec_dir / "sub" / "m.npz"):
        target.write_bytes((spec_dir / "m.npz").read_bytes())
    with pytest.raises(sei.SpecIntegrityError, match="plain sibling"):
        sei.load_spec(path, expected_sha256=sha)


def test_load_spec_rejects_schema_and_missing_matrix_key(tmp_path):
    path, sha, _ = _write_spec(tmp_path, schema="fly-mb-params/v0")
    with pytest.raises(sei.SpecIntegrityError, match="schema_version"):
        sei.load_spec(path, expected_sha256=sha)
    other = tmp_path / "b"
    other.mkdir()
    path, sha, _ = _write_spec(other)
    with pytest.raises(sei.SpecIntegrityError, match="not in spec"):
        sei.load_spec(path, expected_sha256=sha, matrix_key="hb")


def test_module_is_not_wired_into_the_server():
    server = (MCP_DIR / "server.py").read_text(encoding="utf-8")
    assert "sparse_expansion_index" not in server

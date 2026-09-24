import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_training as fbct  # noqa: E402


def _sample(i: int, **metadata) -> fbct.TrainingSample:
    return fbct.TrainingSample(
        sample_id=f"s{i:04d}",
        region_id="r1" if i % 2 else "r2",
        input_text=f"feature {i % 5}",
        expected_label="a" if i % 3 else "b",
        expected_confidence=0.8,
        provenance_refs=(f"ref-{i}",),
        metadata=metadata,
    )


def _split_of(manifest: fbct.DatasetManifest) -> dict[str, str]:
    out = {}
    for name, ids in (("train", manifest.train_ids), ("val", manifest.val_ids), ("test", manifest.test_ids)):
        out.update({sample_id: name for sample_id in ids})
    return out


def test_no_group_metadata_matches_legacy_split():
    samples = [_sample(i) for i in range(200)]
    legacy = fbct.build_dataset_manifest(samples, split_seed="seed-x")
    grouped = fbct.build_dataset_manifest(samples, split_seed="seed-x", group_keys=("cell_type",))
    assert grouped.train_ids == legacy.train_ids
    assert grouped.val_ids == legacy.val_ids
    assert grouped.test_ids == legacy.test_ids
    assert grouped.manifest_id == legacy.manifest_id
    assert grouped.notes["split"]["multi_sample_components"] == 0
    assert "split" not in legacy.notes


def test_cell_types_never_straddle_splits():
    samples = [_sample(i, cell_type=f"T{i % 23}") for i in range(300)]
    manifest = fbct.build_dataset_manifest(samples, split_seed="seed-ct", group_keys=("cell_type",))
    split_of = _split_of(manifest)
    by_type: dict[str, set[str]] = {}
    for sample in samples:
        by_type.setdefault(sample.metadata["cell_type"], set()).add(split_of[sample.sample_id])
    assert all(len(splits) == 1 for splits in by_type.values())
    info = manifest.notes["split"]
    assert info["component_count"] == 23
    assert info["per_sample_split_straddling_components"] > 0  # the leakage the grouping removes
    assert manifest.train_ids and manifest.val_ids and manifest.test_ids
    fbct.assert_grouped_split(samples, manifest, group_keys=("cell_type",))


def test_union_across_keys_is_transitive():
    # s0 and s1 share a cell type; s1 and s2 share a hemilineage -> all three together.
    samples = [
        _sample(0, cell_type="A", hemilineage="H1"),
        _sample(1, cell_type="A", hemilineage="H2"),
        _sample(2, cell_type="B", hemilineage="H2"),
        *[_sample(i, cell_type=f"C{i}") for i in range(3, 60)],
    ]
    components = fbct.split_group_components(samples, group_keys=("cell_type", "hemilineage"))
    assert components["s0000"] == components["s0001"] == components["s0002"]
    manifest = fbct.build_dataset_manifest(samples, split_seed="seed-u", group_keys=("cell_type", "hemilineage"))
    split_of = _split_of(manifest)
    assert split_of["s0000"] == split_of["s0001"] == split_of["s0002"]


@pytest.mark.parametrize("missing", [None, "", "unknown", "NaN", float("nan"), "none"])
def test_unknown_values_do_not_group(missing):
    samples = [_sample(i, cell_type=missing) for i in range(10)]
    components = fbct.split_group_components(samples, group_keys=("cell_type",))
    assert len(set(components.values())) == 10


def test_grouped_split_is_deterministic_and_seed_sensitive():
    samples = [_sample(i, split_group=f"pair-{i // 2}") for i in range(120)]
    first = fbct.build_dataset_manifest(samples, split_seed="seed-a", group_keys=("split_group",))
    again = fbct.build_dataset_manifest(list(reversed(samples)), split_seed="seed-a", group_keys=("split_group",))
    other = fbct.build_dataset_manifest(samples, split_seed="seed-b", group_keys=("split_group",))
    assert set(first.train_ids) == set(again.train_ids)
    assert set(first.test_ids) == set(again.test_ids)
    assert first.train_ids != other.train_ids


def test_assert_grouped_split_detects_straddling():
    samples = [_sample(i, cell_type="same") for i in range(10)]
    legacy = fbct.build_dataset_manifest(samples, split_seed="seed-z")
    with pytest.raises(ValueError, match="straddle"):
        fbct.assert_grouped_split(samples, legacy, group_keys=("cell_type",))

import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_dataset_registry as reg  # noqa: E402


def test_registry_symbols_and_status():
    symbols = [spec.symbol for spec in reg.list_datasets()]
    assert symbols == sorted(["hb", "fw", "banc", "l1em", "fafb", "mc", "mv", "ol"])
    assert [s.symbol for s in reg.list_datasets("active")] == ["fw", "hb"]
    assert all(not s.is_active for s in reg.list_datasets(reg.DatasetStatus.PLANNED))


def test_banc_symbol_normalized_to_lowercase_but_dir_kept_uppercase(tmp_path):
    assert reg.normalize_symbol("BANC") == "banc"
    assert reg.normalize_symbol(" Banc ") == "banc"
    spec = reg.get_dataset("BANC")
    assert spec.snapshot_dir_name == "BANC"
    root = reg.snapshot_version_root("banc", storage_root=tmp_path)
    assert root.parent.name == "BANC"
    assert root.name == "banc_888"


def test_fw_objectives_match_existing_builder_contract():
    assert reg.supported_objectives("fw") == ("connectivity_tier", "neurotransmitter_dominance")
    spec = reg.get_dataset("fw")
    assert spec.pinned_version == "flywire783"
    assert spec.organism_stage is reg.OrganismStage.ADULT
    assert reg.get_dataset("l1em").organism_stage is reg.OrganismStage.LARVAL


def test_unknown_symbol_fails_closed():
    with pytest.raises(reg.DatasetRegistryError) as exc:
        reg.get_dataset("mouse")
    assert exc.value.code is reg.DatasetErrorCode.UNKNOWN_DATASET
    with pytest.raises(reg.DatasetRegistryError):
        reg.normalize_symbol("")


def test_require_objective_error_codes():
    assert reg.require_objective("fw", "connectivity_tier").symbol == "fw"
    with pytest.raises(reg.DatasetRegistryError) as exc:
        reg.require_objective("fw", "region_specialization_tier")
    assert exc.value.code is reg.DatasetErrorCode.OBJECTIVE_NOT_SUPPORTED
    with pytest.raises(reg.DatasetRegistryError) as exc:
        reg.require_objective("fw", "made_up")
    assert exc.value.code is reg.DatasetErrorCode.UNKNOWN_OBJECTIVE


def test_require_active_gates_planned():
    with pytest.raises(reg.DatasetRegistryError) as exc:
        reg.require_active("banc")
    assert exc.value.code is reg.DatasetErrorCode.DATASET_NOT_ACTIVE
    assert reg.require_active("banc", allow_planned=True).symbol == "banc"


def test_unpinned_snapshot_root_fails_closed(tmp_path, monkeypatch):
    unpinned = dataclasses.replace(reg.DATASET_REGISTRY["ol"], pinned_version=None)
    monkeypatch.setitem(reg.DATASET_REGISTRY, "ol", unpinned)
    with pytest.raises(reg.DatasetRegistryError) as exc:
        reg.snapshot_version_root("ol", storage_root=tmp_path)
    assert exc.value.code is reg.DatasetErrorCode.VERSION_UNPINNED


@pytest.mark.parametrize(
    "symbol,version,vocabulary,anatomy_words,citation_words",
    [
        # mc = full male CNS (brain + VNC), mv = MANC (male VNC only): matches the pulled snapshots.
        ("mc", "male-cns_v1.0", reg.RegionVocabulary.MALE_CNS_ROI, ("brain", "ventral nerve cord", "male-cns:v1.0"),
         ("Berg S", "10.1101/2025.10.09.680999", "gs://flyem-male-cns/v1.0")),
        ("mv", "manc_v1.0", reg.RegionVocabulary.MANC_NEUROPIL, ("Nerve Cord (MANC) v1.0", "ventral nerve cord only"),
         ("eLife 13:RP97769", "eLife 13:RP97766", "eLife 13:RP96084", "gs://flyem-manc-exports/v1.0")),
        ("ol", "optic_lobe_v1.1", reg.RegionVocabulary.OPTIC_LOBE_NEUROPIL, ("optic lobe", "optic-lobe:v1.1"),
         ("Nern A", "10.1038/s41586-025-08746-0")),
    ],
)
def test_flyem_snapshots_are_pinned_licensed_and_cited(tmp_path, symbol, version, vocabulary, anatomy_words,
                                                       citation_words):
    spec = reg.get_dataset(symbol)
    assert spec.pinned_version == version
    assert spec.snapshot_dir_name == symbol
    assert spec.region_vocabulary is vocabulary
    assert spec.license == reg.LICENSE_CC_BY_4_0
    assert spec.status is reg.DatasetStatus.PLANNED
    for word in anatomy_words:
        assert word in spec.description
    for word in citation_words:
        assert word in spec.citation
    root = reg.snapshot_version_root(symbol, storage_root=tmp_path)
    assert root.relative_to(tmp_path).as_posix() == f"snapshots/{symbol}/{version}"
    assert "MANC" not in reg.get_dataset("mc").description


def test_split_group_keys_per_dataset():
    assert reg.split_group_keys("banc") == ("cell_type", "hemilineage")
    assert reg.split_group_keys("l1em") == ("split_group",)
    assert reg.split_group_keys("ol") == ("cell_type",)
    assert reg.split_group_keys("fafb") == ()
    assert reg.get_dataset("mc").as_dict()["split_group_keys"] == ["cell_type", "hemilineage"]


def test_registry_validation_requires_pin_for_reviewed_licence(monkeypatch):
    broken = dataclasses.replace(reg.DATASET_REGISTRY["mv"], pinned_version=None)
    monkeypatch.setitem(reg.DATASET_REGISTRY, "mv", broken)
    with pytest.raises(AssertionError, match="pinned snapshot version"):
        reg._validate_registry()


_REAL_ROOT = os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", "")


@pytest.mark.skipif(not _REAL_ROOT or not os.path.isdir(os.path.join(_REAL_ROOT, "snapshots")),
                    reason="real FlyBrain storage root not configured")
@pytest.mark.parametrize("symbol", ["mc", "mv", "ol"])
def test_real_snapshot_manifests_match_registry(symbol):
    import json

    spec = reg.get_dataset(symbol)
    manifest_path = reg.snapshot_version_root(symbol, _REAL_ROOT) / "manifest" / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip(f"{symbol} snapshot not present")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"]["symbol"] == symbol
    assert manifest["dataset"]["version_id"] == spec.pinned_version
    assert manifest["dataset"]["source"]["license"]["spdx_id"] == spec.license


def test_as_dict_is_json_ready():
    payload = reg.get_dataset("hb").as_dict()
    assert payload["status"] == "active"
    assert payload["region_vocabulary"] == "hemibrain_roi"
    assert isinstance(payload["supported_objectives"], list)


def test_integrated_licences_and_objective_lists():
    assert reg.get_dataset("banc").license == reg.LICENSE_CC_BY_4_0
    assert reg.get_dataset("l1em").license == reg.LICENSE_CC_BY_4_0
    assert reg.supported_objectives("l1em") == ("connectivity_tier",)
    assert reg.get_dataset("banc").status is reg.DatasetStatus.PLANNED
    assert reg.get_dataset("l1em").status is reg.DatasetStatus.PLANNED


def test_fafb_is_deferred_and_supports_nothing():
    spec = reg.get_dataset("fafb")
    assert spec.supported_objectives == ()
    assert spec.status is reg.DatasetStatus.PLANNED
    assert "fafb" in reg.NEVER_ACTIVE_SYMBOLS
    with pytest.raises(reg.DatasetRegistryError) as exc:
        reg.require_objective("fafb", "connectivity_tier")
    assert exc.value.code is reg.DatasetErrorCode.OBJECTIVE_NOT_SUPPORTED


def test_registry_validation_refuses_active_fafb(monkeypatch):
    promoted = dataclasses.replace(reg.DATASET_REGISTRY["fafb"], status=reg.DatasetStatus.ACTIVE)
    monkeypatch.setitem(reg.DATASET_REGISTRY, "fafb", promoted)
    with pytest.raises(AssertionError, match="deferred"):
        reg._validate_registry()

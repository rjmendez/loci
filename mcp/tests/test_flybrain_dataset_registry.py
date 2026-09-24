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


def test_unpinned_snapshot_root_fails_closed(tmp_path):
    with pytest.raises(reg.DatasetRegistryError) as exc:
        reg.snapshot_version_root("mc", storage_root=tmp_path)
    assert exc.value.code is reg.DatasetErrorCode.VERSION_UNPINNED


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

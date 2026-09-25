"""Offline tests for flybrain_nt_ground_truth (a synthetic pinned repo under tmp_path; never touches /mnt/f)."""

import hashlib
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_nt_ground_truth as ntgt  # noqa: E402
import flybrain_wiring_features as fwf  # noqa: E402

COMMIT = "0123456789abcdef0123456789abcdef01234567"
NTS = ("acetylcholine", "glutamate", "gaba", "glycine", "dopamine", "serotonin", "octopamine", "tyramine",
       "histamine", "nitric_oxide")


def _gt_row(cell_type, conf, positives=(), negatives=(), species="adult_drosophila_melanogaster",
            hemilineage="HL1", source="Study A"):
    row = {"species": species, "region": "central_brain", "hemilineage": hemilineage, "cell_type": cell_type,
           "neurotransmitter_verified_source": source, "neurotransmitter_verified_evidence": "immuno",
           "neurotransmitter_verified_confidence": conf}
    for nt in NTS:
        row[nt] = 1 if nt in positives else -1 if nt in negatives else 0
    return row


GT_ROWS = [
    _gt_row("TypeA", 5, ["acetylcholine"]),
    _gt_row("TypeA", 4, ["acetylcholine"], source="Study B"),
    _gt_row("TypeB", 4, ["gaba", "nitric_oxide"], hemilineage="HL2"),  # NO is a co-transmitter, ignored
    _gt_row("TypeC", 4, ["glutamate"], hemilineage="HL2"),
    _gt_row("TypeD", 3, ["glutamate"]),  # below the confidence bar
    _gt_row("TypeE", 5, ["acetylcholine", "dopamine"]),  # co-transmission row -> dropped
    _gt_row("TypeF", 4, ["gaba"]),
    _gt_row("TypeF", 5, ["glutamate"], source="Study B"),  # studies disagree -> dropped
    _gt_row("TypeG", 4, ["gaba"]),
    _gt_row("TypeG", 4, [], ["gaba"], source="Study B"),  # negative evidence conflict -> dropped
    _gt_row("TypeH", 5, ["acetylcholine"], species="larval_drosophila_melanogaster"),  # larva -> ignored
    _gt_row("HB_only", 5, ["gaba"], hemilineage="HL3"),  # matched through a secondary (hemibrain) type
    _gt_row("mixedCase", 4, ["glutamate"], hemilineage="HL3"),
]


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def nt_repo(tmp_path, monkeypatch):
    root = tmp_path / "nt-ground-truth"
    repo = root / f"repo-{COMMIT[:8]}"
    _write(repo / ".git" / "HEAD", COMMIT + "\n")
    files = {
        ntgt.GT_DATA: pd.DataFrame(GT_ROWS).to_csv(index=False),
        ntgt.CROSS_MATCHING: "cell_type,hemibrain_type\nTypeA,TypeA\n",
        ntgt.FW_CNN_GT[0]: "root_783,top_nt,cell_type,ito_lee_hemilineage,known_nt,known_nt_source\n1,gaba,TypeA,HL1,a,b\n",
        ntgt.FW_CNN_GT[1]: "species,region,cell_type,hemilineage,known_nt,known_nt_source,known_nt_evidence,"
                           "known_nt_confidence\nadult,midbrain,TypeC,HL2,glutamate,x,y,4\n",
        ntgt.BANC_CNN_GT[0]: "root_id,cell_type\n1,TypeA\n2,TypeB\n",
        ntgt.BANC_CNN_GT[1]: "root_id,cell_type\n",
        ntgt.BANC_CNN_GT[2]: "root_id,cell_type\n",
        ntgt.BANC_CNN_GT[3]: "root_id,cell_type\n",
        ntgt.BANC_REMOVED: "cell_types,super_class\nTypeB,central\n",
        ntgt.MC_CNN_GT[0]: "gt_celltype,cell_type_mcns,neurotransmitter_verified_confidence\n"
                           "TypeA,TypeA_mcns,4\nTypeC,,2\n",
        ntgt.MC_CNN_GT[1]: "cell_type,known_nt,known_nt_confidence\nTypeB,gaba,3\n",
        "LICENSE": "CC-BY-4.0\n",
    }
    pins = {}
    for rel, text in files.items():
        _write(repo / rel, text)
        pins[rel] = hashlib.sha256((repo / rel).read_bytes()).hexdigest()
    monkeypatch.setattr(ntgt, "NT_GT_COMMIT", COMMIT)
    monkeypatch.setattr(ntgt, "PINNED_SHA256", pins)
    _write(root / ntgt.SOURCE_FILE, json.dumps({"commit": COMMIT}))
    return root


# --------------------------------------------------------------------------- source integrity


def test_open_verifies_commit_and_hashes(nt_repo):
    source = ntgt.open_nt_ground_truth(nt_repo)
    assert source.commit == COMMIT and set(source.file_sha256) == set(ntgt.PINNED_SHA256)
    assert source.provenance()["license"] == "CC-BY-4.0"


def test_open_fails_closed(nt_repo, tmp_path):
    repo = nt_repo / f"repo-{COMMIT[:8]}"
    with pytest.raises(ntgt.NtGroundTruthError, match="clone missing"):
        ntgt.open_nt_ground_truth(nt_repo, commit="f" * 40)
    (repo / ".git" / "HEAD").write_text(COMMIT[:8] + "f" * 32)  # same directory, different checkout
    with pytest.raises(ntgt.NtGroundTruthError, match="expected"):
        ntgt.open_nt_ground_truth(nt_repo)
    (repo / ".git" / "HEAD").write_text(COMMIT)
    (repo / ntgt.GT_DATA).write_text("species,cell_type\n", encoding="utf-8")
    with pytest.raises(ntgt.NtGroundTruthError, match="sha256 mismatch"):
        ntgt.open_nt_ground_truth(nt_repo)
    with pytest.raises(ntgt.NtGroundTruthError, match="snapshots"):
        ntgt.open_nt_ground_truth(tmp_path / "snapshots" / "x")


def test_open_requires_source_record(nt_repo):
    (nt_repo / ntgt.SOURCE_FILE).unlink()
    with pytest.raises(ntgt.NtGroundTruthError, match="SOURCE.json"):
        ntgt.open_nt_ground_truth(nt_repo)
    assert ntgt.open_nt_ground_truth(nt_repo, require_source_file=False).commit == COMMIT


def test_pins_cover_every_file_the_module_reads():
    read = {ntgt.GT_DATA, ntgt.CROSS_MATCHING, *ntgt.FW_CNN_GT, *ntgt.BANC_CNN_GT, ntgt.BANC_REMOVED,
            *ntgt.MC_CNN_GT, ntgt.OL_CNN_GT}
    assert read <= set(ntgt.PINNED_SHA256)
    assert all(len(v) == 64 for v in ntgt.PINNED_SHA256.values())


# --------------------------------------------------------------------------- type-level labels


def test_type_level_labels_are_strict(nt_repo):
    source = ntgt.open_nt_ground_truth(nt_repo)
    labels, stats = ntgt.type_level_labels(ntgt.load_gt_rows(source))
    got = dict(zip(labels["gt_type"], labels["label"]))
    assert got == {"TypeA": "acetylcholine", "TypeB": "gaba", "TypeC": "glutamate", "HB_only": "gaba",
                   "mixedCase": "glutamate"}
    assert stats["dropped"] == {"co_transmission_row": 1, "no_fast_nt_row": 0, "studies_disagree": 1,
                                "negative_evidence_conflict": 1}
    assert labels.set_index("gt_type").loc["TypeA", "n_rows"] == 2
    lower, _ = ntgt.type_level_labels(ntgt.load_gt_rows(source), min_confidence=3)
    assert "TypeD" in set(lower["gt_type"]) and "TypeH" not in set(lower["gt_type"])


def test_binary_label():
    assert ntgt.binary_label("acetylcholine") == ntgt.BINARY_CHOLINERGIC
    assert ntgt.binary_label("gaba") == ntgt.binary_label("glutamate") == ntgt.BINARY_INHIBITORY_OR_GLU
    assert ntgt.binary_label("dopamine") is None


def test_hemilineage_nt_table(nt_repo):
    labels, _ = ntgt.type_level_labels(ntgt.load_gt_rows(ntgt.open_nt_ground_truth(nt_repo)))
    table = ntgt.hemilineage_nt_table(labels)
    assert table["HL1"] == "acetylcholine" and table["HL2"] in {"gaba", "glutamate"}


# --------------------------------------------------------------------------- CNN training types + matching


def test_cnn_training_types_per_dataset(nt_repo):
    source = ntgt.open_nt_ground_truth(nt_repo)
    fw, info = ntgt.cnn_training_types(source, "fw")
    assert fw == {"TypeA", "TypeC"} and info["identifiable"]
    banc, info = ntgt.cnn_training_types(source, "banc")
    assert banc == {"TypeA"} and info["removed_types"] == 1  # TypeB was removed from synister_banc training
    mc, _ = ntgt.cnn_training_types(source, "mc")
    assert mc == {"TypeA", "TypeA_mcns", "TypeB"}  # TypeC is below the confidence-3 training bar
    mv, info = ntgt.cnn_training_types(source, "mv")
    assert mv == frozenset() and not info["identifiable"]
    with pytest.raises(ntgt.NtGroundTruthError):
        ntgt.cnn_training_types(source, "nope")


def test_match_types_exact_alias_casefold_and_ambiguity():
    out = ntgt.match_types(["TypeA", "typeb", "Alias", "Amb", "Missing"],
                           ["TypeA", "TypeB", "Real", "amb", "AMB", None, "nan"], alias={"Alias": "Real"})
    got = dict(zip(out["gt_type"], zip(out["dataset_type"], out["match_method"])))
    assert got == {"TypeA": ("TypeA", "exact"), "typeb": ("TypeB", "casefold"), "Alias": ("Real", "alias")}


def test_label_neurons_strict_vs_all_and_secondary_types(nt_repo):
    source = ntgt.open_nt_ground_truth(nt_repo)
    nodes = pd.DataFrame({
        "root_id": [1, 2, 3, 4, 5, 6, 7],
        "cell_type": ["TypeA", "TypeA", "TypeB", "TypeC", "FwName", "MIXEDCASE", None],
        "hemibrain_type": [None, None, None, None, "HB_only", None, "HB_only"],
    })
    strict = ntgt.label_neurons(nodes, dataset="fw", source=source, target=ntgt.TARGET_NT_LITERATURE,
                                id_column="root_id", secondary_type_column="hemibrain_type")
    everything = ntgt.label_neurons(nodes, dataset="fw", source=source, target=ntgt.TARGET_NT_LITERATURE_ALL,
                                    id_column="root_id", secondary_type_column="hemibrain_type")
    all_labels = dict(zip(everything.frame["root_id"], everything.frame["label"]))
    assert all_labels == {1: "acetylcholine", 2: "acetylcholine", 3: "gaba", 4: "glutamate", 5: "gaba",
                          6: "glutamate"}
    assert everything.coverage["match_methods"] == {"exact": 3, "alias": 1, "casefold": 1}
    # TypeA and TypeC were FlyWire CNN ground truth -> removed from the strict target
    assert set(strict.frame["root_id"]) == {3, 5, 6}
    assert strict.coverage["cnn_training"]["matched_types_in_cnn_training"] == 2
    binary = ntgt.label_neurons(nodes, dataset="fw", source=source, target=ntgt.TARGET_NT_LITERATURE_ALL_BINARY,
                                id_column="root_id", secondary_type_column="hemibrain_type")
    assert set(binary.frame["label"]) == {ntgt.BINARY_CHOLINERGIC, ntgt.BINARY_INHIBITORY_OR_GLU}


def test_label_neurons_uses_the_male_cns_type_map(nt_repo):
    source = ntgt.open_nt_ground_truth(nt_repo)
    nodes = pd.DataFrame({"root_id": ["10", "11"], "cell_type": ["TypeA_mcns", "TypeC"]})
    out = ntgt.label_neurons(nodes, dataset="mc", source=source, target=ntgt.TARGET_NT_LITERATURE_ALL,
                             id_column="root_id")
    assert dict(zip(out.frame["root_id"], out.frame["match_method"])) == {"10": "alias", "11": "exact"}
    with pytest.raises(ValueError):
        ntgt.label_neurons(nodes, dataset="mc", source=source, target="nt_ground_truth", id_column="root_id")


# --------------------------------------------------------------------------- exclusions


def test_nt_literature_objectives_are_registered_and_block_lineage_and_partner_nt():
    for target in ntgt.NT_LITERATURE_TARGETS:
        assert target in fwf.registered_objectives()
        blocked = fwf.excluded_feature_names(
            ["hemilineage", "ito_lee_hemilineage", "cell_type", "top_nt", "predicted_nt", "gaba_avg",
             "out_comp__central", "degree__out_n_partners"], target)
        assert set(blocked) == {"hemilineage", "ito_lee_hemilineage", "cell_type", "top_nt", "predicted_nt",
                                "gaba_avg"}
        if hasattr(fwf, "is_nt_objective"):
            assert fwf.is_nt_objective(target)

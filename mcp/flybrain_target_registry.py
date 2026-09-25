"""Label provenance for every FlyBrain real-model target, and the gate that enforces it (R1).

Every (dataset, target) pair a ``flybrain_*_targets`` module can evaluate is
registered here with a ``label_provenance``:

``measured``              the label is an independent measurement (e.g. literature
                          transmitter identity from immunostaining / FISH / RNA-seq)
``curated_morphology``    curated from morphology / anatomy without connectivity (soma
                          position, nerve entry/exit, cell-body-fibre tract)
``connectivity_defined``  the annotation was derived (partly) from the connectome's own
                          wiring (CBLAST / connectivity clustering / motif definitions),
                          or is a statistic of the connectome itself (connectivity or
                          region tiers)
``model_predicted``       the label is the output of a classifier (every dataset NT column)

The gate (``apply_provenance_gate``) applies ONLY to ``measured`` and
``curated_morphology`` targets. For the others the harness criteria are still
computed and reported (``harness_pass``), but ``gate.pass`` is forced False and
the verdict reads ``not_gated``: a connectivity-defined target is reported as
"recovery of connectivity-derived annotations" and a model-predicted target as
"distillation of <classifier>", never as accuracy of the biology
[SYNTHESIS F8; Kapoor 2022 L2; REFORMS, Kapoor 2023; Codex FAQ].
``assert_promotable`` refuses to promote any artifact whose target is not
gateable or did not pass.

``run_gated_evaluation`` is the one entry point the target modules use: it
looks the provenance up (fail closed before any training), stamps it into the
dataset notes, runs ``flybrain_model_eval.run_evaluation``, applies the gate,
re-stamps the saved model artifacts' metrics and rewrites the report with the
provenance next to every number.

Provenance decisions follow the verified synthesis
(``docs/flybrain-research``; research-20260924T2245Z). Open questions (E1, E3,
E4, E5) are carried as ``provenance_uncertain=True`` with a note; they do not
change gating but are shown in every report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import flybrain_nt_ground_truth as ntgt


class LabelProvenance(str, Enum):
    MEASURED = "measured"
    CURATED_MORPHOLOGY = "curated_morphology"
    CONNECTIVITY_DEFINED = "connectivity_defined"
    MODEL_PREDICTED = "model_predicted"


GATEABLE_PROVENANCE: frozenset[LabelProvenance] = frozenset({LabelProvenance.MEASURED,
                                                               LabelProvenance.CURATED_MORPHOLOGY})
VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_NOT_GATED = "not_gated"
PROVENANCE_SCHEMA_VERSION = "flybrain-label-provenance/v1"


class ProvenanceGateError(ValueError):
    """A target without a registered provenance, or a promotion the provenance gate forbids."""


@dataclass(frozen=True)
class TargetProvenance:
    dataset: str
    target: str
    provenance: LabelProvenance
    label_source: str
    citations: tuple[str, ...]
    classifier: str | None = None  # model_predicted only: the classifier being distilled
    provenance_uncertain: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.provenance is LabelProvenance.MODEL_PREDICTED and not self.classifier:
            raise ValueError(f"{self.dataset}/{self.target}: model_predicted needs the classifier name")
        if not self.citations:
            raise ValueError(f"{self.dataset}/{self.target}: a provenance entry needs a citation")

    @property
    def gateable(self) -> bool:
        return self.provenance in GATEABLE_PROVENANCE

    @property
    def reporting_frame(self) -> str:
        if self.provenance is LabelProvenance.CONNECTIVITY_DEFINED:
            return "recovery of connectivity-derived annotations"
        if self.provenance is LabelProvenance.MODEL_PREDICTED:
            return f"distillation of {self.classifier}"
        return f"accuracy against {self.provenance.value.replace('_', ' ')} labels"

    @property
    def claim(self) -> str:
        """Same wording as ``flybrain_eval_stats.provenance_claim`` (the harness side of R1)."""
        if self.provenance is LabelProvenance.CONNECTIVITY_DEFINED:
            return "recovery of connectivity-derived annotations (not biological accuracy)"
        if self.provenance is LabelProvenance.MODEL_PREDICTED:
            return f"distillation of {self.classifier} (not ground-truth accuracy)"
        return self.reporting_frame + (" (provenance uncertain)" if self.provenance_uncertain else "")

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": PROVENANCE_SCHEMA_VERSION, "dataset": self.dataset, "target": self.target,
                "label_provenance": self.provenance.value, "gateable": self.gateable, "gate_applies": self.gateable,
                "claim": self.claim, "reporting_frame": self.reporting_frame, "label_source": self.label_source,
                "citations": list(self.citations), "classifier": self.classifier,
                "provenance_uncertain": self.provenance_uncertain, "note": self.note}

    def notes(self) -> dict[str, Any]:
        """``EvalDataset.notes`` keys the harness reads (``flybrain_eval_stats.provenance_claim``) + the record."""
        return {"label_provenance": self.provenance.value, "label_classifier": self.classifier,
                "provenance_uncertain": self.provenance_uncertain, "label_provenance_record": self.as_dict()}


M, C, K, P = (LabelProvenance.MEASURED, LabelProvenance.CURATED_MORPHOLOGY, LabelProvenance.CONNECTIVITY_DEFINED,
              LabelProvenance.MODEL_PREDICTED)
_TIER_NOTE = ("the label is a statistic of the connectome itself (synapse count / per-neuropil distribution), so a "
              "wiring model recovers a connectivity-derived quantity; kept as plumbing, never gated [SYNTHESIS F4]")
_NT_LIT_SOURCE = ntgt.NT_GT_CITATION


def _nt_literature_entries(dataset: str) -> list[TargetProvenance]:
    out = []
    for target, spec in ntgt.NT_LITERATURE_SPECS.items():
        out.append(TargetProvenance(
            dataset, target, M, spec.describe(),
            ("drosophila_neurotransmitters 2024", "Eckstein 2024", "Lacin 2019"),
            note=("R2 literature ground truth mapped onto this dataset by cell-type name; "
                  + ("the CNN-training-type removal is not identifiable for this dataset (E6)"
                     if dataset == "mv" else "coverage in report notes.nt_literature_coverage"))))
    return out


_ENTRIES: list[TargetProvenance] = [
    # ------------------------------------------------------------------ fw (FlyWire 783)
    TargetProvenance("fw", "super_class", C, "Schlegel 2024 super_class (soma position / nerve entry-exit)",
                     ("Schlegel 2024",)),
    TargetProvenance("fw", "super_class_no_neuropil", C, "Schlegel 2024 super_class, neuropil features removed",
                     ("Schlegel 2024",)),
    TargetProvenance("fw", "flow", C, "Schlegel 2024 flow (afferent / intrinsic / efferent)", ("Schlegel 2024",)),
    TargetProvenance("fw", "cell_class", C, "Schlegel 2024 cell_class", ("Schlegel 2024",), provenance_uncertain=True,
                     note="cell_class groups curated types by anatomy (optic classes are neuropil paths); the "
                          "FlyWire cell_type level itself is partly CBLAST-split (E4)"),
    TargetProvenance("fw", "cell_class_no_neuropil", C, "Schlegel 2024 cell_class, neuropil features removed",
                     ("Schlegel 2024",), provenance_uncertain=True, note="see fw/cell_class (E4)"),
    TargetProvenance("fw", "hemilineage", C, "Schlegel 2024 ito_lee_hemilineage (cell-body-fibre tracts + "
                     "light-level comparison)", ("Schlegel 2024", "Ito 2013", "Lee 2020"),
                     note="headline hemilineage target (R1): assigned without connectivity (inferred from absence "
                          "in the methods [SYNTHESIS F2])"),
    TargetProvenance("fw", "neurotransmitter_dominance", P, "Schlegel 2024 top_nt = per-neuron argmax of synapse-level "
                     "NT predictions", ("Eckstein 2024", "Schlegel 2024"),
                     classifier="the Eckstein 2024 FAFB synapse NT classifier (top_nt)"),
    TargetProvenance("fw", "nt_ground_truth", M, "Schlegel 2024 known_nt (literature; single classical transmitter)",
                     ("Schlegel 2024", "drosophila_neurotransmitters 2024"), provenance_uncertain=True,
                     note="known_nt is the 2024-05 snapshot of the literature table with no confidence filter; "
                          "superseded by nt_literature (R2, confidence >= 4)"),
    TargetProvenance("fw", "connectivity_tier", K, "top quartile of total presynapse count", ("Dorkenwald 2024",),
                     note=_TIER_NOTE),
    *_nt_literature_entries("fw"),
    # ------------------------------------------------------------------ mc (male-CNS v1.0)
    TargetProvenance("mc", "nt_ground_truth", M, "male-CNS v1.0 body-neurotransmitter ground_truth (per type)",
                     ("Berg 2025", "synister_malecns 2025"), provenance_uncertain=True,
                     note="the synister_malecns training table: confidence >= 3 and VNC hemilineage-transmitter "
                          "assignments, so partly inferred rather than measured (E5); superseded by nt_literature"),
    TargetProvenance("mc", "super_class", C, "male-CNS superclass (soma position / nerve entry-exit)",
                     ("Berg 2025",)),
    TargetProvenance("mc", "cell_class", K, "male-CNS class", ("Berg 2025",),
                     note="male-CNS types (and the classes grouping them) used NBLAST + connectivity similarity"),
    TargetProvenance("mc", "connectivity_tier", K, "legacy connectivity tier (total synapses >= q0.75)",
                     ("Berg 2025",), note=_TIER_NOTE),
    TargetProvenance("mc", "region_specialization_tier", K, "legacy region specialization tier",
                     ("Berg 2025",), note=_TIER_NOTE),
    *_nt_literature_entries("mc"),
    # ------------------------------------------------------------------ banc (BANC v888)
    TargetProvenance("banc", "super_class", C, "BANC curated super_class (harmonized)", ("Bates 2025",),
                     provenance_uncertain=True, note="BANC super_class/flow rule is undocumented (E1)"),
    TargetProvenance("banc", "flow", C, "BANC curated flow", ("Bates 2025",), provenance_uncertain=True,
                     note="BANC flow rule is undocumented (E1)"),
    TargetProvenance("banc", "cell_class", K, "BANC curated cell_class", ("Bates 2025",), provenance_uncertain=True,
                     note="BANC types/classes were transferred through NBLAST + connectivity co-clustering matches"),
    TargetProvenance("banc", "connectivity_tier", K, "legacy connectivity tier (outgoing v3 synapses >= q0.75)",
                     ("Bates 2025",), note=_TIER_NOTE),
    TargetProvenance("banc", "neurotransmitter_dominance", P, "per-neuron argmax of BANC neurotransmitter_prediction_v2",
                     ("Bates 2025",), classifier="synister_banc (the BANC fast-NT classifier)"),
    *_nt_literature_entries("banc"),
    # ------------------------------------------------------------------ mv (MANC v1.0)
    TargetProvenance("mv", "cell_class", C, "MANC class (intrinsic / sensory / ascending / descending / motor / "
                     "efferent), harmonized", ("Marin 2024", "Takemura 2024"), provenance_uncertain=True,
                     note="gross class from soma / nerve anatomy; the synthesis groups MANC classes with "
                          "connectivity-derived annotations, so read with care"),
    TargetProvenance("mv", "hemilineage", K, "MANC hemilineage", ("Marin 2024",),
                     note="assigned from soma tract + NBLAST + connectivity clustering, with NT predictions used to "
                          "confirm (R1); the headline hemilineage target is fw/hemilineage"),
    TargetProvenance("mv", "neurotransmitter_dominance", P, "MANC predictedNt", ("Takemura 2024",),
                     classifier="the MANC NT classifier (predictedNt; 187 GT neurons)"),
    TargetProvenance("mv", "connectivity_tier", K, "legacy connectivity tier", ("Takemura 2024",), note=_TIER_NOTE),
    TargetProvenance("mv", "region_specialization_tier", K, "legacy region specialization tier", ("Takemura 2024",),
                     note=_TIER_NOTE),
    *_nt_literature_entries("mv"),
    # ------------------------------------------------------------------ ol (optic-lobe v1.1)
    TargetProvenance("ol", "cell_family", K, "optic-lobe cell-type family", ("Matsliah 2024", "Nern 2025"),
                     note="optic-lobe types were finalised by connectivity (top-5 partner types)"),
    TargetProvenance("ol", "super_class", C, "optic-lobe body class (intrinsic / VPN / VCN / central)",
                     ("Nern 2025",), note="defined by where a neuron's arbors lie; near-definitional with roi features"),
    TargetProvenance("ol", "connectivity_tier", K, "legacy connectivity tier (downstream >= q0.75)", ("Nern 2025",),
                     note=_TIER_NOTE),
    TargetProvenance("ol", "neurotransmitter_dominance", P, "per-neuron predictedNt", ("Nern 2025",),
                     classifier="the optic-lobe synapse NT classifier (predictedNt) [Nern 2025]"),
    TargetProvenance("ol", "nt_ground_truth", M, "type-level consensusNt where ntReference is set", ("Nern 2025",),
                     provenance_uncertain=True,
                     note="consensusNt mixes classifier predictions with curation; superseded by nt_literature"),
    *_nt_literature_entries("ol"),
    # ------------------------------------------------------------------ l1em (Winding 2023)
    TargetProvenance("l1em", "l1em_io_class", K, "S2 celltype collapsed to input/output classes", ("Winding 2023",),
                     provenance_uncertain=True,
                     note="larval S2 types are connectivity-defined (R1); whether the io classes are is open (E3)"),
    TargetProvenance("l1em", "l1em_sensory_modality", C, "sense-organ modality of sensory neurons", ("Winding 2023",)),
    TargetProvenance("l1em", "connectivity_tier", K, "legacy connectivity tier", ("Winding 2023",), note=_TIER_NOTE),
]

TARGET_PROVENANCE: Mapping[tuple[str, str], TargetProvenance] = {}
for _entry in _ENTRIES:
    if (_entry.dataset, _entry.target) in TARGET_PROVENANCE:
        raise RuntimeError(f"duplicate provenance entry {_entry.dataset}/{_entry.target}")
    TARGET_PROVENANCE[(_entry.dataset, _entry.target)] = _entry  # type: ignore[index]


def register_target_provenance(entry: TargetProvenance, *, replace: bool = False) -> TargetProvenance:
    key = (entry.dataset, entry.target)
    if key in TARGET_PROVENANCE and not replace:
        raise ProvenanceGateError(f"provenance already registered for {entry.dataset}/{entry.target}")
    TARGET_PROVENANCE[key] = entry  # type: ignore[index]
    return entry


def target_provenance(dataset: str, target: str) -> TargetProvenance:
    """Fail closed: an unregistered (dataset, target) cannot be evaluated through the gated path."""
    key = (str(dataset).strip().lower(), str(target).strip())
    if key not in TARGET_PROVENANCE:
        raise ProvenanceGateError(
            f"no label provenance registered for {key[0]}/{key[1]}; add it to flybrain_target_registry "
            "(measured / curated_morphology / connectivity_defined / model_predicted)")
    return TARGET_PROVENANCE[key]


def registered_targets(dataset: str | None = None) -> list[tuple[str, str]]:
    return sorted(k for k in TARGET_PROVENANCE if dataset is None or k[0] == dataset)


def provenance_notes(dataset: str, target: str) -> dict[str, Any]:
    """Notes every target module merges into its ``EvalDataset`` (fail closed when unregistered)."""
    return target_provenance(dataset, target).notes()


def check_module_provenance(dataset: str, declared: Mapping[str, str]) -> None:
    """A target module's own provenance table must match the registry exactly (fail closed at import)."""
    for target, value in declared.items():
        registered = target_provenance(dataset, target).provenance.value
        if registered != LabelProvenance(value).value:
            raise ProvenanceGateError(f"{dataset}/{target}: module declares {value}, registry says {registered}")


# =========================================================================== gate


def apply_provenance_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Stamp provenance into ``report`` and force ``gate.pass`` False for non-gateable targets (idempotent)."""
    import flybrain_model_eval as fme

    prov = target_provenance(report["dataset"], report["target"])
    report["label_provenance"] = {**dict(report.get("label_provenance") or {}), **prov.as_dict()}
    for entry in report.get("models", {}).values():
        gate = entry.setdefault("gate", {})
        harness = bool(gate["harness_pass"]) if "harness_pass" in gate else bool(gate.get("pass"))
        gate["harness_pass"] = harness
        gate["label_provenance"] = prov.provenance.value
        gate["gate_applicable"] = prov.gateable
        gate["gate_applies"] = prov.gateable
        gate["claim"] = prov.claim
        gate["reporting_frame"] = prov.reporting_frame
        gate["pass"] = bool(harness and prov.gateable)
        gate["verdict"] = (VERDICT_PASS if gate["pass"] else VERDICT_FAIL) if prov.gateable else VERDICT_NOT_GATED
    rows = [dict(r) for r in report["summary"]] if report.get("summary") else fme.summary_rows(report)
    for row in rows:
        gate = report["models"][row["model"]]["gate"]
        row["gate"] = gate["verdict"]
        row["label_provenance"] = prov.provenance.value
        row["reporting_frame"] = prov.reporting_frame
        row["harness_criteria_met"] = gate["harness_pass"]
        row["provenance_uncertain"] = prov.provenance_uncertain
    report["summary"] = rows
    return report


def assert_promotable(report: Mapping[str, Any], model: str) -> None:
    """Refuse promotion / canary of ``model`` unless its target is gateable and it passed the gate."""
    prov = target_provenance(report["dataset"], report["target"])
    if not prov.gateable:
        raise ProvenanceGateError(
            f"{prov.dataset}/{prov.target} is {prov.provenance.value}: {prov.reporting_frame}; such targets are "
            "reported, never promoted (R1)")
    gate = (report.get("models", {}).get(model) or {}).get("gate") or {}
    if gate.get("label_provenance") != prov.provenance.value:
        raise ProvenanceGateError("report was not passed through apply_provenance_gate")
    if not gate.get("pass"):
        raise ProvenanceGateError(f"{prov.dataset}/{prov.target}/{model} did not pass the gate")


def provenance_banner(report: Mapping[str, Any]) -> str:
    prov = report.get("label_provenance") or target_provenance(report["dataset"], report["target"]).as_dict()
    lines = [
        f"> **Label provenance: `{prov['label_provenance']}`** - read every number below as "
        f"*{prov['reporting_frame']}*. Gate applies: **{'yes' if prov['gateable'] else 'no (reported, never gated)'}**."
        + (" Provenance uncertain." if prov.get("provenance_uncertain") else ""),
        f"> Label source: {prov['label_source']} [{'; '.join(prov['citations'])}]."
        + (f" {prov['note']}" if prov.get("note") else ""),
        "",
        "| model | label provenance | harness criteria met | verdict |",
        "|---|---|---|---|",
    ]
    for row in report.get("summary", []):
        lines.append(f"| {row['model']} | {row.get('label_provenance', prov['label_provenance'])} | "
                     f"{row.get('harness_criteria_met')} | {row['gate']} |")
    return "\n".join(lines) + "\n\n"


def write_gated_report(report: Mapping[str, Any], report_root: str | Path, *, run_label: str = "") -> dict[str, str]:
    """``fme.write_report`` + the provenance banner at the top of report.md."""
    import flybrain_model_eval as fme

    if "label_provenance" not in report:
        raise ProvenanceGateError("apply_provenance_gate must run before the report is written")
    paths = fme.write_report(report, report_root, run_label=run_label)
    md = Path(paths["markdown"])
    text = md.read_text(encoding="utf-8")
    head, _, rest = text.partition("\n")
    md.write_text(head + "\n\n" + provenance_banner(report) + rest.lstrip("\n"), encoding="utf-8")
    return paths


def _restamp_artifacts(report: dict[str, Any], config: Any) -> None:
    """Re-save each saved model so its metrics.json carries the provenance-gated verdict."""
    import flybrain_learners as fl
    import flybrain_model_eval as fme

    root = fme.report_dir(config.report_root, report["dataset"], report["target"], config.run_label) / "models"
    for label, entry in report.get("models", {}).items():
        artifact = entry.get("artifact")
        if not artifact:
            continue
        learner = fl.load_learner(artifact["path"], trusted_root=root,
                                  expected_manifest_sha256=artifact["manifest_sha256"])
        metrics = {"test": {k: v for k, v in entry["test"].items() if k in ("accuracy", "macro_f1", "ece", "log_loss")},
                   "gate": entry["gate"], "label_provenance": report["label_provenance"]}
        manifest = fl.save_learner(learner, artifact["path"], metrics=metrics, overwrite=True)
        entry["artifact"] = {"path": manifest["path"], "manifest_sha256": manifest["manifest_sha256"]}


def run_gated_evaluation(data: Any, config: Any) -> dict[str, Any]:
    """``run_evaluation`` behind the provenance gate (the single entry point for target modules)."""
    import flybrain_model_eval as fme

    prov = target_provenance(data.dataset, data.target)  # fail closed before any training
    data.notes = {**dict(data.notes or {}), **prov.notes()}
    report = fme.run_evaluation(data, config)
    apply_provenance_gate(report)
    if getattr(config, "report_root", None):
        if getattr(config, "save_models", False):
            _restamp_artifacts(report, config)
        write_gated_report(report, config.report_root, run_label=config.run_label)
    return report


def provenance_table() -> list[dict[str, Any]]:
    """Every registered (dataset, target) with its provenance, for docs and the ship step."""
    return [TARGET_PROVENANCE[k].as_dict() for k in registered_targets()]


def main() -> int:  # pragma: no cover - convenience dump
    print(json.dumps(provenance_table(), indent=2))
    return 0


__all__ = [
    "GATEABLE_PROVENANCE",
    "LabelProvenance",
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceGateError",
    "TARGET_PROVENANCE",
    "TargetProvenance",
    "VERDICT_FAIL",
    "VERDICT_NOT_GATED",
    "VERDICT_PASS",
    "apply_provenance_gate",
    "assert_promotable",
    "provenance_banner",
    "provenance_table",
    "register_target_provenance",
    "registered_targets",
    "run_gated_evaluation",
    "target_provenance",
    "write_gated_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Literature neurotransmitter ground truth (R2) for the FlyBrain real-model targets.

The only NT ground truth this project accepts is the literature compilation
``funkelab/drosophila_neurotransmitters`` (CC-BY-4.0; >900 cell types from 71
studies, one row per (cell type, study) with a 0-5 confidence score)
[drosophila_neurotransmitters 2024; Eckstein 2024]. Every per-dataset NT column
(FlyWire ``top_nt``, BANC ``synister_banc``, MANC ``predictedNt``, male-CNS
``synister_malecns``, optic-lobe ``predictedNt``) is a CNN output and is only
ever a *distillation* target (R1, see ``flybrain_target_registry``).

This module

* pins the repository to one commit (``NT_GT_COMMIT``) and the sha256 of every
  file it reads (``PINNED_SHA256``); ``fetch_nt_ground_truth`` clones it
  read-only into ``/mnt/f/.flybrain/cache/nt-ground-truth/`` and records the
  source (``SOURCE.json``); ``open_nt_ground_truth`` fails closed when the
  commit or any hash differs,
* collapses the per-study rows into one type-level label
  (``type_level_labels``): adult rows with confidence >= ``min_confidence``
  (default 4) and exactly one positive fast transmitter; types whose rows
  disagree, or where one row's positive NT is another row's negative evidence,
  are dropped (and counted), never resolved by vote,
* lists the types each dataset's NT CNN was trained on where the repository
  makes that identifiable (``cnn_training_types``), so R2's "drop CNN training
  types" is applied per dataset and its effect is reported,
* maps type-level labels onto one dataset's neurons by type name
  (``label_neurons``) and reports match coverage (``coverage``).

Nitric oxide is excluded from the label set: it is a gaseous co-transmitter
that never appears alone among the confidence >= 4 rows. ``binary_label``
gives the R3 binary view (acetylcholine vs GABA + glutamate).

Target names registered here (label-exclusion objectives; all ``measured``):

``nt_literature``              conf >= 4, the dataset CNN's training types removed (R2, strict)
``nt_literature_binary``       the same rows, acetylcholine vs GABA + glutamate
``nt_literature_all``          conf >= 4 incl. CNN-training types (sensitivity: a wiring model
                               never sees the CNN, so this is not leakage, but it is not
                               comparable with the CNN's own held-out accuracy)
``nt_literature_all_binary``   binary view of ``nt_literature_all``

Nothing here reads a snapshot or writes outside ``<root>``; the cache root is
refused if it lies under a ``snapshots`` directory.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

import flybrain_wiring_features as fwf

NT_GT_REPO_URL = "https://github.com/funkelab/drosophila_neurotransmitters"
NT_GT_COMMIT = "a9417412c8a70fcc9f80a65ca5bc6064eba07be3"
NT_GT_LICENSE = "CC-BY-4.0"
NT_GT_CITATION = ("funkelab/flyconnectome drosophila_neurotransmitters (CC-BY-4.0), "
                  f"commit {NT_GT_COMMIT}; Eckstein, Bates et al. 2024 Cell 187:2574")
DEFAULT_NT_GT_ROOT = "/mnt/f/.flybrain/cache/nt-ground-truth"
SOURCE_FILE = "SOURCE.json"
SOURCE_SCHEMA = "flybrain-nt-ground-truth-source/v1"

GT_DATA = "gt_data.csv"
CROSS_MATCHING = "inst/extdata/cell_type_cross_matching.csv"
FW_CNN_GT = ("gt_sources/bates_2024/202405-flywire_gt_data.csv", "gt_sources/bates_2024/202405-starting_gt_data.csv")
BANC_CNN_GT = ("gt_sources/banc/202505-banc_gt_data.csv", "gt_sources/banc/202506-banc_gt_data.csv",
               "gt_sources/banc/202508-banc_gt_data.csv", "gt_sources/banc/202509-banc_gt_data.csv")
BANC_REMOVED = "gt_sources/banc/banc_cell_types_removed_for_nt.csv"
MC_CNN_GT = ("gt_sources/male_cns/202509-male_cns_gt_data.csv", "gt_sources/male_cns/malecns_extra.csv")
OL_CNN_GT = "gt_sources/nern_et_al_2024/nern_et_al_2024.xlsx"

# sha256 of every file this module reads, at NT_GT_COMMIT (verified by open_nt_ground_truth).
PINNED_SHA256: Mapping[str, str] = {
    GT_DATA: "3608fd604ae37b113c3751f6085d6ede9c832b6148a451c69f09a2a28dccb94e",
    CROSS_MATCHING: "d31a7d9aa33c05dda078131ed97f5d2c3f5b4586c1e23794ca90a550ed0f755b",
    FW_CNN_GT[0]: "7c03fb907b10312c34bee256c2be023d16fb58c19d8696f238ec2fdfed3af0a7",
    FW_CNN_GT[1]: "7667373d6c209ff3b857122578383da8d2a56bd16b843662a235a05d1f6196ac",
    BANC_CNN_GT[0]: "3c4944ba5ccae49feffffae11403c6a2f42686fa6634666940875892cea97ea9",
    BANC_CNN_GT[1]: "4d7f57eaee8a4f1056548b60e367be786a6d1a34696761db6d1c3a92f25030d5",
    BANC_CNN_GT[2]: "4b93cc1925c770db365b3498be8a5cdcf764ed482de89a7b31f0590cc63cda6e",
    BANC_CNN_GT[3]: "4b10884e58118b4a76e7f964f1ea3536c9a0d3016bfb685f45659a6c7e8ffa73",
    BANC_REMOVED: "cec6e1a63ff9a9e29eaec3033586bae80481451f36a86443dee3328785f4f408",
    MC_CNN_GT[0]: "efff14b93613cde46e288113ab7795e534e1a3a04f48252ddbf23dabe033e99b",
    MC_CNN_GT[1]: "8ef45454693de379bdb3081f39119f09cc6b9d35b2278b92b0884548d655b2b1",
    OL_CNN_GT: "52a5fe3d471f39fbd1250c1921177288cac09650b026d70af8b2fdde7a8e4a3c",
    "LICENSE": "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
}

# Fast-acting small-molecule transmitters that can be a label. Nitric oxide is a co-transmitter.
FAST_NT: tuple[str, ...] = ("acetylcholine", "glutamate", "gaba", "glycine", "dopamine", "serotonin", "octopamine",
                            "tyramine", "histamine")
NT_COLUMNS: tuple[str, ...] = FAST_NT + ("nitric_oxide",)
ADULT_SPECIES = "adult_drosophila_melanogaster"
DEFAULT_MIN_CONFIDENCE = 4
BINARY_CHOLINERGIC = "acetylcholine"
BINARY_INHIBITORY_OR_GLU = "gaba_or_glutamate"

TARGET_NT_LITERATURE = "nt_literature"
TARGET_NT_LITERATURE_BINARY = "nt_literature_binary"
TARGET_NT_LITERATURE_ALL = "nt_literature_all"
TARGET_NT_LITERATURE_ALL_BINARY = "nt_literature_all_binary"
NT_LITERATURE_TARGETS: tuple[str, ...] = (TARGET_NT_LITERATURE, TARGET_NT_LITERATURE_BINARY, TARGET_NT_LITERATURE_ALL,
                                          TARGET_NT_LITERATURE_ALL_BINARY)
SUPPORTED_DATASETS: tuple[str, ...] = ("fw", "mc", "banc", "mv", "ol")


class NtGroundTruthError(RuntimeError):
    """The pinned NT ground-truth source failed a fail-closed check."""


# =========================================================================== targets


@dataclass(frozen=True)
class NtLiteratureSpec:
    target: str
    exclude_cnn_training: bool
    binary: bool
    min_confidence: int = DEFAULT_MIN_CONFIDENCE

    def describe(self) -> str:
        return (f"literature NT (drosophila_neurotransmitters @ {NT_GT_COMMIT[:8]}, adult rows, confidence >= "
                f"{self.min_confidence}, one positive fast transmitter per type, conflicting types dropped"
                + ("; the dataset NT CNN's training types removed" if self.exclude_cnn_training else
                   "; CNN-training types KEPT (sensitivity view)")
                + ("; binary acetylcholine vs gaba+glutamate" if self.binary else "") + ")")


NT_LITERATURE_SPECS: Mapping[str, NtLiteratureSpec] = {
    TARGET_NT_LITERATURE: NtLiteratureSpec(TARGET_NT_LITERATURE, True, False),
    TARGET_NT_LITERATURE_BINARY: NtLiteratureSpec(TARGET_NT_LITERATURE_BINARY, True, True),
    TARGET_NT_LITERATURE_ALL: NtLiteratureSpec(TARGET_NT_LITERATURE_ALL, False, False),
    TARGET_NT_LITERATURE_ALL_BINARY: NtLiteratureSpec(TARGET_NT_LITERATURE_ALL_BINARY, False, True),
}

_LINEAGE_AND_TYPE_PATTERNS = ("*hemilineage*", "*lineage*", "ito*", "truman*", "*cell_type*", "type", "*_type",
                              "*supertype*", "*known_nt*", "*gt_*")


def register_nt_literature_objectives() -> None:
    """Register the label-exclusion rules (idempotent): NT patterns + lineage + type identifiers.

    Lineage is excluded because NT is largely fixed per hemilineage [Lacin 2019;
    Eckstein 2024: 88% of hemilineages strongly biased to one NT]; the
    hemilineage -> NT route is scored separately as an oracle baseline, never
    fed to the wiring model. Partner-NT-derived features are banned for every
    NT objective by ``flybrain_wiring_features`` (R5).
    """
    base = fwf.objective_exclusions("nt_ground_truth").patterns
    for target in NT_LITERATURE_TARGETS:
        if hasattr(fwf, "register_nt_objective"):
            fwf.register_nt_objective(target)
        if target in fwf.registered_objectives():
            continue
        fwf.register_objective_exclusions(
            target, tuple(base) + _LINEAGE_AND_TYPE_PATTERNS,
            reason=("label is a literature transmitter assigned per cell type; NT predictions/scores near-copy it, "
                    "hemilineage fixes it for most lineages and type identifiers would look it up"))


register_nt_literature_objectives()


# =========================================================================== source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refuse_snapshots(path: Path) -> None:
    if "snapshots" in Path(path).resolve(strict=False).parts:
        raise NtGroundTruthError(f"refusing to use a path under snapshots/: {path}")


def repo_dir(root: str | Path | None = None, commit: str | None = None) -> Path:
    commit = commit or NT_GT_COMMIT
    return Path(root or DEFAULT_NT_GT_ROOT) / f"repo-{commit[:8]}"


def _git_head(repo: Path) -> str:
    head = (repo / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = repo / ".git" / head.split(None, 1)[1]
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
        packed = repo / ".git" / "packed-refs"
        wanted = head.split(None, 1)[1]
        for line in packed.read_text(encoding="utf-8").splitlines() if packed.is_file() else []:
            parts = line.split()
            if len(parts) == 2 and parts[1] == wanted:
                return parts[0]
        raise NtGroundTruthError(f"cannot resolve {wanted} in {repo}")
    return head


@dataclass(frozen=True)
class NtGroundTruthSource:
    repo: Path
    commit: str
    file_sha256: Mapping[str, str]
    source: Mapping[str, Any] = field(default_factory=dict)

    def path(self, rel: str) -> Path:
        if rel not in self.file_sha256:
            raise NtGroundTruthError(f"{rel} is not a pinned file")
        return self.repo / rel

    def provenance(self) -> dict[str, Any]:
        return {"repo_url": NT_GT_REPO_URL, "commit": self.commit, "license": NT_GT_LICENSE,
                "citation": NT_GT_CITATION, "gt_data_sha256": self.file_sha256[GT_DATA],
                "files_sha256": dict(sorted(self.file_sha256.items()))}


def fetch_nt_ground_truth(root: str | Path | None = None, *, commit: str | None = None,
                          git: str = "git", timeout: int = 600) -> NtGroundTruthSource:
    """Clone (if absent) + check out ``commit`` under ``root``, verify the pins, write ``SOURCE.json``."""
    commit = commit or NT_GT_COMMIT
    root = Path(root or DEFAULT_NT_GT_ROOT)
    _refuse_snapshots(root)
    repo = repo_dir(root, commit)
    if not repo.exists():
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run([git, "clone", "--quiet", NT_GT_REPO_URL, str(repo)], check=True, timeout=timeout)
        subprocess.run([git, "-C", str(repo), "checkout", "--quiet", commit], check=True, timeout=timeout)
    source = open_nt_ground_truth(root, commit=commit, require_source_file=False)
    payload = {"schema_version": SOURCE_SCHEMA, "repo_url": NT_GT_REPO_URL, "commit": commit,
               "license": NT_GT_LICENSE, "citation": NT_GT_CITATION, "repo_dir": str(repo),
               "files_sha256": dict(sorted(source.file_sha256.items())),
               "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (root / SOURCE_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return NtGroundTruthSource(repo, commit, source.file_sha256, payload)


def open_nt_ground_truth(root: str | Path | None = None, *, commit: str | None = None,
                         expected_sha256: Mapping[str, str] | None = None,
                         require_source_file: bool = True) -> NtGroundTruthSource:
    """Verify the local clone (HEAD == ``commit``, every pinned file's sha256) and return it (fail closed).

    Defaults (``DEFAULT_NT_GT_ROOT``, ``NT_GT_COMMIT``, ``PINNED_SHA256``) are read at call time.
    """
    commit = commit or NT_GT_COMMIT
    expected_sha256 = PINNED_SHA256 if expected_sha256 is None else expected_sha256
    root = Path(root or DEFAULT_NT_GT_ROOT)
    _refuse_snapshots(root)
    repo = repo_dir(root, commit)
    if not (repo / ".git").exists():
        raise NtGroundTruthError(f"NT ground-truth clone missing: {repo} (run fetch_nt_ground_truth)")
    head = _git_head(repo)
    if head != commit:
        raise NtGroundTruthError(f"NT ground-truth clone is at {head}, expected {commit}")
    shas: dict[str, str] = {}
    for rel, expected in sorted(expected_sha256.items()):
        path = repo / rel
        if not path.is_file():
            raise NtGroundTruthError(f"pinned file missing: {rel}")
        actual = _sha256(path)
        if actual != expected:
            raise NtGroundTruthError(f"sha256 mismatch for {rel}: expected {expected}, got {actual}")
        shas[rel] = actual
    source: dict[str, Any] = {}
    source_path = root / SOURCE_FILE
    if source_path.is_file():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source.get("commit") != commit:
            raise NtGroundTruthError(f"{SOURCE_FILE} records commit {source.get('commit')}, expected {commit}")
    elif require_source_file:
        raise NtGroundTruthError(f"{source_path} is missing (run fetch_nt_ground_truth to record the source)")
    return NtGroundTruthSource(repo, commit, shas, source)


# =========================================================================== type-level labels


def load_gt_rows(source: NtGroundTruthSource) -> pd.DataFrame:
    frame = pd.read_csv(source.path(GT_DATA), dtype={"cell_type": str, "hemilineage": str, "region": str,
                                                      "species": str})
    missing = [c for c in ("species", "cell_type", "neurotransmitter_verified_confidence", *NT_COLUMNS)
               if c not in frame.columns]
    if missing:
        raise NtGroundTruthError(f"gt_data.csv lacks columns {missing}")
    return frame


def type_level_labels(rows: pd.DataFrame, *, min_confidence: int = DEFAULT_MIN_CONFIDENCE,
                      species: str = ADULT_SPECIES) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One label per cell type from the per-study rows (fail closed on any disagreement).

    A row counts when its species matches and its confidence is >=
    ``min_confidence``. Per row, the positive set is the fast transmitters
    marked 1. A type is kept when every counted row has exactly one positive
    fast transmitter, all rows agree, and no counted row marks that
    transmitter -1. Returns (labels, drop statistics).
    """
    conf = pd.to_numeric(rows["neurotransmitter_verified_confidence"], errors="coerce")
    use = rows[(rows["species"] == species) & (conf >= int(min_confidence)) & rows["cell_type"].notna()].copy()
    use["cell_type"] = use["cell_type"].astype(str).str.strip()
    stats: dict[str, Any] = {"rows_total": int(len(rows)), "rows_used": int(len(use)),
                             "types_considered": int(use["cell_type"].nunique()), "min_confidence": int(min_confidence),
                             "dropped": {"co_transmission_row": 0, "no_fast_nt_row": 0, "studies_disagree": 0,
                                         "negative_evidence_conflict": 0}}
    records = []
    for cell_type, part in use.groupby("cell_type", sort=True):
        positives = []
        reason = None
        for _, row in part.iterrows():
            pos = [nt for nt in FAST_NT if _as_int(row[nt]) == 1]
            if len(pos) > 1:
                reason = "co_transmission_row"
                break
            if not pos:
                reason = reason or "no_fast_nt_row"
                continue
            positives.append(pos[0])
        if reason == "co_transmission_row" or (not positives):
            stats["dropped"][reason or "no_fast_nt_row"] += 1
            continue
        if len(set(positives)) > 1:
            stats["dropped"]["studies_disagree"] += 1
            continue
        label = positives[0]
        if (part[label].map(_as_int) == -1).any():
            stats["dropped"]["negative_evidence_conflict"] += 1
            continue
        hl = sorted({str(v).strip() for v in part.get("hemilineage", pd.Series(dtype=str)).dropna()
                     if str(v).strip() and str(v).strip().lower() not in {"na", "nan"}})
        records.append({
            "gt_type": cell_type, "label": label, "n_rows": int(len(part)), "n_positive_rows": len(positives),
            "max_confidence": int(pd.to_numeric(part["neurotransmitter_verified_confidence"]).max()),
            "sources": "; ".join(sorted(set(part["neurotransmitter_verified_source"].astype(str)))),
            "evidence": "; ".join(sorted(set(part.get("neurotransmitter_verified_evidence",
                                                      pd.Series(dtype=str)).dropna().astype(str)))),
            "gt_hemilineage": hl[0] if len(hl) == 1 else (None if not hl else "|".join(hl)),
            "region": "|".join(sorted(set(part["region"].dropna().astype(str)))) if "region" in part else None,
        })
    out = pd.DataFrame.from_records(records, columns=["gt_type", "label", "n_rows", "n_positive_rows",
                                                      "max_confidence", "sources", "evidence", "gt_hemilineage",
                                                      "region"])
    stats["types_labelled"] = int(len(out))
    stats["types_per_label"] = {str(k): int(v) for k, v in out["label"].value_counts().sort_index().items()}
    return out, stats


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def binary_label(label: Any) -> str | None:
    """R3 binary view: acetylcholine vs (GABA + glutamate); every other transmitter -> None."""
    if label == "acetylcholine":
        return BINARY_CHOLINERGIC
    if label in ("gaba", "glutamate"):
        return BINARY_INHIBITORY_OR_GLU
    return None


# =========================================================================== CNN training types


def _types_from_csv(path: Path, columns: Sequence[str], *, min_conf: int | None = None,
                    conf_column: str | None = None) -> set[str]:
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if min_conf is not None and conf_column in frame.columns:
        frame = frame[pd.to_numeric(frame[conf_column], errors="coerce").fillna(-1) >= min_conf]
    out: set[str] = set()
    for column in columns:
        if column in frame.columns:
            out |= {str(v).strip() for v in frame[column].dropna() if str(v).strip() and str(v).strip() != "NA"}
    return out


def cnn_training_types(source: NtGroundTruthSource, dataset: str) -> tuple[frozenset[str], dict[str, Any]]:
    """Types that were ground truth for ``dataset``'s NT CNN, where the repository makes that identifiable.

    Conservative (over-inclusive) by design: when a classifier's exact
    training list is not published, the ground-truth table that was current
    when it was trained is used [Eckstein 2024; Bates 2025; Nern 2025; SYNTHESIS E6].
    """
    dataset = str(dataset).lower()
    if dataset == "fw":
        types = set()
        for rel in FW_CNN_GT:
            types |= _types_from_csv(source.path(rel), ("cell_type",))
        return frozenset(types), {"identifiable": True, "classifier": "Eckstein 2024 FAFB synapse NT classifier",
                                  "rule": "every cell_type in the 2024-05 FlyWire + starting GT tables "
                                          "(superset of the 356 training types)", "files": list(FW_CNN_GT)}
    if dataset == "banc":
        types = set()
        for rel in BANC_CNN_GT:
            types |= _types_from_csv(source.path(rel), ("cell_type",))
        removed = _types_from_csv(source.path(BANC_REMOVED), ("cell_types",))
        return frozenset(types - removed), {
            "identifiable": True, "classifier": "synister_banc (BANC fast-NT classifier)",
            "rule": "every cell_type in the 2025-05..09 BANC GT tables minus banc_cell_types_removed_for_nt",
            "files": [*BANC_CNN_GT, BANC_REMOVED], "removed_types": len(removed)}
    if dataset == "mc":
        types = _types_from_csv(source.path(MC_CNN_GT[0]), ("gt_celltype", "cell_type_mcns"), min_conf=3,
                                conf_column="neurotransmitter_verified_confidence")
        types |= _types_from_csv(source.path(MC_CNN_GT[1]), ("cell_type",), min_conf=3,
                                 conf_column="known_nt_confidence")
        return frozenset(types), {"identifiable": True, "classifier": "synister_malecns",
                                  "rule": "gt_celltype / cell_type_mcns of the 2025-09 male-CNS GT with confidence "
                                          ">= 3 (the classifier's training threshold) + malecns_extra",
                                  "files": list(MC_CNN_GT)}
    if dataset == "ol":
        table = pd.read_excel(source.path(OL_CNN_GT))
        flag = table["Part_of_training_data"].astype(str).str.strip().str.lower() == "yes"
        types = {str(v).strip() for v in table.loc[flag, "Cell Type"].dropna()}
        expanded = set(types)
        for t in types:  # "R1-R6" style ranges name several types
            m = re.fullmatch(r"R(\d)-R(\d)", t)
            if m:
                expanded |= {f"R{i}" for i in range(int(m.group(1)), int(m.group(2)) + 1)}
        return frozenset(expanded), {"identifiable": True, "classifier": "optic-lobe synapse NT classifier [Nern 2025]",
                                     "rule": "Nern et al. table rows with Part_of_training_data == yes",
                                     "files": [OL_CNN_GT]}
    if dataset == "mv":
        return frozenset(), {"identifiable": False, "classifier": "MANC NT classifier [Takemura 2024]",
                             "rule": "trained on 187 GT neurons (ACh/GABA/Glu); their types are not listed in the "
                                     "repository, so nothing is removed (SYNTHESIS E6)"}
    raise NtGroundTruthError(f"no CNN-training rule for dataset {dataset!r}")


# =========================================================================== mapping onto a dataset


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).casefold()


def load_mc_type_map(source: NtGroundTruthSource) -> dict[str, str]:
    """gt_celltype -> male-CNS type, from the male-CNS GT table (only rows that name both)."""
    frame = pd.read_csv(source.path(MC_CNN_GT[0]), dtype=str)
    frame = frame.dropna(subset=["gt_celltype", "cell_type_mcns"])
    return {str(a).strip(): str(b).strip() for a, b in zip(frame["gt_celltype"], frame["cell_type_mcns"])}


def match_types(gt_types: Iterable[str], dataset_types: Iterable[Any], *,
                alias: Mapping[str, str] | None = None) -> pd.DataFrame:
    """``gt_type -> dataset_type`` by (1) exact name, (2) an explicit alias table, (3) case/space-folded name.

    A folded name that maps to more than one dataset type is ambiguous and not used.
    """
    present = sorted({str(t).strip() for t in dataset_types if t is not None and str(t).strip()
                      and str(t).strip().lower() not in {"nan", "none", "na"}})
    exact = set(present)
    folded: dict[str, list[str]] = {}
    for t in present:
        folded.setdefault(_norm(t), []).append(t)
    rows = []
    for gt in sorted(set(gt_types)):
        if gt in exact:
            rows.append((gt, gt, "exact"))
        elif alias and gt in alias and alias[gt] in exact:
            rows.append((gt, alias[gt], "alias"))
        elif len(folded.get(_norm(gt), [])) == 1:
            rows.append((gt, folded[_norm(gt)][0], "casefold"))
    frame = pd.DataFrame(rows, columns=["gt_type", "dataset_type", "match_method"])
    # one dataset type must not receive two gt types (fail closed: drop both)
    dup = frame["dataset_type"].duplicated(keep=False)
    return frame[~dup].reset_index(drop=True)


@dataclass
class NeuronLabels:
    frame: pd.DataFrame  # id_column, cell_type, gt_type, label, match_method
    coverage: dict[str, Any]
    type_labels: pd.DataFrame


def label_neurons(nodes: pd.DataFrame, *, dataset: str, source: NtGroundTruthSource, target: str,
                  id_column: str, type_column: str = "cell_type",
                  secondary_type_column: str | None = None) -> NeuronLabels:
    """Per-neuron literature NT for ``target`` (one of ``NT_LITERATURE_TARGETS``) + coverage statistics.

    ``secondary_type_column`` (e.g. FlyWire ``hemibrain_type``) is tried for
    literature types that match no primary type: its value is mapped to the
    single primary type carried by those neurons (ambiguous ones are skipped).
    """
    if target not in NT_LITERATURE_SPECS:
        raise ValueError(f"target must be one of {NT_LITERATURE_TARGETS}")
    spec = NT_LITERATURE_SPECS[target]
    rows = load_gt_rows(source)
    types, stats = type_level_labels(rows, min_confidence=spec.min_confidence)
    alias = dict(load_mc_type_map(source)) if dataset == "mc" else {}
    if secondary_type_column is not None:
        pairs = nodes[[type_column, secondary_type_column]].dropna()
        per_secondary = pairs.groupby(secondary_type_column)[type_column].agg(lambda s: sorted(set(map(str, s))))
        for secondary, primaries in per_secondary.items():
            if len(primaries) == 1:
                alias.setdefault(str(secondary).strip(), primaries[0])
    matches = match_types(types["gt_type"], nodes[type_column], alias=alias or None)
    labelled = types.merge(matches, on="gt_type", how="inner")
    cnn_types, cnn_info = cnn_training_types(source, dataset)
    in_cnn = labelled["gt_type"].isin(cnn_types) | labelled["dataset_type"].isin(cnn_types)
    coverage: dict[str, Any] = {
        "dataset": dataset, "target": target, "spec": asdict(spec), "gt": stats,
        "gt_types_labelled": int(len(types)),
        "gt_types_matched": int(len(labelled)),
        "match_methods": {str(k): int(v) for k, v in labelled["match_method"].value_counts().items()},
        "cnn_training": {**cnn_info, "n_types_listed": len(cnn_types),
                         "matched_types_in_cnn_training": int(in_cnn.sum())},
    }
    if spec.exclude_cnn_training:
        labelled = labelled[~in_cnn]
    if spec.binary:
        labelled = labelled.assign(label=labelled["label"].map(binary_label))
        labelled = labelled[labelled["label"].notna()]
    per_type = labelled.set_index("dataset_type")
    typed = nodes[[id_column, type_column]].copy()
    typed["_t"] = typed[type_column].map(lambda v: None if v is None or (isinstance(v, float) and v != v)
                                         else str(v).strip())
    typed = typed[typed["_t"].isin(per_type.index)]
    out = pd.DataFrame({
        id_column: typed[id_column].to_numpy(),
        "cell_type": typed["_t"].to_numpy(),
        "gt_type": per_type.loc[typed["_t"], "gt_type"].to_numpy(),
        "label": per_type.loc[typed["_t"], "label"].to_numpy(),
        "match_method": per_type.loc[typed["_t"], "match_method"].to_numpy(),
        "gt_hemilineage": per_type.loc[typed["_t"], "gt_hemilineage"].to_numpy(),
    })
    coverage.update({
        "types_used": int(len(labelled)),
        "neurons_labelled": int(len(out)),
        "neurons_with_type": int(nodes[type_column].notna().sum()),
        "types_per_label": {str(k): int(v) for k, v in labelled["label"].value_counts().sort_index().items()},
        "neurons_per_label": {str(k): int(v) for k, v in out["label"].value_counts().sort_index().items()},
        # effective label diversity: a few studies / lineages can dominate a small type set
        "top_sources": {str(k): int(v) for k, v in
                        labelled["sources"].str.split("; ").explode().value_counts().head(10).items()},
        "types_per_gt_hemilineage_top": {str(k): int(v) for k, v in
                                         labelled["gt_hemilineage"].value_counts().head(10).items()},
        "n_gt_hemilineages": int(labelled["gt_hemilineage"].nunique()),
        "provenance": source.provenance(),
    })
    return NeuronLabels(out, coverage, labelled.reset_index(drop=True))


def hemilineage_nt_table(type_labels: pd.DataFrame) -> dict[str, str]:
    """Majority literature NT per annotated gt hemilineage (for the R3 hemilineage -> NT oracle baseline)."""
    frame = type_labels.dropna(subset=["gt_hemilineage"])
    frame = frame[~frame["gt_hemilineage"].str.contains(r"\|", regex=True)]
    out = {}
    for hl, part in frame.groupby("gt_hemilineage"):
        counts = part["label"].value_counts()
        out[str(hl)] = str(sorted(counts[counts == counts.max()].index)[0])
    return out


__all__ = [
    "ADULT_SPECIES",
    "BINARY_CHOLINERGIC",
    "BINARY_INHIBITORY_OR_GLU",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_NT_GT_ROOT",
    "FAST_NT",
    "NT_GT_COMMIT",
    "NT_GT_LICENSE",
    "NT_GT_REPO_URL",
    "NT_LITERATURE_SPECS",
    "NT_LITERATURE_TARGETS",
    "NeuronLabels",
    "NtGroundTruthError",
    "NtGroundTruthSource",
    "NtLiteratureSpec",
    "PINNED_SHA256",
    "SUPPORTED_DATASETS",
    "TARGET_NT_LITERATURE",
    "TARGET_NT_LITERATURE_ALL",
    "TARGET_NT_LITERATURE_ALL_BINARY",
    "TARGET_NT_LITERATURE_BINARY",
    "binary_label",
    "cnn_training_types",
    "fetch_nt_ground_truth",
    "hemilineage_nt_table",
    "label_neurons",
    "load_gt_rows",
    "load_mc_type_map",
    "match_types",
    "open_nt_ground_truth",
    "register_nt_literature_objectives",
    "repo_dir",
    "type_level_labels",
]

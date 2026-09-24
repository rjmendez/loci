"""Calibrate brain-cluster gate/shadow thresholds per (dataset, objective).

Runs are grouped by ``(dataset_symbol, objective)``, never by objective alone,
so e.g. hb or BANC runs cannot count toward fw's minimum report count (roadmap
B3 / SC3). A run whose dataset or objective cannot be resolved, or whose
objective is ``custom``/``unknown``, is skipped and never calibrated.

Each calibrated group writes
``braincluster-thresholds-<dataset>-<objective>.json`` and records the
per-run trivial-baseline evidence (model held-out accuracy vs the best trivial
rule, roadmap B1 / AC6) next to the thresholds.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import flybrain_brain_cluster_thresholds as fbthr

RELEASE_PREP_SCHEMA_VERSION = "braincluster-release-prep/v2"
UNRESOLVED = "unknown"
# Objectives that are never calibrated: they carry no stable label contract.
NON_CALIBRATABLE_OBJECTIVES = frozenset({"", "custom", UNRESOLVED})
_SCHEMA_DATASET_RE = re.compile(r"^flybrain-([a-z0-9]+)-training-samples/v[0-9]+$")
_SAMPLE_DATASET_ALIASES = {"flywire": "fw", "flywire783": "fw", "hemibrain": "hb"}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class ReleasePrepConfig:
    runs_root: str | Path
    output_dir: str | Path
    # Minimum eligible reports per (dataset, objective) group.
    min_reports_per_objective: int = 2
    lower_quantile: float = 0.1
    upper_quantile: float = 0.9
    safety_margin: float = 0.02


def _manifest_notes(report: Mapping[str, Any]) -> Mapping[str, Any]:
    dataset_manifest = report.get("dataset_manifest")
    if isinstance(dataset_manifest, Mapping):
        notes = dataset_manifest.get("notes")
        if isinstance(notes, Mapping):
            return notes
    return {}


def _sample_payloads(run_dir: Path) -> list[Mapping[str, Any]]:
    payloads: list[Mapping[str, Any]] = []
    for candidate in sorted(run_dir.glob("*samples*.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, Mapping):
            payloads.append(payload)
    return payloads


def _extract_objective(report: Mapping[str, Any], run_dir: Path) -> str:
    objective = str(_manifest_notes(report).get("objective", "")).strip()
    if objective:
        return objective
    for payload in _sample_payloads(run_dir):
        meta = payload.get("metadata")
        if isinstance(meta, Mapping):
            objective = str(meta.get("objective", "")).strip()
            if objective:
                return objective
        samples = payload.get("samples")
        if isinstance(samples, Sequence):
            labels = {
                str(item.get("expected_label", "")).strip()
                for item in samples
                if isinstance(item, Mapping)
            }
            if labels and all(label.startswith("dominant_") for label in labels):
                return "neurotransmitter_dominance"
            if labels and labels.issubset({"high_connectivity", "baseline_connectivity"}):
                return "connectivity_tier"
    return UNRESOLVED


def _normalize_dataset(raw: Any) -> str:
    from flybrain_dataset_registry import DatasetRegistryError, normalize_symbol

    text = str(raw or "").strip().lower()
    text = _SAMPLE_DATASET_ALIASES.get(text, text)
    if not text or text == UNRESOLVED:
        return UNRESOLVED
    try:
        return normalize_symbol(text)
    except DatasetRegistryError:
        return UNRESOLVED


def _extract_dataset(report: Mapping[str, Any], run_dir: Path) -> str:
    """Dataset symbol from the run notes, the report, or the samples payload; else ``unknown``."""
    symbol = _normalize_dataset(_manifest_notes(report).get("dataset_symbol"))
    if symbol != UNRESOLVED:
        return symbol
    dataset = report.get("dataset")
    if isinstance(dataset, Mapping):
        symbol = _normalize_dataset(dataset.get("dataset_symbol"))
        if symbol != UNRESOLVED:
            return symbol
    for payload in _sample_payloads(run_dir):
        meta = payload.get("metadata")
        if not isinstance(meta, Mapping):
            continue
        symbol = _normalize_dataset(meta.get("dataset_symbol"))
        if symbol != UNRESOLVED:
            return symbol
        match = _SCHEMA_DATASET_RE.match(str(meta.get("schema_version") or ""))
        if match:
            symbol = _normalize_dataset(match.group(1))
            if symbol != UNRESOLVED:
                return symbol
    return UNRESOLVED


def _iter_run_reports(runs_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    reports: list[tuple[Path, dict[str, Any]]] = []
    for report_path in sorted(runs_root.rglob("p0-report.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(report, Mapping):
            reports.append((report_path.parent, dict(report)))
    return reports


def _is_calibration_eligible(report: Mapping[str, Any]) -> bool:
    gate = report.get("gate_report")
    shadow = report.get("shadow_report")
    if not isinstance(gate, Mapping) or not isinstance(shadow, Mapping):
        return False
    gate_metrics = gate.get("metrics")
    shadow_metrics = shadow.get("metrics")
    return isinstance(gate_metrics, Mapping) and isinstance(shadow_metrics, Mapping)


def _baseline_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    baseline = report.get("trivial_baseline")
    if not isinstance(baseline, Mapping) or not baseline:
        gate = report.get("gate_report")
        baseline = gate.get("trivial_baseline") if isinstance(gate, Mapping) else None
    if not isinstance(baseline, Mapping) or not baseline:
        return {"recorded": False, "pass": False}
    return {
        "recorded": True,
        "pass": bool(baseline.get("pass", False)),
        "model_heldout_accuracy": baseline.get("model_heldout_accuracy"),
        "best_trivial_rule": baseline.get("best_trivial_rule"),
        "best_trivial_accuracy": baseline.get("best_trivial_accuracy"),
        "required_accuracy": baseline.get("required_accuracy"),
        "heldout_count": baseline.get("heldout_count"),
    }


def calibrate_thresholds_from_runs(config: ReleasePrepConfig) -> dict[str, Any]:
    runs_root = Path(config.runs_root).resolve(strict=False)
    if not runs_root.exists():
        raise ValueError(f"runs_root not found: {runs_root}")
    if config.min_reports_per_objective < 1:
        raise ValueError("min_reports_per_objective must be >= 1")
    if not (0.0 < config.lower_quantile < config.upper_quantile < 1.0):
        raise ValueError("quantiles must satisfy 0 < lower_quantile < upper_quantile < 1")
    if config.safety_margin < 0.0:
        raise ValueError("safety_margin must be >= 0")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    unresolved: list[dict[str, str]] = []
    discovered = _iter_run_reports(runs_root)
    for run_dir, report in discovered:
        if not _is_calibration_eligible(report):
            continue
        objective = _extract_objective(report, run_dir)
        dataset = _extract_dataset(report, run_dir)
        if dataset == UNRESOLVED or objective in NON_CALIBRATABLE_OBJECTIVES:
            unresolved.append({"run_dir": str(run_dir), "dataset_symbol": dataset, "objective": objective})
            continue
        grouped.setdefault((dataset, objective), []).append(report)

    if not grouped:
        raise ValueError(
            "no eligible p0-report.json files with gate/shadow metrics and a resolved (dataset, objective)"
        )

    output_dir = Path(config.output_dir).resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_outputs: dict[str, Any] = {}
    skipped: dict[str, Any] = {}

    for dataset, objective in sorted(grouped):
        key = f"{dataset}/{objective}"
        rows = grouped[(dataset, objective)]
        evidence = [_baseline_evidence(row) for row in rows]
        if len(rows) < config.min_reports_per_objective:
            skipped[key] = {
                "dataset_symbol": dataset,
                "objective": objective,
                "reason": "insufficient_reports",
                "report_count": len(rows),
                "required_min_reports": config.min_reports_per_objective,
            }
            continue
        calibration = dict(
            fbthr.calibrate_thresholds_from_reports(
                rows,
                lower_quantile=config.lower_quantile,
                upper_quantile=config.upper_quantile,
                safety_margin=config.safety_margin,
            )
        )
        baseline_summary = {
            "runs": evidence,
            "recorded_count": sum(1 for row in evidence if row["recorded"]),
            "pass_count": sum(1 for row in evidence if row["pass"]),
            "all_passed": all(row["pass"] for row in evidence),
        }
        calibration["dataset_symbol"] = dataset
        calibration["objective"] = objective
        calibration["trivial_baseline"] = baseline_summary
        destination = output_dir / f"braincluster-thresholds-{dataset}-{objective}.json"
        destination.write_text(_stable_json(calibration) + "\n", encoding="utf-8")
        group_outputs[key] = {
            "dataset_symbol": dataset,
            "objective": objective,
            "report_count": len(rows),
            "threshold_file": str(destination),
            "calibration_id": calibration["calibration_id"],
            "input_fingerprint": calibration["input_fingerprint"],
            "trivial_baseline_all_passed": baseline_summary["all_passed"],
            "trivial_baseline_pass_count": baseline_summary["pass_count"],
        }

    if not group_outputs:
        raise ValueError("no (dataset, objective) group met minimum report count for calibration")

    return {
        "schema_version": RELEASE_PREP_SCHEMA_VERSION,
        "status": "ok",
        "runs_root": str(runs_root),
        "groups_calibrated": group_outputs,
        "groups_skipped": skipped,
        "unresolved_runs": unresolved,
        "input_report_count": len(discovered),
        "params": {
            "min_reports_per_objective": config.min_reports_per_objective,
            "lower_quantile": config.lower_quantile,
            "upper_quantile": config.upper_quantile,
            "safety_margin": config.safety_margin,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate (dataset, objective)-specific brain-cluster thresholds from run directories."
    )
    parser.add_argument("--runs-root", required=True, help="Root directory containing run subdirectories with p0-report.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for per-(dataset, objective) threshold bundles")
    parser.add_argument("--min-reports-per-objective", type=int, default=2,
                        help="Minimum eligible reports per (dataset, objective) group")
    parser.add_argument("--lower-quantile", type=float, default=0.1)
    parser.add_argument("--upper-quantile", type=float, default=0.9)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--report", help="Optional summary report output path")
    args = parser.parse_args(argv)

    config = ReleasePrepConfig(
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        min_reports_per_objective=args.min_reports_per_objective,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        safety_margin=args.safety_margin,
    )
    try:
        summary = calibrate_thresholds_from_runs(config)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        payload = {
            "schema_version": RELEASE_PREP_SCHEMA_VERSION,
            "status": "error",
            "pass": False,
            "error": str(exc),
        }
        text = _stable_json(payload) + "\n"
        if args.report:
            Path(args.report).write_text(text, encoding="utf-8")
        print(text, end="")
        return 2

    text = _stable_json(summary) + "\n"
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

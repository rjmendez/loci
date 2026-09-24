from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import flybrain_brain_cluster_thresholds as fbthr


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class ReleasePrepConfig:
    runs_root: str | Path
    output_dir: str | Path
    min_reports_per_objective: int = 2
    lower_quantile: float = 0.1
    upper_quantile: float = 0.9
    safety_margin: float = 0.02


def _extract_objective(report: Mapping[str, Any], run_dir: Path) -> str:
    dataset_manifest = report.get("dataset_manifest")
    if isinstance(dataset_manifest, Mapping):
        notes = dataset_manifest.get("notes")
        if isinstance(notes, Mapping):
            objective = str(notes.get("objective", "")).strip()
            if objective:
                return objective
    for candidate in sorted(run_dir.glob("*samples*.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
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
    return "unknown"


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

    grouped: dict[str, list[dict[str, Any]]] = {}
    discovered = _iter_run_reports(runs_root)
    for run_dir, report in discovered:
        if not _is_calibration_eligible(report):
            continue
        objective = _extract_objective(report, run_dir)
        grouped.setdefault(objective, []).append(report)

    if not grouped:
        raise ValueError("no eligible p0-report.json files found with gate/shadow metrics")

    output_dir = Path(config.output_dir).resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    objective_outputs: dict[str, Any] = {}
    skipped: dict[str, Any] = {}

    for objective in sorted(grouped):
        rows = grouped[objective]
        if len(rows) < config.min_reports_per_objective:
            skipped[objective] = {
                "reason": "insufficient_reports",
                "report_count": len(rows),
                "required_min_reports": config.min_reports_per_objective,
            }
            continue
        calibration = fbthr.calibrate_thresholds_from_reports(
            rows,
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
            safety_margin=config.safety_margin,
        )
        destination = output_dir / f"braincluster-thresholds-{objective}.json"
        destination.write_text(_stable_json(calibration) + "\n", encoding="utf-8")
        objective_outputs[objective] = {
            "report_count": len(rows),
            "threshold_file": str(destination),
            "calibration_id": calibration["calibration_id"],
            "input_fingerprint": calibration["input_fingerprint"],
        }

    if not objective_outputs:
        raise ValueError("no objectives met minimum report count for calibration")

    return {
        "schema_version": "braincluster-release-prep/v1",
        "status": "ok",
        "runs_root": str(runs_root),
        "objectives_calibrated": objective_outputs,
        "objectives_skipped": skipped,
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
        description="Calibrate objective-specific brain-cluster thresholds from run directories."
    )
    parser.add_argument("--runs-root", required=True, help="Root directory containing run subdirectories with p0-report.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for objective-specific threshold bundles")
    parser.add_argument("--min-reports-per-objective", type=int, default=2)
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
            "schema_version": "braincluster-release-prep/v1",
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

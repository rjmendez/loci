from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import quantiles
from typing import Any, Mapping, Sequence

import flybrain_brain_cluster as fbc
import flybrain_brain_cluster_training as fbct


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clamp(value: float, *, low: float, high: float) -> float:
    return max(low, min(high, value))


def _quantile(values: Sequence[float], *, p: float) -> float:
    if not values:
        raise ValueError("quantile values must be non-empty")
    if len(values) == 1:
        return float(values[0])
    if not (0.0 < p < 1.0):
        raise ValueError("quantile p must be in (0,1)")
    n = max(2, int(round(1.0 / min(p, 1.0 - p))))
    cuts = quantiles([float(v) for v in values], n=n, method="inclusive")
    index = int(round(p * n)) - 1
    index = max(0, min(len(cuts) - 1, index))
    return float(cuts[index])


def _load_json_value(source: str | Path | Mapping[str, Any] | Sequence[Any]) -> Any:
    if isinstance(source, (dict, list, tuple)):
        return source
    text = str(source).strip()
    if not text:
        raise ValueError("reports source is required")
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    path = Path(text)
    if not path.exists():
        raise ValueError(f"reports source not found: {source}")
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_report_rows(payload: Mapping[str, Any] | Sequence[Any]) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        for key in ("reports", "runs", "items"):
            if key in payload and isinstance(payload[key], Sequence):
                return [dict(item) for item in payload[key] if isinstance(item, Mapping)]
        if "gate_report" in payload and "shadow_report" in payload:
            return [dict(payload)]
        raise ValueError("report payload must contain reports[] or run-level gate_report/shadow_report")
    rows = [dict(item) for item in payload if isinstance(item, Mapping)]
    if not rows:
        raise ValueError("reports payload must include at least one report row")
    return rows


def _as_gate_report(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = row.get("gate_report")
    if isinstance(nested, Mapping):
        return nested
    return row


def _as_shadow_report(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = row.get("shadow_report")
    if isinstance(nested, Mapping):
        return nested
    return row


def _extract_calibration_metrics(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gate_accuracy: list[float] = []
    gate_abstain_rate: list[float] = []
    gate_calibration_error: list[float] = []
    gate_sample_count: list[int] = []

    shadow_decision_match: list[float] = []
    shadow_conf_drift: list[float] = []
    shadow_fail_closed_delta: list[float] = []
    shadow_entropy_drift: list[float] = []
    shadow_concentration: list[float] = []
    shadow_concentration_delta: list[float] = []

    for row in reports:
        gate = _as_gate_report(row)
        shadow = _as_shadow_report(row)

        gate_metrics = gate.get("metrics")
        if isinstance(gate_metrics, Mapping):
            for key, target in (
                ("accuracy", gate_accuracy),
                ("abstain_rate", gate_abstain_rate),
                ("mean_calibration_error", gate_calibration_error),
            ):
                value = gate_metrics.get(key)
                if isinstance(value, (int, float)):
                    target.append(float(value))
            count = gate_metrics.get("sample_count")
            if isinstance(count, int):
                gate_sample_count.append(int(count))

        shadow_metrics = shadow.get("metrics")
        if isinstance(shadow_metrics, Mapping):
            decision = shadow_metrics.get("decision_match_rate")
            if isinstance(decision, (int, float)):
                shadow_decision_match.append(float(decision))

            confidence = shadow_metrics.get("confidence_drift")
            if isinstance(confidence, Mapping):
                mean_abs = confidence.get("mean_abs_delta")
                if isinstance(mean_abs, (int, float)):
                    shadow_conf_drift.append(float(mean_abs))

            fail_closed = shadow_metrics.get("fail_closed_rate")
            if isinstance(fail_closed, Mapping):
                delta = fail_closed.get("delta")
                if isinstance(delta, (int, float)):
                    shadow_fail_closed_delta.append(float(delta))

            entropy = shadow_metrics.get("routing_entropy")
            if isinstance(entropy, Mapping):
                mean_abs_entropy = entropy.get("mean_abs_delta")
                if isinstance(mean_abs_entropy, (int, float)):
                    shadow_entropy_drift.append(float(mean_abs_entropy))

            collapse = shadow_metrics.get("expert_collapse")
            if isinstance(collapse, Mapping):
                candidate = collapse.get("candidate")
                if isinstance(candidate, Mapping):
                    concentration = candidate.get("concentration")
                    if isinstance(concentration, (int, float)):
                        shadow_concentration.append(float(concentration))
                concentration_delta = collapse.get("concentration_delta")
                if isinstance(concentration_delta, (int, float)):
                    shadow_concentration_delta.append(float(concentration_delta))

    if not gate_accuracy or not shadow_decision_match:
        raise ValueError("reports are missing required gate/shadow metrics for calibration")

    return {
        "gate_accuracy": tuple(gate_accuracy),
        "gate_abstain_rate": tuple(gate_abstain_rate),
        "gate_calibration_error": tuple(gate_calibration_error),
        "gate_sample_count": tuple(gate_sample_count),
        "shadow_decision_match": tuple(shadow_decision_match),
        "shadow_conf_drift": tuple(shadow_conf_drift),
        "shadow_fail_closed_delta": tuple(shadow_fail_closed_delta),
        "shadow_entropy_drift": tuple(shadow_entropy_drift),
        "shadow_concentration": tuple(shadow_concentration),
        "shadow_concentration_delta": tuple(shadow_concentration_delta),
    }


def calibrate_thresholds_from_reports(
    reports: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    *,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
    safety_margin: float = 0.02,
) -> dict[str, Any]:
    if not (0.0 < lower_quantile < upper_quantile < 1.0):
        raise ValueError("quantiles must satisfy 0 < lower_quantile < upper_quantile < 1")
    if safety_margin < 0.0:
        raise ValueError("safety_margin must be >= 0")

    payload = _load_json_value(reports)
    rows = _extract_report_rows(payload)
    metrics = _extract_calibration_metrics(rows)

    gate_thresholds = fbct.PromotionGateThresholds(
        min_accuracy=_clamp(
            _quantile(metrics["gate_accuracy"], p=lower_quantile) - safety_margin,
            low=0.0,
            high=1.0,
        ),
        max_abstain_rate=_clamp(
            _quantile(metrics["gate_abstain_rate"], p=upper_quantile) + safety_margin,
            low=0.0,
            high=1.0,
        ),
        max_confidence_calibration_error=_clamp(
            _quantile(metrics["gate_calibration_error"], p=upper_quantile) + safety_margin,
            low=0.0,
            high=1.0,
        ),
        min_samples=max(1, int(round(_quantile([float(v) for v in metrics["gate_sample_count"]], p=lower_quantile)))),
    )

    shadow_thresholds = fbc.ShadowReplayThresholds(
        min_decision_match_rate=_clamp(
            _quantile(metrics["shadow_decision_match"], p=lower_quantile) - safety_margin,
            low=0.0,
            high=1.0,
        ),
        max_mean_abs_confidence_drift=max(
            0.0,
            _quantile(metrics["shadow_conf_drift"], p=upper_quantile) + safety_margin,
        ),
        max_fail_closed_rate_delta=max(
            0.0,
            _quantile(metrics["shadow_fail_closed_delta"], p=upper_quantile) + safety_margin,
        ),
        max_mean_abs_routing_entropy_delta=max(
            0.0,
            _quantile(metrics["shadow_entropy_drift"], p=upper_quantile) + safety_margin,
        ),
        max_selected_expert_concentration=_clamp(
            _quantile(metrics["shadow_concentration"], p=upper_quantile) + safety_margin,
            low=0.0,
            high=1.0,
        ),
        max_selected_expert_concentration_delta=max(
            0.0,
            _quantile(metrics["shadow_concentration_delta"], p=upper_quantile) + safety_margin,
        ),
    )

    fingerprint_source = {
        "rows": rows,
        "params": {
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "safety_margin": safety_margin,
        },
    }
    calibration_fingerprint = _sha256_hex(_stable_json(fingerprint_source))

    return {
        "schema_version": "braincluster-threshold-calibration/v1",
        "calibration_id": f"braincluster-thresholds-{calibration_fingerprint[:16]}",
        "input_report_count": len(rows),
        "input_fingerprint": calibration_fingerprint,
        "params": {
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "safety_margin": safety_margin,
        },
        "thresholds": {
            "gate": gate_thresholds.as_dict(),
            "shadow_replay": shadow_thresholds.as_dict(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate deterministic brain-cluster gate/shadow thresholds from held-out run reports.")
    parser.add_argument("--reports", required=True, help="Path or inline JSON containing reports[] or run reports")
    parser.add_argument("--lower-quantile", type=float, default=0.1)
    parser.add_argument("--upper-quantile", type=float, default=0.9)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--output", help="Optional output JSON file path")
    args = parser.parse_args(argv)
    try:
        result = calibrate_thresholds_from_reports(
            args.reports,
            lower_quantile=args.lower_quantile,
            upper_quantile=args.upper_quantile,
            safety_margin=args.safety_margin,
        )
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        payload = {
            "schema_version": "braincluster-threshold-calibration/v1",
            "status": "error",
            "pass": False,
            "error": str(exc),
        }
        text = _stable_json(payload) + "\n"
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        print(text, end="")
        return 2

    text = _stable_json(result) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

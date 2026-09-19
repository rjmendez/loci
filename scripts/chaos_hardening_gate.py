#!/usr/bin/env python3
"""Executable hardening gates for chaos/adversarial validation artifacts.

Consumes optional inputs:
  - chaos event records (JSON list or JSONL)
  - adversarial harness report JSON

Outputs machine-readable JSON and exits non-zero on failed gates.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1", "ok", "pass", "passed", "success"}:
            return True
        if lowered in {"false", "f", "no", "n", "0", "fail", "failed", "error"}:
            return False
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _read_json_or_jsonl(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if stripped.startswith("["):
        return json.loads(raw)
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_chaos_rows(path: Path) -> list[dict[str, Any]]:
    parsed = _read_json_or_jsonl(path)
    if not isinstance(parsed, list):
        raise ValueError("chaos events must be a JSON array or JSONL records")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(parsed):
        if not isinstance(row, dict):
            raise ValueError(f"chaos event at index {idx} is not a JSON object")
        rows.append(row)
    return rows


def _extract_timeout(row: dict[str, Any]) -> bool | None:
    for key in ("timed_out", "timeout", "is_timeout"):
        val = _coerce_bool(row.get(key))
        if val is not None:
            return val
    for key in ("status", "outcome", "result"):
        value = row.get(key)
        if isinstance(value, str):
            if value.lower() == "timeout":
                return True
    return None


def _extract_retry_count(row: dict[str, Any]) -> int | None:
    for key in ("retry_count", "retries", "retry_attempts"):
        val = _coerce_int(row.get(key))
        if val is not None:
            return max(0, val)
    attempts = _coerce_int(row.get("attempts"))
    if attempts is not None:
        return max(0, attempts - 1)
    return None


def _extract_success(row: dict[str, Any]) -> bool | None:
    val = _coerce_bool(row.get("success"))
    if val is not None:
        return val
    for key in ("status", "outcome", "result"):
        value = row.get(key)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"ok", "success", "passed", "pass"}:
                return True
            if lowered in {"error", "failed", "fail"}:
                return False
    return None


def _extract_duplicate_effect(row: dict[str, Any]) -> bool | None:
    for key in ("duplicate_effect", "duplicate_write", "idempotency_violation"):
        val = _coerce_bool(row.get(key))
        if val is not None:
            return val
    return None


def _extract_provenance_complete(row: dict[str, Any]) -> bool | None:
    direct = _coerce_bool(row.get("provenance_complete"))
    if direct is not None:
        return direct
    tier = row.get("evidence_provenance_tier")
    if isinstance(tier, str):
        return bool(tier.strip())
    return None


@dataclass
class GateResult:
    name: str
    status: str
    observed: dict[str, Any]
    threshold: dict[str, Any]
    reason: str


def _ratio(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return num / den


def evaluate_chaos_gates(
    rows: list[dict[str, Any]],
    *,
    max_timeout_rate: float,
    min_retry_recovery_rate: float,
    max_duplicate_effect_rate: float,
    min_provenance_completeness: float,
) -> list[GateResult]:
    timeout_seen = timeout_true = 0
    retry_seen = retry_success = 0
    duplicate_seen = duplicate_true = 0
    provenance_seen = provenance_true = 0

    for row in rows:
        timeout = _extract_timeout(row)
        if timeout is not None:
            timeout_seen += 1
            timeout_true += int(timeout)

        retries = _extract_retry_count(row)
        success = _extract_success(row)
        if retries is not None and retries > 0:
            retry_seen += 1
            if success is True:
                retry_success += 1

        duplicate = _extract_duplicate_effect(row)
        if duplicate is not None:
            duplicate_seen += 1
            duplicate_true += int(duplicate)

        prov = _extract_provenance_complete(row)
        if prov is not None:
            provenance_seen += 1
            provenance_true += int(prov)

    out: list[GateResult] = []
    timeout_rate = _ratio(timeout_true, timeout_seen)
    if timeout_rate is None:
        out.append(GateResult(
            name="timeouts",
            status="skipped",
            observed={"records_seen": len(rows), "timeout_measurements": 0},
            threshold={"max_timeout_rate": max_timeout_rate},
            reason="No timeout fields found (timed_out/timeout/status=timeout).",
        ))
    else:
        ok = timeout_rate <= max_timeout_rate
        out.append(GateResult(
            name="timeouts",
            status="pass" if ok else "fail",
            observed={"timeouts": timeout_true, "measured": timeout_seen, "timeout_rate": timeout_rate},
            threshold={"max_timeout_rate": max_timeout_rate},
            reason="Timeout rate within threshold." if ok else "Timeout rate exceeded threshold.",
        ))

    retry_rate = _ratio(retry_success, retry_seen)
    if retry_rate is None:
        out.append(GateResult(
            name="retries",
            status="skipped",
            observed={"records_seen": len(rows), "retry_measurements": 0},
            threshold={"min_retry_recovery_rate": min_retry_recovery_rate},
            reason="No retry attempts were measurable (retry_count/retries/attempts).",
        ))
    else:
        ok = retry_rate >= min_retry_recovery_rate
        out.append(GateResult(
            name="retries",
            status="pass" if ok else "fail",
            observed={"retried_records": retry_seen, "recovered_after_retry": retry_success, "recovery_rate": retry_rate},
            threshold={"min_retry_recovery_rate": min_retry_recovery_rate},
            reason="Retry recovery rate within threshold." if ok else "Retry recovery rate below threshold.",
        ))

    duplicate_rate = _ratio(duplicate_true, duplicate_seen)
    if duplicate_rate is None:
        out.append(GateResult(
            name="duplicate_effects",
            status="skipped",
            observed={"records_seen": len(rows), "duplicate_measurements": 0},
            threshold={"max_duplicate_effect_rate": max_duplicate_effect_rate},
            reason="No duplicate-effect fields found (duplicate_effect/idempotency_violation).",
        ))
    else:
        ok = duplicate_rate <= max_duplicate_effect_rate
        out.append(GateResult(
            name="duplicate_effects",
            status="pass" if ok else "fail",
            observed={"duplicate_effects": duplicate_true, "measured": duplicate_seen, "duplicate_effect_rate": duplicate_rate},
            threshold={"max_duplicate_effect_rate": max_duplicate_effect_rate},
            reason="Duplicate effect rate within threshold." if ok else "Duplicate effect rate exceeded threshold.",
        ))

    prov_rate = _ratio(provenance_true, provenance_seen)
    if prov_rate is None:
        out.append(GateResult(
            name="provenance_completeness",
            status="skipped",
            observed={"records_seen": len(rows), "provenance_measurements": 0},
            threshold={"min_provenance_completeness": min_provenance_completeness},
            reason="No provenance fields found (provenance_complete/evidence_provenance_tier).",
        ))
    else:
        ok = prov_rate >= min_provenance_completeness
        out.append(GateResult(
            name="provenance_completeness",
            status="pass" if ok else "fail",
            observed={"complete": provenance_true, "measured": provenance_seen, "completeness_rate": prov_rate},
            threshold={"min_provenance_completeness": min_provenance_completeness},
            reason="Provenance completeness within threshold." if ok else "Provenance completeness below threshold.",
        ))
    return out


def evaluate_adversarial_gates(
    report: dict[str, Any],
    *,
    max_candidate_bypass: int,
) -> list[GateResult]:
    summary = report.get("summary") if isinstance(report, dict) else None
    findings = report.get("findings") if isinstance(report, dict) else None
    if not isinstance(summary, dict):
        return [GateResult(
            name="adversarial_candidate_bypass",
            status="skipped",
            observed={},
            threshold={"max_candidate_bypass": max_candidate_bypass},
            reason="Report has no summary object.",
        )]

    count = summary.get("candidate_bypass_count")
    parsed = _coerce_int(count)
    if parsed is None:
        counts = summary.get("classification_counts")
        if isinstance(counts, dict):
            parsed = _coerce_int(counts.get("candidate_bypass"))
    if parsed is None:
        parsed = 0
        if isinstance(findings, list):
            for item in findings:
                if isinstance(item, dict) and str(item.get("classification", "")).strip().lower() == "candidate_bypass":
                    parsed += 1

    ok = parsed <= max_candidate_bypass
    return [GateResult(
        name="adversarial_candidate_bypass",
        status="pass" if ok else "fail",
        observed={"candidate_bypass_count": parsed},
        threshold={"max_candidate_bypass": max_candidate_bypass},
        reason="Candidate bypass count within threshold." if ok else "Candidate bypass count exceeded threshold.",
    )]


def _result_payload(gates: list[GateResult], *, inputs: dict[str, Any], fail_on_skipped: bool) -> dict[str, Any]:
    failed = [g.name for g in gates if g.status == "fail"]
    skipped = [g.name for g in gates if g.status == "skipped"]
    ok = not failed and not (fail_on_skipped and skipped)
    return {
        "ok": ok,
        "generated_at": _utc_now(),
        "inputs": inputs,
        "summary": {
            "total_gates": len(gates),
            "passed": sum(1 for g in gates if g.status == "pass"),
            "failed": len(failed),
            "skipped": len(skipped),
            "failed_gates": failed,
            "skipped_gates": skipped,
        },
        "gates": [
            {
                "name": g.name,
                "status": g.status,
                "observed": g.observed,
                "threshold": g.threshold,
                "reason": g.reason,
            }
            for g in gates
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chaos-events", type=Path, help="JSON/JSONL event records from chaos runs")
    ap.add_argument("--adversarial-report", type=Path, help="JSON report from scripts/redteam/loci_adversarial_harness.py")
    ap.add_argument("--max-timeout-rate", type=float, default=0.05)
    ap.add_argument("--min-retry-recovery-rate", type=float, default=0.80)
    ap.add_argument("--max-duplicate-effect-rate", type=float, default=0.0)
    ap.add_argument("--min-provenance-completeness", type=float, default=0.99)
    ap.add_argument("--max-candidate-bypass", type=int, default=0)
    ap.add_argument("--fail-on-skipped", action="store_true", help="treat skipped gates as failures")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.chaos_events and not args.adversarial_report:
        print("{}", end="")
        print("\nchaos_hardening_gate: provide --chaos-events and/or --adversarial-report", file=sys.stderr)
        return 2

    gates: list[GateResult] = []
    inputs: dict[str, Any] = {
        "chaos_events": str(args.chaos_events) if args.chaos_events else None,
        "adversarial_report": str(args.adversarial_report) if args.adversarial_report else None,
    }

    if args.chaos_events:
        rows = _load_chaos_rows(args.chaos_events)
        gates.extend(evaluate_chaos_gates(
            rows,
            max_timeout_rate=args.max_timeout_rate,
            min_retry_recovery_rate=args.min_retry_recovery_rate,
            max_duplicate_effect_rate=args.max_duplicate_effect_rate,
            min_provenance_completeness=args.min_provenance_completeness,
        ))
        inputs["chaos_events_rows"] = len(rows)

    if args.adversarial_report:
        report = json.loads(args.adversarial_report.read_text(encoding="utf-8"))
        gates.extend(evaluate_adversarial_gates(
            report,
            max_candidate_bypass=args.max_candidate_bypass,
        ))

    payload = _result_payload(gates, inputs=inputs, fail_on_skipped=args.fail_on_skipped)
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

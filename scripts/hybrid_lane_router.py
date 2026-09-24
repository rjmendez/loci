from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_record_route_event():
    """Load mcp/route_audit.py by file path.

    Putting the repo root or mcp/ on sys.path from a script would let the repo's
    mcp/ dir compete with the installed `mcp` SDK package, and let mcp/*.py
    modules (server, verify, compact, ...) shadow unrelated imports.
    """
    try:
        path = Path(__file__).resolve().parent.parent / "mcp" / "route_audit.py"
        spec = importlib.util.spec_from_file_location("_loci_route_audit", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.record_route_event
    except Exception:
        return None


record_route_event = _load_record_route_event()


_VALID_PRIORITIES = {"P0", "P1", "P2"}
_VALID_TASK_CLASSES = {"integration", "tooling", "reasoning", "fanout", "speculative", "recovery"}
_VALID_EVIDENCE = {"none", "standard", "strict"}
_VALID_BROWNOUT = {"L0", "L1", "L2", "L3", "L4"}


@dataclass(frozen=True)
class RouteDecision:
    lane_id: str
    reason: str
    degraded: bool = False



def _record_decision(decision: "RouteDecision") -> None:
    if record_route_event is None:
        return
    try:
        record_route_event(
            tier="hybrid_lane",
            route=decision.lane_id,
            reason=decision.reason,
            degraded=decision.degraded,
            source="hybrid_lane_router",
        )
    except Exception:
        # Routing is a pure decision; auditing it must never change the answer.
        pass

def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "y"}:
            return True
        if norm in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"cannot parse bool from {value!r}")


def validate_dispatch_request(request: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "task_id": str,
        "session_id": str,
        "priority": str,
        "critical_path": (bool, str),
        "task_class": str,
        "idempotency_key": str,
        "max_latency_ms": int,
        "max_cost_usd": (int, float),
        "quality_floor": (int, float),
        "evidence_required": str,
    }

    for field, expected_type in required.items():
        if field not in request:
            errors.append(f"missing:{field}")
            continue
        value = request[field]
        if not isinstance(value, expected_type):
            errors.append(f"type:{field}")

    if "priority" in request and request["priority"] not in _VALID_PRIORITIES:
        errors.append("value:priority")
    if "task_class" in request and request["task_class"] not in _VALID_TASK_CLASSES:
        errors.append("value:task_class")
    if "evidence_required" in request and request["evidence_required"] not in _VALID_EVIDENCE:
        errors.append("value:evidence_required")
    if "max_latency_ms" in request and isinstance(request["max_latency_ms"], int) and request["max_latency_ms"] <= 0:
        errors.append("range:max_latency_ms")
    if "max_cost_usd" in request and isinstance(request["max_cost_usd"], (int, float)) and request["max_cost_usd"] <= 0:
        errors.append("range:max_cost_usd")
    if "quality_floor" in request and isinstance(request["quality_floor"], (int, float)):
        q = float(request["quality_floor"])
        if q < 0.0 or q > 1.0:
            errors.append("range:quality_floor")
    if "idempotency_key" in request and isinstance(request["idempotency_key"], str) and len(request["idempotency_key"]) < 16:
        errors.append("length:idempotency_key")
    return errors


def _brownout_degrades_to_local_only(brownout_level: str) -> bool:
    return brownout_level in {"L3", "L4"}


def choose_lane(
    request: dict[str, Any],
    *,
    upstream_confidence: float | None = None,
    brownout_level: str = "L0",
    local_healthy: bool = True,
    copilot_healthy: bool = True,
) -> RouteDecision:
    errors = validate_dispatch_request(request)
    if errors:
        raise ValueError(f"invalid dispatch request: {', '.join(errors)}")
    if brownout_level not in _VALID_BROWNOUT:
        raise ValueError(f"invalid brownout level: {brownout_level}")

    critical_path = _to_bool(request["critical_path"])
    task_class = request["task_class"]

    if critical_path:
        decision = RouteDecision("copilot-critical", "critical_path")
        _record_decision(decision)
        return decision

    if _brownout_degrades_to_local_only(brownout_level):
        if task_class in {"integration", "tooling"}:
            decision = RouteDecision("copilot-general", "brownout_keep_integration", degraded=True)
            _record_decision(decision)
            return decision
        if not local_healthy:
            decision = RouteDecision("local-recovery", "brownout_local_unhealthy", degraded=True)
            _record_decision(decision)
            return decision
        decision = RouteDecision("local-batch", "brownout_local_only", degraded=True)
        _record_decision(decision)
        return decision

    if task_class in {"integration", "tooling"}:
        if not copilot_healthy:
            decision = RouteDecision("local-recovery", "copilot_unhealthy")
            _record_decision(decision)
            return decision
        decision = RouteDecision("copilot-general", "integration_or_tooling")
        _record_decision(decision)
        return decision

    if task_class == "speculative":
        confidence = 0.0 if upstream_confidence is None else float(upstream_confidence)
        if confidence >= 0.8:
            if not local_healthy:
                decision = RouteDecision("local-recovery", "speculative_local_unhealthy", degraded=True)
                _record_decision(decision)
                return decision
            decision = RouteDecision("local-speculative", "speculation_ok")
            _record_decision(decision)
            return decision
        decision = RouteDecision("local-batch", "speculation_confidence_too_low")
        _record_decision(decision)
        return decision

    if task_class in {"reasoning", "fanout"}:
        if not local_healthy:
            decision = RouteDecision("local-recovery", "reasoning_local_unhealthy", degraded=True)
            _record_decision(decision)
            return decision
        decision = RouteDecision("local-batch", "reasoning_or_fanout")
        _record_decision(decision)
        return decision

    if task_class == "recovery":
        decision = RouteDecision("local-recovery", "recovery_class")
        _record_decision(decision)
        return decision

    decision = RouteDecision("local-recovery", "fallback_unclassified", degraded=True)
    _record_decision(decision)
    return decision


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("input JSON must be an object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic router for hybrid Copilot/local lanes")
    parser.add_argument("--request-json", required=True, help="Path to dispatch_request JSON")
    parser.add_argument("--confidence", type=float, default=None, help="Upstream confidence [0.0,1.0]")
    parser.add_argument("--brownout-level", default="L0", choices=sorted(_VALID_BROWNOUT))
    parser.add_argument("--local-unhealthy", action="store_true")
    parser.add_argument("--copilot-unhealthy", action="store_true")
    args = parser.parse_args(argv)

    request = _read_json(args.request_json)
    errors = validate_dispatch_request(request)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}))
        return 1

    decision = choose_lane(
        request,
        upstream_confidence=args.confidence,
        brownout_level=args.brownout_level,
        local_healthy=not args.local_unhealthy,
        copilot_healthy=not args.copilot_unhealthy,
    )
    print(json.dumps({"ok": True, "lane_id": decision.lane_id, "reason": decision.reason, "degraded": decision.degraded}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

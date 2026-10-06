from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from scripts.swarm_supervisor import plan_source_routing, supervise_and_correct, supervise_findings


class MissionCoordinator:
    """Fail-closed coordinator wrapper around the existing supervisor layer."""

    def __init__(
        self,
        task: Any,
        findings: Any,
        sources: Any | None = None,
        gen_fn: Callable[..., Any] | None = None,
        evidence_fn: Callable[..., Any] | None = None,
        worker_fn: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.task = self._stringify(task)
        self.findings = self._normalize_findings(findings)
        self.sources = self._normalize_sources(sources)
        self.gen_fn = gen_fn
        self.evidence_fn = evidence_fn
        self.worker_fn = worker_fn
        self.kwargs = kwargs

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        return str(value).strip()

    @staticmethod
    def _normalize_findings(findings: Any) -> list[dict[str, Any]]:
        if findings is None:
            return []
        if isinstance(findings, dict):
            findings = [findings]
        if not isinstance(findings, list):
            return []
        normalized: list[dict[str, Any]] = []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            normalized.append(dict(finding))
        return normalized

    @staticmethod
    def _normalize_sources(sources: Any) -> list[dict[str, Any]]:
        if sources is None:
            return []
        if not isinstance(sources, list):
            return []
        normalized: list[dict[str, Any]] = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            name = str(source.get("name") or source.get("source") or "").strip()
            if not name:
                continue
            normalized.append({
                "name": name,
                "capability": str(source.get("capability") or source.get("description") or "").strip(),
            })
        return normalized

    @staticmethod
    def _verdict_clean(verdict: dict[str, Any]) -> bool:
        return all(bool(verdict.get(key, True)) for key in ("on_task", "supported", "source_appropriate"))

    def _route_payload(self, plan: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(plan, dict):
            return {"task": self.task, "preferred_sources": [], "source_policy": ""}
        decision_tree = plan.get("decision_tree") or []
        assignments = plan.get("assignments") or []
        preferred_sources: list[str] = []
        for assignment in assignments:
            if not isinstance(assignment, dict):
                continue
            source = str(assignment.get("source") or "").strip()
            if source:
                preferred_sources.append(source)
        if not preferred_sources:
            for node in decision_tree:
                if not isinstance(node, dict):
                    continue
                preferred_sources.extend(str(src).strip() for src in (node.get("preferred_sources") or []) if str(src).strip())
        return {
            "task": str(plan.get("task") or self.task),
            "preferred_sources": preferred_sources,
            "source_policy": str(plan.get("source_policy") or ""),
            "decision_tree": decision_tree,
            "assignments": assignments,
        }

    def run(self) -> dict[str, Any]:
        task = self.task
        findings = self.findings
        sources = self.sources
        if not task:
            return {
                "status": "blocked",
                "reason": "missing task objective",
                "route": {"task": task, "preferred_sources": [], "source_policy": "", "decision_tree": [], "assignments": []},
                "verdicts": [],
                "replan": {"needed": True, "reason": "task objective is empty"},
            }
        if not callable(self.gen_fn):
            return {
                "status": "blocked",
                "reason": "generator function missing; fail-closed mission gate cannot validate outputs",
                "route": self._route_payload({"task": task, "decision_tree": [], "assignments": [], "source_policy": "No source can be routed without a generator."}),
                "verdicts": [],
                "replan": {"needed": True, "reason": "generator missing"},
            }

        try:
            plan = plan_source_routing(task, sources, self.gen_fn, **self.kwargs)
        except Exception as exc:  # pragma: no cover - defensive path
            return {
                "status": "blocked",
                "reason": f"routing plan failed: {exc}",
                "route": {"task": task, "preferred_sources": [], "source_policy": "", "decision_tree": [], "assignments": []},
                "verdicts": [],
                "replan": {"needed": True, "reason": f"routing failed: {exc}"},
            }

        route = self._route_payload(plan)
        try:
            result = supervise_findings(
                task,
                findings,
                plan,
                self.gen_fn,
                evidence_fn=self.evidence_fn,
                strict_tripwire=True,
            )
        except Exception as exc:
            return {
                "status": "blocked",
                "reason": f"mission gate could not evaluate findings: {exc}",
                "route": route,
                "verdicts": [],
                "replan": {"needed": True, "reason": f"verification failed: {exc}"},
            }

        verdicts = result.get("verdicts") or []
        flagged = [v for v in verdicts if not self._verdict_clean(v)]
        if result.get("tripwire_triggered") or flagged:
            replan = {"needed": True, "reason": "unsupported, off-task, or wrong-source finding detected"}
            if callable(self.worker_fn):
                try:
                    repair = supervise_and_correct(
                        task,
                        findings,
                        plan,
                        self.gen_fn,
                        self.worker_fn,
                        evidence_fn=self.evidence_fn,
                        max_rounds=1,
                        strict_tripwire=True,
                        **self.kwargs,
                    )
                except Exception as exc:
                    repair = {"converged": False, "error": str(exc), "verdicts": verdicts, "findings": findings}
                replan["attempted"] = True
                replan["converged"] = bool(repair.get("converged"))
                replan["plan"] = repair.get("audit_trail")[-1].get("replanned_routing_plan") if isinstance(repair.get("audit_trail"), list) and repair.get("audit_trail") else plan
                replan["result"] = repair
                if repair.get("converged"):
                    return {
                        "status": "accepted",
                        "reason": "worker repair brought outputs back within mission scope",
                        "route": route,
                        "verdicts": repair.get("verdicts") or verdicts,
                        "replan": replan,
                    }
                return {
                    "status": "rejected",
                    "reason": "mission gate failed: outputs drifted from task objective or lacked verified evidence",
                    "route": route,
                    "verdicts": repair.get("verdicts") or verdicts,
                    "replan": replan,
                }
            return {
                "status": "rejected",
                "reason": "mission gate failed: outputs drifted from task objective or lacked verified evidence",
                "route": route,
                "verdicts": verdicts,
                "replan": replan,
            }

        return {
            "status": "accepted",
            "reason": "all findings stayed on-task, were supported, and matched the source routing plan",
            "route": route,
            "verdicts": verdicts,
            "replan": {"needed": False, "reason": "no correction or route change required"},
        }


def mission_gate(
    task: Any,
    findings: Any,
    sources: Any | None = None,
    gen_fn: Callable[..., Any] | None = None,
    evidence_fn: Callable[..., Any] | None = None,
    worker_fn: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility wrapper for the fail-closed coordinator gate."""
    return MissionCoordinator(
        task,
        findings,
        sources,
        gen_fn=gen_fn,
        evidence_fn=evidence_fn,
        worker_fn=worker_fn,
        **kwargs,
    ).run()


def coordinate_research_task(
    task: Any,
    findings: Any,
    sources: Any | None = None,
    gen_fn: Callable[..., Any] | None = None,
    evidence_fn: Callable[..., Any] | None = None,
    worker_fn: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Coordinate a research/training task through routing and mission-gate checks."""
    return mission_gate(task, findings, sources, gen_fn=gen_fn, evidence_fn=evidence_fn, worker_fn=worker_fn, **kwargs)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed coordinator wrapper around swarm_supervisor routing and evidence checks.")
    parser.add_argument("--task", default="", help="Task objective to enforce.")
    parser.add_argument("--sources", default="[]", help="JSON list of source objects with name/capability.")
    parser.add_argument("--findings", default="[]", help="JSON list of finding dicts to evaluate.")
    parser.add_argument("--payload", default="", help="Optional JSON payload containing task, sources, findings, and optional gen_fn metadata.")
    args = parser.parse_args(argv)

    payload = {}
    if args.payload:
        try:
            payload = json.loads(args.payload)
        except Exception:
            payload = {}
    if not payload:
        payload = {
            "task": args.task,
            "sources": json.loads(args.sources) if args.sources else [],
            "findings": json.loads(args.findings) if args.findings else [],
        }

    task = payload.get("task")
    sources = payload.get("sources")
    findings = payload.get("findings")

    def gen_fn(**_kwargs: Any) -> Any:
        return json.dumps({
            "task": str(task),
            "fail_open": False,
            "decision_tree": [{
                "sub_need": "mission validation",
                "preferred_sources": [str(item.get("name") or "").strip() for item in (sources or []) if str(item.get("name") or "").strip()],
                "fallback_sources": [],
                "rationale": "fallback plan for CLI validation",
            }],
            "assignments": [{"sub_need": "mission validation", "source": str(item.get("name") or "").strip(), "rationale": "source match for evaluation"} for item in (sources or []) if str(item.get("name") or "").strip()],
            "source_policy": "Use the task's routed sources; if the source is wrong, reject the mission.",
        })

    result = mission_gate(task, findings, sources, gen_fn=gen_fn)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"accepted", "rejected"} else 1


if __name__ == "__main__":
    raise SystemExit(_main())

"""Mission-level coordinator: a fail-closed gate above the swarm supervisor.

The supervisor underneath is fail-open on purpose: when its generator fails or answers with something it cannot
parse, it returns a default routing plan and neutral all-true verdicts and sets ``fail_open``. That is right for a
helper and wrong for a gate, so this wrapper treats every ``fail_open`` as ``blocked`` and never as ``accepted``.

``status`` is one of:

* ``accepted``: the findings were verified on-task, supported and from the right sources (possibly after one worker repair);
* ``rejected``: they were verified and failed;
* ``blocked``: the gate could not decide (no objective, no generator, no findings, malformed findings, a degraded
  plan or verification, an exception). Blocked is never a pass.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from scripts.swarm_supervisor import plan_source_routing, supervise_and_correct, supervise_findings

EXIT_ACCEPTED, EXIT_REJECTED, EXIT_BLOCKED = 0, 1, 2


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
        self.findings, self.dropped_findings = self._normalize_findings(findings)
        self.sources = self._normalize_sources(sources)
        self.gen_fn = gen_fn
        self.evidence_fn = evidence_fn
        self.worker_fn = worker_fn
        self.kwargs = kwargs

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _normalize_findings(findings: Any) -> tuple[list[dict[str, Any]], int]:
        """``(findings, dropped)``: a single dict is one finding; anything that is not a dict is dropped and counted,
        because a gate that silently discards its input would accept an empty list."""
        if findings is None:
            return [], 0
        if isinstance(findings, dict):
            findings = [findings]
        if not isinstance(findings, list):
            return [], 1
        kept = [dict(f) for f in findings if isinstance(f, dict)]
        return kept, len(findings) - len(kept)

    @staticmethod
    def _normalize_sources(sources: Any) -> list[dict[str, Any]]:
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
            return {"task": self.task, "preferred_sources": [], "source_policy": "", "decision_tree": [], "assignments": []}
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

    def _blocked(self, reason: str, *, route: dict[str, Any] | None = None, verdicts: list | None = None) -> dict[str, Any]:
        return {
            "status": "blocked",
            "reason": reason,
            "route": route if route is not None else self._route_payload(None),
            "verdicts": verdicts or [],
            "replan": {"needed": True, "reason": reason},
        }

    @staticmethod
    def _replanned_routing_plan(repair: dict[str, Any], fallback: Any) -> Any:
        trail = repair.get("audit_trail")
        if isinstance(trail, list) and trail and isinstance(trail[-1], dict):
            return trail[-1].get("replanned_routing_plan", fallback)
        return fallback

    def run(self) -> dict[str, Any]:
        task, findings = self.task, self.findings
        if not task:
            return self._blocked("missing task objective")
        if not callable(self.gen_fn):
            return self._blocked("generator function missing; fail-closed mission gate cannot validate outputs")
        if self.dropped_findings:
            return self._blocked(f"malformed findings: {self.dropped_findings} item(s) are not objects, so the gate cannot evaluate them")
        if not findings:
            return self._blocked("no findings to evaluate")

        try:
            plan = plan_source_routing(task, self.sources, self.gen_fn, **self.kwargs)
        except Exception as exc:
            return self._blocked(f"routing plan failed: {exc}")
        route = self._route_payload(plan)
        if not isinstance(plan, dict) or plan.get("fail_open"):
            return self._blocked("routing plan degraded (supervisor fail-open): source routing cannot be enforced", route=route)

        try:
            result = supervise_findings(task, findings, plan, self.gen_fn, evidence_fn=self.evidence_fn, strict_tripwire=True)
        except Exception as exc:
            return self._blocked(f"mission gate could not evaluate findings: {exc}", route=route)
        if not isinstance(result, dict) or result.get("fail_open"):
            return self._blocked("verification degraded (supervisor fail-open): findings could not be verified", route=route)

        verdicts = result.get("verdicts") or []
        inconclusive = sum(1 for v in verdicts if v.get("evidence_inconclusive"))
        if inconclusive:
            return self._blocked(f"evidence check inconclusive for {inconclusive} finding(s): the claim was not verified",
                                 route=route, verdicts=verdicts)
        flagged = [v for v in verdicts if not self._verdict_clean(v)]        # a tripped wire is an unsupported verdict too
        if not flagged:
            return {
                "status": "accepted",
                "reason": "all findings stayed on-task, were supported, and matched the source routing plan",
                "route": route,
                "verdicts": verdicts,
                "replan": {"needed": False, "reason": "no correction or route change required"},
            }

        replan: dict[str, Any] = {"needed": True, "reason": "unsupported, off-task, or wrong-source finding detected"}
        rejected = {
            "status": "rejected",
            "reason": "mission gate failed: outputs drifted from task objective or lacked verified evidence",
            "route": route,
            "verdicts": verdicts,
            "replan": replan,
        }
        if not callable(self.worker_fn):
            return rejected
        try:
            repair = supervise_and_correct(
                task, findings, plan, self.gen_fn, self.worker_fn,
                evidence_fn=self.evidence_fn, max_rounds=1, strict_tripwire=True, **self.kwargs,
            )
        except Exception as exc:
            repair = {"converged": False, "error": str(exc), "verdicts": verdicts, "findings": findings}
        # A repair that degraded to fail-open has verified nothing: it cannot turn a rejection into an acceptance.
        # supervise_and_correct reviews from scratch before it repairs anything, so "converged" after zero rounds
        # means the model flagged the findings and then, asked again, did not: nothing was fixed, only its opinion
        # changed, and a gate does not accept on that.
        converged = bool(repair.get("converged")) and not repair.get("fail_open")
        if converged and int(repair.get("rounds") or 0) < 1:
            converged = False
            replan["reason"] = "reviews disagreed (flagged, then clean on re-review with nothing repaired): not trusted"
        replan.update(attempted=True, converged=converged, plan=self._replanned_routing_plan(repair, plan), result=repair)
        rejected["verdicts"] = repair.get("verdicts") or verdicts
        if converged:
            return {
                "status": "accepted",
                "reason": "worker repair brought outputs back within mission scope",
                "route": route,
                "verdicts": rejected["verdicts"],
                "replan": replan,
            }
        return rejected


def mission_gate(
    task: Any,
    findings: Any,
    sources: Any | None = None,
    gen_fn: Callable[..., Any] | None = None,
    evidence_fn: Callable[..., Any] | None = None,
    worker_fn: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """The fail-closed coordinator gate."""
    return MissionCoordinator(
        task, findings, sources, gen_fn=gen_fn, evidence_fn=evidence_fn, worker_fn=worker_fn, **kwargs,
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


def _default_gen_fn() -> Callable[..., Any] | None:
    """The local model through ``llm_local.generate`` (the supervisor calls it with ``prompt=``, ``fmt=`` ...), or
    None when it cannot be imported, in which case the gate blocks rather than inventing a plan."""
    try:
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp"))
        from llm_local import generate
        return generate
    except Exception:
        return None


def _main(argv: list[str] | None = None, gen_fn: Callable[..., Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed coordinator gate around swarm_supervisor routing and evidence checks. "
                                                 "Exit 0 accepted, 1 rejected, 2 blocked or bad input.")
    parser.add_argument("--task", default="", help="Task objective to enforce.")
    parser.add_argument("--sources", default="[]", help="JSON list of source objects with name/capability.")
    parser.add_argument("--findings", default="[]", help="JSON list of finding dicts to evaluate.")
    parser.add_argument("--payload", default="", help="JSON object with task, sources and findings (overrides the three options).")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.payload) if args.payload else {
            "task": args.task, "sources": json.loads(args.sources or "[]"), "findings": json.loads(args.findings or "[]")}
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
    except ValueError as exc:                        # json.JSONDecodeError is a ValueError
        print(json.dumps({"status": "blocked", "reason": f"bad input: {exc}"}, sort_keys=True))
        return EXIT_BLOCKED
    result = mission_gate(payload.get("task"), payload.get("findings"), payload.get("sources"),
                          gen_fn=gen_fn if gen_fn is not None else _default_gen_fn())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return {"accepted": EXIT_ACCEPTED, "rejected": EXIT_REJECTED}.get(result.get("status"), EXIT_BLOCKED)


if __name__ == "__main__":
    raise SystemExit(_main())

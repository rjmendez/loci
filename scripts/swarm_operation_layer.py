"""Grounded runtime contracts for the swarm operation layer.

This module is intentionally lightweight: it codifies the observable contracts already
present in ``scripts/swarm_supervisor.py`` and ``scripts/swarm_escalate.py`` so the
runtime can expose a single, documented layer for supervision, agent composition,
reusable prompt scaffolds, and observability.

The structures below are documentation-shaped data, not a second execution engine.
They are intentionally aligned to the existing policy and output schema in the codebase.
"""

from __future__ import annotations

from typing import Any, Dict

SUPERVISOR_ROLE_SPEC: Dict[str, Any] = {
    "purpose": "Coordinate decompose -> cheap-answer -> triage -> escalate -> synthesize while preserving provenance and fail-open behavior.",
    "owned_responsibilities": [
        "constrain task scope and keep the root question visible",
        "build the subtask tree and map subtasks to preferred evidence sources",
        "apply budget checks (calls, tokens, cost) through SupervisorBudget",
        "gate findings by confidence, duplicate similarity, contradiction and source fit",
        "trigger re-plan or worker correction only on flagged findings",
        "preserve provenance by tracking source, evidence, and rationale in each finding",
        "fail open: keep a degraded but machine-readable result if a stage breaks",
    ],
    "runtime_hooks": {
        "source_routing": "scripts.swarm_supervisor.plan_source_routing",
        "budget": "scripts.swarm_supervisor.SupervisorBudget",
        "findings_review": "scripts.swarm_supervisor.supervise_findings",
        "repair_loop": "scripts.swarm_supervisor.supervise_and_correct",
        "swarm_pipeline": "scripts.swarm_escalate.SwarmConfig",
        "mcp_wrapper": "mcp.llm_tools.swarm_reason",
    },
    "decision_policy": {
        "similar_subtasks_threshold": 0.50,
        "duplicate_like_answer_threshold": 0.82,
        "default_confidence_trigger": "low",
        "escalation_is_fail_open": True,
        "replan_on_stall": True,
    },
}

AGENT_ENSEMBLE_SPEC: Dict[str, Any] = {
    "composition": [
        {
            "role": "supervisor",
            "responsibility": "owns the end-to-end plan, evidence routing, and budget guardrails",
            "runtime_entry": "scripts.swarm_supervisor.supervise_and_correct",
        },
        {
            "role": "decomposer",
            "responsibility": "turns a topic into atomic subtasks and source-fit decisions",
            "runtime_entry": "scripts.swarm_escalate.decompose_subtasks",
        },
        {
            "role": "cheap_worker",
            "responsibility": "answers the majority of subtasks on the cheap tier with bounded token use",
            "runtime_entry": "scripts.swarm_escalate.answer_subtasks",
        },
        {
            "role": "triage_critic",
            "responsibility": "flags unsupported, wrong-source, or low-confidence findings",
            "runtime_entry": "scripts.swarm_supervisor.supervise_findings",
        },
        {
            "role": "escalator",
            "responsibility": "re-answers only the flagged slice under stronger model settings",
            "runtime_entry": "scripts.swarm_escalate.escalate_findings",
        },
        {
            "role": "synthesizer",
            "responsibility": "merges validated evidence into a single machine-readable result",
            "runtime_entry": "scripts.swarm_escalate.synthesize_swarm",
        },
    ],
    "interaction_rules": [
        "Supervisor decides which findings need repair and which can pass through.",
        "Only flagged findings are sent back to the worker for redispatching.",
        "Escalation is selective; it never converts a whole run into a full retry.",
        "The synthesize stage consumes the validated lineage and emits final structured JSON.",
    ],
}

PROMPT_TEMPLATES: Dict[str, str] = {
    "decompose": """You are the swarm decomposer.\nTask: {topic}\nGoal: break the task into atomic subtasks that can be answered independently.\nConstraints:\n- Use the current reasoning policy: fail-open, bounded fanout, no hidden assumptions.\n- Prefer evidence-fit over broad-but-weak search.\n- Return structured JSON with task, fail_open, decision_tree, assignments, and source_policy.\n- Preserve provenance with each subtask's preferred source and rationale.\n""",
    "triage": """You are the swarm triage critic.\nReview the findings against the plan and the current task.\nTask: {topic}\nPlan: {plan_json}\nFindings:\n{findings_json}\nReturn JSON with verdicts for every finding: finding_index, on_task, supported, source_appropriate, guidance, rationale.\nDo not invent evidence; only judge whether the finding fits the task and the routing plan.\n""",
    "escalate": """You are the escalation specialist.\nReview only the flagged findings and produce a stronger answer while staying within the current budget.\nTask: {topic}\nFlagged findings: {flagged_findings_json}\nCurrent rationale: {rationale_json}\nRequirements:\n- keep the answer grounded in the evidence actually provided\n- explain why the original answer failed or was mis-routed\n- if a repair is impossible, return a conservative answer and leave the finding degraded rather than fabricating certainty\n""",
    "synthesize": """You are the swarm synthesizer.\nCombine validated findings into one structured result for downstream use.\nTask: {topic}\nValidated findings: {findings_json}\nConstraints:\n- summarize only supported, relevant, and provenance-preserving evidence\n- cite confidence, model, and tier_reached for each retained finding\n- mark degraded runs explicitly rather than silently smoothing over failures\n- output JSON only\n""",
}

OBSERVABILITY_DASHBOARD: Dict[str, Any] = {
    "core_views": [
        {
            "name": "run_summary",
            "purpose": "topic, counts, confidence distribution, degraded flag, seed metadata",
            "source": "scripts.swarm_escalate.SwarmConfig + final swarm payload",
        },
        {
            "name": "triage_watch",
            "purpose": "flagged_indices, escalation_reasons, similar_pairs, duplicate_like findings",
            "source": "scripts.swarm_supervisor.supervise_findings",
        },
        {
            "name": "budget_health",
            "purpose": "calls, prompt_tokens, completion_tokens, total_tokens, cost_usd, max_* limits",
            "source": "scripts.swarm_supervisor.SupervisorBudget.snapshot",
        },
        {
            "name": "escalation_flow",
            "purpose": "escalated_count, escalation_rate, model choices, tier_reached transitions",
            "source": "scripts.swarm_escalate + mcp.llm_tools._degraded_swarm_result",
        },
    ],
    "metrics": {
        "fanout_count": "number of subtasks produced by decomposition",
        "escalated_count": "number of findings that needed escalation",
        "escalation_rate": "escalated_count / fanout_count, or a per-run ratio",
        "confidence_distribution": "low/medium/high counts across findings",
        "degraded": "true when a stage must fail open or a parser drops a result",
        "parse_failures": "count of JSON parse failures or malformed model output",
        "latency_ms": "elapsed run time per stage",
        "budget_cost_usd": "SupervisorBudget.cost_usd with current token accounting",
        "seed_count": "number of independent swarm seeds merged for the output",
        "lineage_length": "how many provenance-bearing records are retained in the run",
    },
    "artifact_schema": {
        "summary": "top-level run summary with counts and confidence",
        "triage": "flagged indices, escalation reasons, similar-pair detection",
        "tiers": "cheap + escalate stage metrics and model metadata",
        "budget": "SupervisorBudget snapshot for hard cap tracking",
        "lineage": "ordered evidence trail for each finding and its provenance",
    },
}


def runtime_contract() -> Dict[str, Any]:
    """Return the coupled operation-layer contract for runtime docs and dashboards."""
    return {
        "supervisor": SUPERVISOR_ROLE_SPEC,
        "ensemble": AGENT_ENSEMBLE_SPEC,
        "prompt_templates": PROMPT_TEMPLATES,
        "observability": OBSERVABILITY_DASHBOARD,
    }

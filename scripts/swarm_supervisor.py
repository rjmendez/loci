#!/usr/bin/env python3
"""Advisory supervisor for decision-tree source routing and finding review.

This module is deliberately standalone so it can be imported by swarm/deep-think
pipelines later without coupling to their runtimes. The supervisor is fail-open:
model, timeout, JSON-shape, or worker failures return conservative advisory
metadata rather than blocking work.
"""
from __future__ import annotations

import concurrent.futures
import json
from typing import Any, Callable


GenFn = Callable[..., Any]
EvidenceFn = Callable[..., Any]
WorkerFn = Callable[..., Any]
DEFAULT_MAX_TREE_NODES = 25


def _json_default(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


def _source_names(available_sources: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for source in available_sources or []:
        if isinstance(source, dict):
            name = str(source.get("name") or "").strip()
            if name:
                names.append(name)
    return names


def _safe_sources(available_sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for source in available_sources or []:
        if not isinstance(source, dict):
            continue
        name = str(source.get("name") or "").strip()
        if not name:
            continue
        out.append({
            "name": name,
            "capability": str(source.get("capability") or source.get("description") or "").strip(),
        })
    return out


def _call_generate(gen_fn: GenFn, prompt: str, *, max_tokens: int = 1200, temperature: float = 0.1) -> Any:
    attempts = (
        {"prompt": prompt, "fmt": "json", "max_tokens": max_tokens, "temperature": temperature},
        {"prompt": prompt, "format": "json", "max_tokens": max_tokens, "temperature": temperature},
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        {"prompt": prompt},
    )
    for kwargs in attempts:
        try:
            return gen_fn(**kwargs)
        except TypeError:
            continue
        except Exception as exc:
            return {"ok": False, "text": "", "error": str(exc)}
    return {"ok": False, "text": "", "error": "generate signature mismatch"}


def _response_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for key in ("text", "response", "content", "output"):
            value = raw.get(key)
            if isinstance(value, str):
                return value
        message = raw.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not starts:
        raise ValueError("no JSON object found")
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escape = False
    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack:
                raise ValueError("unbalanced JSON")
            open_char = stack.pop()
            if (open_char, char) not in (("{", "}"), ("[", "]")):
                raise ValueError("mismatched JSON delimiters")
            if not stack:
                return json.loads(text[start:idx + 1])
    raise ValueError("unterminated JSON")


def _default_routing_plan(task: str, available_sources: list[dict[str, Any]], why: str = "fail-open default") -> dict[str, Any]:
    names = _source_names(available_sources)
    return {
        "task": task,
        "fail_open": True,
        "decision_tree": [{
            "sub_need": "unclassified evidence need",
            "preferred_sources": names,
            "fallback_sources": names,
            "rationale": f"{why}; use every available source and do not enforce a routing preference.",
        }],
        "assignments": [
            {"sub_need": "unclassified evidence need", "source": name, "rationale": f"{why}; retained as an available source."}
            for name in names
        ],
        "source_policy": "No source was preferred because supervisor planning degraded; workers should cite evidence and cross-check where possible.",
    }


def _normalize_node(item: dict[str, Any], *, allow_nested: bool = False) -> dict[str, Any] | None:
    sub_need = str(item.get("sub_need") or item.get("need") or "").strip()
    preferred = item.get("preferred_sources") or item.get("sources") or item.get("source") or []
    if isinstance(preferred, str):
        preferred = [preferred]
    if not sub_need or not isinstance(preferred, list):
        return None
    node: dict[str, Any] = {
        "sub_need": sub_need,
        "preferred_sources": [str(src) for src in preferred if str(src).strip()],
        "fallback_sources": [str(src) for src in (item.get("fallback_sources") or []) if str(src).strip()],
        "rationale": str(item.get("rationale") or "").strip(),
    }
    if allow_nested:
        fallback_condition = str(item.get("fallback_condition") or "").strip()
        if fallback_condition:
            node["fallback_condition"] = fallback_condition
        children = item.get("children") or []
        if isinstance(children, list):
            node["children"] = [
                child for child in (_normalize_node(raw, allow_nested=True) for raw in children if isinstance(raw, dict))
                if child is not None
            ]
    return node


def _normalize_routing_plan(
    obj: Any,
    task: str,
    available_sources: list[dict[str, Any]],
    *,
    allow_nested: bool = False,
) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("routing response is not an object")
    plan = dict(obj)
    plan.setdefault("task", task)
    plan["fail_open"] = bool(plan.get("fail_open", False))
    if not isinstance(plan.get("decision_tree"), list) or not plan["decision_tree"]:
        raise ValueError("routing plan missing decision_tree")
    normalized_tree: list[dict[str, Any]] = []
    for item in plan["decision_tree"]:
        if not isinstance(item, dict):
            continue
        node = _normalize_node(item, allow_nested=allow_nested)
        if node is not None:
            normalized_tree.append(node)
    if not normalized_tree:
        raise ValueError("routing plan has no usable decision_tree entries")
    plan["decision_tree"] = normalized_tree
    assignments = plan.get("assignments")
    if not isinstance(assignments, list):
        assignments = []
        for item in normalized_tree:
            for source in item["preferred_sources"]:
                assignments.append({"sub_need": item["sub_need"], "source": source, "rationale": item["rationale"]})
    plan["assignments"] = [item for item in assignments if isinstance(item, dict)]
    return plan


def _count_nodes(nodes: list[dict[str, Any]]) -> int:
    total = 0
    stack = list(nodes or [])
    while stack:
        node = stack.pop()
        total += 1
        children = node.get("children") or []
        if isinstance(children, list):
            stack.extend(child for child in children if isinstance(child, dict))
    return total


def _expand_node(
    node: dict[str, Any],
    *,
    task: str,
    safe_sources: list[dict[str, str]],
    gen_fn: GenFn,
    depth_remaining: int,
    node_budget: int,
) -> tuple[dict[str, Any], int]:
    if depth_remaining <= 1 or node_budget <= 1:
        return dict(node), 1
    prompt = f"""You are expanding one source-routing decision-tree node.
Return ONLY valid JSON with this schema:
{{
  "sub_need": string,
  "preferred_sources": [string],
  "fallback_condition": string,
  "fallback_sources": [string],
  "rationale": string,
  "children": [{{"sub_need": string, "preferred_sources": [string], "fallback_condition": string, "fallback_sources": [string], "rationale": string}}]
}}

Rules:
- Keep the same core sub_need unless a clearer wording is needed.
- Add children only for genuinely separate evidence checks under this node.
- Add fallback_condition as a plain-language trigger for trying fallback_sources.
- Do not invent sources outside Available sources.
- Return at most {max(0, node_budget - 1)} child nodes in this response.

Task: {task}
Available sources: {_json_default(safe_sources)}
Node to expand: {_json_default(node)}
"""
    raw = _call_generate(gen_fn, prompt, max_tokens=1200, temperature=0.1)
    expanded = _normalize_node(_extract_json(_response_text(raw)), allow_nested=True)
    if expanded is None:
        raise ValueError("expanded node was unusable")
    if _count_nodes([expanded]) > node_budget:
        raise ValueError("expanded routing tree exceeds node cap")
    children = expanded.get("children") or []
    if children and depth_remaining > 2:
        per_child_budget = max(1, (node_budget - 1) // max(1, len(children)))
        deeper_children: list[dict[str, Any] | None] = [None] * len(children)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(children))) as ex:
            futures = {
                ex.submit(
                    _expand_node,
                    child,
                    task=task,
                    safe_sources=safe_sources,
                    gen_fn=gen_fn,
                    depth_remaining=depth_remaining - 1,
                    node_budget=per_child_budget,
                ): idx
                for idx, child in enumerate(children)
            }
            for fut in concurrent.futures.as_completed(futures):
                child, _used = fut.result()
                deeper_children[futures[fut]] = child
        expanded["children"] = [child for child in deeper_children if child is not None]
    used = _count_nodes([expanded])
    if used > node_budget:
        raise ValueError("expanded routing tree exceeds node cap")
    return expanded, used


def _neutral_verdicts(findings: list[dict[str, Any]], why: str = "fail-open default") -> dict[str, Any]:
    return {
        "fail_open": True,
        "verdicts": [{
            "finding_index": idx,
            "on_task": True,
            "supported": True,
            "source_appropriate": True,
            "guidance": "",
            "rationale": why,
        } for idx, _finding in enumerate(findings or [])],
    }


def _normalize_supervision(obj: Any, findings: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("supervision response is not an object")
    verdicts = obj.get("verdicts")
    if not isinstance(verdicts, list):
        raise ValueError("supervision response missing verdicts")
    by_index: dict[int, dict[str, Any]] = {}
    for item in verdicts:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("finding_index"))
        except Exception:
            continue
        if 0 <= index < len(findings):
            needs_guidance = not all(bool(item.get(key, True)) for key in ("on_task", "supported", "source_appropriate"))
            by_index[index] = {
                "finding_index": index,
                "on_task": bool(item.get("on_task", True)),
                "supported": bool(item.get("supported", True)),
                "source_appropriate": bool(item.get("source_appropriate", True)),
                "guidance": str(item.get("guidance") or ("Review and redirect this worker." if needs_guidance else "")).strip(),
                "rationale": str(item.get("rationale") or "").strip(),
            }
    if len(by_index) != len(findings):
        for idx in range(len(findings)):
            by_index.setdefault(idx, {
                "finding_index": idx,
                "on_task": True,
                "supported": True,
                "source_appropriate": True,
                "guidance": "",
                "rationale": "missing model verdict; neutral fail-open fill",
            })
    return {"fail_open": bool(obj.get("fail_open", False)), "verdicts": [by_index[idx] for idx in range(len(findings))]}


def plan_source_routing(
    task: str,
    available_sources: list[dict[str, Any]],
    gen_fn: GenFn,
    max_depth: int = 1,
    *,
    max_nodes: int = DEFAULT_MAX_TREE_NODES,
) -> dict[str, Any]:
    """Return a structured source-routing decision tree for a task.

    ``max_depth=1`` preserves the original flat, one-call behavior. Larger depths
    expand each flat node through separate concurrent model calls, and fail open
    back to that flat plan if recursive expansion degrades.
    """
    safe_sources = _safe_sources(available_sources)
    prompt = f"""You are a supervisor/orchestrator planning source use before workers start.
Return ONLY valid JSON with this schema:
{{
  "task": string,
  "fail_open": false,
  "decision_tree": [{{"sub_need": string, "preferred_sources": [string], "fallback_sources": [string], "rationale": string}}],
  "assignments": [{{"sub_need": string, "source": string, "rationale": string}}],
  "source_policy": string
}}

Rules:
- Match each evidence need to the source whose capability best fits it.
- If the task asks for local/GPS/photo-verified observations, prioritize a source with GPS-filterable photo-verified observations.
- Use encyclopedic sources for background/taxonomy only, not proof of local observation.
- Use general web search for recent/local mentions only when structured local evidence is unavailable or as a supplement.

Task: {task}
Available sources: {_json_default(safe_sources)}
"""
    raw = _call_generate(gen_fn, prompt, max_tokens=1300, temperature=0.1)
    try:
        obj = _extract_json(_response_text(raw))
        flat_plan = _normalize_routing_plan(obj, task, safe_sources)
        depth = max(1, int(max_depth))
        if depth <= 1:
            return flat_plan
        try:
            budget = max(1, int(max_nodes))
            if len(flat_plan["decision_tree"]) > budget:
                raise ValueError("flat routing tree exceeds node cap")
            expanded_tree: list[dict[str, Any] | None] = [None] * len(flat_plan["decision_tree"])
            per_node_budget = max(1, budget // max(1, len(flat_plan["decision_tree"])))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(flat_plan["decision_tree"]))) as ex:
                futures = {
                    ex.submit(
                        _expand_node,
                        node,
                        task=task,
                        safe_sources=safe_sources,
                        gen_fn=gen_fn,
                        depth_remaining=depth,
                        node_budget=per_node_budget,
                    ): idx
                    for idx, node in enumerate(flat_plan["decision_tree"])
                }
                for fut in concurrent.futures.as_completed(futures):
                    node, _used = fut.result()
                    expanded_tree[futures[fut]] = node
            tree = [node for node in expanded_tree if node is not None]
            if _count_nodes(tree) > budget:
                raise ValueError("expanded routing tree exceeds node cap")
            expanded_plan = dict(flat_plan)
            expanded_plan["decision_tree"] = tree
            return expanded_plan
        except Exception:
            flat_copy = dict(flat_plan)
            flat_copy["fail_open"] = True
            flat_copy["source_policy"] = str(flat_copy.get("source_policy") or "") + " Recursive expansion degraded; using flat routing plan."
            return flat_copy
    except Exception as exc:
        return _default_routing_plan(task, safe_sources, f"supervisor routing degraded: {exc}")


def _apply_evidence_verdict(verdict: dict[str, Any], _finding: dict[str, Any], evidence_result: Any) -> dict[str, Any]:
    updated = dict(verdict)
    if isinstance(evidence_result, bool):
        supported = evidence_result
        rationale = "evidence_fn returned supported" if supported else "evidence_fn returned unsupported"
    elif isinstance(evidence_result, dict):
        value = evidence_result.get("supported")
        if value is None:
            value = evidence_result.get("ok")
        if value is None:
            value = evidence_result.get("verdict")
        if isinstance(value, str):
            supported = value.lower() in {"supported", "true", "yes", "ok", "pass"}
        else:
            supported = bool(value)
        rationale = str(evidence_result.get("rationale") or evidence_result.get("reason") or evidence_result.get("evidence") or "").strip()
    else:
        supported = bool(evidence_result)
        rationale = f"evidence_fn returned {type(evidence_result).__name__}"
    updated["supported"] = supported
    if not supported and not updated.get("guidance"):
        updated["guidance"] = "Re-check the claim against retrieved evidence and remove or qualify unsupported details."
    if rationale:
        prior = str(updated.get("rationale") or "").strip()
        updated["rationale"] = f"{prior}; evidence_fn: {rationale}" if prior else f"evidence_fn: {rationale}"
    updated["evidence_checked"] = True
    return updated


def make_loci_evidence_fn(loci_verify_fn: Callable[..., Any], *, investigation_id: str | None = None) -> EvidenceFn:
    """Adapt an injected Loci verifier/grounding callable into an evidence_fn."""
    def evidence_fn(*, finding: dict[str, Any], **_: Any) -> dict[str, Any]:
        claim = str(finding.get("claim") or finding.get("text") or "")
        context = str(finding.get("evidence") or finding.get("context") or "")
        kwargs: dict[str, Any] = {"claim": claim, "context": context}
        if investigation_id:
            kwargs["investigation_id"] = investigation_id
        try:
            raw = loci_verify_fn(**kwargs)
        except TypeError:
            raw = loci_verify_fn(claim, context=context)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return {"supported": False, "rationale": raw}
        if isinstance(raw, dict):
            verdict = str(raw.get("verdict") or raw.get("supported") or raw.get("status") or "").lower()
            supported = raw.get("supported")
            if supported is None:
                supported = verdict in {"supported", "true", "yes", "pass", "grounded"}
            return {
                "supported": bool(supported),
                "rationale": str(raw.get("rationale") or raw.get("reason") or raw.get("summary") or raw),
            }
        return {"supported": bool(raw), "rationale": "loci evidence adapter returned non-structured result"}
    return evidence_fn


def supervise_findings(
    task: str,
    findings: list[dict[str, Any]],
    routing_plan: dict[str, Any],
    gen_fn: GenFn,
    evidence_fn: EvidenceFn | None = None,
) -> dict[str, Any]:
    """Return advisory per-finding verdicts against task, evidence, and routing plan."""
    safe_findings = [finding for finding in (findings or []) if isinstance(finding, dict)]
    prompt = f"""You are a supervisor monitoring worker findings mid-run.
Return ONLY valid JSON with this schema:
{{
  "fail_open": false,
  "verdicts": [{{"finding_index": integer, "on_task": boolean, "supported": boolean, "source_appropriate": boolean, "guidance": string, "rationale": string}}]
}}

Evaluate every finding by index:
- on_task: answers the original task/sub-need rather than drifting.
- supported: specific claims are backed by the finding's cited/evidence text; unsupported exact counts or confirmations must be false.
- source_appropriate: source matches the routing plan's best-fit source for that sub-need.
- guidance: empty only when all three booleans are true; otherwise tell the worker exactly how to redirect.

Task: {task}
Routing plan: {_json_default(routing_plan)}
Findings: {_json_default(safe_findings)}
"""
    raw = _call_generate(gen_fn, prompt, max_tokens=1600, temperature=0.0)
    try:
        obj = _extract_json(_response_text(raw))
        result = _normalize_supervision(obj, safe_findings)
        if evidence_fn is None:
            return result
        verdicts = []
        for finding, verdict in zip(safe_findings, result["verdicts"]):
            try:
                evidence_result = evidence_fn(finding=finding, task=task, routing_plan=routing_plan, verdict=verdict)
            except TypeError:
                evidence_result = evidence_fn(finding, task, routing_plan)
            verdicts.append(_apply_evidence_verdict(verdict, finding, evidence_result))
        result["verdicts"] = verdicts
        result["evidence_grounded"] = True
        return result
    except Exception as exc:
        return _neutral_verdicts(safe_findings, f"supervisor review degraded: {exc}")


def _verdict_clean(verdict: dict[str, Any]) -> bool:
    return all(bool(verdict.get(key, True)) for key in ("on_task", "supported", "source_appropriate"))


def _flatten_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stack = list(reversed(nodes or []))
    while stack:
        node = stack.pop()
        out.append(node)
        children = node.get("children") or []
        if isinstance(children, list):
            stack.extend(reversed([child for child in children if isinstance(child, dict)]))
    return out


def _routing_node_for_finding(finding: dict[str, Any], routing_plan: dict[str, Any]) -> dict[str, Any]:
    sub_need = str(finding.get("sub_need") or "").lower()
    source = str(finding.get("source") or "").lower()
    nodes = _flatten_nodes(routing_plan.get("decision_tree") or [])
    for node in nodes:
        node_need = str(node.get("sub_need") or "").lower()
        if sub_need and (sub_need in node_need or node_need in sub_need):
            return node
    for node in nodes:
        preferred = [str(src).lower() for src in node.get("preferred_sources") or []]
        fallback = [str(src).lower() for src in node.get("fallback_sources") or []]
        if source and source in preferred + fallback:
            return node
    return nodes[0] if nodes else {}


def _call_worker(
    worker_fn: WorkerFn,
    *,
    task: str,
    finding: dict[str, Any],
    guidance: str,
    routing_node: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    context = {"task": task, "finding": finding, "guidance": guidance, "routing_node": routing_node, "round": round_index}
    try:
        corrected = worker_fn(**context)
    except TypeError:
        corrected = worker_fn(finding, guidance, routing_node)
    if not isinstance(corrected, dict):
        raise ValueError("worker_fn must return a finding dict")
    return corrected


def supervise_and_correct(
    task: str,
    findings: list[dict[str, Any]],
    routing_plan: dict[str, Any],
    gen_fn: GenFn,
    worker_fn: WorkerFn,
    max_rounds: int = 2,
    evidence_fn: EvidenceFn | None = None,
) -> dict[str, Any]:
    """Supervise, re-dispatch flagged findings, and return a visible audit trail.

    Any supervisor or worker failure fails open: original findings are returned
    untouched with ``fail_open=True`` instead of blocking or dropping work.
    """
    original_findings = [dict(finding) for finding in (findings or []) if isinstance(finding, dict)]
    current_findings = [dict(finding) for finding in original_findings]
    audit: list[dict[str, Any]] = [
        {"finding_index": idx, "original_finding": dict(finding), "rounds": [], "converged": False}
        for idx, finding in enumerate(original_findings)
    ]
    try:
        rounds = max(0, int(max_rounds))
        verdict_result = supervise_findings(task, current_findings, routing_plan, gen_fn, evidence_fn=evidence_fn)
        for round_index in range(rounds + 1):
            verdicts = verdict_result["verdicts"]
            flagged = [idx for idx, verdict in enumerate(verdicts) if not _verdict_clean(verdict)]
            for idx, verdict in enumerate(verdicts):
                entry = {
                    "round": round_index,
                    "finding": dict(current_findings[idx]),
                    "verdict": dict(verdict),
                    "guidance": str(verdict.get("guidance") or ""),
                }
                if _verdict_clean(verdict):
                    audit[idx]["converged"] = True
                audit[idx]["rounds"].append(entry)
            if not flagged:
                return {"fail_open": False, "findings": current_findings, "verdicts": verdicts, "audit_trail": audit, "converged": True, "rounds": round_index}
            if round_index >= rounds:
                for idx in flagged:
                    audit[idx]["unconverged"] = True
                return {"fail_open": False, "findings": current_findings, "verdicts": verdicts, "audit_trail": audit, "converged": False, "rounds": round_index}
            corrected_by_index: dict[int, dict[str, Any]] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(flagged))) as ex:
                futures = {
                    ex.submit(
                        _call_worker,
                        worker_fn,
                        task=task,
                        finding=current_findings[idx],
                        guidance=str(verdicts[idx].get("guidance") or ""),
                        routing_node=_routing_node_for_finding(current_findings[idx], routing_plan),
                        round_index=round_index + 1,
                    ): idx
                    for idx in flagged
                }
                for fut in concurrent.futures.as_completed(futures):
                    corrected_by_index[futures[fut]] = fut.result()
            for idx, corrected in corrected_by_index.items():
                current_findings[idx] = corrected
                audit[idx]["rounds"][-1]["corrected_finding"] = dict(corrected)
                audit[idx]["rounds"][-1]["routing_node"] = _routing_node_for_finding(corrected, routing_plan)
            verdict_result = supervise_findings(task, current_findings, routing_plan, gen_fn, evidence_fn=evidence_fn)
    except Exception as exc:
        return {
            "fail_open": True,
            "error": str(exc),
            "findings": original_findings,
            "verdicts": _neutral_verdicts(original_findings, f"closed-loop supervisor degraded: {exc}")["verdicts"],
            "audit_trail": audit,
            "converged": False,
        }
    return {
        "fail_open": True,
        "findings": original_findings,
        "verdicts": _neutral_verdicts(original_findings, "closed-loop supervisor reached an unexpected state")["verdicts"],
        "audit_trail": audit,
        "converged": False,
    }

#!/usr/bin/env python3
"""Tiered local swarm reasoning with cheap fan-out, bounded escalation, and fail-open synthesis.

This is the wide-fan-out sibling of ``local_deep_think.py``. The goal is not one
long chain on one model, but many small, cheap passes in parallel: decompose a topic
into atomic subtasks, answer them concurrently on a fast local tier, escalate only the
uncertain or conflicting slice to a stronger model, then synthesize one structured JSON
object downstream tools can consume directly.

The value proposition is throughput and selective spend. When most subtasks are easy,
a cheap tier should answer most of them and the stronger tiers only pay for the hard
residue. The triage heuristics here are deliberately bounded and testable rather than
magical: self-reported confidence, parse failures, and simple similarity/contradiction
checks on near-duplicate subtasks.

Every lane is fail-open: dead decomposition, cheap, escalation, or synthesis tiers
return a degraded but well-formed result and NEVER raise. A broken tier lowers answer
quality and marks more findings low-confidence; it does not abort the run. Limitations
are explicit too: the contradiction detector is lexical, confidence is self-reported,
and synthesis falls back to a mechanical summary when the strong tier cannot produce
valid JSON.

Usage:
  python3 scripts/swarm_escalate.py "topic here"
  python3 scripts/swarm_escalate.py "topic here" --fanout-count 30 --pretty
  python3 scripts/swarm_escalate.py "topic here" --subtask "Check auth" --subtask "Check cache"
  python3 scripts/swarm_escalate.py "topic here" --subtasks-file subtasks.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional


_CONFIDENCE_LEVELS = ("low", "medium", "high")
_DEFAULT_CHEAP_MODEL = os.environ.get("LOCI_SWARM_CHEAP_MODEL", "qwen2.5:3b")
_DEFAULT_ESCALATE_MODEL = os.environ.get("LOCI_SWARM_ESCALATE_MODEL", "qwen3.8:latest")
_DEFAULT_SYNTHESIZE_MODEL = os.environ.get("LOCI_SWARM_SYNTHESIZE_MODEL", "qwen3.8:latest")
_DEFAULT_DECOMPOSE_MODEL = os.environ.get("LOCI_SWARM_DECOMPOSE_MODEL", "")
_DEFAULT_STIGMERGIC_CONSENSUS = os.environ.get("LOCI_SWARM_STIGMERGIC_CONSENSUS", "").strip().lower() in {"1", "true", "yes", "on"}
_STOPWORDS = {
    "a", "an", "and", "are", "for", "how", "in", "is", "of", "on", "or", "the",
    "this", "to", "what", "when", "where", "which", "why", "with",
}
_NEGATIVE_MARKERS = (" no ", " not ", " never ", " absent ", " missing ", " false ", " unsupported ")
_POSITIVE_MARKERS = (
    " yes ", " present ", " true ", " supported ", " enabled ", " works ",
    " require ", " requires ",
)


@dataclass
class SwarmConfig:
    topic: str
    cheap_model: str = _DEFAULT_CHEAP_MODEL
    escalate_model: str = _DEFAULT_ESCALATE_MODEL
    synthesize_model: str = _DEFAULT_SYNTHESIZE_MODEL
    decompose_model: str = _DEFAULT_DECOMPOSE_MODEL
    subtasks: Optional[list[str]] = None
    fanout_count: int = 20
    escalate_confidences: tuple[str, ...] = ("low",)
    subtask_similarity_threshold: float = 0.50
    answer_similarity_threshold: float = 0.82
    decompose_max_tokens: int = 1200
    answer_max_tokens: int = 320
    synthesize_max_tokens: int = 1400
    subtask_prompt_chars: int = 1500
    synthesis_subtask_chars: int = 240
    synthesis_answer_chars: int = 400
    stigmergic_consensus: bool = _DEFAULT_STIGMERGIC_CONSENSUS
    stigmergic_ttl_minutes: float = 60.0


@dataclass
class SwarmFinding:
    subtask: str
    answer: str
    confidence: str
    tier_reached: str
    model: str
    ok: bool
    parse_ok: bool
    why: str = ""
    escalation_attempted: bool = False
    escalation_reasons: list[str] = field(default_factory=list)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_paths() -> None:
    mcp = _repo_root() / "mcp"
    if str(mcp) not in sys.path:
        sys.path.insert(0, str(mcp))


def _batched_generate(prompts: list[str], *, model: str, max_tokens: int,
                      fmt: Optional[str] = None) -> list[dict]:
    _ensure_paths()
    import batched_gen

    return batched_gen.generate_batch(prompts, model=model, max_tokens=max_tokens, fmt=fmt)


def _fail_batch(size: int, why: str) -> list[dict]:
    return [{"text": "", "ok": False, "why": why} for _ in range(max(0, size))]


def _call_batch(batch_fn: Callable[..., list[dict]], prompts: list[str], *, model: str,
                max_tokens: int, fmt: Optional[str] = None) -> list[dict]:
    prompts = [str(prompt) for prompt in (prompts or [])]
    if not prompts:
        return []
    try:
        rows = batch_fn(prompts, model=model, max_tokens=max_tokens, fmt=fmt)
    except Exception as exc:
        return _fail_batch(len(prompts), str(exc))
    rows = list(rows or [])
    if len(rows) < len(prompts):
        rows.extend(_fail_batch(len(prompts) - len(rows), "missing batch rows"))
    return [row if isinstance(row, dict) else {"text": str(row or ""), "ok": False} for row in rows[:len(prompts)]]


def _call_one(batch_fn: Callable[..., list[dict]], prompt: str, *, model: str,
              max_tokens: int, fmt: Optional[str] = None) -> dict:
    rows = _call_batch(batch_fn, [prompt], model=model, max_tokens=max_tokens, fmt=fmt)
    return rows[0] if rows else {"text": "", "ok": False, "why": "empty batch result"}


def _extract_json_object(text: str):
    cooked = str(text or "").strip()
    if not cooked:
        return None
    try:
        return json.loads(cooked)
    except Exception:
        pass
    starts = [idx for idx, char in enumerate(cooked) if char == "{"]
    for start in starts:
        depth = 0
        for end in range(start, len(cooked)):
            char = cooked[end]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    snippet = cooked[start:end + 1]
                    try:
                        return json.loads(snippet)
                    except Exception:
                        break
    return None


def _normalize_confidence(value: str, default: str = "low") -> str:
    cooked = str(value or "").strip().lower()
    return cooked if cooked in _CONFIDENCE_LEVELS else default


def _coerce_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _normalize_subtasks(raw_subtasks: list, *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for entry in raw_subtasks or []:
        text = str(entry or "").strip()
        key = re.sub(r"\s+", " ", text.lower())
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit > 0 and len(out) >= limit:
            break
    return out


def _fallback_subtasks(topic: str) -> list[str]:
    return [str(topic or "").strip() or "Answer the original topic directly."]


def decompose_subtasks(config: SwarmConfig, batch_fn: Callable[..., list[dict]]) -> tuple[list[str], dict]:
    supplied = _normalize_subtasks(config.subtasks or [], limit=0)
    if supplied:
        return supplied, {"source": "supplied", "requested": config.fanout_count, "degraded": False}

    model = config.decompose_model or config.cheap_model
    prompt = (
        "You are the FAN-OUT decomposition stage in a local reasoning swarm.\n"
        f"Topic: {config.topic}\n"
        f"Return exactly {max(1, int(config.fanout_count))} atomic micro-subtasks that can be answered independently.\n"
        "Keep each subtask concrete, non-overlapping when possible, and phrased as a direct question or check.\n"
        'Return ONLY JSON: {"subtasks":["..."]}'
    )
    raw = _call_one(batch_fn, prompt, model=model, max_tokens=config.decompose_max_tokens, fmt="json")
    obj = _extract_json_object(str(raw.get("text", ""))) if raw.get("ok") else None
    subtasks = _normalize_subtasks((obj or {}).get("subtasks") or [], limit=max(1, int(config.fanout_count)))
    if subtasks:
        return subtasks, {
            "source": "model",
            "model": model,
            "requested": config.fanout_count,
            "degraded": False,
        }
    return _fallback_subtasks(config.topic), {
        "source": "fallback",
        "model": model,
        "requested": config.fanout_count,
        "degraded": True,
        "error": raw.get("why") or "unparseable decomposition output",
    }


def _truncate(text: str, max_chars: int) -> str:
    """Bound embedded text so a huge subtask/answer never crowds out the model's output
    budget or blows the synthesis prompt past context length [pattern: evidence_chars in
    local_deep_think.py]. Marks truncation explicitly rather than silently cutting text."""
    cooked = str(text or "")
    if max_chars <= 0 or len(cooked) <= max_chars:
        return cooked
    return cooked[:max_chars] + "\u2026 [truncated]"


def _answer_prompt(topic: str, subtask: str, max_chars: int = 1500) -> str:
    return (
        "You are the answer stage in a local reasoning swarm.\n"
        f"Topic: {topic}\n"
        f"Subtask: {_truncate(subtask, max_chars)}\n"
        "Answer only this subtask. Be brief and concrete.\n"
        'Return ONLY JSON: {"answer":"...","confidence":"high|medium|low"}'
    )


def _parse_answer_result(raw: dict, *, subtask: str, model: str, tier: str) -> SwarmFinding:
    text = str((raw or {}).get("text") or "").strip()
    why = str((raw or {}).get("why") or "").strip()
    if not (raw or {}).get("ok"):
        return SwarmFinding(
            subtask=subtask,
            answer=text,
            confidence="low",
            tier_reached=tier,
            model=model,
            ok=False,
            parse_ok=False,
            why=why or "generation failed",
        )

    obj = _extract_json_object(text)
    if not isinstance(obj, dict):
        return SwarmFinding(
            subtask=subtask,
            answer=text,
            confidence="low",
            tier_reached=tier,
            model=model,
            ok=False,
            parse_ok=False,
            why="unparseable_json",
        )

    answer = str(obj.get("answer") or obj.get("text") or "").strip()
    confidence = _normalize_confidence(obj.get("confidence"), default="low")
    parse_ok = bool(answer)
    return SwarmFinding(
        subtask=subtask,
        answer=answer or text,
        confidence=confidence,
        tier_reached=tier,
        model=model,
        ok=parse_ok,
        parse_ok=parse_ok,
        why="" if parse_ok else "empty_answer",
    )


def answer_subtasks(topic: str, subtasks: list[str], *, model: str,
                    config: SwarmConfig, batch_fn: Callable[..., list[dict]],
                    tier: str) -> tuple[list[SwarmFinding], dict]:
    prompts = [_answer_prompt(topic, subtask, config.subtask_prompt_chars) for subtask in subtasks]
    raw_rows = _call_batch(batch_fn, prompts, model=model, max_tokens=config.answer_max_tokens, fmt="json")
    findings = [
        _parse_answer_result(raw, subtask=subtask, model=model, tier=tier)
        for subtask, raw in zip(subtasks, raw_rows)
    ]
    return findings, {
        "tier": tier,
        "model": model,
        "count": len(findings),
        "ok": sum(1 for item in findings if item.ok),
        "degraded": sum(1 for item in findings if not item.ok),
    }


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return {token for token in tokens if len(token) > 2 and token not in _STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _answer_signature(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def _answer_polarity(text: str) -> str:
    cooked = f" {_answer_signature(text)} "
    if " does not " in cooked or " do not " in cooked or " not require " in cooked:
        return "negative"
    negative = any(marker in cooked for marker in _NEGATIVE_MARKERS)
    positive = any(marker in cooked for marker in _POSITIVE_MARKERS)
    if negative and not positive:
        return "negative"
    if positive and not negative:
        return "positive"
    return "unknown"


def triage_findings(findings: list[SwarmFinding], config: SwarmConfig) -> dict:
    reasons: dict[int, set[str]] = {idx: set() for idx in range(len(findings))}
    similar_pairs: list[dict] = []
    confidence_triggers = {item.lower() for item in config.escalate_confidences}

    for idx, finding in enumerate(findings):
        if not finding.ok:
            reasons[idx].add("generation_failed")
        if not finding.parse_ok:
            reasons[idx].add("parse_failure")
        if finding.confidence.lower() in confidence_triggers:
            reasons[idx].add(f"confidence:{finding.confidence.lower()}")

    subtask_tokens = [_tokenize(item.subtask) for item in findings]
    answer_tokens = [_tokenize(item.answer) for item in findings]
    for left in range(len(findings)):
        for right in range(left + 1, len(findings)):
            similarity = _jaccard(subtask_tokens[left], subtask_tokens[right])
            if similarity < config.subtask_similarity_threshold:
                continue
            pair = {
                "left": left,
                "right": right,
                "subtask_similarity": round(similarity, 4),
            }
            left_sig = _answer_signature(findings[left].answer)
            right_sig = _answer_signature(findings[right].answer)
            answer_similarity = _jaccard(answer_tokens[left], answer_tokens[right])
            if left_sig and left_sig == right_sig:
                reasons[left].add("duplicate_similar_subtask")
                reasons[right].add("duplicate_similar_subtask")
                pair["reason"] = "duplicate"
                pair["answer_similarity"] = 1.0
                similar_pairs.append(pair)
                continue
            if answer_similarity >= config.answer_similarity_threshold:
                reasons[left].add("duplicate_like_similar_subtask")
                reasons[right].add("duplicate_like_similar_subtask")
                pair["reason"] = "duplicate_like"
                pair["answer_similarity"] = round(answer_similarity, 4)
                similar_pairs.append(pair)
                continue
            left_polarity = _answer_polarity(findings[left].answer)
            right_polarity = _answer_polarity(findings[right].answer)
            if left_polarity != "unknown" and right_polarity != "unknown" and left_polarity != right_polarity:
                reasons[left].add("contradiction_similar_subtask")
                reasons[right].add("contradiction_similar_subtask")
                pair["reason"] = "contradiction"
                pair["answer_similarity"] = round(answer_similarity, 4)
                similar_pairs.append(pair)

    flagged_indices = sorted(idx for idx, why in reasons.items() if why)
    return {
        "flagged_indices": flagged_indices,
        "flagged_count": len(flagged_indices),
        "escalation_reasons": {str(idx): sorted(list(why)) for idx, why in reasons.items() if why},
        "similar_pairs": similar_pairs,
    }


def escalate_findings(findings: list[SwarmFinding], triage: dict, *, config: SwarmConfig,
                      batch_fn: Callable[..., list[dict]]) -> tuple[list[SwarmFinding], dict]:
    flagged = [int(idx) for idx in (triage or {}).get("flagged_indices") or []]
    if not flagged:
        return list(findings), {
            "attempted": 0,
            "succeeded": 0,
            "failed_open": 0,
            "model": config.escalate_model,
        }

    prompts = [
        _answer_prompt(config.topic, findings[idx].subtask, config.subtask_prompt_chars)
        for idx in flagged
    ]
    raw_rows = _call_batch(
        batch_fn,
        prompts,
        model=config.escalate_model,
        max_tokens=config.answer_max_tokens,
        fmt="json",
    )
    updated = list(findings)
    succeeded = 0
    failed_open = 0
    by_index = (triage or {}).get("escalation_reasons") or {}
    for idx, raw in zip(flagged, raw_rows):
        escalated = _parse_answer_result(
            raw,
            subtask=findings[idx].subtask,
            model=config.escalate_model,
            tier="escalated",
        )
        escalated.escalation_attempted = True
        escalated.escalation_reasons = list(by_index.get(str(idx), []))
        current = updated[idx]
        current.escalation_attempted = True
        current.escalation_reasons = list(escalated.escalation_reasons)
        should_replace = escalated.ok or (bool(escalated.answer.strip()) and not current.answer.strip())
        if should_replace:
            updated[idx] = escalated
            succeeded += 1
        else:
            failed_open += 1
            if escalated.why:
                current.escalation_reasons.append(f"escalation_failed:{escalated.why}")
    return updated, {
        "attempted": len(flagged),
        "succeeded": succeeded,
        "failed_open": failed_open,
        "model": config.escalate_model,
    }


def apply_consensus_gate(findings: list[SwarmFinding], triage: dict, config: SwarmConfig) -> tuple[dict, dict]:
    if not config.stigmergic_consensus:
        return triage, {"enabled": False}
    try:
        import stigmergic_consensus

        gate_config = stigmergic_consensus.StigmergicConfig(
            enabled=True,
            ttl_minutes=float(config.stigmergic_ttl_minutes),
        )
        result = stigmergic_consensus.apply_stigmergic_consensus(
            findings,
            triage,
            gate_config,
        )
        return result["triage"], result["consensus"]
    except Exception as exc:
        report = {"enabled": True, "degraded": True, "error": str(exc)[:300]}
        return triage, report


def _public_finding(finding: SwarmFinding, *, subtask_chars: int = 0, answer_chars: int = 0) -> dict:
    subtask = finding.subtask
    answer = finding.answer
    if subtask_chars > 0:
        subtask = _truncate(subtask, subtask_chars)
    if answer_chars > 0:
        answer = _truncate(answer, answer_chars)
    payload = {
        "subtask": subtask,
        "answer": answer,
        "confidence": finding.confidence,
        "tier_reached": finding.tier_reached,
        "model": finding.model,
        "ok": finding.ok,
    }
    if finding.escalation_attempted:
        payload["escalation_attempted"] = True
    if finding.escalation_reasons:
        payload["escalation_reasons"] = list(finding.escalation_reasons)
    if finding.why:
        payload["why"] = finding.why
    return payload


def _fallback_summary(topic: str, findings: list[dict], stats: dict) -> str:
    good = [item for item in findings if item.get("answer")]
    fragments = [f"{item['subtask']}: {item['answer']}" for item in good[:3]]
    lead = (
        f"Swarm processed {stats['fanout_count']} subtasks for '{topic}' and escalated "
        f"{stats['escalated_count']} of them."
    )
    return lead if not fragments else f"{lead} Key findings: {' | '.join(fragments)}"


def synthesize_swarm(config: SwarmConfig, findings: list[SwarmFinding], stats: dict,
                     batch_fn: Callable[..., list[dict]]) -> dict:
    public_findings = [_public_finding(item) for item in findings]
    # The prompt sent to the synthesis model uses a separately truncated view: with dozens
    # of subtasks (especially ones carrying large embedded evidence, e.g. a full diff) the
    # untruncated findings list can blow past the model's context window on its own, before
    # the model ever emits a token of output. Bounding what's re-embedded here keeps the
    # synthesis prompt's size roughly independent of how large any one subtask/answer is,
    # while the full-fidelity public_findings above is still what callers get back.
    prompt_findings = [
        _public_finding(
            item,
            subtask_chars=config.synthesis_subtask_chars,
            answer_chars=config.synthesis_answer_chars,
        )
        for item in findings
    ]
    prompt = (
        "You are the SYNTHESIZE stage in a tiered local reasoning swarm.\n"
        "Combine the final per-subtask answers into one direct, evidence-aware JSON object.\n"
        "Do not invent findings absent from the inputs.\n"
        'Return ONLY JSON with this schema: {"schema_version":1,"topic":"...",'
        '"findings":[{"subtask":"...","answer":"...","confidence":"low|medium|high",'
        '"tier_reached":"cheap|escalated|synthesized"}],"summary":"...",'
        '"stats":{"fanout_count":0,"escalated_count":0,"escalation_rate":0.0}}\n\n'
        f"TOPIC: {config.topic}\n"
        f"FINAL_FINDINGS: {json.dumps(prompt_findings, indent=2)}\n"
        f"STATS: {json.dumps(stats, indent=2)}"
    )
    raw = _call_one(
        batch_fn,
        prompt,
        model=config.synthesize_model,
        max_tokens=config.synthesize_max_tokens,
        fmt="json",
    )
    obj = _extract_json_object(str(raw.get("text", ""))) if raw.get("ok") else None
    summary = ""
    if isinstance(obj, dict):
        summary = str(obj.get("summary") or "").strip()
    result = {
        "schema_version": 1,
        "topic": config.topic,
        "findings": public_findings,
        "summary": summary or _fallback_summary(config.topic, prompt_findings, stats),
        "stats": {
            "fanout_count": int(stats.get("fanout_count") or 0),
            "escalated_count": int(stats.get("escalated_count") or 0),
            "escalation_rate": float(stats.get("escalation_rate") or 0.0),
        },
        "degraded": not bool(summary),
        "synthesis": {
            "model": config.synthesize_model,
            "ok": bool(summary),
            "degraded": not bool(summary),
        },
    }
    if raw.get("why"):
        result["synthesis"]["error"] = raw.get("why")
    return result


def validate_swarm_result(result: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["result must be a dict"]
    if result.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if not isinstance(result.get("topic"), str) or not result.get("topic", "").strip():
        errors.append("topic must be a non-empty string")
    if not isinstance(result.get("summary"), str) or not result.get("summary", "").strip():
        errors.append("summary must be a non-empty string")
    findings = result.get("findings")
    if not isinstance(findings, list) or not findings:
        errors.append("findings must be a non-empty list")
    else:
        for idx, finding in enumerate(findings):
            if not isinstance(finding, dict):
                errors.append(f"findings[{idx}] must be a dict")
                continue
            if not str(finding.get("subtask") or "").strip():
                errors.append(f"findings[{idx}].subtask missing")
            if not isinstance(finding.get("answer"), str):
                errors.append(f"findings[{idx}].answer must be a string")
            if finding.get("confidence") not in _CONFIDENCE_LEVELS:
                errors.append(f"findings[{idx}].confidence invalid")
            if finding.get("tier_reached") not in {"cheap", "escalated", "synthesized"}:
                errors.append(f"findings[{idx}].tier_reached invalid")
    stats = result.get("stats")
    if not isinstance(stats, dict):
        errors.append("stats must be a dict")
    else:
        fanout_count = stats.get("fanout_count")
        escalated_count = stats.get("escalated_count")
        escalation_rate = stats.get("escalation_rate")
        if not isinstance(fanout_count, int) or fanout_count < 0:
            errors.append("stats.fanout_count invalid")
        if not isinstance(escalated_count, int) or escalated_count < 0:
            errors.append("stats.escalated_count invalid")
        if not isinstance(escalation_rate, (int, float)) or escalation_rate < 0:
            errors.append("stats.escalation_rate invalid")
        if isinstance(fanout_count, int) and isinstance(escalated_count, int) and fanout_count >= 0:
            expected = _coerce_ratio(escalated_count, fanout_count)
            if isinstance(escalation_rate, (int, float)) and abs(float(escalation_rate) - expected) > 1e-9:
                errors.append("stats.escalation_rate inconsistent with counts")
    return errors


def run_swarm(config: SwarmConfig, deps: Optional[dict] = None) -> dict:
    deps = deps or {"generate_batch": _batched_generate}
    batch_fn = deps["generate_batch"]

    subtasks, decomposition = decompose_subtasks(config, batch_fn)
    cheap_findings, cheap_report = answer_subtasks(
        config.topic,
        subtasks,
        model=config.cheap_model,
        config=config,
        batch_fn=batch_fn,
        tier="cheap",
    )
    triage = triage_findings(cheap_findings, config)
    triage, consensus_report = apply_consensus_gate(cheap_findings, triage, config)
    final_findings, escalate_report = escalate_findings(cheap_findings, triage, config=config, batch_fn=batch_fn)
    stats = {
        "fanout_count": len(subtasks),
        "escalated_count": int(escalate_report["attempted"]),
        "escalation_rate": _coerce_ratio(int(escalate_report["attempted"]), len(subtasks)),
    }
    result = synthesize_swarm(config, final_findings, stats, batch_fn)
    result["decomposition"] = decomposition
    result["triage"] = triage
    result["stigmergic_consensus"] = consensus_report
    result["tiers"] = {
        "cheap": cheap_report,
        "escalate": escalate_report,
    }
    result["lineage"] = [asdict(item) for item in final_findings]
    return result


def _split_csv(raw: Optional[str]) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in str(raw or "").split(",") if item.strip())


def _load_subtasks_file(path: str) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(raw)
    except Exception:
        obj = None
    if isinstance(obj, list):
        return [str(item).strip() for item in obj if str(item).strip()]
    return [line.strip() for line in raw.splitlines() if line.strip()]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("topic", help="Topic/question to reason over.")
    ap.add_argument("--cheap-model", default=_DEFAULT_CHEAP_MODEL)
    ap.add_argument("--escalate-model", default=_DEFAULT_ESCALATE_MODEL)
    ap.add_argument("--synthesize-model", default=_DEFAULT_SYNTHESIZE_MODEL)
    ap.add_argument(
        "--decompose-model",
        default=_DEFAULT_DECOMPOSE_MODEL,
        help="Model for subtask decomposition. Default: cheap model when unset.",
    )
    ap.add_argument("--fanout-count", type=int, default=20)
    ap.add_argument(
        "--escalate-confidences",
        default="low",
        help="Comma-separated confidences that force escalation. Default: low",
    )
    ap.add_argument("--subtasks-file", help="JSON array or newline-delimited subtasks.")
    ap.add_argument("--subtask", action="append", default=[], help="Repeatable pre-supplied subtask.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    ap.add_argument(
        "--stigmergic-consensus",
        action="store_true",
        help="Enable deterministic stigmergic gating before escalation.",
    )
    ap.add_argument(
        "--stigmergic-ttl-minutes",
        type=float,
        default=60.0,
        help="TTL for stigmergic findings before they require escalation. Default: 60.",
    )
    return ap.parse_args(argv)


def _resolve_config(args: argparse.Namespace) -> SwarmConfig:
    supplied = list(args.subtask or [])
    if args.subtasks_file:
        supplied.extend(_load_subtasks_file(args.subtasks_file))
    return SwarmConfig(
        topic=args.topic,
        cheap_model=args.cheap_model,
        escalate_model=args.escalate_model,
        synthesize_model=args.synthesize_model,
        decompose_model=args.decompose_model,
        subtasks=supplied or None,
        fanout_count=max(1, int(args.fanout_count)),
        escalate_confidences=_split_csv(args.escalate_confidences) or ("low",),
        stigmergic_consensus=bool(args.stigmergic_consensus or _DEFAULT_STIGMERGIC_CONSENSUS),
        stigmergic_ttl_minutes=max(0.0, float(args.stigmergic_ttl_minutes)),
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = _resolve_config(args)
    result = run_swarm(config)
    print(json.dumps(result, indent=2 if args.pretty else None))
    return 0 if not validate_swarm_result(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())

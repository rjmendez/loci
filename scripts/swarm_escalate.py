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
  python3 scripts/swarm_escalate.py "topic here" --fanout-count 10 --seeds 4 --pretty
  python3 scripts/swarm_escalate.py "topic here" --subtask "Check auth" --subtask "Check cache"
  python3 scripts/swarm_escalate.py "topic here" --subtasks-file subtasks.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional


_CONFIDENCE_LEVELS = ("low", "medium", "high")
_DEFAULT_CHEAP_MODEL = os.environ.get("LOCI_SWARM_CHEAP_MODEL", "qwen2.5:3b")
# Existing/default code path (no opt-in tier flags set): unchanged from before the
# 2026-09-16 tier work. Do not change these without also changing the default behavior
# for callers that pass no flags at all.
_DEFAULT_ESCALATE_MODEL = os.environ.get("LOCI_SWARM_ESCALATE_MODEL", "qwen3.8:latest")
_DEFAULT_SYNTHESIZE_MODEL = os.environ.get("LOCI_SWARM_SYNTHESIZE_MODEL", "qwen3.8:latest")
_DEFAULT_DECOMPOSE_MODEL = os.environ.get("LOCI_SWARM_DECOMPOSE_MODEL", "")
_DEFAULT_STIGMERGIC_CONSENSUS = os.environ.get("LOCI_SWARM_STIGMERGIC_CONSENSUS", "").strip().lower() in {"1", "true", "yes", "on"}
# Opt-in tier: chosen from a live 12-model / 4-task (code, math, JSON, security) benchmark
# on 2026-09-16 across the machine's full local Ollama library (see ~/.loci notes /
# session history). These only apply once the caller explicitly opts into one of the new
# tier flags (seeds>1, synthesize_think, safety_check, self_consistency_samples>1,
# reduce_group_size>0, escalate_with_prior_context) AND does not explicitly pass their
# own escalate_model/synthesize_model -- see compute_tier_active()/resolve_tier_models()
# below, which _resolve_config() (CLI) and mcp/llm_tools.py's swarm_reason (MCP tool) both
# call before a SwarmConfig even exists, so an explicit override is never second-guessed
# even when it happens to equal the legacy default (e.g. --escalate-model qwen3.8:latest
# --seeds 2).
_TIER_ESCALATE_MODEL = "heretic-llama31-8b-instruct:latest"
_TIER_SYNTHESIZE_MODEL = "hf.co/slevinw/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M"
_TIER_DECOMPOSE_MAX_TOKENS = 2200
_TIER_SYNTHESIZE_MAX_TOKENS = 2200
_STOPWORDS = {
    "a", "an", "and", "are", "for", "how", "in", "is", "of", "on", "or", "the",
    "this", "to", "what", "when", "where", "which", "why", "with",
}
_NEGATIVE_MARKERS = (" no ", " not ", " never ", " absent ", " missing ", " false ", " unsupported ")
_POSITIVE_MARKERS = (
    " yes ", " present ", " true ", " supported ", " enabled ", " works ",
    " require ", " requires ",
)
_MULTI_SEED_SYNTHESIS_MAX_CHARS = 18000


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0", "false", "False")


def compute_tier_active(*, seeds: int = 1, synthesize_think: bool = False, safety_check: bool = False,
                        self_consistency_samples: int = 1, reduce_group_size: int = 0,
                        escalate_with_prior_context: bool = False) -> bool:
    """True once the caller has opted into any new reasoning-tier knob. Callers that
    build a SwarmConfig indirectly (CLI, MCP tool) should call this BEFORE resolving
    escalate_model/synthesize_model, so an explicit model override is never confused
    with an implicit legacy default (see resolve_tier_models())."""
    return (
        int(seeds or 1) > 1
        or bool(synthesize_think)
        or bool(safety_check)
        or int(self_consistency_samples or 1) > 1
        or int(reduce_group_size or 0) > 0
        or bool(escalate_with_prior_context)
    )


def resolve_tier_models(*, escalate_model: Optional[str], synthesize_model: Optional[str],
                        tier_active: bool) -> tuple[str, str]:
    """Resolve the effective escalate/synthesize models from explicit (possibly None
    or empty) caller input plus tier_active. An explicit, non-empty value always wins
    -- even if it equals the legacy default -- so it is never silently upgraded."""
    resolved_escalate = str(escalate_model or "").strip() or (
        _TIER_ESCALATE_MODEL if tier_active else _DEFAULT_ESCALATE_MODEL
    )
    resolved_synthesize = str(synthesize_model or "").strip() or (
        _TIER_SYNTHESIZE_MODEL if tier_active else _DEFAULT_SYNTHESIZE_MODEL
    )
    return resolved_escalate, resolved_synthesize


@dataclass
class SwarmConfig:
    topic: str
    cheap_model: str = _DEFAULT_CHEAP_MODEL
    escalate_model: str = _DEFAULT_ESCALATE_MODEL
    synthesize_model: str = _DEFAULT_SYNTHESIZE_MODEL
    decompose_model: str = _DEFAULT_DECOMPOSE_MODEL
    escalate_model_explicit: bool = False
    synthesize_model_explicit: bool = False
    subtasks: Optional[list[str]] = None
    fanout_count: int = 20
    seeds: int = 1
    escalate_confidences: tuple[str, ...] = ("low",)
    subtask_similarity_threshold: float = 0.50
    answer_similarity_threshold: float = 0.82
    # Existing/default token budget (pre-2026-09-16 tier work). Bumped only when an
    # opt-in tier flag is set; see __post_init__. There is no CLI/MCP override for this
    # field, so the equality check below can never conflate an explicit override with
    # an implicit default.
    decompose_max_tokens: int = 1200
    answer_max_tokens: int = 320
    # Existing/default token budget (pre-2026-09-16 tier work). Same note as
    # decompose_max_tokens above: no CLI/MCP override exists for this field.
    synthesize_max_tokens: int = 1400
    subtask_prompt_chars: int = 1500
    synthesis_subtask_chars: int = 240
    synthesis_answer_chars: int = 400
    # Opt-in "high-end" mode for the single synthesis call: enables Ollama reasoning
    # (`think`) on models that support it, at a much larger token budget (see
    # synthesize_think_max_tokens). Live-benchmarked 2026-09-16: enabling `think` on a
    # thinking-capable model without a MUCH bigger budget is unsafe -- Ollama's
    # num_predict is a budget SHARED between the hidden reasoning trace and the visible
    # answer, and a 25B model burned an entire 1500-token budget on reasoning alone for
    # one of five held-out prompts, returning a genuinely empty answer. Defaults to False
    # (matches every prior run); synthesize_swarm() also auto-retries with think=False at
    # the normal budget if a think=True attempt comes back unparseable/empty, so turning
    # this on can only add latency on a bad draw, never break the fail-open guarantee.
    synthesize_think: bool = field(default_factory=lambda: _env_bool("LOCI_SWARM_SYNTHESIZE_THINK"))
    synthesize_think_max_tokens: int = 4000
    safety_check: bool = field(default_factory=lambda: _env_bool("LOCI_SWARM_SAFETY_CHECK"))
    self_consistency_samples: int = 1
    escalate_with_prior_context: bool = False
    reduce_group_size: int = 0
    stigmergic_consensus: bool = _DEFAULT_STIGMERGIC_CONSENSUS
    stigmergic_ttl_minutes: float = 60.0

    def __post_init__(self) -> None:
        # decompose_max_tokens/synthesize_max_tokens have no CLI/MCP override path (no
        # --decompose-max-tokens/--synthesize-max-tokens flag exists), so an
        # equality-based upgrade here cannot conflate an explicit caller override with
        # an implicit default. escalate_model/synthesize_model ARE externally
        # overridable, so CLI/MCP callers set the *_model_explicit flags during
        # construction; direct SwarmConfig callers can set them too when pinning a
        # legacy-default tag intentionally. See _resolve_config() and
        # mcp/llm_tools.py's swarm_reason().
        tier_active = compute_tier_active(
            seeds=self.seeds,
            synthesize_think=self.synthesize_think,
            safety_check=self.safety_check,
            self_consistency_samples=self.self_consistency_samples,
            reduce_group_size=self.reduce_group_size,
            escalate_with_prior_context=self.escalate_with_prior_context,
        )
        if not tier_active:
            return
        if not self.escalate_model_explicit and self.escalate_model == _DEFAULT_ESCALATE_MODEL:
            self.escalate_model = _TIER_ESCALATE_MODEL
        if not self.synthesize_model_explicit and self.synthesize_model == _DEFAULT_SYNTHESIZE_MODEL:
            self.synthesize_model = _TIER_SYNTHESIZE_MODEL
        if self.decompose_max_tokens == 1200:
            self.decompose_max_tokens = _TIER_DECOMPOSE_MAX_TOKENS
        if self.synthesize_max_tokens == 1400:
            self.synthesize_max_tokens = _TIER_SYNTHESIZE_MAX_TOKENS


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
                      fmt: Optional[str] = None, think: bool = False,
                      endpoint_role: Optional[str] = None) -> list[dict]:
    _ensure_paths()
    import batched_gen

    return batched_gen.generate_batch(prompts, model=model, max_tokens=max_tokens, fmt=fmt,
                                      think=think, endpoint_role=endpoint_role)


def _fail_batch(size: int, why: str) -> list[dict]:
    return [{"text": "", "ok": False, "why": why} for _ in range(max(0, size))]


def _call_batch(batch_fn: Callable[..., list[dict]], prompts: list[str], *, model: str,
                max_tokens: int, fmt: Optional[str] = None, think: bool = False) -> list[dict]:
    prompts = [str(prompt) for prompt in (prompts or [])]
    if not prompts:
        return []
    try:
        rows = batch_fn(prompts, model=model, max_tokens=max_tokens, fmt=fmt, think=think)
    except Exception as exc:
        return _fail_batch(len(prompts), str(exc))
    rows = list(rows or [])
    if len(rows) < len(prompts):
        rows.extend(_fail_batch(len(prompts) - len(rows), "missing batch rows"))
    return [row if isinstance(row, dict) else {"text": str(row or ""), "ok": False} for row in rows[:len(prompts)]]


def _call_one(batch_fn: Callable[..., list[dict]], prompt: str, *, model: str,
              max_tokens: int, fmt: Optional[str] = None, think: bool = False) -> dict:
    rows = _call_batch(batch_fn, [prompt], model=model, max_tokens=max_tokens, fmt=fmt, think=think)
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


def decompose_subtasks(config: SwarmConfig, batch_fn: Callable[..., list[dict]], *,
                       seed_index: int = 0, seed_count: int = 1) -> tuple[list[str], dict]:
    supplied = _normalize_subtasks(config.subtasks or [], limit=0)
    if supplied:
        return supplied, {"source": "supplied", "requested": config.fanout_count, "degraded": False}

    model = config.decompose_model or config.cheap_model
    diversity_line = ""
    if seed_count > 1:
        diversity_line = (
            f"Independent seed: {seed_index + 1}/{seed_count}. Prefer a slightly different "
            "decomposition angle than other parallel runs while staying concrete.\n"
        )
    prompt = (
        "You are the FAN-OUT decomposition stage in a local reasoning swarm.\n"
        f"Topic: {config.topic}\n"
        f"{diversity_line}"
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


def _escalation_prompt(topic: str, finding: SwarmFinding, *, include_prior: bool,
                       max_chars: int = 1500) -> str:
    if not include_prior:
        return _answer_prompt(topic, finding.subtask, max_chars)
    return (
        "You are the escalation stage in a local reasoning swarm.\n"
        f"Topic: {topic}\n"
        f"Subtask: {_truncate(finding.subtask, max_chars)}\n"
        "A weaker cheap-tier model already attempted this subtask. Critique it, "
        "correct any mistakes, and improve the answer without inventing evidence.\n"
        f"PRIOR_WEAK_ANSWER: {_truncate(finding.answer, max_chars)}\n"
        f"PRIOR_CONFIDENCE: {finding.confidence}\n"
        "Return ONLY JSON: {\"answer\":\"...\",\"confidence\":\"high|medium|low\"}"
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


def self_consistency_findings(findings: list[SwarmFinding], triage: dict, *, config: SwarmConfig,
                              batch_fn: Callable[..., list[dict]]) -> tuple[list[SwarmFinding], dict, dict]:
    samples = max(1, int(config.self_consistency_samples or 1))
    if samples <= 1:
        return list(findings), triage, {
            "enabled": False,
            "samples": samples,
            "attempted": 0,
            "majority": 0,
            "disagreement": 0,
            "model": config.cheap_model,
        }

    reasons_by_index = {
        str(idx): list(reasons)
        for idx, reasons in ((triage or {}).get("escalation_reasons") or {}).items()
    }
    flagged = [
        int(idx)
        for idx in ((triage or {}).get("flagged_indices") or [])
        if "confidence:low" in reasons_by_index.get(str(idx), [])
    ]
    if not flagged:
        return list(findings), triage, {
            "enabled": True,
            "samples": samples,
            "attempted": 0,
            "majority": 0,
            "disagreement": 0,
            "model": config.cheap_model,
        }

    prompts = [
        _answer_prompt(config.topic, findings[idx].subtask, config.subtask_prompt_chars)
        for idx in flagged
        for _ in range(samples)
    ]
    raw_rows = _call_batch(
        batch_fn,
        prompts,
        model=config.cheap_model,
        max_tokens=config.answer_max_tokens,
        fmt="json",
    )

    updated = list(findings)
    new_reasons = {str(idx): list(reasons) for idx, reasons in reasons_by_index.items()}
    majority = 0
    disagreement = 0
    for offset, idx in enumerate(flagged):
        sample_rows = raw_rows[offset * samples:(offset + 1) * samples]
        parsed = [
            _parse_answer_result(raw, subtask=findings[idx].subtask, model=config.cheap_model, tier="cheap")
            for raw in sample_rows
        ]
        buckets: dict[str, list[SwarmFinding]] = {}
        for item in parsed:
            if item.ok and item.answer.strip():
                buckets.setdefault(_answer_signature(item.answer), []).append(item)
        winner_key = ""
        winner_items: list[SwarmFinding] = []
        if buckets:
            winner_key, winner_items = max(buckets.items(), key=lambda pair: (len(pair[1]), _finding_sort_key(pair[1][0])))
        if winner_key and len(winner_items) > samples / 2:
            chosen = max(winner_items, key=_finding_sort_key)
            updated[idx] = chosen
            majority += 1
            existing = [
                reason for reason in new_reasons.get(str(idx), [])
                if not str(reason).startswith("confidence:")
            ]
            if chosen.confidence.lower() not in {item.lower() for item in config.escalate_confidences}:
                if existing:
                    new_reasons[str(idx)] = existing
                else:
                    new_reasons.pop(str(idx), None)
            else:
                reasons = new_reasons[str(idx)] = existing
                if f"confidence:{chosen.confidence.lower()}" not in reasons:
                    reasons.append(f"confidence:{chosen.confidence.lower()}")
        else:
            disagreement += 1
            reasons = new_reasons.setdefault(str(idx), [])
            if "self_consistency_disagreement" not in reasons:
                reasons.append("self_consistency_disagreement")

    new_flagged = sorted(int(idx) for idx, reasons in new_reasons.items() if reasons)
    new_triage = dict(triage or {})
    new_triage["flagged_indices"] = new_flagged
    new_triage["flagged_count"] = len(new_flagged)
    new_triage["escalation_reasons"] = {str(idx): sorted(reasons) for idx, reasons in new_reasons.items() if reasons}
    return updated, new_triage, {
        "enabled": True,
        "samples": samples,
        "attempted": len(flagged),
        "majority": majority,
        "disagreement": disagreement,
        "model": config.cheap_model,
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
        _escalation_prompt(
            config.topic,
            findings[idx],
            include_prior=bool(config.escalate_with_prior_context),
            max_chars=config.subtask_prompt_chars,
        )
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


def _group_findings_for_reduce(prompt_findings: list[dict], group_size: int) -> list[list[dict]]:
    size = max(1, int(group_size or 1))
    return [prompt_findings[idx:idx + size] for idx in range(0, len(prompt_findings), size)]


def _fallback_group_summary(group: list[dict], group_index: int) -> dict:
    fragments = []
    for item in group:
        subtask = str(item.get("subtask") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if subtask or answer:
            fragments.append(f"{subtask}: {answer}".strip(": "))
    return {
        "subtask": f"Reduced finding group {group_index + 1}",
        "answer": " | ".join(fragments[:5]) or "Group summary unavailable.",
        "confidence": "low",
        "tier_reached": "synthesized",
        "model": "",
        "ok": False,
    }


def _reduce_findings_for_synthesis(config: SwarmConfig, prompt_findings: list[dict],
                                   stats: dict, batch_fn: Callable[..., list[dict]]) -> tuple[list[dict], dict]:
    group_size = max(0, int(config.reduce_group_size or 0))
    if group_size <= 0 or len(prompt_findings) <= group_size:
        return prompt_findings, {
            "enabled": False,
            "group_size": group_size,
            "input_count": len(prompt_findings),
            "group_count": 0,
            "failed_open": 0,
        }

    groups = _group_findings_for_reduce(prompt_findings, group_size)
    prompts = [
        (
            "You are an intermediate REDUCE stage in a tiered local reasoning swarm.\n"
            "Summarize this group of per-subtask findings without dropping material facts.\n"
            "Do not add claims absent from the inputs.\n"
            'Return ONLY JSON: {"summary":"...","confidence":"high|medium|low"}\n\n'
            f"TOPIC: {config.topic}\n"
            f"GROUP_INDEX: {idx + 1}/{len(groups)}\n"
            f"GROUP_FINDINGS: {json.dumps(group, indent=2)}\n"
            f"OVERALL_STATS: {json.dumps(stats, indent=2)}"
        )
        for idx, group in enumerate(groups)
    ]
    rows = _call_batch(
        batch_fn,
        prompts,
        model=config.synthesize_model,
        max_tokens=config.synthesize_max_tokens,
        fmt="json",
    )
    reduced: list[dict] = []
    failed_open = 0
    for idx, (group, raw) in enumerate(zip(groups, rows)):
        obj = _extract_json_object(str((raw or {}).get("text") or "")) if (raw or {}).get("ok") else None
        summary = str((obj or {}).get("summary") or "").strip() if isinstance(obj, dict) else ""
        if summary:
            reduced.append({
                "subtask": f"Reduced finding group {idx + 1} ({len(group)} findings)",
                "answer": summary,
                "confidence": _normalize_confidence((obj or {}).get("confidence"), default="medium"),
                "tier_reached": "synthesized",
                "model": config.synthesize_model,
                "ok": True,
            })
        else:
            failed_open += 1
            reduced.append(_fallback_group_summary(group, idx))
    return reduced, {
        "enabled": True,
        "group_size": group_size,
        "input_count": len(prompt_findings),
        "group_count": len(groups),
        "output_count": len(reduced),
        "failed_open": failed_open,
        "model": config.synthesize_model,
    }


def _guardian_check(text: str) -> dict:
    _ensure_paths()
    from guardian import check_injection_risk
    return check_injection_risk(text)


def _safety_flag(text: str, guardian_fn: Callable[[str], dict]) -> dict:
    try:
        result = guardian_fn(text)
    except Exception as exc:
        return {"flagged": False, "why": f"guardian failed open: {exc}"[:200]}
    if not isinstance(result, dict):
        return {"flagged": False, "why": "guardian failed open: invalid result"}
    flagged = bool(result.get("flagged"))
    if flagged:
        why = result.get("error") or f"guardian verdict {result.get('verdict') or 'Yes'}"
    elif result.get("ok") is False:
        why = f"guardian failed open: {result.get('error') or 'unknown error'}"
    else:
        why = f"guardian verdict {result.get('verdict') or 'No'}"
    return {"flagged": flagged, "why": str(why)[:200]}


def _confidence_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(value or "").lower(), -1)


def _tier_rank(value: str) -> int:
    return {"cheap": 0, "escalated": 1, "synthesized": 2}.get(str(value or "").lower(), -1)


def _finding_sort_key(finding: SwarmFinding) -> tuple[int, int, int, int, int]:
    return (
        1 if finding.ok else 0,
        1 if finding.parse_ok else 0,
        _confidence_rank(finding.confidence),
        _tier_rank(finding.tier_reached),
        len(str(finding.answer or "")),
    )


def _merge_findings(findings: list[SwarmFinding]) -> tuple[list[SwarmFinding], dict]:
    merged: list[SwarmFinding] = []
    seen: dict[tuple[str, str], int] = {}
    dropped = 0
    for finding in findings:
        key = (_answer_signature(finding.subtask), _answer_signature(finding.answer))
        if key in seen:
            dropped += 1
            idx = seen[key]
            if _finding_sort_key(finding) > _finding_sort_key(merged[idx]):
                merged[idx] = finding
            continue
        seen[key] = len(merged)
        merged.append(finding)
    return merged, {
        "input_count": len(findings),
        "merged_count": len(merged),
        "deduped_count": dropped,
    }


def _bound_findings_for_synthesis(config: SwarmConfig, findings: list[SwarmFinding]) -> tuple[list[SwarmFinding], dict]:
    bounded: list[SwarmFinding] = []
    total_chars = 2  # opening/closing brackets for the JSON list
    dropped = 0
    for finding in findings:
        prompt_view = _public_finding(
            finding,
            subtask_chars=config.synthesis_subtask_chars,
            answer_chars=config.synthesis_answer_chars,
        )
        serialized = json.dumps(prompt_view, separators=(",", ":"), sort_keys=True)
        next_total = total_chars + len(serialized) + (1 if bounded else 0)
        if bounded and next_total > _MULTI_SEED_SYNTHESIS_MAX_CHARS:
            dropped += 1
            continue
        bounded.append(finding)
        total_chars = next_total
    return bounded, {
        "input_count": len(findings),
        "kept_count": len(bounded),
        "dropped_count": dropped,
        "prompt_chars": total_chars,
        "max_prompt_chars": _MULTI_SEED_SYNTHESIS_MAX_CHARS,
    }


def _run_single_seed(config: SwarmConfig, batch_fn: Callable[..., list[dict]], *, seed_index: int = 0,
                     seed_count: int = 1) -> dict:
    subtasks, decomposition = decompose_subtasks(config, batch_fn, seed_index=seed_index, seed_count=seed_count)
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
    consistent_findings, triage, self_consistency_report = self_consistency_findings(
        cheap_findings,
        triage,
        config=config,
        batch_fn=batch_fn,
    )
    final_findings, escalate_report = escalate_findings(consistent_findings, triage, config=config, batch_fn=batch_fn)
    return {
        "seed": seed_index,
        "subtasks": subtasks,
        "decomposition": decomposition,
        "cheap_report": cheap_report,
        "triage": triage,
        "consensus_report": consensus_report,
        "self_consistency_report": self_consistency_report,
        "final_findings": final_findings,
        "escalate_report": escalate_report,
    }


def _run_multi_seed(config: SwarmConfig, batch_fn: Callable[..., list[dict]],
                    guardian_fn: Callable[[str], dict]) -> dict:
    ordered: list[dict | None] = [None] * int(config.seeds)
    seed_errors: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(config.seeds))) as ex:
        futures = {
            ex.submit(_run_single_seed, config, batch_fn, seed_index=seed_idx, seed_count=int(config.seeds)): seed_idx
            for seed_idx in range(int(config.seeds))
        }
        for fut in concurrent.futures.as_completed(futures):
            seed_idx = futures[fut]
            try:
                ordered[seed_idx] = fut.result()
            except Exception as exc:
                seed_errors.append({"seed": seed_idx, "error": str(exc)})

    completed = [entry for entry in ordered if isinstance(entry, dict)]
    if not completed:
        return _run_single_seed_result(config, batch_fn, guardian_fn)

    merged_findings, merge_report = _merge_findings([
        finding
        for entry in completed
        for finding in entry["final_findings"]
    ])
    synthesis_findings, synthesis_bound = _bound_findings_for_synthesis(config, merged_findings)
    escalated_count = sum(1 for item in merged_findings if item.escalation_attempted)
    stats = {
        "fanout_count": len(merged_findings),
        "escalated_count": escalated_count,
        "escalation_rate": _coerce_ratio(escalated_count, len(merged_findings)),
        "seed_count": int(config.seeds),
        "seed_failures": len(seed_errors),
    }
    result = synthesize_swarm(config, synthesis_findings, stats, batch_fn, guardian_fn)
    result["findings"] = [_public_finding(item) for item in merged_findings]
    result["stats"] = stats
    result["decomposition"] = {
        "source": "multi-seed",
        "requested": config.fanout_count,
        "degraded": any(bool(entry["decomposition"].get("degraded")) for entry in completed),
        "seeds": [
            {
                "seed": entry["seed"],
                **entry["decomposition"],
            }
            for entry in completed
        ],
    }
    result["triage"] = {
        "flagged_indices": [],
        "flagged_count": sum(int(entry["triage"].get("flagged_count") or 0) for entry in completed),
        "escalation_reasons": {
            str(entry["seed"]): entry["triage"].get("escalation_reasons") or {}
            for entry in completed
        },
        "similar_pairs": [
            {
                "seed": entry["seed"],
                **pair,
            }
            for entry in completed
            for pair in (entry["triage"].get("similar_pairs") or [])
        ],
    }
    result["stigmergic_consensus"] = {
        "enabled": bool(config.stigmergic_consensus),
        "per_seed": {
            str(entry["seed"]): entry["consensus_report"]
            for entry in completed
        },
    }
    result["tiers"] = {
        "cheap": {
            "tier": "cheap",
            "model": config.cheap_model,
            "count": sum(int(entry["cheap_report"]["count"]) for entry in completed),
            "ok": sum(int(entry["cheap_report"]["ok"]) for entry in completed),
            "degraded": sum(int(entry["cheap_report"]["degraded"]) for entry in completed),
        },
        "escalate": {
            "attempted": sum(int(entry["escalate_report"]["attempted"]) for entry in completed),
            "succeeded": sum(int(entry["escalate_report"]["succeeded"]) for entry in completed),
            "failed_open": sum(int(entry["escalate_report"]["failed_open"]) for entry in completed),
            "model": config.escalate_model,
        },
    }
    if int(config.self_consistency_samples or 1) > 1:
        result["tiers"]["self_consistency"] = {
            "enabled": True,
            "samples": max(1, int(config.self_consistency_samples or 1)),
            "attempted": sum(int(entry["self_consistency_report"]["attempted"]) for entry in completed),
            "majority": sum(int(entry["self_consistency_report"]["majority"]) for entry in completed),
            "disagreement": sum(int(entry["self_consistency_report"]["disagreement"]) for entry in completed),
            "model": config.cheap_model,
        }
    result["lineage"] = [asdict(item) for item in merged_findings]
    result["multi_seed"] = {
        "enabled": True,
        "seed_count": int(config.seeds),
        "completed_seeds": len(completed),
        "failed_seeds": len(seed_errors),
        "errors": seed_errors,
        "merge": merge_report,
        "synthesis_input": synthesis_bound,
        "per_seed": [
            {
                "seed": entry["seed"],
                "subtask_count": len(entry["subtasks"]),
                "decomposition": entry["decomposition"],
                "cheap": entry["cheap_report"],
                "triage": entry["triage"],
                "consensus": entry["consensus_report"],
                "self_consistency": entry["self_consistency_report"],
                "escalate": entry["escalate_report"],
            }
            for entry in completed
        ],
    }
    if len(synthesis_findings) != len(merged_findings):
        result["synthesis"]["input_truncated"] = True
    return result


def synthesize_swarm(config: SwarmConfig, findings: list[SwarmFinding], stats: dict,
                     batch_fn: Callable[..., list[dict]],
                     guardian_fn: Callable[[str], dict] = _guardian_check) -> dict:
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
    prompt_findings, reduce_report = _reduce_findings_for_synthesis(config, prompt_findings, stats, batch_fn)
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
    used_think = False
    raw = {}
    if config.synthesize_think:
        # High-end pass: let the model reason before answering, with a budget generous
        # enough that reasoning traces don't starve the visible answer (see SwarmConfig's
        # synthesize_think comment for the live-measured "too small a budget" failure mode).
        raw = _call_one(
            batch_fn,
            prompt,
            model=config.synthesize_model,
            max_tokens=config.synthesize_think_max_tokens,
            fmt="json",
            think=True,
        )
        used_think = True
        obj = _extract_json_object(str(raw.get("text", ""))) if raw.get("ok") else None
        if not (isinstance(obj, dict) and str(obj.get("summary") or "").strip()):
            # The think=True attempt came back empty/unparseable (e.g. the reasoning
            # trace ate the whole budget on a hard prompt) -- fall back to a normal,
            # non-reasoning call at the usual budget rather than surface a degraded
            # result when a plain call would have worked fine.
            used_think = False
            raw = {}
    if not raw:
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
            "think": used_think,
            "ok": bool(summary),
            "degraded": not bool(summary),
        },
    }
    if reduce_report.get("enabled"):
        result["synthesis"]["reduce"] = reduce_report
    if raw.get("why"):
        result["synthesis"]["error"] = raw.get("why")
    if config.safety_check:
        result["safety_flag"] = _safety_flag(result["summary"], guardian_fn)
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
    guardian_fn = deps.get("guardian_check") or _guardian_check
    if int(config.seeds or 1) > 1:
        try:
            return _run_multi_seed(config, batch_fn, guardian_fn)
        except Exception:
            return _run_single_seed_result(config, batch_fn, guardian_fn)

    return _run_single_seed_result(config, batch_fn, guardian_fn)


def _run_single_seed_result(config: SwarmConfig, batch_fn: Callable[..., list[dict]],
                            guardian_fn: Callable[[str], dict] = _guardian_check) -> dict:
    seed = _run_single_seed(config, batch_fn)

    subtasks = seed["subtasks"]
    decomposition = seed["decomposition"]
    cheap_report = seed["cheap_report"]
    triage = seed["triage"]
    consensus_report = seed["consensus_report"]
    final_findings = seed["final_findings"]
    escalate_report = seed["escalate_report"]
    stats = {
        "fanout_count": len(subtasks),
        "escalated_count": int(escalate_report["attempted"]),
        "escalation_rate": _coerce_ratio(int(escalate_report["attempted"]), len(subtasks)),
    }
    result = synthesize_swarm(config, final_findings, stats, batch_fn, guardian_fn)
    result["decomposition"] = decomposition
    result["triage"] = triage
    result["stigmergic_consensus"] = consensus_report
    result["tiers"] = {
        "cheap": cheap_report,
        "escalate": escalate_report,
    }
    if int(config.self_consistency_samples or 1) > 1:
        result["tiers"]["self_consistency"] = seed["self_consistency_report"]
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
    ap.add_argument(
        "--escalate-model",
        default=None,
        help=(
            "Model for the escalation tier. Default: qwen3.8:latest, unless a new "
            "tier flag (--seeds>1, --synthesize-think, --safety-check, "
            "--self-consistency-samples>1, --reduce-group-size>0, "
            "--escalate-with-prior-context) is set, in which case it defaults to a "
            "faster benchmarked 8B model instead. An explicit value here always wins, "
            "even if it equals the legacy default."
        ),
    )
    ap.add_argument(
        "--synthesize-model",
        default=None,
        help=(
            "Model for the synthesis tier. Default: qwen3.8:latest, unless a new "
            "tier flag is set (see --escalate-model), in which case it defaults to a "
            "larger benchmarked model instead. An explicit value here always wins, "
            "even if it equals the legacy default."
        ),
    )
    ap.add_argument(
        "--decompose-model",
        default=_DEFAULT_DECOMPOSE_MODEL,
        help="Model for subtask decomposition. Default: cheap model when unset.",
    )
    ap.add_argument("--fanout-count", type=int, default=20)
    ap.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Independent swarm seeds to run in parallel before one merged synthesis. Default: 1",
    )
    ap.add_argument(
        "--escalate-confidences",
        default="low",
        help="Comma-separated confidences that force escalation. Default: low",
    )
    ap.add_argument("--subtasks-file", help="JSON array or newline-delimited subtasks.")
    ap.add_argument("--subtask", action="append", default=[], help="Repeatable pre-supplied subtask.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    ap.add_argument(
        "--synthesize-think",
        action="store_true",
        default=None,
        help=(
            "Enable Ollama reasoning ('think') for the single synthesis call, at a much "
            "larger token budget, with automatic fallback to a normal call if the "
            "reasoning attempt returns empty/unparseable. Default: env "
            "LOCI_SWARM_SYNTHESIZE_THINK, else off."
        ),
    )
    ap.add_argument(
        "--safety-check",
        action="store_true",
        default=None,
        help=(
            "Opt into an advisory Granite Guardian annotation on the final summary. "
            "Never blocks, drops, or rewrites findings. Default: env LOCI_SWARM_SAFETY_CHECK, else off."
        ),
    )
    ap.add_argument(
        "--self-consistency-samples",
        type=int,
        default=1,
        help=(
            "Opt into cheap-tier self-consistency for low-confidence subtasks before "
            "escalation. 1 preserves current behavior."
        ),
    )
    ap.add_argument(
        "--escalate-with-prior-context",
        action="store_true",
        default=False,
        help="Include the weaker cheap-tier answer in escalation prompts for critique-and-improve.",
    )
    ap.add_argument(
        "--reduce-group-size",
        type=int,
        default=0,
        help="Opt into hierarchical synthesis reduce groups. 0 preserves current flat synthesis.",
    )
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

    synthesize_think = (
        bool(args.synthesize_think) if args.synthesize_think is not None
        else _env_bool("LOCI_SWARM_SYNTHESIZE_THINK")
    )
    safety_check = (
        bool(args.safety_check) if args.safety_check is not None
        else _env_bool("LOCI_SWARM_SAFETY_CHECK")
    )
    seeds = max(1, int(args.seeds))
    self_consistency_samples = max(1, int(args.self_consistency_samples))
    reduce_group_size = max(0, int(args.reduce_group_size))
    escalate_with_prior_context = bool(args.escalate_with_prior_context)

    # Tier gating: resolve escalate_model/synthesize_model BEFORE constructing
    # SwarmConfig, using the raw --escalate-model/--synthesize-model CLI input (None
    # when the flag is unset). This is what lets an explicit --escalate-model
    # qwen3.8:latest survive even when --seeds/--safety-check/etc. are also set --
    # resolve_tier_models() never has to guess whether a value came from the caller
    # or from a default, because None only ever means "unset".
    tier_active = compute_tier_active(
        seeds=seeds,
        synthesize_think=synthesize_think,
        safety_check=safety_check,
        self_consistency_samples=self_consistency_samples,
        reduce_group_size=reduce_group_size,
        escalate_with_prior_context=escalate_with_prior_context,
    )
    escalate_model, synthesize_model = resolve_tier_models(
        escalate_model=args.escalate_model,
        synthesize_model=args.synthesize_model,
        tier_active=tier_active,
    )

    kwargs = dict(
        topic=args.topic,
        cheap_model=args.cheap_model,
        escalate_model=escalate_model,
        synthesize_model=synthesize_model,
        decompose_model=args.decompose_model,
        escalate_model_explicit=bool(str(args.escalate_model or "").strip()),
        synthesize_model_explicit=bool(str(args.synthesize_model or "").strip()),
        subtasks=supplied or None,
        fanout_count=max(1, int(args.fanout_count)),
        seeds=seeds,
        escalate_confidences=_split_csv(args.escalate_confidences) or ("low",),
        self_consistency_samples=self_consistency_samples,
        escalate_with_prior_context=escalate_with_prior_context,
        reduce_group_size=reduce_group_size,
        synthesize_think=synthesize_think,
        safety_check=safety_check,
        stigmergic_consensus=bool(args.stigmergic_consensus or _DEFAULT_STIGMERGIC_CONSENSUS),
        stigmergic_ttl_minutes=max(0.0, float(args.stigmergic_ttl_minutes)),
    )
    return SwarmConfig(**kwargs)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = _resolve_config(args)
    result = run_swarm(config)
    print(json.dumps(result, indent=2 if args.pretty else None))
    return 0 if not validate_swarm_result(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())

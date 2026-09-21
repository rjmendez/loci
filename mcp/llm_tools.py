"""Local-model and embedding MCP wrappers, split from server.py.

These tools hold no server state: register() only needs the shared FastMCP
instance, and server.py re-exports them for in-process callers and tests.
Sibling imports stay inside function bodies because the tool ``llm_local``
would shadow the sibling module at module scope.
"""
import json
import logging
import os
import sys
import threading
from importlib import util as importlib_util
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger("loci-mcp")
_SWARM_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "swarm_escalate.py"
_SWARM_MODULE_NAME = "_loci_scripts_swarm_escalate"
_SWARM_MODULE = None


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read int env config with bounded fail-open fallback."""
    try:
        value = int((os.environ.get(name, "") or "").strip() or default)
    except Exception:
        value = int(default)
    return max(minimum, min(maximum, value))


_SWARM_MAX_FANOUT = _env_int("LOCI_SWARM_MAX_FANOUT", 64, minimum=1, maximum=256)
_SWARM_MAX_SEEDS = _env_int("LOCI_SWARM_MAX_SEEDS", 8, minimum=1, maximum=32)
_SWARM_MAX_SELF_CONSISTENCY_SAMPLES = _env_int(
    "LOCI_SWARM_MAX_SELF_CONSISTENCY_SAMPLES", 5, minimum=1, maximum=16
)
_SWARM_MAX_REDUCE_GROUP_SIZE = _env_int(
    "LOCI_SWARM_MAX_REDUCE_GROUP_SIZE", 32, minimum=1, maximum=128
)
_SWARM_MAX_INFLIGHT = _env_int("LOCI_SWARM_MAX_INFLIGHT", 2, minimum=1, maximum=32)
_SWARM_INFLIGHT = threading.BoundedSemaphore(value=_SWARM_MAX_INFLIGHT)


def _bounded_int(value, *, default: int, minimum: int, maximum: int) -> tuple[int, int]:
    """Return (requested, bounded) integer pair."""
    try:
        requested = int(value)
    except Exception:
        requested = int(default)
    bounded = max(minimum, min(maximum, requested))
    return requested, bounded


def _env_bool(name: str) -> bool:
    """Mirror scripts/swarm_escalate.py's env-driven SwarmConfig defaults for
    synthesize_think/safety_check so tier-gating decisions made here (before a
    SwarmConfig even exists) match what the dataclass would resolve to on its own."""
    return os.environ.get(name, "") not in ("", "0", "false", "False")


def _coerce_labels(labels) -> list:
    """Normalize malformed tool input to a safe list without raising."""
    if labels is None:
        return []
    if isinstance(labels, list):
        return labels
    if isinstance(labels, (tuple, set)):
        return list(labels)
    return [labels]


def _load_swarm_escalate():
    global _SWARM_MODULE
    if _SWARM_MODULE is not None:
        return _SWARM_MODULE
    spec = importlib_util.spec_from_file_location(_SWARM_MODULE_NAME, _SWARM_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {_SWARM_SCRIPT_PATH}")
    module = importlib_util.module_from_spec(spec)
    sys.modules[_SWARM_MODULE_NAME] = module
    spec.loader.exec_module(module)
    _SWARM_MODULE = module
    return module


def _degraded_swarm_result(topic: str, *, fanout_count: int, error: str) -> dict:
    try:
        requested = max(0, int(fanout_count))
    except Exception:
        requested = 0
    cooked_topic = str(topic or "").strip() or "unknown topic"
    finding = {
        "subtask": cooked_topic,
        "answer": "",
        "confidence": "low",
        "tier_reached": "synthesized",
        "model": "",
        "ok": False,
        "why": "wrapper_exception",
    }
    return {
        "schema_version": 1,
        "topic": cooked_topic,
        "findings": [finding],
        "summary": f"Swarm reasoning degraded for '{cooked_topic}' before completion.",
        "stats": {
            "fanout_count": requested,
            "escalated_count": 0,
            "escalation_rate": 0.0,
        },
        "degraded": True,
        "synthesis": {
            "model": "",
            "ok": False,
            "degraded": True,
            "error": str(error or "unknown swarm wrapper error"),
        },
        "decomposition": {
            "source": "wrapper_fallback",
            "requested": requested,
            "degraded": True,
            "error": str(error or "unknown swarm wrapper error"),
        },
        "triage": {
            "flagged_indices": [0],
            "flagged_count": 1,
            "escalation_reasons": {"0": ["wrapper_exception"]},
            "similar_pairs": [],
        },
        "tiers": {
            "cheap": {"tier": "cheap", "model": "", "count": 1, "ok": 0, "degraded": 1},
            "escalate": {"attempted": 0, "succeeded": 0, "failed_open": 0, "model": ""},
        },
        "lineage": [{
            "subtask": cooked_topic,
            "answer": "",
            "confidence": "low",
            "tier_reached": "synthesized",
            "model": "",
            "ok": False,
            "parse_ok": False,
            "why": "wrapper_exception",
            "escalation_attempted": False,
            "escalation_reasons": ["wrapper_exception"],
        }],
    }


def llm_local(prompt: str, model: str = "", fmt: Optional[str] = None,
              max_tokens: int = 256, temperature: float = 0.2, keep_alive: str = "30m") -> str:
    """
    Generate with a local Ollama model for cheap high-volume work that should
    avoid Claude tokens. Leave ``model`` unset to use the configured generation
    model (``backends.ollama_gen_model()`` / ``[ollama].gen_model``) — hard-coding
    a specific tag here previously caused every unset-model call to silently
    request a tag that may not exist on the configured generation host, which
    fails open by falling through to a different (weaker/slower) tier instead
    of the one actually intended.

    ``keep_alive`` keeps the model resident (default ``'30m'``) to avoid the
    ~70s cold load; keep it long on hot paths. ``fmt='json'`` constrains and
    validates JSON output. Fail-open: errors, timeouts, or invalid JSON return
    ``ok=False`` so callers can fall back upstream.

    Returns JSON ``{text, ok, model}``.
    """
    import llm_local as _llm
    return json.dumps(_llm.generate(prompt, model=model, fmt=fmt, max_tokens=max_tokens,
                                    temperature=temperature, keep_alive=keep_alive), indent=2)


def generate_batch(prompts: list, model: Optional[str] = None, max_tokens: int = 256,
                   fmt: Optional[str] = None) -> str:
    """
    Generate many prompts at once for fan-out stages.

    Uses a batched OpenAI-compatible server (vLLM/TGI at ``VLLM_BASE_URL``)
    when configured; otherwise fails open to sequential Ollama via
    ``llm_local``. Returns a JSON list of ``{text, ok}`` aligned 1:1 to
    ``prompts``. Failed prompts return ``{text:'', ok:False}``; the tool does
    not raise.
    """
    import batched_gen
    return json.dumps(batched_gen.generate_batch(list(prompts or []), model=model,
                                                 max_tokens=max_tokens, fmt=fmt), indent=2)


def query_expand(query: str, n_queries: int = 3, n_keywords: int = 6) -> str:
    """
    Expand a search query with local-model paraphrases and domain keywords
    before embedding retrieval. Runs on the GPU with near-zero Claude-token
    cost. Fail-open: if the local model is unavailable, returns the original
    query with ``degraded=True``.

    Returns JSON ``{queries, keywords, degraded}``.
    """
    import query_expand as _qe
    return json.dumps(_qe.expand(query, n_queries=n_queries, n_keywords=n_keywords), indent=2)


def _finding_provenance_context(investigation_id: Optional[str], finding_id: Optional[str]):
    """Look up a stored finding's own provenance tier plus its investigation's
    other findings, to thread into ``verify.verify_finding``'s provenance
    firewall. Fail-open: any lookup problem (missing investigation/finding,
    corrupt storage) returns ``(None, None)`` so the caller falls back to
    verify_finding's own legacy-default behavior instead of raising or wrongly
    gating a claim it could not resolve.
    """
    if not investigation_id or not finding_id:
        return None, None
    try:
        from inv_store import _inv_dir, _read_jsonl
        from provenance_firewall import normalize_provenance_tier
        findings = _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl")
        target = None
        evidence_rows = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            if target is None and str(f.get("id") or "") == str(finding_id):
                target = f
            else:
                evidence_rows.append(f)
        if target is None:
            return None, None
        return normalize_provenance_tier(target), evidence_rows
    except Exception as exc:
        logger.debug("verify_finding: provenance lookup failed (fail-open): %r", exc)
        return None, None


def verify_finding(claim: str,
                   context: str = "",
                   investigation_id: Optional[str] = None,
                   finding_id: Optional[str] = None,
                   auto_promote_procedures: bool = False) -> str:
    """
    Adversarially verify a claim with a local-model skeptic: keep it only if the
    skeptic cannot refute it. Optional ``context`` can hold code, file refs, or
    other evidence. If ``context`` is empty and ``investigation_id`` is given,
    best-effort RAG grounding is pulled fail-open.

    Skeptical by default: returns ``confirmed`` only when the skeptic fails to
    refute the claim, otherwise ``refuted`` or ``uncertain``. If the model is
    unavailable or output is unparseable, returns ``verdict='uncertain'`` with
    ``degraded=True``.

    When ``investigation_id`` and ``finding_id`` identify a stored finding, its
    own provenance tier and the investigation's other findings are threaded
    into the provenance firewall, so a ``model_asserted`` finding with no
    independent (human/tool/deterministic) supporting evidence is reported
    ``uncertain`` instead of reaching the model verifier and possibly being
    confirmed on its own say-so.

    Pass ``auto_promote_procedures=True`` with both ``investigation_id`` and
    ``finding_id`` to opt into procedure auto-promotion for action-shaped
    confirmed findings. Default False preserves existing write behavior.

    Returns JSON ``{verdict, refutation, confidence, degraded}``.
    """
    import verify as _v
    candidate_provenance_tier, evidence_rows = _finding_provenance_context(investigation_id, finding_id)
    return json.dumps(_v.verify_finding(claim, context=context,
                                        investigation_id=investigation_id,
                                        finding_id=finding_id,
                                        candidate_provenance_tier=candidate_provenance_tier,
                                        evidence_rows=evidence_rows,
                                        auto_promote_procedures=auto_promote_procedures), indent=2)


def classify_text(text: str, labels: list) -> str:
    """
    Pick the best label from ``labels`` for ``text`` with the local model. This
    is a cheap gate/router in place of a classifier agent. Fail-open: returns
    ``label=None`` and ``degraded=True`` if the model is unavailable or emits an
    out-of-set label.

    Returns JSON ``{label, degraded}``.
    """
    import text_ops as _to
    return json.dumps(_to.classify(text, _coerce_labels(labels)), indent=2)


def compress_text(text: str, max_chars: int = 600) -> str:
    """
    Condense ``text`` to ``<= max_chars`` with the local model, e.g. before a
    Claude synthesis stage. Fail-open: if the model is unavailable, returns a
    character truncation with ``degraded=True``.

    Returns JSON ``{text, degraded}``.
    """
    import text_ops as _to
    return json.dumps(_to.compress(text, max_chars=max_chars), indent=2)


def semantic_dedup(items: list, threshold: float = 0.88, text_key: Optional[str] = None) -> str:
    """
    Cluster near-duplicate items by local embedding cosine similarity. No
    generation model is used, so token cost stays near zero. Use it after fanout
    so downstream synthesis sees one representative per cluster.

    ``items`` may be strings or dicts; dict text is pulled from ``text_key`` or
    from ``text/content/summary/title``. ``threshold`` is the duplicate floor
    (default ``0.88``; raise it to be stricter). Fail-open: if embeddings are
    unavailable, nothing is dropped and ``degraded=True``.

    Returns JSON ``{clusters:[{rep_index, member_indices, text}], kept:[...],
    dropped:int, degraded}``.
    """
    import embed_ops
    result = embed_ops.dedup(items or [], threshold=threshold, key=text_key)
    # Fail-open is preserved (nothing dropped), but make the silent degradation
    # observable: without embeddings, semantic_dedup returns every item unchanged.
    if result.get("degraded") and len(items or []) > 1:
        logger.warning("semantic_dedup degraded (embeddings unavailable) — %d items "
                       "returned unchanged, nothing deduped.", len(items or []))
    return json.dumps(result, indent=2)


def semantic_relevance(texts: list, topic: str) -> str:
    """
    Score each text's cosine relevance to ``topic`` on the local embedding path.
    Use it as a cheap gate/router before Claude. No generation model is used.

    Returns JSON ``{scores:[float|None], degraded}``; scores align with
    ``texts``. ``None`` means embeddings were unavailable and ``degraded=True``.
    """
    if not topic or not str(topic).strip():
        return json.dumps({"scores": [None] * len(texts or []), "degraded": True,
                           "error": "topic must not be empty"})
    import embed_ops
    return json.dumps(embed_ops.relevance(list(texts or []), topic), indent=2)


def ground(
    title: str,
    focus: str = "",
    case_ids: Optional[list] = None,
    entities: Optional[list] = None,
    code_refs: Optional[list] = None,
    budget_chars: int = 4000,
    allow_keyword: bool = False,
    graph_available: bool = False,
    mode: Literal["normal", "compact"] = "normal",
) -> str:
    """
    Build a compact, provenance-tagged grounding block for a task. Call it once
    before fan-out, then inject the block into every agent prompt so agents do
    not each re-query Loci.

    Retrieval is structured-first and embedding-independent where possible:
    named cases (``investigation_load``), exact entities
    (``investigation_entity_lookup``), code graph when ``graph_available``,
    semantic RAG, curated ``MEMORY.md``, then optional keyword FTS. Every lane
    is fail-open: dead sources set ``degraded=True`` instead of aborting.

    Prefer this to piecing together individual ``investigation_*`` or RAG calls
    when preparing a workflow. The block is read-only reference, not ground
    truth: consumers must still verify against live code/data and cite the
    ``[tag]`` they rely on.

    Args:
        title: Short task title (drives retrieval).
        focus: Optional longer task description.
        case_ids: Named investigation IDs to load.
        entities: Exact entity IDs to look up (O(1), no embedding).
        code_refs: Symbol names for code-graph grounding (used only if graph_available).
        budget_chars: Max characters in the assembled block (default 4000).
        allow_keyword: Enable the noisy keyword/FTS fallback lane (default off).
        graph_available: Enable the code-graph lane (default off; requires the
            LadybugDB graph).
        mode: "normal" (default) for the legacy block, or "compact" for terse tagged lines.

    Returns:
        JSON ``{block, sources, chars, degraded}``.
    """
    if not title or not title.strip():
        return json.dumps({"error": "title must not be empty",
                           "block": "", "sources": [], "chars": 0, "degraded": True})
    import grounding
    task = {
        "title": title, "focus": focus or "",
        "caseIds": case_ids or [], "entities": entities or [],
        "codeRefs": code_refs or [],
    }
    opts = {
        "budgetChars": budget_chars,
        "allowKeyword": allow_keyword,
        "graphAvailable": graph_available,
    }
    if mode == "compact":
        opts["mode"] = "compact"
    return json.dumps(grounding.ground(task, opts), indent=2)


def swarm_reason(topic: str,
                 fanout_count: int = 20,
                 seeds: int = 1,
                 cheap_model: str = "",
                 escalate_model: str = "",
                 synthesize_model: str = "",
                 decompose_model: str = "",
                 subtasks: Optional[list] = None,
                 escalate_confidences: Optional[list] = None,
                 synthesize_think: Optional[bool] = None,
                 safety_check: Optional[bool] = None,
                 self_consistency_samples: int = 1,
                 escalate_with_prior_context: bool = False,
                 reduce_group_size: int = 0,
                 stigmergic_consensus: bool = False,
                 stigmergic_ttl_minutes: float = 60.0) -> str:
    """
    Run the 4-stage local swarm reasoner: cheap fan-out, triage, selective
    escalation, then synthesis. Returns the structured JSON result with
    ``schema_version``, ``findings``, ``summary``, ``stats``, and diagnostics.

    seeds: opt-in number of independent decompose->cheap->triage->escalate rounds
        to run concurrently before one merged synthesis. Default 1 preserves the
        historic single-seed behavior exactly.

    synthesize_think: opt-in "high-end" mode for the single synthesis call -- enables
        Ollama reasoning at a much larger token budget, with automatic fallback to a
        normal call if the reasoning attempt returns empty/unparseable (live-verified:
        some thinking-capable models can burn an entire large budget on reasoning alone
        for some prompts). None -> SwarmConfig's own env-driven default (off unless
        LOCI_SWARM_SYNTHESIZE_THINK is set).

    safety_check: opt-in advisory Granite Guardian annotation on the final summary.
        It never blocks, drops, or rewrites findings. None -> SwarmConfig's own
        env-driven default (off unless LOCI_SWARM_SAFETY_CHECK is set).

    self_consistency_samples: opt-in cheap-tier majority sampling for low-confidence
        subtasks before escalation. 1 preserves historic behavior.

    escalate_with_prior_context: opt-in critique-and-improve escalation prompt that
        includes the weaker cheap-tier answer. Default False preserves prompts.

    reduce_group_size: opt-in hierarchical synthesis grouping for large fan-outs. 0
        preserves the flat synthesis prompt.

    stigmergic_consensus: opt-in cross-run stigmergic consensus gate over swarm
        findings. Default False preserves the historic single-run result.

    stigmergic_ttl_minutes: time-to-live for stigmergic consensus trail entries when
        stigmergic_consensus is enabled. Ignored otherwise.

    escalate_model / synthesize_model: leave empty ("") to use the existing default
        model for the current opt-in tier -- an empty value here is never treated as
        an explicit override, so it still upgrades correctly when seeds, self-
        consistency, safety_check, synthesize_think, escalate_with_prior_context, or
        reduce_group_size opt into the new reasoning tier (see swarm_escalate.py's
        SwarmConfig for the exact gating). Passing a non-empty model name here always
        wins over both the legacy default and the tier default.

    Fail-open: import/runtime/validation errors return degraded JSON instead of
    raising, so downstream MCP clients can still inspect one well-formed result.
    """
    if not _SWARM_INFLIGHT.acquire(blocking=False):
        return json.dumps(
            _degraded_swarm_result(
                topic,
                fanout_count=fanout_count,
                error=(
                    f"swarm_reason busy: global inflight limit {_SWARM_MAX_INFLIGHT} reached; "
                    "try again later"
                ),
            ),
            indent=2,
        )
    try:
        swarm = _load_swarm_escalate()
        requested_fanout, resolved_fanout = _bounded_int(
            fanout_count, default=20, minimum=1, maximum=_SWARM_MAX_FANOUT
        )
        requested_seeds, resolved_seeds = _bounded_int(
            seeds, default=1, minimum=1, maximum=_SWARM_MAX_SEEDS
        )
        requested_samples, resolved_samples = _bounded_int(
            self_consistency_samples,
            default=1,
            minimum=1,
            maximum=_SWARM_MAX_SELF_CONSISTENCY_SAMPLES,
        )
        requested_reduce_group_size, resolved_reduce_group_size = _bounded_int(
            reduce_group_size,
            default=0,
            minimum=0,
            maximum=_SWARM_MAX_REDUCE_GROUP_SIZE,
        )
        resolved_prior_context = bool(escalate_with_prior_context)
        resolved_synthesize_think = (
            bool(synthesize_think) if synthesize_think is not None
            else _env_bool("LOCI_SWARM_SYNTHESIZE_THINK")
        )
        resolved_safety_check = (
            bool(safety_check) if safety_check is not None
            else _env_bool("LOCI_SWARM_SAFETY_CHECK")
        )
        # Tier gating mirrors scripts/swarm_escalate.py's _resolve_config: the new 8B/27B
        # models only replace the legacy qwen3.8:latest default when the caller opts into
        # one of the new tier knobs, and an explicit escalate_model/synthesize_model
        # (even if it equals the legacy default) always wins over both defaults.
        tier_active = (
            resolved_seeds > 1
            or resolved_synthesize_think
            or resolved_safety_check
            or resolved_samples > 1
            or resolved_reduce_group_size > 0
            or resolved_prior_context
        )
        resolved_escalate_model = str(escalate_model or "").strip() or (
            swarm._TIER_ESCALATE_MODEL if tier_active else swarm._DEFAULT_ESCALATE_MODEL
        )
        resolved_synthesize_model = str(synthesize_model or "").strip() or (
            swarm._TIER_SYNTHESIZE_MODEL if tier_active else swarm._DEFAULT_SYNTHESIZE_MODEL
        )
        kwargs = dict(
            topic=str(topic or "").strip(),
            cheap_model=str(cheap_model or "") or swarm._DEFAULT_CHEAP_MODEL,
            escalate_model=resolved_escalate_model,
            synthesize_model=resolved_synthesize_model,
            decompose_model=str(decompose_model or ""),
            escalate_model_explicit=bool(str(escalate_model or "").strip()),
            synthesize_model_explicit=bool(str(synthesize_model or "").strip()),
            subtasks=_coerce_labels(subtasks) or None,
            fanout_count=resolved_fanout,
            seeds=resolved_seeds,
            auto_parallel=False,
            escalate_confidences=tuple(
                str(item).strip().lower()
                for item in _coerce_labels(escalate_confidences or ("low",))
                if str(item).strip()
            ) or ("low",),
            synthesize_think=resolved_synthesize_think,
            safety_check=resolved_safety_check,
            self_consistency_samples=resolved_samples,
            escalate_with_prior_context=resolved_prior_context,
            reduce_group_size=resolved_reduce_group_size,
            stigmergic_consensus=bool(stigmergic_consensus),
            stigmergic_ttl_minutes=max(0.0, float(stigmergic_ttl_minutes)),
        )
        config = swarm.SwarmConfig(**kwargs)
        result = swarm.run_swarm(config)
        errors = list(swarm.validate_swarm_result(result))
        if errors:
            raise ValueError("; ".join(errors))
        limits_applied = {}
        if resolved_fanout != requested_fanout:
            limits_applied["fanout_count"] = {"requested": requested_fanout, "applied": resolved_fanout}
        if resolved_seeds != requested_seeds:
            limits_applied["seeds"] = {"requested": requested_seeds, "applied": resolved_seeds}
        if resolved_samples != requested_samples:
            limits_applied["self_consistency_samples"] = {
                "requested": requested_samples,
                "applied": resolved_samples,
            }
        if resolved_reduce_group_size != requested_reduce_group_size:
            limits_applied["reduce_group_size"] = {
                "requested": requested_reduce_group_size,
                "applied": resolved_reduce_group_size,
            }
        if limits_applied:
            result["orchestration_limits"] = {
                "global_inflight_limit": _SWARM_MAX_INFLIGHT,
                "applied": limits_applied,
            }
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.warning("swarm_reason degraded for topic %r: %s", topic, exc)
        return json.dumps(
            _degraded_swarm_result(topic, fanout_count=fanout_count, error=str(exc)),
            indent=2,
        )
    finally:
        _SWARM_INFLIGHT.release()


def adversarial_review(findings: list,
                       mode: Literal["redteam", "gaps"] = "redteam",
                       context: str = "",
                       domain: str = "") -> str:
    """
    Adversarially review a SET of findings with a local red-team model. Two modes:
    ``redteam`` critiques each finding as an attacker (attack path, preconditions,
    impact, how to confirm); ``gaps`` runs one completeness pass over the whole set
    (missing attack surface, unverified claims, highest-value next probes).

    Routes to ``backends.ollama_redteam_model()`` — an uncensored/abliterated local
    model by default — because aligned models soften "attack this" prompts. For "is
    this ONE claim true?" use ``verify_finding`` instead; this tool deliberately does
    not reimplement refutation.

    Fail-open: an empty set or an unavailable model returns a well-formed result with
    ``degraded=True``; a single finding's failure never sinks the batch.

    Returns JSON: redteam -> ``{mode, model, results:[{finding, exploitable, attack,
    preconditions, impact, confirm, degraded}], degraded}``; gaps -> ``{mode, model,
    gaps, next_probes, summary, degraded}``.
    """
    import adversarial as _adv
    return json.dumps(_adv.adversarial_review(findings, mode=mode, context=context,
                                              domain=domain), indent=2)


def offload_tool_loop(task: str, allowed_tools: Optional[list] = None,
                      max_steps: int = 8, max_tool_calls: int = 8,
                      max_elapsed_s: float = 120.0, max_output_bytes: int = 32768,
                      model: str = "", investigation_id: Optional[str] = None,
                      dry_run: bool = False) -> str:
    """
    Let the LOCAL model work a multi-step, read-only tool loop so the calling cloud model
    does not spend tokens on the intermediate steps. The local model emits one JSON intent
    per turn; each is validated against a deny-by-default registry of read-only tools
    (investigation_search, investigation_entity_lookup, investigation_list,
    investigation_load, memory_health, code_graph_query), executed under budgets, and
    fed back as untrusted data. ``allowed_tools`` can only NARROW that set (as can the
    LOCI_OFFLOAD_TOOLS env var); ``investigation_id`` pins every call to one investigation.

    Budgets are clamped to hard ceilings (20 steps, 20 tool calls, 300 s, 256 KiB). No
    cloud model is ever called from inside the loop: a run that cannot finish returns
    ``status="fallback"`` with a compact ``handoff`` for the caller to continue from.
    Every run writes a JSONL audit trail (path in ``audit.path``). Fail-open: never raises.

    Returns JSON ``{status: done|fallback, reason, answer (done), handoff (fallback),
    steps, budget, metrics, audit, model, lane}``. ``answer`` is the local model's own
    unverified claim. ``metrics.est_tokens_*`` are bytes/4 ESTIMATES, not billed tokens.
    """
    import offload_loop as _ol
    return json.dumps(_ol.offload_entry(
        task, allowed_tools=allowed_tools, max_steps=max_steps,
        max_tool_calls=max_tool_calls, max_elapsed_s=max_elapsed_s,
        max_output_bytes=max_output_bytes, model=model, investigation_id=investigation_id,
        dry_run=dry_run), indent=2, default=str)


def register(mcp):
    """Register every local-model passthrough tool on the shared FastMCP instance."""
    for fn in (
        llm_local,
        generate_batch,
        query_expand,
        verify_finding,
        adversarial_review,
        classify_text,
        compress_text,
        semantic_dedup,
        semantic_relevance,
        ground,
        swarm_reason,
        offload_tool_loop,
    ):
        mcp.tool()(fn)

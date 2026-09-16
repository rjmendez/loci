#!/usr/bin/env python3
"""Local multi-tier reasoning chain for Loci.

This is the local-Ollama sibling of ``deep_think_loci``: slow but high quality and
private, with an option to use abliterated models for red-team evaluation. It mirrors
the same load-bearing shape:

Init -> Ideate (2-3 diverse local models, grounded by Loci RAG + the cosine gate) ->
Write (one dedicated persistence function stores ideas and returns real finding_ids) ->
Verify (reuse ``mcp/verify.py``) -> optional RedTeam -> Synthesize.

Why the dedicated writer matters: the Workflow version proved that multi-step agents
silently skip or fabricate store confirmations. Here, generation models never touch
``investigation_store`` directly; one Python function owns every write, so persistence
is deterministic and the returned finding_ids are real.

Usage:
  python3 scripts/local_deep_think.py "topic here"
  python3 scripts/local_deep_think.py "topic here" --collections loci_memory --red-team
  python3 scripts/local_deep_think.py "topic here" --ideate-models llama3.1-agent:latest,qwen3.8:latest
  python3 scripts/local_deep_think.py "topic here" --no-self-reflect
  python3 scripts/local_deep_think.py "topic here" --no-learn-procedures
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional


LOG = logging.getLogger("local_deep_think")
_MODEL_SAFE_RE = re.compile(r"[^a-z0-9]+")
_TEXT_KEYS = ("text", "snippet", "content", "chunk_text", "summary", "body", "passage")
_DEFAULT_IDEATE_MODELS = "llama3.1-agent:latest,qwen3.8:latest"
_DEFAULT_VERIFY_MODEL = "qwen3.8:latest"
_DEFAULT_SYNTH_MODEL = "qwen3.8:latest"
_DEFAULT_REFLECT_MODEL = "qwen3.8:latest"
_PROCEDURE_LEARNING_MIN_CONFIDENCE = 0.75


@dataclass
class ChainConfig:
    topic: str
    investigation_id: str
    title: str
    collections: list[str]
    ideate_models: list[str]
    verify_model: str
    synthesize_model: str
    self_reflect_model: str
    redteam_model: str
    learn_procedures: bool = True
    self_reflect: bool = True
    red_team: bool = False
    ideas_per_model: int = 3
    retrieval_limit: int = 8
    ground_threshold: float = 0.59
    evidence_chars: int = 4200
    max_lineage: int = 12


@dataclass
class Idea:
    claim: str
    rationale: str
    confidence: str
    evidence_ids: list[str]
    model: str


@dataclass
class StoredFinding:
    finding_id: str
    text: str
    tier: str
    model: str
    derived_from: list[str] = field(default_factory=list)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_paths() -> None:
    root = _repo_root()
    for entry in (root / "mcp", root / "deep_think_loci" / "grounding"):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)


def _runtime() -> SimpleNamespace:
    _ensure_paths()
    backends = importlib.import_module("backends")
    backends.load_env(_repo_root())
    return SimpleNamespace(
        backends=backends,
        llm_local=importlib.import_module("llm_local"),
        model_json=importlib.import_module("model_json"),
        qdrant_ops=importlib.import_module("qdrant_ops"),
        server=importlib.import_module("server"),
        verify=importlib.import_module("verify"),
        ground_gate=importlib.import_module("ground_gate"),
    )


def _backends_module() -> SimpleNamespace:
    """Import just ``backends``, not the full runtime.

    ``_runtime()`` eagerly imports ``server``/``verify``/``ground_gate`` etc.
    for the production dependency-injection path in ``run_chain``. Config
    lookups only need ``backends`` and must not drag in unrelated modules
    (``mcp/server.py`` requires ``python-dotenv``, which isn't installed in
    every environment this script's config helpers run in, e.g. minimal
    CI/test environments) or trip the broad ``except Exception`` below into
    silently returning the default.
    """
    _ensure_paths()
    mod = importlib.import_module("backends")
    mod.load_env(_repo_root())
    return mod


def _model_json_module():
    """Import just ``model_json``, for the same reason as ``_backends_module``."""
    _ensure_paths()
    return importlib.import_module("model_json")


def _cfg_value(key: str, default: str = "") -> str:
    try:
        return str(_backends_module()._cfg("ollama", key, default) or default)
    except Exception:
        return default


def _split_csv(raw: Optional[str]) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _safe_model_tag(model: str) -> str:
    text = _MODEL_SAFE_RE.sub("-", (model or "").lower()).strip("-")
    return text or "unknown-model"


def _slug(text: str, limit: int = 48) -> str:
    cooked = _MODEL_SAFE_RE.sub("-", (text or "").lower()).strip("-")
    cooked = re.sub(r"-{2,}", "-", cooked)
    return cooked[:limit].strip("-") or "topic"


def _default_investigation_id(topic: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"local-deep-think-{_slug(topic, 28)}-{stamp}"


def _resolve_models(args: argparse.Namespace) -> ChainConfig:
    ideate_models = _split_csv(
        args.ideate_models
        or os.environ.get("LOCI_LOCAL_DEEP_THINK_IDEATE_MODELS")
        or _cfg_value("deep_think_ideate_models", _DEFAULT_IDEATE_MODELS)
    )
    verify_model = (
        args.verify_model
        or os.environ.get("LOCI_LOCAL_DEEP_THINK_VERIFY_MODEL")
        or _cfg_value("deep_think_verify_model", _DEFAULT_VERIFY_MODEL)
    )
    synthesize_model = (
        args.synthesize_model
        or os.environ.get("LOCI_LOCAL_DEEP_THINK_SYNTHESIZE_MODEL")
        or _cfg_value("deep_think_synthesize_model", _DEFAULT_SYNTH_MODEL)
    )
    self_reflect_model = (
        args.self_reflect_model
        or os.environ.get("LOCI_LOCAL_DEEP_THINK_SELF_REFLECT_MODEL")
        or _cfg_value("deep_think_self_reflect_model", _DEFAULT_REFLECT_MODEL)
        or synthesize_model
    )
    try:
        redteam_model = (
            args.redteam_model
            or os.environ.get("LOCI_OLLAMA_REDTEAM_MODEL")
            or _backends_module().ollama_redteam_model()
        )
    except Exception:
        redteam_model = (
            args.redteam_model
            or os.environ.get("LOCI_OLLAMA_REDTEAM_MODEL")
            or "hf.co/slevinw/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M"
        )

    collections = _split_csv(
        args.collections
        or os.environ.get("LOCI_LOCAL_DEEP_THINK_COLLECTIONS")
        or _cfg_value("deep_think_collections", "loci_memory")
    ) or ["loci_memory"]

    return ChainConfig(
        topic=args.topic,
        investigation_id=args.investigation_id or _default_investigation_id(args.topic),
        title=args.title or f"Local deep think: {args.topic[:80]}",
        collections=collections,
        ideate_models=ideate_models or _split_csv(_DEFAULT_IDEATE_MODELS),
        verify_model=verify_model,
        synthesize_model=synthesize_model,
        self_reflect_model=self_reflect_model,
        redteam_model=redteam_model,
        learn_procedures=not bool(args.no_learn_procedures),
        self_reflect=not bool(args.no_self_reflect),
        red_team=bool(args.red_team),
        ideas_per_model=max(1, int(args.ideas_per_model)),
        retrieval_limit=max(1, int(args.retrieval_limit)),
        ground_threshold=float(args.ground_threshold),
        evidence_chars=max(1000, int(args.evidence_chars)),
        max_lineage=max(1, int(args.max_lineage)),
    )


def _parse_json(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        obj = json.loads(raw)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _call_generate(gen_fn: Callable, prompt: str, *, model: str,
                   fmt: str = "json", max_tokens: int = 700,
                   temperature: float = 0.2) -> dict:
    attempts = (
        {"prompt": prompt, "model": model, "fmt": fmt, "max_tokens": max_tokens, "temperature": temperature},
        {"prompt": prompt, "model": model, "fmt": fmt, "max_tokens": max_tokens},
        {"prompt": prompt, "fmt": fmt, "max_tokens": max_tokens},
    )
    for kw in attempts:
        try:
            return gen_fn(**kw)
        except TypeError:
            continue
        except Exception as exc:
            return {"text": "", "ok": False, "why": str(exc)}
    return {"text": "", "ok": False, "why": "generate signature mismatch"}


def _extract_json_object(text: str):
    try:
        return _model_json_module().extract_json_object(text)
    except Exception:
        return None


def _normalize_candidates(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        text = ""
        for key in _TEXT_KEYS:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        if not text:
            continue
        origin = str(row.get("origin") or row.get("collection") or "unknown")
        ident = str(row.get("id") or f"{origin}:{len(out)}")
        pair = (origin, ident)
        if pair in seen:
            continue
        seen.add(pair)
        out.append({
            "id": ident,
            "origin": origin,
            "score": float(row.get("score") or 0.0),
            "text": text,
        })
    return out


def retrieve_candidates(query: str, collections: list[str], limit: int,
                        search_fn: Callable[..., list[dict]]) -> tuple[list[dict], dict]:
    gathered: list[dict] = []
    report = {"collections": [], "errors": [], "hits": 0}
    for collection in collections:
        try:
            hits = search_fn(query=query, collection_name=collection, limit=limit)
            normalized = _normalize_candidates(hits)
            gathered.extend(normalized)
            report["collections"].append({"name": collection, "hits": len(normalized)})
        except Exception as exc:
            LOG.warning("retrieval skipped for %s: %s", collection, exc)
            report["errors"].append({"collection": collection, "error": str(exc)})
    deduped = _normalize_candidates(gathered)
    report["hits"] = len(deduped)
    return deduped, report


def make_search_fn(rt: SimpleNamespace) -> Callable[..., list[dict]]:
    direct = rt.qdrant_ops._qdrant_search_collection

    def _search(*, query: str, collection_name: str, limit: int) -> list[dict]:
        try:
            return direct(query=query, collection_name=collection_name, limit=limit)
        except Exception as exc:
            text = str(exc)
            if "qdrant_unavailable" not in text and "No module named 'qdrant_client'" not in text:
                raise
            return _qdrant_rest_search(rt, query=query, collection_name=collection_name, limit=limit)

    return _search


def _qdrant_rest_search(rt: SimpleNamespace, *, query: str, collection_name: str, limit: int) -> list[dict]:
    """HTTP fallback for thin hosts missing qdrant_client.

    The dedicated Qdrant helper in mcp/qdrant_ops.py is still the preferred path because it
    handles sparse fusion and cross-encoder reranking. This lightweight REST path exists so
    the standalone script stays genuinely runnable when the host has the repo code but not the
    optional qdrant_client wheel installed: retrieve dense hits, then let the existing cosine
    grounding gate drop off-topic bleed before any model reasons over them.
    """
    try:
        import requests
    except Exception as exc:  # pragma: no cover - requests exists in the live runtime
        raise RuntimeError(f"qdrant_requests_unavailable: {exc}") from exc

    url, api_key = rt.backends.qdrant()
    if not url:
        raise RuntimeError("qdrant_unavailable")
    vector = rt.qdrant_ops._embed(query)
    if not vector:
        raise RuntimeError("embedding_unavailable")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key

    using = None
    try:
        info = requests.get(f"{url.rstrip('/')}/collections/{collection_name}", headers=headers, timeout=10)
        info.raise_for_status()
        vectors = ((info.json().get("result") or {}).get("config") or {}).get("params", {}).get("vectors")
        if isinstance(vectors, dict):
            using = "dense" if "dense" in vectors else next(iter(vectors), None)
    except Exception as exc:
        LOG.warning("collection info lookup failed open for %s: %s", collection_name, exc)

    body = {
        "query": vector,
        "limit": max(limit * 3, limit),
        "with_payload": True,
    }
    if using:
        body["using"] = using
    resp = requests.post(
        f"{url.rstrip('/')}/collections/{collection_name}/points/query",
        headers=headers,
        json=body,
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json().get("result") or {}
    points = payload.get("points") if isinstance(payload, dict) else payload
    rows = []
    for point in points or []:
        point_payload = dict(point.get("payload") or {})
        rows.append({
            "id": str(point_payload.get("id") or point.get("id") or ""),
            "score": float(point.get("score") or 0.0),
            **point_payload,
            "origin": collection_name,
        })
    rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    return rows[:limit]


def gate_candidates(query: str, candidates: list[dict], threshold: float,
                    gate_fn: Callable[..., dict]) -> tuple[list[dict], dict]:
    if not candidates:
        return [], {"mode": "empty", "kept": 0, "dropped": 0}
    try:
        result = gate_fn(query, [{"id": c["id"], "text": c["text"]} for c in candidates], threshold)
        kept_ids = {str(item.get("id")) for item in (result.get("kept") or [])}
        kept = [c for c in candidates if c["id"] in kept_ids]
        return kept, {
            "mode": result.get("mode", f"cosine>={threshold}"),
            "kept": len(kept),
            "dropped": len(candidates) - len(kept),
        }
    except Exception as exc:
        LOG.warning("ground gate failed open: %s", exc)
        # Fail-open: keep the raw retrieval results rather than crashing the chain.
        return candidates, {
            "mode": f"failed-open-raw:{type(exc).__name__}",
            "kept": len(candidates),
            "dropped": 0,
        }


def _render_context(candidates: list[dict], max_chars: int) -> tuple[str, list[str]]:
    chunks: list[str] = []
    used_ids: list[str] = []
    total = 0
    for cand in candidates:
        piece = f"[{cand['id']}] ({cand['origin']} score={cand['score']:.3f}) {cand['text']}"
        if chunks and total + len(piece) + 2 > max_chars:
            break
        chunks.append(piece)
        used_ids.append(cand["id"])
        total += len(piece) + 2
    return "\n\n".join(chunks) if chunks else "(no retrieved evidence)", used_ids


def _normalize_confidence(value: str, default: str = "medium") -> str:
    cooked = str(value or "").strip().lower()
    return cooked if cooked in {"low", "medium", "high"} else default


def _coerce_score(value, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _normalize_ideas(obj: dict, model: str, limit: int) -> list[Idea]:
    ideas = obj.get("ideas") if isinstance(obj, dict) else None
    if not isinstance(ideas, list):
        return []
    out: list[Idea] = []
    for entry in ideas:
        if not isinstance(entry, dict):
            continue
        claim = str(entry.get("claim") or "").strip()
        if not claim:
            continue
        rationale = str(entry.get("rationale") or entry.get("why") or "").strip()
        evidence_ids = [str(item).strip() for item in (entry.get("evidence_ids") or []) if str(item).strip()]
        out.append(Idea(
            claim=claim,
            rationale=rationale,
            confidence=_normalize_confidence(entry.get("confidence")),
            evidence_ids=evidence_ids,
            model=model,
        ))
        if len(out) >= limit:
            break
    return out


def ideate(topic: str, *, candidates: list[dict], config: ChainConfig,
           gen_fn: Callable) -> tuple[list[Idea], list[dict]]:
    context, context_ids = _render_context(candidates, config.evidence_chars)
    results: list[Idea] = []
    reports: list[dict] = []
    for model in config.ideate_models:
        prompt = (
            "You are the IDEATE tier in a grounded reasoning chain.\n"
            f"Topic: {topic}\n"
            f"Return exactly {config.ideas_per_model} candidate findings or hypotheses.\n"
            "Use ONLY the evidence below. No persistence. No markdown.\n"
            'Return ONLY JSON: {"ideas":[{"claim":"...","rationale":"...","confidence":"low|medium|high","evidence_ids":["id"]}]}\n\n'
            f"EVIDENCE:\n{context}\n"
        )
        raw = _call_generate(gen_fn, prompt, model=model, max_tokens=900)
        obj = _extract_json_object(str(raw.get("text", ""))) if raw.get("ok") else None
        ideas = _normalize_ideas(obj or {}, model, config.ideas_per_model)
        if not ideas:
            why = raw.get("why") or "unparseable or empty ideation output"
            LOG.warning("ideate skipped for %s: %s", model, why)
            reports.append({"model": model, "ok": False, "ideas": 0, "error": why})
            continue
        for item in ideas:
            if not item.evidence_ids:
                item.evidence_ids = list(context_ids)
        results.extend(ideas)
        reports.append({"model": model, "ok": True, "ideas": len(ideas)})
    return results, reports


def _format_idea_text(idea: Idea) -> str:
    evidence = ", ".join(idea.evidence_ids) if idea.evidence_ids else "(none)"
    return (
        f"Generator model: {idea.model}\n"
        f"Claim: {idea.claim}\n"
        f"Rationale: {idea.rationale or '(none)'}\n"
        f"Grounding candidate ids: {evidence}"
    )


def persist_record(store_fn: Callable[..., str], *, investigation_id: str, text: str,
                   source: str, confidence: str = "medium", tags: Optional[list[str]] = None,
                   derived_from: Optional[list[str]] = None, finding_type: str = "inferred") -> Optional[str]:
    # One dedicated function owns ALL investigation_store writes. This is the same reliability
    # fix validated by deep_think_loci: generators reason, but only the writer persists, so a
    # model can never silently skip or fabricate a store confirmation.
    try:
        raw = store_fn(
            investigation_id=investigation_id,
            finding_type=finding_type,
            text=text,
            source=source,
            confidence=confidence,
            tags=",".join(tags or []),
            derived_from=derived_from or None,
        )
        obj = _parse_json(raw)
        if obj.get("stored") and obj.get("finding_id"):
            return str(obj["finding_id"])
        LOG.warning("store skipped: %s", obj.get("error") or raw)
    except Exception as exc:
        LOG.warning("store failed open: %s", exc)
    return None


def write_ideas(ideas: list[Idea], *, config: ChainConfig, store_fn: Callable[..., str]) -> list[StoredFinding]:
    stored: list[StoredFinding] = []
    for idea in ideas:
        model_tag = _safe_model_tag(idea.model)
        fid = persist_record(
            store_fn,
            investigation_id=config.investigation_id,
            text=_format_idea_text(idea),
            source=f"scripts/local_deep_think.py#ideate/{idea.model}",
            confidence=idea.confidence,
            tags=["local-deep-think", "ideate", f"model:{model_tag}"],
        )
        if fid:
            stored.append(StoredFinding(finding_id=fid, text=idea.claim, tier="ideate", model=idea.model))
    return stored


def load_ground_truth(load_fn: Callable[..., str], investigation_id: str,
                      expected_ids: list[str]) -> dict[str, dict]:
    try:
        raw = load_fn(investigation_id=investigation_id,
                      last_n_findings=max(20, len(expected_ids) * 4),
                      include_retracted=True)
        obj = _parse_json(raw)
        rows = obj.get("recent_findings") or obj.get("findings") or []
        wanted = set(expected_ids)
        return {str(row.get("id")): row for row in rows if str(row.get("id")) in wanted}
    except Exception as exc:
        LOG.warning("ground-truth reload failed open: %s", exc)
        return {}


def _bind_model(gen_fn: Callable, model: str) -> Callable[..., dict]:
    def _wrapped(prompt: str, *, fmt: Optional[str] = None, max_tokens: int = 256):
        return _call_generate(gen_fn, prompt, model=model, fmt=fmt or "json", max_tokens=max_tokens)
    return _wrapped


def _procedure_learning_gate(verdict: dict, *, enabled: bool) -> dict:
    confidence = _coerce_score(verdict.get("confidence"))
    status = str(verdict.get("verdict") or "uncertain")
    degraded = bool(verdict.get("degraded"))
    report = {
        "enabled": enabled,
        "eligible": False,
        "attempted": False,
        "verdict": status,
        "confidence": confidence,
        "threshold": _PROCEDURE_LEARNING_MIN_CONFIDENCE,
        "degraded": degraded,
    }
    if not enabled:
        report["reason"] = "disabled"
        return report
    if status != "confirmed":
        report["reason"] = "verdict_not_confirmed"
        return report
    if degraded:
        report["reason"] = "verification_degraded"
        return report
    if confidence < _PROCEDURE_LEARNING_MIN_CONFIDENCE:
        report["reason"] = "confidence_below_threshold"
        return report
    report["eligible"] = True
    report["reason"] = "eligible"
    return report


def maybe_learn_procedure(*, investigation_id: str, finding_id: str, verify_verdict: dict,
                          gen_fn: Callable, learn_fn: Optional[Callable[..., dict]] = None) -> dict:
    """Fail-open wrapper around procedure-learning promotion for verified findings."""
    hook = learn_fn
    if hook is None:
        try:
            hook = importlib.import_module("procedure_learning").maybe_promote_to_procedure
        except Exception as exc:
            LOG.warning("procedure learning unavailable for %s: %s", finding_id, exc)
            return {"promoted": False, "reason": "procedure_learning_unavailable", "degraded": True}
    try:
        result = hook(
            investigation_id=investigation_id,
            finding_id=finding_id,
            verify_verdict=verify_verdict,
            gen_fn=gen_fn,
        )
    except Exception as exc:
        LOG.warning("procedure learning failed open for %s: %s", finding_id, exc)
        return {"promoted": False, "reason": "unexpected_error", "degraded": True}
    if not isinstance(result, dict):
        LOG.warning("procedure learning returned non-dict for %s", finding_id)
        return {"promoted": False, "reason": "non_dict_result", "degraded": True}
    if result.get("degraded"):
        LOG.warning(
            "procedure learning degraded for %s: %s",
            finding_id,
            result.get("reason") or "degraded",
        )
    return result


def verify_findings(stored_ideas: list[StoredFinding], *, topic: str, config: ChainConfig,
                    search_fn: Callable[..., list[dict]], gate_fn: Callable[..., dict],
                    verify_fn: Callable[..., dict], gen_fn: Callable,
                    store_fn: Callable[..., str],
                    learn_fn: Optional[Callable[..., dict]] = None) -> tuple[list[StoredFinding], list[dict]]:
    survivors: list[StoredFinding] = []
    reports: list[dict] = []
    verify_gen = _bind_model(gen_fn, config.verify_model)
    for item in stored_ideas:
        query = f"{topic}\n{item.text}"
        raw_hits, retrieval = retrieve_candidates(query, config.collections, config.retrieval_limit, search_fn)
        gated_hits, gate = gate_candidates(item.text, raw_hits, config.ground_threshold, gate_fn)
        context, used_ids = _render_context(gated_hits, config.evidence_chars)
        verdict = verify_fn(item.text, context=context, gen_fn=verify_gen)
        status = str(verdict.get("verdict") or "uncertain")
        degraded = bool(verdict.get("degraded"))
        procedure_learning = _procedure_learning_gate(verdict, enabled=config.learn_procedures)
        if procedure_learning["eligible"]:
            learn_result = maybe_learn_procedure(
                investigation_id=config.investigation_id,
                finding_id=item.finding_id,
                verify_verdict=verdict,
                gen_fn=verify_gen,
                learn_fn=learn_fn,
            )
            procedure_learning = {
                **procedure_learning,
                **learn_result,
                "attempted": True,
            }
        reports.append({
            "finding_id": item.finding_id,
            "claim": item.text,
            "verdict": status,
            "degraded": degraded,
            "retrieval": retrieval,
            "gate": gate,
            "context_ids": used_ids,
            "procedure_learning": procedure_learning,
        })
        if status != "confirmed":
            continue
        evidence_note = ", ".join(used_ids) if used_ids else "(none)"
        fid = persist_record(
            store_fn,
            investigation_id=config.investigation_id,
            text=(
                f"Verified claim from {item.finding_id} (generator {item.model})\n"
                f"Claim: {item.text}\n"
                f"Verifier model: {config.verify_model}\n"
                f"Evidence ids: {evidence_note}\n"
                f"Refutation attempt: {verdict.get('refutation', '')}"
            ),
            source=f"scripts/local_deep_think.py#verify/{config.verify_model}",
            confidence="high" if not degraded else "medium",
            tags=["local-deep-think", "verify", f"model:{_safe_model_tag(config.verify_model)}"],
            derived_from=[item.finding_id],
        )
        if fid:
            survivors.append(StoredFinding(
                finding_id=fid,
                text=item.text,
                tier="verify",
                model=config.verify_model,
                derived_from=[item.finding_id],
            ))
    return survivors, reports


def _normalize_critiques(obj: dict) -> list[dict]:
    critiques = obj.get("critiques") if isinstance(obj, dict) else None
    if not isinstance(critiques, list):
        return []
    out = []
    for entry in critiques:
        if not isinstance(entry, dict):
            continue
        target = str(entry.get("target_finding_id") or "").strip()
        attack = str(entry.get("attack") or "").strip()
        if not target or not attack:
            continue
        out.append({
            "target_finding_id": target,
            "attack": attack,
            "severity": _normalize_confidence(entry.get("severity"), default="medium"),
        })
    return out


def red_team_findings(survivors: list[StoredFinding], *, config: ChainConfig,
                      gen_fn: Callable, store_fn: Callable[..., str]) -> tuple[list[StoredFinding], dict]:
    if not config.red_team:
        return [], {"enabled": False, "stored": 0}
    if not survivors:
        return [], {"enabled": True, "stored": 0, "skipped": "no surviving findings"}

    payload = [
        {"finding_id": item.finding_id, "claim": item.text, "model": item.model, "tier": item.tier}
        for item in survivors
    ]
    # Deliberately uses an abliterated/heretic model: aligned models tend to refuse or soften
    # explicit attack-framed critique prompts, and this is the one tier where hard adversarial
    # language is the whole point.
    prompt = (
        "You are the RED-TEAM tier. Attack each surviving finding or proposal as an adversary would.\n"
        "Do not soften, refuse, or reframe into generic safety advice. Focus on bypasses, missing "
        "preconditions, abuse paths, and how the claim could be wrong.\n"
        'Return ONLY JSON: {"critiques":[{"target_finding_id":"...","attack":"...","severity":"low|medium|high"}]}\n\n'
        f"TARGETS:\n{json.dumps(payload, indent=2)}"
    )
    raw = _call_generate(gen_fn, prompt, model=config.redteam_model, max_tokens=900)
    obj = _extract_json_object(str(raw.get("text", ""))) if raw.get("ok") else None
    critiques = _normalize_critiques(obj or {})
    if not critiques:
        return [], {
            "enabled": True,
            "stored": 0,
            "error": raw.get("why") or "unparseable or empty red-team output",
        }

    stored: list[StoredFinding] = []
    for entry in critiques:
        fid = persist_record(
            store_fn,
            investigation_id=config.investigation_id,
            text=(
                f"Red-team critique for {entry['target_finding_id']}\n"
                f"Target claim: {next((s.text for s in survivors if s.finding_id == entry['target_finding_id']), '')}\n"
                f"Model: {config.redteam_model}\n"
                f"Attack: {entry['attack']}"
            ),
            source=f"scripts/local_deep_think.py#red-team/{config.redteam_model}",
            confidence=entry["severity"],
            tags=["local-deep-think", "red-team", f"model:{_safe_model_tag(config.redteam_model)}"],
            derived_from=[entry["target_finding_id"]],
        )
        if fid:
            stored.append(StoredFinding(
                finding_id=fid,
                text=entry["attack"],
                tier="red-team",
                model=config.redteam_model,
                derived_from=[entry["target_finding_id"]],
            ))
    return stored, {"enabled": True, "stored": len(stored)}


def _normalize_synthesis(obj: dict) -> dict:
    if not isinstance(obj, dict):
        return {}
    summary = str(obj.get("summary") or obj.get("synthesis") or "").strip()
    if not summary:
        return {}
    support = [str(item).strip() for item in (obj.get("supporting_finding_ids") or []) if str(item).strip()]
    key_findings = []
    for entry in (obj.get("key_findings") or []):
        if isinstance(entry, dict):
            key_findings.append({
                "finding_id": str(entry.get("finding_id") or "").strip(),
                "why": str(entry.get("why") or "").strip(),
            })
    risks = [str(item).strip() for item in (obj.get("risks") or []) if str(item).strip()]
    next_steps = [str(item).strip() for item in (obj.get("next_steps") or []) if str(item).strip()]
    return {
        "summary": summary,
        "supporting_finding_ids": support,
        "key_findings": key_findings,
        "risks": risks,
        "next_steps": next_steps,
    }


def _render_synthesis_text(synthesis: dict) -> str:
    text = synthesis["summary"]
    if synthesis["key_findings"]:
        text += "\n\nKey findings:\n" + "\n".join(
            f"- {entry['finding_id']}: {entry['why']}" for entry in synthesis["key_findings"] if entry["finding_id"]
        )
    if synthesis["risks"]:
        text += "\n\nRisks:\n" + "\n".join(f"- {item}" for item in synthesis["risks"])
    if synthesis["next_steps"]:
        text += "\n\nNext steps:\n" + "\n".join(f"- {item}" for item in synthesis["next_steps"])
    return text


def _normalize_critique_lines(text: str) -> list[str]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    bullet_lines = [line for line in lines if line.lstrip().startswith(("-", "*"))]
    return (bullet_lines or lines)[:4]


def _self_reflect_synthesis(topic: str, *, synthesis: dict, evidence: list[dict],
                            config: ChainConfig, gen_fn: Callable) -> tuple[dict, dict]:
    report = {
        "enabled": True,
        "model": config.self_reflect_model,
        "iterations": 1,
        "revised": False,
    }
    critique_prompt = (
        "You are the CRITIQUE step in a bounded self-reflection loop.\n"
        f"Topic: {topic}\n"
        f"Synthesized answer:\n{_render_synthesis_text(synthesis)}\n\n"
        "Available evidence:\n"
        f"{json.dumps(evidence, indent=2)}\n\n"
        "What in this synthesized answer is weakest, least verified, most likely to be "
        "an unsupported claim, most likely to have a logical gap, or most likely to "
        "miss an important constraint? Be specific and cite finding_id values when visible. "
        "Reply with ONLY 2-4 short bullet lines of plain text, one concern per line."
    )
    critique_raw = _call_generate(
        gen_fn, critique_prompt, model=config.self_reflect_model, fmt="text", max_tokens=500
    )
    if not critique_raw.get("ok"):
        report["error"] = critique_raw.get("why") or "critique call failed"
        return synthesis, report
    critique_lines = _normalize_critique_lines(critique_raw.get("text", ""))
    if not critique_lines:
        report["error"] = "empty critique"
        return synthesis, report
    critique_text = "\n".join(critique_lines)
    report["critique"] = critique_text

    revise_prompt = (
        "You are the REVISE step in the same bounded self-reflection loop.\n"
        f"Topic: {topic}\n"
        "Revise the synthesized answer to fix the concrete problems in the critique while "
        "staying grounded in the available evidence. Do not invent new evidence, and keep "
        "the answer direct.\n"
        "Cite lineage by referencing finding_id, tier, and model in the summary itself.\n"
        'Return ONLY JSON: {"summary":"...","supporting_finding_ids":["id"],'
        '"key_findings":[{"finding_id":"id","why":"..."}],"risks":["..."],"next_steps":["..."]}\n\n'
        f"ORIGINAL SYNTHESIS:\n{json.dumps(synthesis, indent=2)}\n\n"
        f"CRITIQUE:\n{critique_text}\n\n"
        f"AVAILABLE EVIDENCE:\n{json.dumps(evidence, indent=2)}"
    )
    revise_raw = _call_generate(
        gen_fn, revise_prompt, model=config.self_reflect_model, max_tokens=1400
    )
    revised_obj = _extract_json_object(str(revise_raw.get("text", ""))) if revise_raw.get("ok") else None
    revised = _normalize_synthesis(revised_obj or {})
    if not revised:
        report["error"] = revise_raw.get("why") or "unparseable revise output"
        return synthesis, report
    report["revised"] = True
    return revised, report


def synthesize(topic: str, *, survivors: list[StoredFinding], critiques: list[StoredFinding],
               config: ChainConfig, gen_fn: Callable, store_fn: Callable[..., str]) -> dict:
    evidence = [
        {"finding_id": item.finding_id, "tier": item.tier, "model": item.model, "text": item.text,
         "derived_from": item.derived_from}
        for item in (survivors + critiques)
    ]
    prompt = (
        "You are the SYNTHESIZE tier in a grounded local reasoning chain.\n"
        "Produce the strongest grounded answer you can from the stored findings below.\n"
        "Cite lineage by referencing finding_id, tier, and model in the summary itself.\n"
        'Return ONLY JSON: {"summary":"...","supporting_finding_ids":["id"],'
        '"key_findings":[{"finding_id":"id","why":"..."}],"risks":["..."],"next_steps":["..."]}\n\n'
        f"TOPIC: {topic}\n\nEVIDENCE:\n{json.dumps(evidence, indent=2)}"
    )
    raw = _call_generate(gen_fn, prompt, model=config.synthesize_model, max_tokens=1400)
    obj = _extract_json_object(str(raw.get("text", ""))) if raw.get("ok") else None
    synthesis = _normalize_synthesis(obj or {})
    if not synthesis:
        summary = "No synthesis produced; synthesis tier degraded."
        return {
            "summary": summary,
            "finding_id": None,
            "supporting_finding_ids": [],
            "self_reflection": {"enabled": config.self_reflect, "revised": False},
        }

    reflection_report = {"enabled": False, "revised": False}
    source_suffix = f"synthesize/{config.synthesize_model}"
    if config.self_reflect:
        synthesis, reflection_report = _self_reflect_synthesis(
            topic, synthesis=synthesis, evidence=evidence, config=config, gen_fn=gen_fn
        )
        if reflection_report.get("revised"):
            source_suffix = f"self-reflect/{config.self_reflect_model}"
    final_model = config.self_reflect_model if reflection_report.get("revised") else config.synthesize_model
    tags = ["local-deep-think", "synthesize", f"model:{_safe_model_tag(final_model)}"]
    if reflection_report.get("revised"):
        tags.append("self-reflect")

    derived = synthesis["supporting_finding_ids"][:config.max_lineage]
    if not derived:
        derived = [item.finding_id for item in survivors[:config.max_lineage]]
    text = _render_synthesis_text(synthesis)
    fid = persist_record(
        store_fn,
        investigation_id=config.investigation_id,
        text=text,
        source=f"scripts/local_deep_think.py#{source_suffix}",
        confidence="high",
        tags=tags,
        derived_from=derived,
    )
    synthesis["finding_id"] = fid
    synthesis["self_reflection"] = reflection_report
    return synthesis


def run_chain(config: ChainConfig, deps: Optional[dict] = None) -> dict:
    rt = None if deps else _runtime()
    deps = deps or {
        "generate": rt.llm_local.generate,
        "search_collection": make_search_fn(rt),
        "gate": rt.ground_gate.gate,
        "start": rt.server.investigation_start,
        "load": rt.server.investigation_load,
        "store": rt.server.investigation_store,
        "verify": rt.verify.verify_finding,
    }

    start_obj = _parse_json(deps["start"](
        investigation_id=config.investigation_id,
        title=config.title,
        context=(
            "Standalone local deep-think chain: ideate with diverse local models, "
            "persist through a dedicated writer, verify with mcp/verify.py, "
            "optionally auto-learn procedures from confirmed findings, "
            "optionally red-team with an abliterated model, synthesize, then run "
            "one bounded critique+revise self-reflection pass."
        ),
    ))

    raw_hits, retrieval = retrieve_candidates(
        config.topic, config.collections, config.retrieval_limit, deps["search_collection"]
    )
    gated_hits, gate = gate_candidates(config.topic, raw_hits, config.ground_threshold, deps["gate"])
    ideas, ideate_reports = ideate(config.topic, candidates=gated_hits, config=config, gen_fn=deps["generate"])
    stored_ideas = write_ideas(ideas, config=config, store_fn=deps["store"])
    ground_truth = load_ground_truth(deps["load"], config.investigation_id, [item.finding_id for item in stored_ideas])
    survivors, verify_reports = verify_findings(
        stored_ideas, topic=config.topic, config=config, search_fn=deps["search_collection"],
        gate_fn=deps["gate"], verify_fn=deps["verify"], gen_fn=deps["generate"], store_fn=deps["store"],
        learn_fn=deps.get("learn_procedure"),
    )
    critiques, redteam_report = red_team_findings(
        survivors, config=config, gen_fn=deps["generate"], store_fn=deps["store"]
    )
    synthesis = synthesize(
        config.topic, survivors=survivors, critiques=critiques,
        config=config, gen_fn=deps["generate"], store_fn=deps["store"]
    )
    return {
        "investigation_id": config.investigation_id,
        "title": config.title,
        "topic": config.topic,
        "collections": config.collections,
        "start_status": start_obj.get("status"),
        "retrieval": retrieval,
        "ground_gate": gate,
        "ideate": {
            "reports": ideate_reports,
            "idea_count": len(ideas),
            "stored_count": len(stored_ideas),
            "stored_finding_ids": [item.finding_id for item in stored_ideas],
        },
        "verify": {
            "model": config.verify_model,
            "reports": verify_reports,
            "survivor_count": len(survivors),
            "survivor_finding_ids": [item.finding_id for item in survivors],
            "procedure_learning_enabled": config.learn_procedures,
        },
        "red_team": redteam_report,
        "synthesis": synthesis,
        "ground_truth_found": sorted(ground_truth),
        "lineage": [asdict(item) for item in survivors + critiques],
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Local-Ollama deep-think chain for Loci with fail-open procedure auto-learning."
    )
    ap.add_argument("topic", help="Topic/question to reason over.")
    ap.add_argument("--investigation-id")
    ap.add_argument("--title")
    ap.add_argument("--collections", help="Comma-separated Qdrant collections. Default: loci_memory")
    ap.add_argument("--ideate-models", help="Comma-separated ideation models.")
    ap.add_argument("--verify-model")
    ap.add_argument("--synthesize-model")
    ap.add_argument(
        "--self-reflect-model",
        help="Model for the bounded critique+revise pass. Default: qwen3.8:latest",
    )
    ap.add_argument(
        "--no-self-reflect",
        action="store_true",
        help="Disable the bounded critique+revise pass after synthesis.",
    )
    ap.add_argument("--redteam-model")
    ap.add_argument(
        "--no-learn-procedures",
        action="store_true",
        help="Disable auto-promotion of confirmed high-confidence action-shaped findings.",
    )
    ap.add_argument("--red-team", action="store_true", help="Enable the adversarial red-team tier.")
    ap.add_argument("--ideas-per-model", type=int, default=3)
    ap.add_argument("--retrieval-limit", type=int, default=8)
    ap.add_argument("--ground-threshold", type=float, default=0.59)
    ap.add_argument("--evidence-chars", type=int, default=4200)
    ap.add_argument("--max-lineage", type=int, default=12)
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = _resolve_models(args)
    result = run_chain(config)
    print(json.dumps(result, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Investigation lifecycle MCP tools — split out of server.py (P2b of the split).

The 11 tools here are the investigation lifecycle surface: create/load/share/export
and the read-side views over an investigation. Storage primitives come from
inv_store; the handful of server-side collaborators these tools still need
(graph upsert, lifecycle folding, event log, self-check, qdrant upsert) are
INJECTED through register() rather than imported, so this module never imports
server and no import cycle exists.

The memory root is injected the same way inv_store takes it — a lambda closing over
server's global — so tests that rebind it to a tmpdir keep steering these tools too.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from ladybug_ops import _ladybug_upsert_investigation
from inv_store import (
    _append_jsonl,
    _inv_dir,
    _load_manifest,
    _load_retracted_ids,
    _now,
    _read_jsonl,
    _save_manifest,
    _node_numeric_confidence,
    _NEUTRAL_NUMERIC_CONFIDENCE,
)

logger = logging.getLogger("loci-mcp")

# --- injected by register() -------------------------------------------------
_get_memory_dir = None
_apply_lifecycle = None
_compute_self_check = None
_event_log_append = None
_qdrant_upsert = None


def _root():
    """The investigation memory root, resolved through the injected accessor."""
    return _get_memory_dir()


def investigation_start(
    investigation_id: str,
    title: str,
    context: Optional[str] = None,
) -> str:
    """
    Create or resume an investigation. Call at the start of any session to
    initialize the manifest. Idempotent — resuming an existing ID returns
    the current manifest without overwriting it.

    Args:
        investigation_id: Short identifier — ticket number, case ID, or a
                          descriptive slug (e.g. "RQ41919026", "pww-actor-2026").
        title: One-line description of the investigation.
        context: Optional background to record on first creation only.

    Returns:
        JSON: {"status": "created"|"resumed", "manifest": {id, title, context, status,
               created_at, updated_at, hypothesis, open_questions, next_step,
               checked_sources, finding_counts, closed_at, closed_summary}}

        The investigation ID is at result["manifest"]["id"], NOT result["investigation_id"].
        Example extraction: inv_id = json.loads(result)["manifest"]["id"]
    """
    existing = _load_manifest(investigation_id)
    if existing:
        _ladybug_upsert_investigation(investigation_id, existing.get("title", ""))
        return json.dumps({"status": "resumed", "manifest": existing}, indent=2)

    manifest = {
        "id": investigation_id,
        "title": title,
        "context": context or "",
        "status": "active",
        "created_at": _now(),
        "updated_at": _now(),
        "hypothesis": None,
        "open_questions": [],
        "next_step": None,
        "checked_sources": {},
        "finding_counts": {"observed": 0, "inferred": 0, "assumed": 0, "gap": 0},
        "closed_at": None,
        "closed_summary": None,
        "owner": "",
        "acl": [],
        "summary_l1": [],
        "summary_l2": "",
    }
    _save_manifest(manifest)
    _ladybug_upsert_investigation(investigation_id, title)
    logger.info("Created investigation %s", investigation_id)
    return json.dumps({"status": "created", "manifest": manifest}, indent=2)


# Access rows are text-less AND the newest rows, so any findings[-N:] slice fills up with them.
_NON_FINDING_RECORD_TYPES = frozenset({"access"})


def _only_findings(records: list) -> list:
    """Drop the non-finding rows from a raw findings.jsonl read."""
    return [r for r in records
            if isinstance(r, dict)
            and (r.get("record_type") or r.get("type") or "") not in _NON_FINDING_RECORD_TYPES]


def investigation_load(
    investigation_id: str,
    last_n_findings: int = 20,
    include_retracted: bool = False,
    requesting_agent_id: Optional[str] = None,
    fidelity: str = "full",
) -> str:
    """
    Retrieve manifest and recent findings for an investigation.
    Use at session start to recover context without re-running all previous
    tool calls. The manifest contains hypothesis, open questions, checked
    sources, and next step — everything needed to resume cleanly.

    Soft-tombstoned (retracted) findings are excluded by default so a known
    hallucination and its contaminated lineage don't re-enter recall. The data
    is never lost — pass ``include_retracted=True`` to see them, and the count
    of excluded findings is always reported as ``excluded_retracted``.

    Args:
        investigation_id: Investigation identifier.
        last_n_findings: How many recent findings to include (default 20).
        include_retracted: Include soft-retracted findings (default False).
        requesting_agent_id: Optional agent_id of the requesting agent. When
                             provided and the investigation has a non-empty ACL,
                             findings are filtered to those authored by agents
                             in the ACL or by the requesting agent itself.
        fidelity: Controls how much detail is returned. One of:
                  "full"    — existing behavior: returns manifest + all recent findings.
                  "summary" — returns manifest + summary_l1 (bullets) + summary_l2
                              (paragraph) instead of full findings list. Useful when
                              context window is constrained.
                  "brief"   — returns manifest + summary_l2 only (single paragraph).
                              Most compact form; good for quick orientation.

    Returns:
        JSON with manifest, total finding count, recent findings, and
        ``excluded_retracted`` (count of findings filtered out).
        When fidelity is "summary" or "brief", the ``recent_findings`` key is
        omitted and replaced with ``summary_l1`` and/or ``summary_l2``.
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({
            "error": f"Investigation '{investigation_id}' not found. Call investigation_start first."
        })

    # Ensure summary fields exist (backwards-compatible with manifests created before this feature)
    summary_l1 = manifest.get("summary_l1") or []
    summary_l2 = manifest.get("summary_l2") or ""

    if fidelity == "brief":
        return json.dumps({
            "manifest": manifest,
            "fidelity": "brief",
            "summary_l2": summary_l2,
        }, indent=2)

    if fidelity == "summary":
        findings = _only_findings(_read_jsonl(_inv_dir(investigation_id) / "findings.jsonl"))
        all_retracted = _load_retracted_ids(investigation_id)
        total_retracted = len(all_retracted)
        retracted = set() if include_retracted else all_retracted
        excluded_retracted = 0
        if retracted:
            kept = [f for f in findings if str(f.get("id", "")) not in retracted]
            excluded_retracted = len(findings) - len(kept)
            findings = kept
        return json.dumps({
            "manifest": manifest,
            "fidelity": "summary",
            "total_findings": len(findings),
            "summary_l1": summary_l1,
            "summary_l2": summary_l2,
            "excluded_retracted": excluded_retracted,
            "total_retracted": total_retracted,
            "include_retracted": include_retracted,
        }, indent=2)

    findings = _only_findings(_read_jsonl(_inv_dir(investigation_id) / "findings.jsonl"))
    all_retracted = _load_retracted_ids(investigation_id)
    total_retracted = len(all_retracted)
    retracted = set() if include_retracted else all_retracted
    excluded_retracted = 0
    if retracted:
        kept = [f for f in findings if str(f.get("id", "")) not in retracted]
        excluded_retracted = len(findings) - len(kept)
        findings = kept

    acl = manifest.get("acl") or []
    if requesting_agent_id and acl:
        acl_set = set(acl)
        findings = [
            f for f in findings
            if f.get("authored_by", "") == requesting_agent_id
            or f.get("authored_by", "") in acl_set
        ]

    recent = findings[-last_n_findings:]
    # Append-log overrides win, else stored/default "open"; findings without these fields read open and not-stale.
    _apply_lifecycle(recent, investigation_id)

    payload = {
        "manifest": manifest,
        "fidelity": "full",
        "total_findings": len(findings),
        "recent_findings": recent,
        "excluded_retracted": excluded_retracted,
        "total_retracted": total_retracted,
        "include_retracted": include_retracted,
    }
    verifications = _verification_summary(investigation_id)
    if verifications:
        payload["verifications"] = verifications
    return json.dumps(payload, indent=2)


def _verification_summary(investigation_id: str) -> Optional[dict]:
    """Fold finding_verifications.jsonl into something a reader will actually see.

    investigation_verify_all writes adversarial verdicts to a separate log so they
    never bloat the findings scan (inv_store.py). The side effect was that NOTHING
    read them: a census of the tool-audit log found the verdicts' only other
    would-be consumer, memory_self_check, invoked 0 times, while
    investigation_store ran 299 times. A skeptic refuting a stored finding at 0.95
    confidence was landing in a file with no reader.

    This surfaces it where someone is already looking. Returns None when there are
    no verdicts, so the 140 investigations without any are unchanged.

    ``degraded`` verdicts are counted separately and NEVER as refutations: a
    degraded result means no model was reached, which is not a judgement about
    the finding.
    """
    try:
        rows = _read_jsonl(_inv_dir(investigation_id) / "finding_verifications.jsonl")
    except Exception as exc:
        logger.debug("verification summary unreadable for %s: %r", investigation_id, exc)
        return None
    if not rows:
        return None

    # Append-only: last verdict per finding wins, same rule as finding_updates.
    latest: dict = {}
    for r in rows:
        if isinstance(r, dict) and r.get("finding_id"):
            latest[str(r["finding_id"])] = r

    counts = {"refuted": 0, "confirmed": 0, "uncertain": 0, "degraded": 0}
    refuted: list[dict] = []
    for fid, r in latest.items():
        if r.get("degraded"):
            counts["degraded"] += 1
            continue
        verdict = str(r.get("verdict") or "uncertain")
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict == "refuted":
            try:
                conf = float(r.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            refuted.append({"finding_id": fid, "confidence": conf, "ts": r.get("ts")})

    refuted.sort(key=lambda x: -x["confidence"])
    out = {"counts": counts, "verified_findings": len(latest)}
    if refuted:
        out["refuted"] = refuted
        out["hint"] = ("adversarial verdicts, advisory only — they do NOT change a "
                       "finding's resolution. Review, then finding_resolve if warranted.")
    return out


def investigation_as_of(
    investigation_id: str,
    as_of_timestamp: str,
) -> str:
    """
    Return findings from an investigation as they were believed at a specific point in time.

    A finding is included when BOTH of the following hold:
      - created_at_ts <= as_of_epoch  (the finding existed by that moment)
      - valid_until is null OR valid_until >= as_of_timestamp  (it was still believed valid)

    This supports bi-temporal analysis: you can reconstruct the investigation's
    knowledge state at any historical moment, even after findings have been
    superseded or retracted.

    Args:
        investigation_id: Investigation identifier.
        as_of_timestamp: ISO8601 timestamp (e.g. "2024-01-15T10:30:00+00:00").
                         Findings created after this moment are excluded, and
                         findings whose valid_until is before this moment are
                         also excluded.

    Returns:
        JSON: {
          "investigation_id": "<id>",
          "as_of": "<as_of_timestamp>",
          "findings": [...],
          "count": <int>
        }
        On error: {"error": "<message>"}
    """
    try:
        manifest = _load_manifest(investigation_id)
        if not manifest:
            return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

        try:
            as_of_dt = datetime.fromisoformat(as_of_timestamp)
            # Make timezone-aware if naive (assume UTC)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
            as_of_epoch = int(as_of_dt.timestamp())
        except (ValueError, TypeError) as exc:
            return json.dumps({"error": f"Invalid as_of_timestamp: {exc}"})

        findings_path = _inv_dir(investigation_id) / "findings.jsonl"
        all_findings = _read_jsonl(findings_path)

        result_findings = []
        for f in all_findings:
            created_at_ts = f.get("created_at_ts")
            if created_at_ts is None:
                # Older findings without created_at_ts: include them (fail-open)
                pass
            elif int(created_at_ts) > as_of_epoch:
                continue

            valid_until = f.get("valid_until")
            if valid_until is not None:
                try:
                    vu_dt = datetime.fromisoformat(str(valid_until))
                    if vu_dt.tzinfo is None:
                        vu_dt = vu_dt.replace(tzinfo=timezone.utc)
                    if vu_dt < as_of_dt:
                        continue
                except (ValueError, TypeError) as exc:
                    logger.debug("investigation_as_of: valid_until parse failed (fail-open): %r", exc)
                    pass  # fail-open: include if valid_until can't be parsed

            result_findings.append(f)

        return json.dumps({
            "investigation_id": investigation_id,
            "as_of": as_of_timestamp,
            "findings": result_findings,
            "count": len(result_findings),
        }, indent=2)
    except Exception as exc:
        logger.exception("investigation_as_of failed")
        return json.dumps({"error": f"investigation_as_of failed: {exc}"})


def investigation_note(
    investigation_id: str,
    field: str,
    value: str,
) -> str:
    """
    Update a manifest field for the investigation. Use to track the working
    hypothesis, next action, open questions, and which sources have been checked.

    Args:
        investigation_id: Investigation identifier.
        field: One of:
               context             — overwrite the investigation context (corrects stale
                                     framing set at creation time).
               hypothesis          — current working hypothesis (overwrite).
               next_step           — recommended next action (overwrite).
               open_question_add   — append a question to the open list.
               open_question_remove — remove a question from the open list.
               checked_source      — mark a source as checked; format as
                                     "tool_name: one-line summary of what was found".
               closed_summary      — close the investigation with a final summary.

    Returns:
        JSON with the updated manifest.
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    if field in ("context", "hypothesis", "next_step"):
        stripped = value.strip() if value else ""
        if not stripped:
            return json.dumps({"error": f"Field '{field}' must not be empty or whitespace-only."})
        manifest[field] = stripped
        manifest[f"{field}_ts"] = _now()
    elif field == "open_question_add":
        if value not in manifest["open_questions"]:
            manifest["open_questions"].append(value)
    elif field == "open_question_remove":
        manifest["open_questions"] = [q for q in manifest["open_questions"] if q != value]
    elif field == "checked_source":
        parts = value.rsplit(":", 1)
        tool = parts[0].strip()
        summary = parts[1].strip() if len(parts) > 1 else ""
        if not summary:
            return json.dumps({"error": "checked_source summary must not be empty"})
        manifest["checked_sources"].setdefault(tool, []).append({"summary": summary, "ts": _now()})
    elif field == "closed_summary":
        manifest["closed_summary"] = value
        manifest["status"] = "closed"
        manifest["closed_at"] = _now()
    else:
        return json.dumps({
            "error": (
                f"Unknown field '{field}'. Valid: context, hypothesis, next_step, "
                "open_question_add, open_question_remove, checked_source, closed_summary"
            )
        })

    _save_manifest(manifest)
    _event_log_append({
        "op": "note",
        "investigation_id": investigation_id,
        "field": field,
    })
    return json.dumps({"updated": field, "manifest": manifest}, indent=2)


def investigation_reflect(investigation_id: str) -> str:
    """
    Synthesize the current state of an investigation. Returns a structured
    summary of what has been established, what is still open, and what has
    not been checked. Call before write actions, at handoff points, or when
    context is growing long.

    Args:
        investigation_id: Investigation identifier.

    Returns:
        JSON reflection: finding breakdown, open questions, gaps, hypothesis,
        checked vs unchecked sources, and most recent findings per type.
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    findings = _only_findings(_read_jsonl(_inv_dir(investigation_id) / "findings.jsonl"))
    retracted = _load_retracted_ids(investigation_id)
    excluded_retracted = 0
    if retracted:
        kept = [f for f in findings if str(f.get("id", "")) not in retracted]
        excluded_retracted = len(findings) - len(kept)
        findings = kept

    by_type: dict[str, list] = {"observed": [], "inferred": [], "assumed": [], "gap": []}
    for f in findings:
        by_type.setdefault(f.get("type", "observed"), []).append(f)

    # Fail-open: degrades to empty lists, never blocks reflect.
    checks = _compute_self_check(investigation_id)
    self_check = {
        "unsupported_observed": [
            {
                "refs": v.refs,
                "excerpt": v.subject_excerpt,
                "rationale": v.rationale,
                "decision": v.decision,
                "confidence": v.confidence,
            }
            for v in checks["unsupported_observed"]
        ],
        "contradictions": [
            {
                "refs": v.refs,
                "excerpt": v.subject_excerpt,
                "rationale": v.rationale,
                "decision": v.decision,
                "confidence": v.confidence,
            }
            for v in checks["contradictions"]
        ],
        "hallucination_candidates": checks.get("hallucination_candidates", []),
    }

    entity_counts: dict[str, dict[str, int]] = {
        "ips": {}, "emails": {}, "hostnames": {}, "hashes": {}, "cves": {}
    }
    for f in findings:
        for etype, vals in (f.get("entities") or {}).items():
            if etype in entity_counts:
                for v in (vals or []):
                    v = str(v).lower()
                    if v:
                        entity_counts[etype][v] = entity_counts[etype].get(v, 0) + 1
    key_entities = {
        etype: sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
        for etype, freq in entity_counts.items()
        if freq
    }

    # Persisted to the manifest so investigation_load need not re-read findings; fail-open to the deterministic summary.
    summary_l1: list[str] = []
    summary_l2: str = ""
    try:
        last_20 = findings[-20:]
        try:
            from memcheck import llm as _llm
            if _llm.llm_available() and last_20:
                context_bullets = "\n".join(
                    f"- [{f.get('type', '?')}] {str(f.get('text', ''))[:300]}"
                    for f in last_20
                )
                l1_prompt = (
                    f"Investigation: {manifest['title']}\n"
                    f"Recent findings (up to 20):\n{context_bullets}\n\n"
                    "Produce exactly 5-7 concise key-point bullet strings that capture "
                    "the most important things known so far. Each bullet should be a "
                    "single sentence under 120 characters. Reply with ONLY a JSON array "
                    "of strings, no other text. Example: [\"First point.\", \"Second point.\"]"
                )
                # json_mode=False on purpose: Ollama's format=json coerces the reply to an OBJECT, and this prompt needs a bare ARRAY.
                l1_raw = _llm.call_llm(l1_prompt, timeout=60.0)
                if l1_raw:
                    try:
                        parsed = json.loads(l1_raw)
                        # Accept a bare array, or one wrapped in a single-key object.
                        if isinstance(parsed, dict):
                            for _v in parsed.values():
                                if isinstance(_v, list):
                                    parsed = _v
                                    break
                        if isinstance(parsed, list):
                            summary_l1 = [str(b) for b in parsed if str(b).strip()][:7]
                    except Exception as _l1_exc:
                        logger.debug("investigation_reflect: L1 summary JSON parse failed (fail-open): %r", _l1_exc)

                if summary_l1:
                    l2_prompt = (
                        f"Investigation: {manifest['title']}\n"
                        f"Key points:\n" + "\n".join(f"- {b}" for b in summary_l1) + "\n\n"
                        "Write a 2-3 sentence 'state of knowledge' paragraph summarising "
                        "what is established, what is uncertain, and what remains to check. "
                        "Be direct and concise. Reply with only the paragraph text."
                    )
                    l2_raw = _llm.call_llm(l2_prompt, timeout=60.0)
                    if l2_raw:
                        summary_l2 = l2_raw.strip()
        except Exception as exc:
            logger.debug("investigation_reflect: LLM summary ladder failed (fail-open): %r", exc)

        # Deterministic fallback when LLM is unavailable or failed to produce output
        if not summary_l1:
            summary_l1 = [
                str(f.get("text", ""))[:100]
                for f in findings[-5:]
                if str(f.get("text", "")).strip()
            ]
        if not summary_l2:
            n = len(findings)
            latest_text = str(findings[-1].get("text", "")) if findings else ""
            summary_l2 = (
                f"Investigation with {n} finding{'s' if n != 1 else ''}."
                + (f" Latest: {latest_text[:200]}" if latest_text else "")
            )

        # Persist to manifest (write-through cache via _save_manifest)
        manifest["summary_l1"] = summary_l1
        manifest["summary_l2"] = summary_l2
        _save_manifest(manifest)
    except Exception as exc:
        logger.debug("investigation_reflect: summary ladder persist failed (fail-open): %r", exc)  # fail-open: summary generation never breaks reflect

    return json.dumps({
        "investigation_id": investigation_id,
        "title": manifest["title"],
        "status": manifest["status"],
        "hypothesis": manifest["hypothesis"],
        "next_step": manifest["next_step"],
        "finding_counts": {t: len(v) for t, v in by_type.items()},
        "open_questions": manifest["open_questions"],
        "checked_sources": manifest["checked_sources"],
        "gaps": [f["text"] for f in by_type.get("gap", [])],
        "recent_per_type": {t: entries[-3:] for t, entries in by_type.items() if entries},
        "key_entities": key_entities,
        "excluded_retracted": excluded_retracted,
        "self_check": self_check,
        "summary_l1": summary_l1,
        "summary_l2": summary_l2,
    }, indent=2)


def investigation_finding_provenance(
    finding_id: str,
    investigation_id: str,
) -> str:
    """
    Trace a finding back through its derivation chain to root observed evidence.

    Follows the ``derived_from`` links stored on each finding, walking up the
    chain until it reaches findings with no parent (root observations) or a
    cycle is detected. Returns the full chain so the analyst can verify that
    an inference is actually grounded in observed data and not built on top of
    another inference or assumption.

    A chain that terminates in an ``assumed`` finding — rather than ``observed``
    data — means the conclusion is a hypothesis chain, not an evidence chain.

    Args:
        finding_id: The ID of the finding to trace.
        investigation_id: The investigation containing the finding.

    Returns:
        JSON with the chain from the target finding to its root evidence,
        each node annotated with its type, confidence, source, and text.
    """
    # Not _inv_dir(): it creates the directory, so a bad id would silently leave an empty investigation behind.
    inv_path = _root() / investigation_id
    if not inv_path.is_dir():
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    findings_by_id: dict[str, dict] = {
        str(f.get("id", "")): f
        for f in _read_jsonl(inv_path / "findings.jsonl")
        if f.get("id")
    }

    target = findings_by_id.get(finding_id)
    if not target:
        return json.dumps({"error": f"Finding '{finding_id}' not found in '{investigation_id}'"})

    chain: list[dict] = []
    visited: set[str] = set()
    current_id = finding_id
    _MAX_CHAIN_DEPTH = 5

    while current_id and current_id not in visited and len(chain) < _MAX_CHAIN_DEPTH:
        visited.add(current_id)
        node = findings_by_id.get(current_id)
        if not node:
            chain.append({"finding_id": current_id, "error": "not_found"})
            break
        chain.append({
            "finding_id": current_id,
            "ts": node.get("ts"),
            "record_type": node.get("record_type") or node.get("type"),
            "confidence": node.get("confidence"),
            "numeric_confidence": _node_numeric_confidence(node),
            "source": node.get("source"),
            "text": str(node.get("text", ""))[:400],
            "derived_from": node.get("derived_from", []),
        })
        # Only the first parent is followed: grounded_in_observed reflects the first-listed branch only.
        parents = node.get("derived_from") or []
        if not parents:
            break
        current_id = str(parents[0]) if parents else None

    root = chain[-1] if chain else None
    root_type = root.get("record_type") if root else None
    grounded = root_type == "observed"

    # A finding without numeric_confidence resolves from its own confidence label
    # (_node_numeric_confidence); an unstamped record is not worth 1.0.
    try:
        aggregate_confidence = 1.0
        for node_entry in chain:
            if "error" not in node_entry:
                nc = node_entry.get("numeric_confidence", _NEUTRAL_NUMERIC_CONFIDENCE)
                try:
                    aggregate_confidence *= float(nc)
                except (TypeError, ValueError) as exc:
                    logger.debug("investigation_finding_provenance: fail-open swallow: %r", exc)
        aggregate_confidence = round(aggregate_confidence, 6)
    except Exception:
        aggregate_confidence = None

    return json.dumps({
        "investigation_id": investigation_id,
        "chain_length": len(chain),
        "grounded_in_observed": grounded,
        "grounding_assessment": (
            "fully grounded" if grounded
            else f"chain terminates in '{root_type}' — not directly observed evidence"
        ),
        "aggregate_confidence": aggregate_confidence,
        "chain": chain,
    }, indent=2, default=str)


def investigation_list(
    limit: int = 30,
    offset: int = 0,
    summary: bool = True,
) -> str:
    """
    List investigations with status and finding counts, most recently
    updated first.

    Bounded by default to avoid overflowing the tool-result token cap: only
    `limit` investigations are returned starting at `offset`, and `summary`
    mode returns a compact record per investigation (id, title, status,
    finding_counts, updated_at). Set summary=False for the full record
    (created_at, open_questions_count, hypothesis, visibility, tier_counts)
    and/or raise limit to page through or fetch everything.

    Args:
        limit: Max investigations to return (default 30). Use 0 or a negative
            value for no limit (return all remaining). Note: offset is still
            honored when limit<=0, so this returns everything *starting at*
            offset, not the entire list.
        offset: Number of investigations to skip from the front (default 0).
            Always applied, including when limit<=0.
        summary: If True (default), return only compact fields; if False,
            return the full record including tier counts.

    Returns:
        JSON: {"investigations": [...], "total": N, "limit": ..., "offset": ...}
    """
    # Coerce BEFORE any early return, so the empty-root path and the normal path echo the same normalized ints.
    try:
        offset = max(0, int(offset))
    except (TypeError, ValueError):
        offset = 0
    # None collapses into the documented limit<=0 no-limit case; garbage falls back to the default.
    if limit is None:
        limit = 0
    else:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 30

    # None and '' must preserve summary=True: only 'false'/'0'/'no' may select the overflow-prone full mode.
    if summary is None:
        summary = True
    elif isinstance(summary, str):
        summary = summary.strip().lower() not in ("false", "0", "no")
    else:
        summary = bool(summary)

    if not _root().exists():
        return json.dumps({"investigations": [], "total": 0, "limit": limit, "offset": offset})

    # Filter to real investigation dirs first: pagination is over investigations, and the findings scan only runs for the page returned.
    entries = []
    for d in sorted(_root().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        manifest = _load_manifest(d.name)
        if manifest:
            entries.append((d, manifest))

    total = len(entries)

    if limit <= 0:
        page = entries[offset:]
    else:
        page = entries[offset:offset + limit]

    investigations = []
    for d, manifest in page:
        record = {
            "id": manifest["id"],
            "title": manifest["title"],
            "status": manifest["status"],
            "updated_at": manifest["updated_at"],
            "finding_counts": manifest["finding_counts"],
        }
        if not summary:
            tier_counts = {"hot": 0, "warm": 0, "cold": 0}
            try:
                findings_path = d / "findings.jsonl"
                for f in _read_jsonl(findings_path):
                    t = f.get("tier", "warm")
                    if t in tier_counts:
                        tier_counts[t] += 1
                    else:
                        tier_counts["warm"] += 1  # default for legacy findings
            except Exception as exc:
                logger.debug("investigation_list: tier count scan failed (fail-open): %r", exc)
                pass  # fail-open
            # acl only feeds visibility, a full-mode-only field, so the summary path skips the lookup.
            acl = manifest.get("acl") or []
            record.update({
                "created_at": manifest["created_at"],
                "open_questions_count": len(manifest["open_questions"]),
                "hypothesis": manifest["hypothesis"],
                "visibility": "shared" if acl else "private",
                "tier_counts": tier_counts,
            })
        investigations.append(record)

    return json.dumps({
        "investigations": investigations,
        "total": total,
        "limit": limit,
        "offset": offset,
    }, indent=2)


def investigation_share(
    investigation_id: str,
    agent_ids: list,
) -> str:
    """
    Grant read/write access to an investigation for one or more agents.
    Adds the given agent_ids to the investigation's ACL (access control list).
    Idempotent — adding an agent already in the ACL has no effect.

    Args:
        investigation_id: Investigation identifier.
        agent_ids: List of agent_id strings to add to the ACL.

    Returns:
        JSON: {"shared_with": [...], "total_acl": N}
        On error: {"error": "<message>"}
    """
    try:
        manifest = _load_manifest(investigation_id)
        if not manifest:
            return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

        current_acl = list(manifest.get("acl") or [])
        current_set = set(current_acl)
        added = []
        for agent_id in (agent_ids or []):
            if agent_id and agent_id not in current_set:
                current_acl.append(agent_id)
                current_set.add(agent_id)
                added.append(agent_id)

        manifest["acl"] = current_acl
        _save_manifest(manifest)

        return json.dumps({
            "shared_with": added,
            "total_acl": len(current_acl),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def investigation_unshare(
    investigation_id: str,
    agent_ids: list,
) -> str:
    """
    Revoke access to an investigation for one or more agents.
    Removes the given agent_ids from the investigation's ACL.
    Idempotent — removing an agent not in the ACL has no effect.

    Args:
        investigation_id: Investigation identifier.
        agent_ids: List of agent_id strings to remove from the ACL.

    Returns:
        JSON: {"removed": [...], "total_acl": N}
        On error: {"error": "<message>"}
    """
    try:
        manifest = _load_manifest(investigation_id)
        if not manifest:
            return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

        current_acl = list(manifest.get("acl") or [])
        remove_set = set(agent_ids or [])
        removed = [a for a in current_acl if a in remove_set]
        current_acl = [a for a in current_acl if a not in remove_set]

        manifest["acl"] = current_acl
        _save_manifest(manifest)

        return json.dumps({
            "removed": removed,
            "total_acl": len(current_acl),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def investigation_export(
    investigation_id: str,
    include_embeddings: bool = False,
) -> str:
    """
    Export an investigation as a portable JSON bundle suitable for archival or
    transfer to another Loci instance.

    Bundles the manifest, all findings, conflicts, and entities into a single
    JSON string.  The ``include_embeddings`` parameter is accepted for forward
    compatibility but embeddings are not yet included in the bundle (future work).

    Args:
        investigation_id: Investigation identifier to export.
        include_embeddings: Reserved for future use — embeddings are not yet
                            included.  Pass ``True`` to opt-in once supported.

    Returns:
        JSON: {"exported": true, "investigation_id": str, "bundle": {...},
               "finding_count": int, "size_bytes": int}
        On error: {"error": str}
    """
    try:
        manifest = _load_manifest(investigation_id)
        if not manifest:
            return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

        inv_dir = _inv_dir(investigation_id)

        findings = _read_jsonl(inv_dir / "findings.jsonl")
        conflicts = _read_jsonl(inv_dir / "conflicts.jsonl")
        entities = _read_jsonl(inv_dir / "entities.jsonl")

        bundle = {
            "schema_version": "1.0",
            "exported_at": _now(),
            "manifest": manifest,
            "findings": findings,
            "conflicts": conflicts,
            "entities": entities,
        }

        bundle_str = json.dumps(bundle)
        size_bytes = len(bundle_str.encode("utf-8"))

        return json.dumps({
            "exported": True,
            "investigation_id": investigation_id,
            "bundle": bundle,
            "finding_count": len(findings),
            "size_bytes": size_bytes,
        })
    except Exception as exc:
        logger.warning("investigation_export failed: %s", exc)
        return json.dumps({"error": f"Export failed: {exc}"})


def investigation_import(
    bundle_json: str,
    new_title: Optional[str] = None,
) -> str:
    """
    Import an investigation bundle (produced by ``investigation_export``) into
    this Loci instance under a brand-new investigation ID.

    A fresh UUID is always assigned — the original investigation ID is preserved
    in the manifest as ``imported_from``.  Findings are re-indexed into Qdrant
    on a best-effort basis (fail-open: Qdrant may be unavailable).

    Args:
        bundle_json: The JSON string produced by ``investigation_export`` (the
                     value of the ``bundle`` key, or the whole export response).
        new_title: Optional override for the investigation title.  When omitted
                   the original title from the bundle is used.

    Returns:
        JSON: {"imported": true, "new_investigation_id": str,
               "original_investigation_id": str, "findings_imported": int,
               "qdrant_indexed": int}
        On error: {"error": str}
    """
    _MAX_BUNDLE_BYTES = 10 * 1024 * 1024  # 10 MB
    try:
        raw_bytes = bundle_json.encode("utf-8") if isinstance(bundle_json, str) else bundle_json
        if len(raw_bytes) > _MAX_BUNDLE_BYTES:
            return json.dumps({"error": "bundle too large"})

        try:
            data = json.loads(bundle_json)
        except Exception as exc:
            return json.dumps({"error": f"Invalid JSON in bundle_json: {exc}"})

        # Accept either the raw bundle or the whole export response.
        if "bundle" in data and isinstance(data.get("bundle"), dict):
            data = data["bundle"]

        schema_version = data.get("schema_version")
        if schema_version != "1.0":
            return json.dumps({"error": f"Unsupported schema_version: {schema_version!r}. Expected '1.0'."})

        required_keys = {"manifest", "findings"}
        missing = required_keys - set(data.keys())
        if missing:
            return json.dumps({"error": f"Bundle is missing required keys: {sorted(missing)}"})

        src_manifest = data["manifest"]
        if not isinstance(src_manifest, dict):
            return json.dumps({"error": "Bundle manifest is not a dict."})

        original_id = src_manifest.get("id", "unknown")

        new_id = str(uuid.uuid4())

        now = _now()
        new_manifest = dict(src_manifest)
        new_manifest["id"] = new_id
        new_manifest["created_at"] = now
        new_manifest["updated_at"] = now
        new_manifest["imported_from"] = original_id
        if new_title:
            new_manifest["title"] = new_title

        inv_dir = _inv_dir(new_id)
        _save_manifest(new_manifest)

        findings = data.get("findings") or []
        if not isinstance(findings, list):
            findings = []

        findings_path = inv_dir / "findings.jsonl"
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            f = dict(finding)
            f["investigation_id"] = new_id
            _append_jsonl(findings_path, f)

        conflicts = data.get("conflicts")
        if conflicts and isinstance(conflicts, list):
            conflicts_path = inv_dir / "conflicts.jsonl"
            for entry in conflicts:
                if isinstance(entry, dict):
                    _append_jsonl(conflicts_path, entry)

        entities = data.get("entities")
        if entities and isinstance(entities, list):
            entities_path = inv_dir / "entities.jsonl"
            for entry in entities:
                if isinstance(entry, dict):
                    _append_jsonl(entities_path, entry)

        # Re-index findings into Qdrant (fail-open).
        qdrant_indexed = 0
        import_ts = int(datetime.now(timezone.utc).timestamp())
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            text = str(finding.get("text") or "").strip()
            finding_id = str(finding.get("id") or "")
            if not text or not finding_id:
                continue
            try:
                # Index the whole finding, as the native store path does
                # (server._store_finding). A hand-built payload dropped
                # created_at_ts, and every age check reads that key: ranking
                # decay skipped the row and the retention purge could not match
                # it, so imported findings outranked native ones forever.
                payload = dict(finding)
                payload.update({
                    "investigation_id": new_id,
                    "type": finding.get("type") or finding.get("record_type") or "observed",
                    "source": finding.get("source") or "",
                    "confidence": finding.get("confidence") or "medium",
                    "tags": finding.get("tags") or [],
                })
                if not payload.get("created_at_ts"):
                    # A bundle with no age at all: substitute import time and say so,
                    # so a reader can tell a real age from a stand-in.
                    payload["created_at_ts"] = import_ts
                    payload["age_source"] = "imported"
                _qdrant_upsert(finding_id, text, payload)
                qdrant_indexed += 1
            except Exception as exc:
                logger.debug("investigation_import: qdrant upsert skipped for %s: %s", finding_id, exc)

        return json.dumps({
            "imported": True,
            "new_investigation_id": new_id,
            "original_investigation_id": original_id,
            "findings_imported": len(findings),
            "qdrant_indexed": qdrant_indexed,
        })
    except Exception as exc:
        logger.warning("investigation_import failed: %s", exc)
        return json.dumps({"error": f"Import failed: {exc}"})


def register(mcp, get_memory_dir, deps):
    """Inject deps and register every investigation tool on the shared FastMCP instance.

    ``deps`` supplies the server-side collaborators that stayed in server.py; passing
    them explicitly (rather than importing server) is what keeps this module acyclic.
    """
    global _get_memory_dir, _apply_lifecycle, _compute_self_check
    global _event_log_append, _qdrant_upsert
    _get_memory_dir = get_memory_dir
    _apply_lifecycle = deps["_apply_lifecycle"]
    _compute_self_check = deps["_compute_self_check"]
    _event_log_append = deps["_event_log_append"]
    _qdrant_upsert = deps["_qdrant_upsert"]
    for fn in (
        investigation_start,
        investigation_load,
        investigation_as_of,
        investigation_note,
        investigation_reflect,
        investigation_finding_provenance,
        investigation_list,
        investigation_share,
        investigation_unshare,
        investigation_export,
        investigation_import,
    ):
        mcp.tool()(fn)

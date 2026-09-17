#!/usr/bin/env python3
"""Propose reflection-loop findings as human-gated GitHub issues.

The module is standalone and deliberately fail-open: queue, dashboard, dedup,
rejection-ledger, or GitHub adapter failures return audit metadata instead of
crashing the caller. Dashboard rendering is the default sink; creating a public
issue additionally requires an explicit promotion id, ``confirm=True``, and a
repository allowlist match.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


IssueSearchFn = Callable[..., Any]
DedupFn = Callable[..., Any]
GhFn = Callable[..., Any]

DEFAULT_QUEUE_PATH = Path(__file__).with_name("issue_proposals_queue.jsonl")
DEFAULT_DASHBOARD_PATH = Path(__file__).with_name("issue_proposals_dashboard.md")
DEFAULT_REJECTION_PATH = Path(__file__).with_name("issue_proposal_rejections.jsonl")
DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_MAX_PROPOSALS = 5
FUZZY_TITLE_THRESHOLD = 0.88


def _json_default(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    text = str(value or "").strip().lower()
    if text in {"critical", "certain"}:
        return 1.0
    if text == "high":
        return 0.9
    if text == "medium":
        return 0.6
    if text == "low":
        return 0.3
    try:
        return max(0.0, min(1.0, float(text)))
    except Exception:
        return default


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", title.lower())).strip()


def _normalize_evidence_key(item: str) -> str:
    return re.sub(r"\s+", " ", item.strip().lower())


def _fingerprint(repo: str, finding: dict[str, Any], title: str, evidence: list[str]) -> str:
    """Stable identity for dedup: repo + normalized evidence, not rendered prose.

    Rewording the same finding's title/body/confidence must not change the
    fingerprint, so identity is derived from structured, stable identifiers
    (a finding ID if present, otherwise the normalized evidence references
    such as file:line or signature) rather than the natural-language text.
    """
    finding_id = str(finding.get("finding_id") or finding.get("id") or "").strip().lower()
    if finding_id:
        identity_parts = [finding_id]
    else:
        identity_parts = sorted({_normalize_evidence_key(item) for item in evidence if item.strip()})
    if not identity_parts:
        # No stable identifiers available (e.g. evidence-less findings that
        # were not filtered upstream); fall back to the normalized title so
        # the function still returns a deterministic value.
        identity_parts = [_normalize_title(title)]
    payload = "\n".join([repo.strip().lower(), *identity_parts])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _normalize_repo(repo: str) -> str:
    return _normalize_text(repo)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _evidence_raw_items(finding: dict[str, Any]) -> list[Any]:
    raw_items: list[Any] = []
    for key in (
        "evidence",
        "evidence_refs",
        "supporting_evidence",
        "citations",
        "loci_finding_ids",
        "logs",
    ):
        raw_items.extend(_as_list(finding.get(key)))
    for key in (
        "error_signature",
        "signature",
        "crash_state",
        "file",
        "file_path",
        "path",
        "symbol",
        "function",
        "line",
        "lines",
        "location",
        "log_excerpt",
        "finding_id",
    ):
        if finding.get(key) not in (None, ""):
            raw_items.append({key: finding.get(key)})
    return raw_items


def _evidence_items(finding: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in _evidence_raw_items(finding):
        if isinstance(item, dict):
            text = str(
                item.get("text")
                or item.get("line")
                or item.get("ref")
                or item.get("id")
                or item.get("signature")
                or item.get("error_signature")
                or item.get("crash_state")
                or item.get("file")
                or item.get("file_path")
                or item.get("path")
                or item.get("symbol")
                or item.get("location")
                or item
            ).strip()
        else:
            text = str(item or "").strip()
        if text:
            out.append(text)
    return out


# Patterns that mark evidence text as verifiable/traceable rather than prose.
_EVIDENCE_FILE_LINE_RE = re.compile(r"\b[\w./-]+\.[a-zA-Z0-9]+:\d+\b")
_EVIDENCE_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_EVIDENCE_FINDING_ID_RE = re.compile(r"\b(?:FND|CVE|GHSA|CWE)-[A-Za-z0-9-]+\b", re.I)
_EVIDENCE_SIGNATURE_RE = re.compile(r"\b(?:sha(?:1|256|512)?:[0-9a-f]{7,64}|(?=[0-9a-f]*\d)[0-9a-f]{7,64})\b", re.I)
_EVIDENCE_STRUCTURED_KEYS = ("finding_id", "id", "ref", "sha", "signature", "line", "path", "file")


def _has_concrete_evidence(finding: dict[str, Any]) -> bool:
    """Require actually verifiable/traceable evidence, not just prose.

    Accepts: a file path + line number reference, a finding ID (UUID or
    FND/CVE/GHSA/CWE-style identifier), a hash/signature, or a structured
    evidence object carrying one of those as a distinct field. Plain
    natural-language sentences (regardless of length) are rejected.
    """
    raw_items: list[Any] = []
    for key in ("evidence", "evidence_refs", "supporting_evidence", "citations", "loci_finding_ids", "logs"):
        raw_items.extend(_as_list(finding.get(key)))
    for raw_item in raw_items:
        if isinstance(raw_item, dict):
            for struct_key in _EVIDENCE_STRUCTURED_KEYS:
                if str(raw_item.get(struct_key) or "").strip():
                    return True
            text = str(raw_item.get("text") or "").strip()
        else:
            text = str(raw_item or "").strip()
        if not text:
            continue
        if _EVIDENCE_FILE_LINE_RE.search(text):
            return True
        if _EVIDENCE_UUID_RE.search(text):
            return True
        if _EVIDENCE_FINDING_ID_RE.search(text):
            return True
        if _EVIDENCE_SIGNATURE_RE.search(text):
            return True
    return False


def _finding_title(finding: dict[str, Any]) -> str:
    for key in ("issue_title", "title", "claim", "summary"):
        value = str(finding.get(key) or "").strip()
        if value:
            return value[:180]
    return "Reflection-loop finding needs human review"


def _finding_body_seed(finding: dict[str, Any]) -> str:
    for key in ("issue_body", "body", "description", "text", "claim", "summary"):
        value = str(finding.get(key) or "").strip()
        if value:
            return value
    return _finding_title(finding)


def _compose_body(
    finding: dict[str, Any],
    *,
    repo: str,
    confidence: float,
    evidence: list[str],
    proposal_id: str,
) -> str:
    lines = [
        "> Machine-generated proposal from the Loci reflection loop. A human maintainer must review before any action.",
        "> Dashboard-first mode is the default; filing requires explicit promotion plus repository allowlisting.",
        "",
        f"Proposal ID: `{proposal_id}`",
        f"Target repository: `{repo}`",
        f"Confidence: `{confidence:.2f}`",
        "",
        "## Proposed issue",
        _finding_body_seed(finding),
        "",
        "## Evidence",
    ]
    lines.extend(f"- {item}" for item in evidence)
    source = str(finding.get("source") or finding.get("investigation_id") or "").strip()
    if source:
        lines.extend(["", f"Source: `{source}`"])
    lines.extend(["", "_No GitHub issue was opened by proposing this item._"])
    return "\n".join(lines)


def _proposal_from_finding(finding: dict[str, Any], repo: str) -> dict[str, Any]:
    confidence = _safe_float(finding.get("numeric_confidence", finding.get("confidence", 0.0)))
    evidence = _evidence_items(finding)
    title = _finding_title(finding)
    fingerprint = _fingerprint(repo, finding, title, evidence)
    body = _compose_body(finding, repo=repo, confidence=confidence, evidence=evidence, proposal_id=fingerprint)
    return {
        "proposal_id": fingerprint,
        "title": title,
        "body": body,
        "repo": repo,
        "confidence": confidence,
        "evidence": evidence,
        "source_finding": dict(finding),
        "fingerprint": fingerprint,
        "labels": ["loci-reflection", "machine-generated-proposal"],
        "status": "proposed",
    }


def _read_jsonl(path: str | Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    if path is None:
        return [], [], False
    target = Path(path)
    if not target.exists():
        return [], [], False
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        for line_no, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except Exception as exc:
                errors.append({"line": line_no, "error": str(exc)})
        return rows, errors, bool(errors)
    except Exception as exc:
        return [], [{"error": f"jsonl read degraded: {exc}"}], True


def _write_jsonl(path: str | Path | None, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    if path is None:
        return [], False
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
        return [], False
    except Exception as exc:
        return [{"error": f"jsonl write degraded: {exc}"}], True


def _append_jsonl(path: str | Path | None, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    if path is None or not rows:
        return [], False
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return [], False
    except Exception as exc:
        return [{"error": f"jsonl append degraded: {exc}"}], True


def _queue_row_from_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": proposal.get("created_at") or _now_iso(),
        "status": proposal.get("status") or "proposed",
        "proposal_id": proposal.get("proposal_id") or proposal.get("fingerprint"),
        "repo": proposal.get("repo"),
        "title": proposal.get("title"),
        "body": proposal.get("body"),
        "confidence": proposal.get("confidence"),
        "evidence": proposal.get("evidence", []),
        "fingerprint": proposal.get("fingerprint"),
        "labels": proposal.get("labels", []),
        "rejection_reason": proposal.get("rejection_reason"),
        "rejected_at": proposal.get("rejected_at"),
        "opened_at": proposal.get("opened_at"),
        "opened_url": proposal.get("opened_url"),
    }


def _read_queue(queue_path: str | Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    rows, errors, degraded = _read_jsonl(queue_path)
    normalized = []
    for row in rows:
        normalized.append(_queue_row_from_proposal(row))
    return normalized, errors, degraded


def _append_queue(queue_path: str | Path | None, proposals: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    rows = [_queue_row_from_proposal(proposal) for proposal in proposals]
    return _append_jsonl(queue_path, rows)


def _read_rejection_ledger(rejection_path: str | Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    return _read_jsonl(rejection_path)


def _rejection_entry(proposal: dict[str, Any], *, reason: str = "") -> dict[str, Any]:
    return {
        "proposal_id": proposal.get("proposal_id") or proposal.get("fingerprint"),
        "fingerprint": proposal.get("fingerprint"),
        "repo": proposal.get("repo"),
        "title": proposal.get("title"),
        "reason": reason or "rejected_by_human",
        "rejected_at": _now_iso(),
    }


def _render_dashboard(
    queue_rows: list[dict[str, Any]],
    *,
    dashboard_path: str | Path | None,
    rejection_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    if dashboard_path is None:
        return [], False
    target = Path(dashboard_path)
    pending = sorted(
        [row for row in queue_rows if str(row.get("status") or "proposed") == "proposed"],
        key=lambda item: (-_safe_float(item.get("confidence")), str(item.get("title") or "").lower()),
    )
    opened = sorted(
        [row for row in queue_rows if str(row.get("status") or "") == "opened"],
        key=lambda item: str(item.get("opened_at") or item.get("created_at") or ""),
        reverse=True,
    )
    rejected = sorted(
        [row for row in queue_rows if str(row.get("status") or "") == "rejected"],
        key=lambda item: str(item.get("rejected_at") or item.get("created_at") or ""),
        reverse=True,
    )

    def render_rows(rows: list[dict[str, Any]], *, include_status: bool = False) -> list[str]:
        if not rows:
            return ["_None._", ""]
        lines: list[str] = []
        for row in rows:
            proposal_id = row.get("proposal_id") or row.get("fingerprint") or "unknown"
            lines.extend([
                f"### `{proposal_id}` — {row.get('title') or 'Untitled proposal'}",
                f"- Repo: `{row.get('repo') or ''}`",
                f"- Confidence: `{_safe_float(row.get('confidence')):.2f}`",
                f"- Fingerprint: `{row.get('fingerprint') or proposal_id}`",
            ])
            if include_status:
                lines.append(f"- Status: `{row.get('status') or 'proposed'}`")
            if row.get("opened_url"):
                lines.append(f"- Opened issue: {row.get('opened_url')}")
            if row.get("rejection_reason"):
                lines.append(f"- Rejection reason: {row.get('rejection_reason')}")
            lines.extend(["", "#### Evidence"])
            evidence = row.get("evidence") or []
            lines.extend(f"- {item}" for item in evidence)
            lines.extend(["", "#### Proposed body", "", row.get("body") or "", ""])
        return lines

    lines = [
        "# Loci issue proposal dashboard",
        "",
        "Dashboard-first mode is the default. No issue is filed from this artifact unless a human promotes a specific proposal ID and reruns with `--confirm-open` and an explicit `--allow-repo`.",
        "",
        f"Generated at: `{_now_iso()}`",
        "",
        "## Pending proposals",
        "",
        *render_rows(pending),
        "## Opened proposals",
        "",
        *render_rows(opened, include_status=True),
        "## Rejected proposals",
        "",
        *render_rows(rejected, include_status=True),
        "## Rejection ledger",
        "",
    ]
    if rejection_rows:
        for row in sorted(rejection_rows, key=lambda item: str(item.get("rejected_at") or ""), reverse=True):
            lines.append(
                f"- `{row.get('proposal_id') or row.get('fingerprint')}` `{row.get('fingerprint')}` `{row.get('repo')}` — {row.get('reason') or 'rejected_by_human'}"
            )
    else:
        lines.append("_None._")
    lines.append("")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines), encoding="utf-8")
        return [], False
    except Exception as exc:
        return [{"error": f"dashboard render degraded: {exc}"}], True


def _call_existing_issues(existing_issues_fn: IssueSearchFn | None, repo: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    if existing_issues_fn is None:
        return [], [], False
    attempts = (
        {"repo": repo, "state": "open"},
        {"repo": repo},
        {},
    )
    for kwargs in attempts:
        try:
            raw = existing_issues_fn(**kwargs)
            if isinstance(raw, dict):
                raw = raw.get("issues") or raw.get("items") or raw.get("results") or []
            return [item for item in _as_list(raw) if isinstance(item, dict)], [], False
        except TypeError:
            continue
        except Exception as exc:
            return [], [{"error": f"existing issue lookup degraded: {exc}"}], True
    return [], [{"error": "existing issue lookup signature mismatch"}], True


def _title_duplicate(proposal: dict[str, Any], candidates: list[dict[str, Any]], source: str) -> dict[str, Any] | None:
    title = _normalize_title(proposal["title"])
    for item in candidates:
        candidate_title = _normalize_title(str(item.get("title") or ""))
        if not candidate_title:
            continue
        ratio = difflib.SequenceMatcher(None, title, candidate_title).ratio()
        if title == candidate_title or ratio >= FUZZY_TITLE_THRESHOLD:
            return {"source": source, "match": item, "score": ratio, "method": "exact_or_fuzzy_title"}
        title_tokens = set(title.split())
        candidate_tokens = set(candidate_title.split())
        if title_tokens and candidate_tokens:
            jaccard = len(title_tokens & candidate_tokens) / len(title_tokens | candidate_tokens)
            if jaccard >= 0.82:
                return {"source": source, "match": item, "score": jaccard, "method": "token_title"}
    return None


def _custom_duplicate(
    proposal: dict[str, Any],
    candidates: list[dict[str, Any]],
    source: str,
    dedup_fn: DedupFn | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], bool]:
    if dedup_fn is None or not candidates:
        return None, [], False
    try:
        raw = dedup_fn(proposal=proposal, candidates=candidates, source=source)
    except TypeError:
        try:
            raw = dedup_fn(proposal, candidates)
        except Exception as exc:
            return None, [{"error": f"{source} semantic dedup degraded: {exc}"}], True
    except Exception as exc:
        return None, [{"error": f"{source} semantic dedup degraded: {exc}"}], True
    if isinstance(raw, bool):
        return ({"source": source, "match": {}, "score": 1.0, "method": "dedup_fn"} if raw else None), [], False
    if isinstance(raw, dict):
        duplicate = bool(raw.get("duplicate") or raw.get("is_duplicate"))
        if duplicate:
            return {
                "source": source,
                "match": raw.get("match") or raw.get("candidate") or {},
                "score": _safe_float(raw.get("score"), 1.0),
                "method": str(raw.get("method") or "dedup_fn"),
            }, [], False
        if raw.get("degraded"):
            return None, [{"error": f"{source} semantic dedup degraded"}], True
    return None, [], False


def _dedup_match(
    proposal: dict[str, Any],
    local_queue: list[dict[str, Any]],
    existing_issues: list[dict[str, Any]],
    dedup_fn: DedupFn | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], bool]:
    all_errors: list[dict[str, Any]] = []
    any_degraded = False
    for source, candidates in (("local_queue", local_queue), ("open_issues", existing_issues)):
        for item in candidates:
            if str(item.get("fingerprint") or "") == proposal["fingerprint"]:
                return {"source": source, "match": item, "score": 1.0, "method": "fingerprint"}, all_errors, any_degraded
        match = _title_duplicate(proposal, candidates, source)
        if match:
            return match, all_errors, any_degraded
        custom_match, errors, degraded = _custom_duplicate(proposal, candidates, source, dedup_fn)
        all_errors.extend(errors)
        any_degraded = any_degraded or degraded
        if custom_match:
            return custom_match, all_errors, any_degraded
    return None, all_errors, any_degraded


def make_loci_dedup_fn(loci_semantic_dedup_fn: Callable[..., Any], *, threshold: float = 0.88) -> DedupFn:
    """Adapt an injected Loci ``semantic_dedup`` callable into ``dedup_fn``.

    The deterministic evidence fingerprint and title checks still run first;
    this adapter is the semantic backstop for differently worded duplicates.
    """

    def dedup_fn(*, proposal: dict[str, Any], candidates: list[dict[str, Any]], source: str = "") -> dict[str, Any]:
        items = [{"text": "\n".join(proposal.get("evidence") or [proposal.get("title", "")]), "candidate": proposal}]
        items.extend({"text": "\n".join(candidate.get("evidence") or [candidate.get("title", "")]), "candidate": candidate} for candidate in candidates)
        try:
            raw = loci_semantic_dedup_fn(items=items, text_key="text", threshold=threshold)
        except TypeError:
            raw = loci_semantic_dedup_fn(items, "text", threshold)
        if not isinstance(raw, dict):
            return {"duplicate": False, "method": "loci_semantic_dedup"}
        for cluster in raw.get("clusters") or []:
            members = [int(idx) for idx in cluster.get("member_indices") or [] if str(idx).isdigit()]
            if 0 in members and len(members) > 1:
                match_idx = next(idx for idx in members if idx != 0)
                return {
                    "duplicate": True,
                    "match": candidates[match_idx - 1] if 0 < match_idx <= len(candidates) else {},
                    "score": threshold,
                    "method": f"loci_semantic_dedup:{source}" if source else "loci_semantic_dedup",
                }
        return {"duplicate": False, "method": "loci_semantic_dedup", "degraded": bool(raw.get("degraded"))}

    return dedup_fn


def propose_issues(
    findings: list[dict[str, Any]],
    repo: str,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    existing_issues_fn: IssueSearchFn | None = None,
    dedup_fn: DedupFn | None = None,
    *,
    queue_path: str | Path | None = DEFAULT_QUEUE_PATH,
    dashboard_path: str | Path | None = DEFAULT_DASHBOARD_PATH,
    rejection_path: str | Path | None = DEFAULT_REJECTION_PATH,
) -> dict[str, Any]:
    """Rank and persist dashboard proposals, returning accepted and rejected audit."""

    audit: list[dict[str, Any]] = []
    fail_open = False
    local_queue, queue_errors, queue_degraded = _read_queue(queue_path)
    rejection_rows, rejection_errors, rejection_degraded = _read_rejection_ledger(rejection_path)
    existing_issues, issue_errors, issue_degraded = _call_existing_issues(existing_issues_fn, repo)
    fail_open = fail_open or queue_degraded or rejection_degraded or issue_degraded
    audit.extend({"action": "degraded", **error} for error in queue_errors + rejection_errors + issue_errors)
    rejected_fingerprints = {
        str(row.get("fingerprint") or row.get("proposal_id") or "")
        for row in rejection_rows
        if str(row.get("fingerprint") or row.get("proposal_id") or "")
    }

    candidates: list[dict[str, Any]] = []
    for idx, raw_finding in enumerate(findings or []):
        finding = raw_finding if isinstance(raw_finding, dict) else {"text": str(raw_finding)}
        proposal = _proposal_from_finding(finding, repo)
        if not _has_concrete_evidence(finding):
            audit.append({"finding_index": idx, "action": "dropped", "reason": "missing_concrete_evidence", "title": proposal["title"]})
            continue
        if proposal["fingerprint"] in rejected_fingerprints:
            audit.append({
                "finding_index": idx,
                "action": "dropped",
                "reason": "previously_rejected",
                "proposal_id": proposal["proposal_id"],
                "fingerprint": proposal["fingerprint"],
                "title": proposal["title"],
            })
            continue
        if proposal["confidence"] < float(min_confidence):
            audit.append({
                "finding_index": idx,
                "action": "dropped",
                "reason": "below_confidence_floor",
                "confidence": proposal["confidence"],
                "min_confidence": float(min_confidence),
                "title": proposal["title"],
            })
            continue
        match, errors, degraded = _dedup_match(proposal, local_queue + candidates, existing_issues, dedup_fn)
        fail_open = fail_open or degraded
        audit.extend({"action": "degraded", **error} for error in errors)
        if match:
            audit.append({
                "finding_index": idx,
                "action": "deduped",
                "reason": "duplicate",
                "against": match["source"],
                "method": match["method"],
                "score": match["score"],
                "title": proposal["title"],
                "proposal_id": proposal["proposal_id"],
            })
            continue
        candidates.append(proposal)
        audit.append({
            "finding_index": idx,
            "action": "candidate",
            "reason": "passed_gates",
            "title": proposal["title"],
            "confidence": proposal["confidence"],
            "proposal_id": proposal["proposal_id"],
        })

    ranked = sorted(candidates, key=lambda item: (-item["confidence"], item["title"].lower()))
    accepted = ranked[: max(0, int(max_proposals))]
    for capped in ranked[len(accepted):]:
        audit.append({"action": "dropped", "reason": "max_proposals_cap", "title": capped["title"], "max_proposals": int(max_proposals)})
    append_errors, append_degraded = _append_queue(queue_path, accepted)
    fail_open = fail_open or append_degraded
    audit.extend({"action": "degraded", **error} for error in append_errors)
    queue_rows = local_queue + [_queue_row_from_proposal(proposal) for proposal in accepted]
    dashboard_errors, dashboard_degraded = _render_dashboard(queue_rows, dashboard_path=dashboard_path, rejection_rows=rejection_rows)
    fail_open = fail_open or dashboard_degraded
    audit.extend({"action": "degraded", **error} for error in dashboard_errors)
    for proposal in accepted:
        audit.append({
            "action": "proposed",
            "reason": "rendered_to_dashboard",
            "title": proposal["title"],
            "fingerprint": proposal["fingerprint"],
            "proposal_id": proposal["proposal_id"],
        })

    dropped = [item for item in audit if item.get("action") == "dropped"]
    deduped = [item for item in audit if item.get("action") == "deduped"]
    return {
        "fail_open": fail_open,
        "dry_run": True,
        "repo": repo,
        "queue_path": str(queue_path) if queue_path is not None else None,
        "dashboard_path": str(dashboard_path) if dashboard_path is not None else None,
        "rejection_path": str(rejection_path) if rejection_path is not None else None,
        "proposals": accepted,
        "dropped": dropped,
        "deduped": deduped,
        "audit_trail": audit,
    }


def _allowlist_match(repo: str, allowed_repos: list[str] | tuple[str, ...] | set[str] | None) -> bool:
    normalized = {_normalize_repo(item) for item in (allowed_repos or []) if _normalize_repo(item)}
    return _normalize_repo(repo) in normalized


def _load_targeted_proposal(queue_rows: list[dict[str, Any]], proposal_id: str) -> dict[str, Any] | None:
    for row in queue_rows:
        if str(row.get("proposal_id") or row.get("fingerprint") or "") == str(proposal_id):
            return row
    return None


def open_proposed_issues(
    proposals: list[dict[str, Any]],
    gh_fn: GhFn,
    *,
    confirm: bool = False,
    promotion_id: str | None = None,
    allowed_repos: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    """Open a single promoted proposal only when confirmed and allowlisted."""

    if not promotion_id:
        return {
            "fail_open": False,
            "dry_run": True,
            "opened": [],
            "would_open": [],
            "audit_trail": [{"action": "skipped", "reason": "promotion_id_required"}],
        }
    proposal = _load_targeted_proposal([_queue_row_from_proposal(item) for item in proposals or []], promotion_id)
    if proposal is None:
        return {
            "fail_open": False,
            "dry_run": True,
            "opened": [],
            "would_open": [],
            "audit_trail": [{"action": "skipped", "reason": "proposal_not_found", "proposal_id": promotion_id}],
        }
    if not confirm:
        return {
            "fail_open": False,
            "dry_run": True,
            "opened": [],
            "would_open": [{"repo": proposal.get("repo"), "title": proposal.get("title"), "body": proposal.get("body"), "labels": proposal.get("labels", [])}],
            "audit_trail": [{"action": "dry_run", "reason": "confirm_false", "proposal_id": proposal.get("proposal_id")}],
        }
    if not _allowlist_match(str(proposal.get("repo") or ""), allowed_repos):
        return {
            "fail_open": False,
            "dry_run": True,
            "opened": [],
            "would_open": [{"repo": proposal.get("repo"), "title": proposal.get("title"), "body": proposal.get("body"), "labels": proposal.get("labels", [])}],
            "audit_trail": [{
                "action": "skipped",
                "reason": "repo_not_allowlisted",
                "proposal_id": proposal.get("proposal_id"),
                "repo": proposal.get("repo"),
            }],
        }
    try:
        try:
            result = gh_fn(repo=proposal.get("repo"), title=proposal.get("title"), body=proposal.get("body"), labels=proposal.get("labels", []))
        except TypeError:
            result = gh_fn(proposal)
    except Exception as exc:
        return {
            "fail_open": True,
            "dry_run": False,
            "opened": [],
            "would_open": [],
            "audit_trail": [{
                "action": "degraded",
                "reason": "gh_fn_failed",
                "title": proposal.get("title"),
                "proposal_id": proposal.get("proposal_id"),
                "error": str(exc),
            }],
        }
    return {
        "fail_open": False,
        "dry_run": False,
        "opened": [result],
        "would_open": [],
        "audit_trail": [{"action": "opened", "title": proposal.get("title"), "proposal_id": proposal.get("proposal_id"), "result": result}],
    }


def promote_proposal(
    proposal_id: str,
    *,
    queue_path: str | Path | None = DEFAULT_QUEUE_PATH,
    dashboard_path: str | Path | None = DEFAULT_DASHBOARD_PATH,
    rejection_path: str | Path | None = DEFAULT_REJECTION_PATH,
    gh_fn: GhFn,
    confirm: bool = False,
    allowed_repos: list[str] | tuple[str, ...] | set[str] | None = None,
    existing_issues_fn: IssueSearchFn | None = None,
    dedup_fn: DedupFn | None = None,
) -> dict[str, Any]:
    audit: list[dict[str, Any]] = []
    fail_open = False
    queue_rows, queue_errors, queue_degraded = _read_queue(queue_path)
    rejection_rows, rejection_errors, rejection_degraded = _read_rejection_ledger(rejection_path)
    fail_open = fail_open or queue_degraded or rejection_degraded
    audit.extend({"action": "degraded", **error} for error in queue_errors + rejection_errors)

    proposal = _load_targeted_proposal(queue_rows, proposal_id)
    if proposal is None:
        return {"fail_open": fail_open, "dry_run": True, "opened": [], "would_open": [], "audit_trail": audit + [{"action": "skipped", "reason": "proposal_not_found", "proposal_id": proposal_id}]}
    if str(proposal.get("status") or "") == "rejected":
        return {"fail_open": fail_open, "dry_run": True, "opened": [], "would_open": [], "audit_trail": audit + [{"action": "skipped", "reason": "proposal_previously_rejected", "proposal_id": proposal_id}]}
    if str(proposal.get("status") or "") == "opened":
        return {"fail_open": fail_open, "dry_run": True, "opened": [], "would_open": [], "audit_trail": audit + [{"action": "skipped", "reason": "proposal_already_opened", "proposal_id": proposal_id}]}
    rejected_fingerprints = {
        str(row.get("fingerprint") or row.get("proposal_id") or "")
        for row in rejection_rows
        if str(row.get("fingerprint") or row.get("proposal_id") or "")
    }
    if str(proposal.get("fingerprint") or proposal.get("proposal_id") or "") in rejected_fingerprints:
        return {"fail_open": fail_open, "dry_run": True, "opened": [], "would_open": [], "audit_trail": audit + [{"action": "skipped", "reason": "proposal_previously_rejected", "proposal_id": proposal_id}]}

    would_open = [{
        "repo": proposal.get("repo"),
        "title": proposal.get("title"),
        "body": proposal.get("body"),
        "labels": proposal.get("labels", []),
    }]
    if not confirm:
        audit.append({"action": "dry_run", "reason": "confirm_false", "proposal_id": proposal_id})
        return {"fail_open": fail_open, "dry_run": True, "opened": [], "would_open": would_open, "audit_trail": audit}
    if not _allowlist_match(str(proposal.get("repo") or ""), allowed_repos):
        audit.append({
            "action": "skipped",
            "reason": "repo_not_allowlisted",
            "proposal_id": proposal_id,
            "repo": proposal.get("repo"),
        })
        return {
            "fail_open": fail_open,
            "dry_run": True,
            "opened": [],
            "would_open": would_open,
            "audit_trail": audit,
        }
    if existing_issues_fn is not None:
        existing_issues, issue_errors, issue_degraded = _call_existing_issues(existing_issues_fn, str(proposal.get("repo") or ""))
        fail_open = fail_open or issue_degraded
        audit.extend({"action": "degraded", **error} for error in issue_errors)
        match, errors, degraded = _dedup_match(proposal, [], existing_issues, dedup_fn)
        fail_open = fail_open or degraded
        audit.extend({"action": "degraded", **error} for error in errors)
        if match:
            return {
                "fail_open": fail_open,
                "dry_run": True,
                "opened": [],
                "would_open": would_open,
                "audit_trail": audit + [{
                    "action": "skipped",
                    "reason": "duplicate_existing_open_issue",
                    "proposal_id": proposal_id,
                    "against": match["source"],
                    "method": match["method"],
                }],
            }
        if fail_open:
            return {
                "fail_open": True,
                "dry_run": True,
                "opened": [],
                "would_open": would_open,
                "audit_trail": audit + [{"action": "skipped", "reason": "promotion_fail_open_blocks_issue_creation", "proposal_id": proposal_id}],
            }

    opened_result = open_proposed_issues(
        [proposal],
        gh_fn,
        confirm=True,
        promotion_id=proposal_id,
        allowed_repos=allowed_repos,
    )
    audit.extend(opened_result["audit_trail"])
    if opened_result["fail_open"]:
        return {"fail_open": True, "dry_run": False, "opened": [], "would_open": [], "audit_trail": audit}

    opened_url = ""
    if opened_result["opened"] and isinstance(opened_result["opened"][0], dict):
        opened_url = str(opened_result["opened"][0].get("url") or "")
    updated_rows = []
    for row in queue_rows:
        if str(row.get("proposal_id") or row.get("fingerprint") or "") != proposal_id:
            updated_rows.append(row)
            continue
        changed = dict(row)
        changed["status"] = "opened"
        changed["opened_at"] = _now_iso()
        changed["opened_url"] = opened_url
        updated_rows.append(_queue_row_from_proposal(changed))
    write_errors, write_degraded = _write_jsonl(queue_path, updated_rows)
    fail_open = fail_open or write_degraded
    audit.extend({"action": "degraded", **error} for error in write_errors)
    dashboard_errors, dashboard_degraded = _render_dashboard(updated_rows, dashboard_path=dashboard_path, rejection_rows=rejection_rows)
    fail_open = fail_open or dashboard_degraded
    audit.extend({"action": "degraded", **error} for error in dashboard_errors)
    return {"fail_open": fail_open, "dry_run": False, "opened": opened_result["opened"], "would_open": [], "audit_trail": audit}


def record_proposal_rejection(
    proposal_id: str,
    *,
    queue_path: str | Path | None = DEFAULT_QUEUE_PATH,
    dashboard_path: str | Path | None = DEFAULT_DASHBOARD_PATH,
    rejection_path: str | Path | None = DEFAULT_REJECTION_PATH,
    reason: str = "",
) -> dict[str, Any]:
    audit: list[dict[str, Any]] = []
    fail_open = False
    queue_rows, queue_errors, queue_degraded = _read_queue(queue_path)
    rejection_rows, rejection_errors, rejection_degraded = _read_rejection_ledger(rejection_path)
    fail_open = fail_open or queue_degraded or rejection_degraded
    audit.extend({"action": "degraded", **error} for error in queue_errors + rejection_errors)

    proposal = _load_targeted_proposal(queue_rows, proposal_id)
    if proposal is None:
        return {"fail_open": fail_open, "recorded": [], "audit_trail": audit + [{"action": "skipped", "reason": "proposal_not_found", "proposal_id": proposal_id}]}
    fingerprint = str(proposal.get("fingerprint") or proposal.get("proposal_id") or "")
    existing_rejection = next(
        (row for row in rejection_rows if str(row.get("fingerprint") or row.get("proposal_id") or "") == fingerprint),
        None,
    )
    entry = existing_rejection or _rejection_entry(proposal, reason=reason)
    if existing_rejection is None:
        append_errors, append_degraded = _append_jsonl(rejection_path, [entry])
        fail_open = fail_open or append_degraded
        audit.extend({"action": "degraded", **error} for error in append_errors)
        if existing_rejection is None:
            rejection_rows = rejection_rows + [entry]
    updated_rows = []
    for row in queue_rows:
        if str(row.get("proposal_id") or row.get("fingerprint") or "") != proposal_id:
            updated_rows.append(row)
            continue
        changed = dict(row)
        changed["status"] = "rejected"
        changed["rejection_reason"] = reason or entry.get("reason") or "rejected_by_human"
        changed["rejected_at"] = entry.get("rejected_at") or _now_iso()
        updated_rows.append(_queue_row_from_proposal(changed))
    write_errors, write_degraded = _write_jsonl(queue_path, updated_rows)
    fail_open = fail_open or write_degraded
    audit.extend({"action": "degraded", **error} for error in write_errors)
    dashboard_errors, dashboard_degraded = _render_dashboard(updated_rows, dashboard_path=dashboard_path, rejection_rows=rejection_rows)
    fail_open = fail_open or dashboard_degraded
    audit.extend({"action": "degraded", **error} for error in dashboard_errors)
    audit.append({"action": "rejected", "reason": changed.get("rejection_reason"), "proposal_id": proposal_id, "fingerprint": fingerprint})
    return {"fail_open": fail_open, "recorded": [entry], "audit_trail": audit}


def gh_issue_search_fn(*, repo: str, state: str = "open", limit: int = 100) -> list[dict[str, Any]]:
    """Return GitHub issues through the authenticated ``gh`` CLI."""
    completed = subprocess.run(
        ["gh", "issue", "list", "--repo", repo, "--state", state, "--limit", str(limit), "--json", "number,title,body,url"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    raw = json.loads(completed.stdout or "[]")
    return raw if isinstance(raw, list) else []


def gh_create_issue_fn(*, repo: str, title: str, body: str, labels: list[str] | None = None) -> dict[str, Any]:
    """Create a GitHub issue through ``gh``; callers must gate this with confirm."""
    cmd = ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", body]
    for label in labels or []:
        cmd.extend(["--label", label])
    completed = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"url": completed.stdout.strip()}


def _load_findings(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    raw = json.loads(text)
    if isinstance(raw, dict):
        raw = raw.get("findings") or raw.get("items") or []
    return [item for item in raw if isinstance(item, dict)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Propose Loci reflection findings as human-gated GitHub issues.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--findings-json", help="JSON/JSONL findings file; omitted means no findings.")
    parser.add_argument("--queue-path", default=str(DEFAULT_QUEUE_PATH))
    parser.add_argument("--dashboard-path", default=str(DEFAULT_DASHBOARD_PATH))
    parser.add_argument("--rejection-path", default=str(DEFAULT_REJECTION_PATH))
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--max-proposals", type=int, default=DEFAULT_MAX_PROPOSALS)
    parser.add_argument("--confirm-open", action="store_true", help="Actually open a promoted GitHub issue.")
    parser.add_argument("--live-existing-issues", action="store_true", help="Use gh to deduplicate a promoted proposal against open issues.")
    parser.add_argument("--allow-repo", action="append", default=[], help="Explicit repository allowlist entry. Repeatable.")
    parser.add_argument("--rejection-reason", default="", help="Reason recorded with --reject-proposal.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--promote-proposal", help="Proposal ID to promote into a real GitHub issue.")
    action.add_argument("--reject-proposal", help="Proposal ID to persist into the rejection ledger.")
    args = parser.parse_args(argv)

    if args.reject_proposal:
        result = record_proposal_rejection(
            args.reject_proposal,
            queue_path=args.queue_path,
            dashboard_path=args.dashboard_path,
            rejection_path=args.rejection_path,
            reason=args.rejection_reason,
        )
        print(json.dumps({"rejection_result": result}, ensure_ascii=False, indent=2))
        return 0

    if args.promote_proposal:
        existing_fn = gh_issue_search_fn if args.live_existing_issues and args.confirm_open else None
        result = promote_proposal(
            args.promote_proposal,
            queue_path=args.queue_path,
            dashboard_path=args.dashboard_path,
            rejection_path=args.rejection_path,
            gh_fn=gh_create_issue_fn,
            confirm=args.confirm_open,
            allowed_repos=args.allow_repo,
            existing_issues_fn=existing_fn,
        )
        if args.live_existing_issues and not args.confirm_open:
            result["audit_trail"].append({"action": "skipped", "reason": "live_existing_issues_requires_confirm_open"})
        print(json.dumps({"promotion_result": result}, ensure_ascii=False, indent=2))
        return 0

    result = propose_issues(
        _load_findings(args.findings_json),
        args.repo,
        args.min_confidence,
        args.max_proposals,
        None,
        queue_path=args.queue_path,
        dashboard_path=args.dashboard_path,
        rejection_path=args.rejection_path,
    )
    if args.live_existing_issues and not args.confirm_open:
        result["audit_trail"].append({
            "action": "skipped",
            "reason": "live_existing_issues_requires_confirm_open",
        })
    if args.confirm_open and result["fail_open"]:
        open_result = {
            "fail_open": True,
            "dry_run": True,
            "opened": [],
            "would_open": [{"repo": p.get("repo"), "title": p.get("title"), "body": p.get("body"), "labels": p.get("labels", [])} for p in result["proposals"]],
            "audit_trail": [{"action": "skipped", "reason": "proposal_fail_open_blocks_confirmed_issue_creation"}],
        }
    else:
        open_result = open_proposed_issues(result["proposals"], gh_create_issue_fn, confirm=args.confirm_open)
    print(json.dumps({"proposal_result": result, "open_result": open_result}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

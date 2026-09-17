#!/usr/bin/env python3
"""Propose reflection-loop findings as human-gated GitHub issues.

The module is standalone and deliberately fail-open: queue, dedup, or GitHub
adapter failures return audit metadata instead of crashing the caller. Dry-run is
the default; creating public issues requires ``confirm=True``.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable


IssueSearchFn = Callable[..., Any]
DedupFn = Callable[..., Any]
GhFn = Callable[..., Any]

DEFAULT_QUEUE_PATH = Path(__file__).with_name("issue_proposals_queue.jsonl")
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


def _fingerprint(title: str, body: str) -> str:
    payload = f"{_normalize_title(title)}\n{body.strip().lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _evidence_items(finding: dict[str, Any]) -> list[str]:
    raw_items: list[Any] = []
    for key in ("evidence", "evidence_refs", "supporting_evidence", "citations", "loci_finding_ids", "logs"):
        raw_items.extend(_as_list(finding.get(key)))
    out: list[str] = []
    for item in raw_items:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("line") or item.get("ref") or item.get("id") or item).strip()
        else:
            text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _has_concrete_evidence(finding: dict[str, Any]) -> bool:
    for item in _evidence_items(finding):
        if re.search(r"\b[\w./-]+:\d+\b", item):
            return True
        if re.search(r"\b[0-9a-f]{8}-[0-9a-f-]{13,}\b", item, flags=re.I):
            return True
        if len(item.split()) >= 4:
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


def _compose_body(finding: dict[str, Any], *, repo: str, confidence: float, evidence: list[str]) -> str:
    lines = [
        "> Machine-generated proposal from the Loci reflection loop. A human maintainer must review before any action.",
        "",
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
    body = _compose_body(finding, repo=repo, confidence=confidence, evidence=evidence)
    return {
        "title": title,
        "body": body,
        "repo": repo,
        "confidence": confidence,
        "evidence": evidence,
        "source_finding": dict(finding),
        "fingerprint": _fingerprint(title, body),
        "labels": ["loci-reflection", "machine-generated-proposal"],
    }


def _read_queue(queue_path: str | Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    if queue_path is None:
        return [], [], False
    path = Path(queue_path)
    if not path.exists():
        return [], [], False
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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
        return [], [{"error": f"queue read degraded: {exc}"}], True


def _append_queue(queue_path: str | Path | None, proposals: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    if queue_path is None or not proposals:
        return [], False
    path = Path(queue_path)
    rows = [{
        "status": "proposed",
        "repo": proposal["repo"],
        "title": proposal["title"],
        "body": proposal["body"],
        "confidence": proposal["confidence"],
        "evidence": proposal["evidence"],
        "fingerprint": proposal["fingerprint"],
        "labels": proposal.get("labels", []),
    } for proposal in proposals]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return [], False
    except Exception as exc:
        return [{"error": f"queue append degraded: {exc}"}], True


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

    The deterministic title/fingerprint checks still run first; this adapter is
    the semantic backstop for differently worded duplicates.
    """
    def dedup_fn(*, proposal: dict[str, Any], candidates: list[dict[str, Any]], source: str = "") -> dict[str, Any]:
        items = [{"text": f"{proposal.get('title', '')}\n{proposal.get('body', '')}", "candidate": proposal}]
        items.extend({"text": f"{candidate.get('title', '')}\n{candidate.get('body', '')}", "candidate": candidate} for candidate in candidates)
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
) -> dict[str, Any]:
    """Rank and persist issue proposals, returning accepted and rejected audit.

    Accepted items are appended to the local proposal queue. Re-running with the
    same queue makes the call idempotent because those items deduplicate before
    they can be appended again.
    """
    audit: list[dict[str, Any]] = []
    fail_open = False
    local_queue, queue_errors, queue_degraded = _read_queue(queue_path)
    existing_issues, issue_errors, issue_degraded = _call_existing_issues(existing_issues_fn, repo)
    fail_open = fail_open or queue_degraded or issue_degraded
    audit.extend({"action": "degraded", **error} for error in queue_errors + issue_errors)

    candidates: list[dict[str, Any]] = []
    for idx, raw_finding in enumerate(findings or []):
        finding = raw_finding if isinstance(raw_finding, dict) else {"text": str(raw_finding)}
        proposal = _proposal_from_finding(finding, repo)
        if not _has_concrete_evidence(finding):
            audit.append({"finding_index": idx, "action": "dropped", "reason": "missing_concrete_evidence", "title": proposal["title"]})
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
            })
            continue
        candidates.append(proposal)
        audit.append({"finding_index": idx, "action": "candidate", "reason": "passed_gates", "title": proposal["title"], "confidence": proposal["confidence"]})

    ranked = sorted(candidates, key=lambda item: (-item["confidence"], item["title"].lower()))
    accepted = ranked[: max(0, int(max_proposals))]
    for capped in ranked[len(accepted):]:
        audit.append({"action": "dropped", "reason": "max_proposals_cap", "title": capped["title"], "max_proposals": int(max_proposals)})
    append_errors, append_degraded = _append_queue(queue_path, accepted)
    fail_open = fail_open or append_degraded
    audit.extend({"action": "degraded", **error} for error in append_errors)
    for proposal in accepted:
        audit.append({"action": "proposed", "reason": "queued_for_human_review", "title": proposal["title"], "fingerprint": proposal["fingerprint"]})

    dropped = [item for item in audit if item.get("action") == "dropped"]
    deduped = [item for item in audit if item.get("action") == "deduped"]
    return {
        "fail_open": fail_open,
        "dry_run": True,
        "repo": repo,
        "queue_path": str(queue_path) if queue_path is not None else None,
        "proposals": accepted,
        "dropped": dropped,
        "deduped": deduped,
        "audit_trail": audit,
    }


def open_proposed_issues(proposals: list[dict[str, Any]], gh_fn: GhFn, *, confirm: bool = False) -> dict[str, Any]:
    """Open accepted proposals only when explicitly confirmed.

    With ``confirm=False`` this is a pure dry-run and never calls ``gh_fn``.
    """
    audit: list[dict[str, Any]] = []
    if not confirm:
        return {
            "fail_open": False,
            "dry_run": True,
            "opened": [],
            "would_open": [{"repo": p.get("repo"), "title": p.get("title"), "body": p.get("body"), "labels": p.get("labels", [])} for p in proposals or []],
            "audit_trail": [{"action": "dry_run", "reason": "confirm_false", "title": p.get("title")} for p in proposals or []],
        }
    opened: list[Any] = []
    fail_open = False
    for proposal in proposals or []:
        try:
            try:
                result = gh_fn(repo=proposal.get("repo"), title=proposal.get("title"), body=proposal.get("body"), labels=proposal.get("labels", []))
            except TypeError:
                result = gh_fn(proposal)
            opened.append(result)
            audit.append({"action": "opened", "title": proposal.get("title"), "result": result})
        except Exception as exc:
            fail_open = True
            audit.append({"action": "degraded", "reason": "gh_fn_failed", "title": proposal.get("title"), "error": str(exc)})
    return {"fail_open": fail_open, "dry_run": False, "opened": opened, "would_open": [], "audit_trail": audit}


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
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--max-proposals", type=int, default=DEFAULT_MAX_PROPOSALS)
    parser.add_argument("--confirm-open", action="store_true", help="Actually open GitHub issues after proposing.")
    parser.add_argument("--live-existing-issues", action="store_true", help="Use gh to deduplicate against open issues.")
    args = parser.parse_args(argv)

    existing_fn = gh_issue_search_fn if args.live_existing_issues and args.confirm_open else None
    result = propose_issues(
        _load_findings(args.findings_json),
        args.repo,
        args.min_confidence,
        args.max_proposals,
        existing_fn,
        queue_path=args.queue_path,
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
    print(json.dumps({"proposal_result": result, "open_result": open_result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

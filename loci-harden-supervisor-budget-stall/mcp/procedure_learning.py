"""Automatic procedure promotion + outcome recording for confirmed findings.

This module closes the manual loop between:
  1. verifying that a finding is true, and
  2. remembering to promote repeatable action-shaped findings into procedure memory,
     then later recording real-world execution outcomes against them.

The promotion path is intentionally narrow and fail-open:
  - only a confirmed, non-degraded verification may promote;
  - a cheap local classification/heuristic gate decides whether the finding reads
    like a repeatable action/runbook step rather than a one-off observation;
  - any model / I/O / Qdrant problem returns a degraded result instead of raising.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional

from inv_store import _atomic_write_text, _inv_dir, _locked_file, _read_jsonl, _save_manifest
from qdrant_ops import _qdrant_upsert
from text_ops import classify

logger = logging.getLogger("loci-mcp")

GenFn = Callable[..., dict]

_PROCEDURE_LABEL = "procedure"
_NON_PROCEDURE_LABEL = "non_procedure"
_ACTION_RE = re.compile(
    r"^(?:"
    r"run|rerun|restart|reboot|rotate|drain|cordon|reload|rebuild|recreate|reindex|"
    r"clear|flush|delete|remove|set|reset|update|enable|disable|install|uninstall|"
    r"verify|check|retry|revoke|grant|migrate|patch|apply|scale|roll\s+back|rollback"
    r")\b",
    re.I,
)
_ACTION_EFFECT_RE = re.compile(
    r"\b(?:restarting|running|rerunning|resetting|clearing|flushing|rotating|draining|"
    r"disabling|enabling|rebuilding|recreating|reindexing|patching|migrating)\b.*\b"
    r"(?:fix(?:es|ed)?|clear(?:s|ed)?|restore(?:s|d)?|resolve(?:s|d)?|unblock(?:s|ed)?)\b",
    re.I,
)
_PROCEDURE_CUE_RE = re.compile(
    r"\b(?:runbook|playbook|workaround|procedure|follow these steps|steps?:|first,|then\b|1\.)",
    re.I,
)
_NON_ACTION_RE = re.compile(
    r"^(?:the logs|the log|logs|telemetry|metrics|trace|observed|we saw|there is|there are|"
    r"the service|the host|the system|an error|errors?)\b",
    re.I,
)


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coerce_int(value) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, coerced)


def _is_procedure_finding(finding: dict) -> bool:
    if not isinstance(finding, dict):
        return False
    kind = str(finding.get("record_type") or finding.get("type") or "").strip().lower()
    return kind == "procedure"


def _verification_confirmed(verify_verdict) -> bool:
    if isinstance(verify_verdict, dict):
        verdict = str(verify_verdict.get("verdict") or "").strip().lower()
        if verdict != "confirmed":
            return False
        return not bool(verify_verdict.get("degraded"))
    return str(verify_verdict or "").strip().lower() == "confirmed"


def _heuristic_procedure_shape(text: str) -> Optional[bool]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return False
    if _ACTION_RE.search(normalized):
        return True
    if _ACTION_EFFECT_RE.search(normalized):
        return True
    if _PROCEDURE_CUE_RE.search(normalized):
        return True
    if _NON_ACTION_RE.search(normalized):
        return False
    return None


def _classify_procedure_shape(text: str, gen_fn: Optional[GenFn] = None) -> tuple[bool, bool, str]:
    heuristic = _heuristic_procedure_shape(text)
    if heuristic is True:
        return True, False, "heuristic_action_shape"
    if heuristic is False:
        return False, False, "heuristic_non_action_shape"

    prompt_text = (
        "Decide whether this text describes a repeatable action, fix, or runbook step "
        "that an agent could execute again, versus a one-off observation.\n"
        f"Text: {text}"
    )
    result = classify(prompt_text, [_PROCEDURE_LABEL, _NON_PROCEDURE_LABEL], gen_fn=gen_fn)
    label = str(result.get("label") or "")
    degraded = bool(result.get("degraded"))
    if label == _PROCEDURE_LABEL:
        return True, degraded, "classifier_action_shape"
    if label == _NON_PROCEDURE_LABEL:
        return False, degraded, "classifier_non_action_shape"
    return False, True, "classifier_unavailable"


def _find_finding(findings: list[dict], finding_id: str) -> dict | None:
    for finding in findings:
        if str(finding.get("id") or "") == finding_id:
            return finding
    return None


def _fresh_manifest(inv_dir: Path) -> dict | None:
    manifest_path = inv_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None
    if "owner" not in manifest:
        manifest["owner"] = ""
    if "acl" not in manifest:
        manifest["acl"] = []
    return manifest


def _normalized_procedure_meta(existing) -> dict:
    current = existing if isinstance(existing, dict) else {}
    return {
        "preconditions": str(current.get("preconditions") or ""),
        "steps": str(current.get("steps") or ""),
        "postconditions": str(current.get("postconditions") or ""),
        "success_count": _coerce_int(current.get("success_count")),
        "attempt_count": _coerce_int(current.get("attempt_count")),
    }


def _rewrite_findings(path: Path, findings: list[dict]) -> None:
    lines = "\n".join(json.dumps(f) for f in findings)
    _atomic_write_text(path, lines + ("\n" if lines else ""))


def maybe_promote_to_procedure(investigation_id: str,
                               finding_id: str,
                               verify_verdict,
                               gen_fn: Optional[GenFn] = None) -> dict:
    """Promote a confirmed action-shaped finding into procedure memory.

    Returns ``{promoted: bool, reason: str, degraded: bool}``. Never raises.
    """
    try:
        if not _verification_confirmed(verify_verdict):
            return {"promoted": False, "reason": "verification_not_confirmed", "degraded": False}

        inv_dir = _inv_dir(investigation_id)
        findings_path = inv_dir / "findings.jsonl"
        findings = _read_jsonl(findings_path)
        finding = _find_finding(findings, str(finding_id))
        if finding is None:
            return {"promoted": False, "reason": "finding_not_found", "degraded": False}
        if _is_procedure_finding(finding):
            return {"promoted": False, "reason": "already_procedure", "degraded": False}

        is_action, degraded, shape_reason = _classify_procedure_shape(
            str(finding.get("text") or ""), gen_fn=gen_fn,
        )
        if not is_action:
            return {"promoted": False, "reason": shape_reason, "degraded": degraded}

        lock_path = inv_dir / ".lock"
        manifest_degraded = False
        qdrant_degraded = False
        with _locked_file(lock_path, "a+", exclusive=True):
            findings = _read_jsonl(findings_path)
            target = _find_finding(findings, str(finding_id))
            if target is None:
                return {"promoted": False, "reason": "finding_not_found", "degraded": False}
            if _is_procedure_finding(target):
                return {"promoted": False, "reason": "already_procedure", "degraded": False}

            old_type = str(target.get("record_type") or target.get("type") or "observed").strip() or "observed"
            target["record_type"] = "procedure"
            target["type"] = "procedure"
            target["procedure_meta"] = _normalized_procedure_meta(target.get("procedure_meta"))

            _rewrite_findings(findings_path, findings)

            try:
                manifest = _fresh_manifest(inv_dir)
                if manifest is not None:
                    counts = manifest.setdefault("finding_counts", {})
                    if old_type != "procedure":
                        counts[old_type] = max(0, _coerce_int(counts.get(old_type)) - 1)
                    counts["procedure"] = _coerce_int(counts.get("procedure")) + 1
                    _save_manifest(manifest)
                else:
                    manifest_degraded = True
            except Exception as exc:
                manifest_degraded = True
                logger.warning("maybe_promote_to_procedure: manifest update failed (fail-open): %s", exc)

        try:
            _qdrant_upsert(str(finding_id), str(target.get("text") or ""), target)
        except Exception as exc:
            qdrant_degraded = True
            logger.debug("maybe_promote_to_procedure: qdrant upsert failed (fail-open): %r", exc)

        degraded = degraded or manifest_degraded or qdrant_degraded
        if manifest_degraded:
            reason = "promoted_manifest_degraded"
        elif qdrant_degraded:
            reason = "promoted_qdrant_degraded"
        else:
            reason = "promoted"
        return {"promoted": True, "reason": reason, "degraded": degraded}

    except Exception as exc:
        logger.warning("maybe_promote_to_procedure: unexpected error (fail-open): %s", exc)
        return {"promoted": False, "reason": "unexpected_error", "degraded": True}


def record_execution_outcome(investigation_id: str,
                             finding_id: str,
                             success,
                             source: str = "auto",
                             gen_fn: Optional[GenFn] = None) -> dict:
    """Ensure a finding is a procedure, then delegate outcome recording to procedure_attempt."""
    try:
        promoted = maybe_promote_to_procedure(
            investigation_id=investigation_id,
            finding_id=finding_id,
            verify_verdict={"verdict": "confirmed", "degraded": False},
            gen_fn=gen_fn,
        )
        if not promoted.get("promoted") and promoted.get("reason") != "already_procedure":
            return {
                "recorded": False,
                "source": source,
                "reason": str(promoted.get("reason") or "not_procedure"),
                "degraded": bool(promoted.get("degraded")),
            }

        import server as _server

        raw = _server.procedure_attempt(
            investigation_id=investigation_id,
            finding_id=finding_id,
            success=_coerce_bool(success),
        )
        try:
            result = json.loads(raw)
        except Exception:
            return {
                "recorded": False,
                "source": source,
                "reason": "procedure_attempt_non_json",
                "degraded": True,
            }
        if not isinstance(result, dict):
            return {
                "recorded": False,
                "source": source,
                "reason": "procedure_attempt_non_dict",
                "degraded": True,
            }
        if "error" in result:
            return {
                "recorded": False,
                "source": source,
                "reason": str(result.get("error") or "procedure_attempt_error"),
                "degraded": True,
            }
        return {
            **result,
            "recorded": True,
            "source": source,
            "degraded": bool(promoted.get("degraded")),
        }
    except Exception as exc:
        logger.warning("record_execution_outcome: unexpected error (fail-open): %s", exc)
        return {
            "recorded": False,
            "source": source,
            "reason": "unexpected_error",
            "degraded": True,
        }

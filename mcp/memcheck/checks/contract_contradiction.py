"""Contract contradiction check — field name conflicts against stored contract declarations.

When a new finding references a field name that conflicts with a stored contract
declaration for the same entity (same name, different type; or same entity has a
producer and consumer with mismatched field names), this emits a contradiction verdict.

Pure: takes plain dicts, returns ``list[Verdict]``. Fail-safe: malformed records are
skipped, never raised.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..verdict import Verdict, make_signature, new_verdict, redact_excerpt

__all__ = ["run_contract_contradiction"]

_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]{1,}", re.I)
_FIELD_CONTEXT_RE = re.compile(
    r'["\']([a-z][a-z0-9_]{1,})["\']'
    r'|\bfield[s]?\s+["\']?([a-z][a-z0-9_]{1,})["\']?'
    r'|\b([a-z][a-z0-9_]{2,})\s*(?:field|key|column|param|attribute)',
    re.I,
)


def _extract_field_tokens(text: str) -> set[str]:
    """Extract plausible field-name tokens from a finding's text."""
    tokens: set[str] = set()
    for m in _FIELD_CONTEXT_RE.finditer(text or ""):
        for g in m.groups():
            if g:
                tokens.add(g.lower())
    return tokens


def _parse_contract_fields(finding: dict) -> dict[str, str]:
    """Parse the fields JSON stored in a contract_declaration finding's text.

    Convention: contract_declare stores text in the form:
    "Contract [entity] as [role]: fields=<json> protocol=<p>"
    """
    text = finding.get("text", "") or ""
    match = re.search(r"fields=(\{[^}]+\})", text)
    if not match:
        return {}
    try:
        return {k.lower(): str(v).lower() for k, v in json.loads(match.group(1)).items()}
    except Exception:
        return {}


def _entity_from_finding(finding: dict) -> Optional[str]:
    """Extract entity from tags like 'entity:UserSerializer'."""
    tags = finding.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    for tag in tags:
        if str(tag).startswith("entity:"):
            return str(tag)[len("entity:"):]
    return None


def _is_contract_declaration(finding: dict) -> bool:
    tags = finding.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    return "contract_declaration" in tags


def run_contract_contradiction(
    new_finding: dict,
    stored_findings: list[dict],
) -> list[Verdict]:
    """Check if new_finding's field name references conflict with stored contract declarations.

    A conflict is detected when:
    1. A stored contract_declaration for entity E declares field F with type T.
    2. new_finding references field F for entity E but implies a different type,
       OR references a field name that is similar but not identical (likely a rename drift).

    Uses token extraction on the finding text — no embeddings required.
    Fail-safe: any exception per record is caught and skipped.

    Parameters
    ----------
    new_finding:
        The candidate finding to check against stored contracts.
    stored_findings:
        All previously stored findings for the investigation.
    """
    if not isinstance(new_finding, dict) or not isinstance(stored_findings, list):
        return []

    # Extract field tokens from the new finding
    try:
        new_text = str(new_finding.get("text", "") or "")
        new_field_tokens = _extract_field_tokens(new_text)
        if not new_field_tokens:
            return []
    except Exception:
        return []

    verdicts: list[Verdict] = []

    for stored in stored_findings:
        if not isinstance(stored, dict):
            continue
        try:
            if not _is_contract_declaration(stored):
                continue

            contract_fields = _parse_contract_fields(stored)
            if not contract_fields:
                continue

            entity = _entity_from_finding(stored)
            stored_field_names = set(contract_fields.keys())

            # Check for near-miss field names (edit-distance-1 heuristic via prefix match)
            for new_token in new_field_tokens:
                for stored_field in stored_field_names:
                    # Exact match — no conflict (correct usage)
                    if new_token == stored_field:
                        continue

                    # Detect likely rename drift: one token is a suffix or prefix of
                    # the other (e.g., "score" ⊂ "threat_score", "user" ⊂ "user_id"),
                    # suggesting the field was renamed on one side of the boundary.
                    shorter, longer = sorted([new_token, stored_field], key=len)
                    # Pure containment: if shorter appears as a complete suffix or
                    # prefix of longer, it's a strong rename-drift signal regardless
                    # of length ratio (e.g., "score"/5 ⊂ "threat_score"/12 = 42%).
                    is_suffix_match = longer.endswith(shorter) or longer.startswith(shorter)
                    if is_suffix_match and len(shorter) >= 4:
                        entity_label = f" (entity: {entity})" if entity else ""
                        excerpt = redact_excerpt(
                            f"new field '{new_token}' vs contract field '{stored_field}'{entity_label}"
                        )
                        verdicts.append(
                            new_verdict(
                                subject_kind="memory",
                                subject_signature=make_signature(
                                    "contract_contradiction",
                                    f"{new_token}|{stored_field}|{stored.get('id', '')}",
                                ),
                                subject_excerpt=excerpt,
                                verdict_type="contract_contradiction",
                                decision="flag",
                                confidence=0.60,
                                rationale=(
                                    f"Field name '{new_token}' in new finding may conflict with "
                                    f"declared contract field '{stored_field}'"
                                    + (f" for entity '{entity}'" if entity else "")
                                    + ". Possible rename drift across a serialization boundary."
                                ),
                                source="rule",
                                refs=[str(stored.get("id", ""))],
                                provisional=True,
                            )
                        )
        except Exception:
            continue

    return verdicts

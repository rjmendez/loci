"""Verdict schema for the memcheck core.

A ``Verdict`` is an accumulated judgment about a *subject* — a CLI action, an
LLM output, or a hermes memory/finding. The dataclass is wire-compatible with
the Rust ``EnforcementMemory`` struct so the same qdrant collection
(``hermes_verdicts``) can be shared between the Python and Rust layers.

Pure stdlib — no third-party imports. The qdrant/mnemosyne backends layer on
top of this without changing the schema.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

__all__ = [
    "Verdict",
    "new_verdict",
    "make_signature",
    "redact_excerpt",
]

# Subject kinds, decisions, and sources are kept as plain strings (mirroring the
# Rust enums serialized as strings) rather than Python enums, to stay
# wire-compatible and trivially JSON-serializable.

_WHITESPACE_RE = re.compile(r"\s+")


def _now_iso() -> str:
    """Current UTC time as an ISO8601 string with a trailing ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _normalize(text: str) -> str:
    """Lowercase + collapse all runs of whitespace to a single space, stripped."""
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def make_signature(subject_kind: str, text: str) -> str:
    """Stable signature for a subject.

    sha256 of ``f"{subject_kind}:{normalized}"`` where ``normalized`` is the
    text lowercased with whitespace collapsed. Stable across runs and machines
    so the same subject always maps to the same signature (and therefore the
    same qdrant point id downstream).
    """
    normalized = _normalize(text)
    digest = hashlib.sha256(f"{subject_kind}:{normalized}".encode("utf-8"))
    return digest.hexdigest()


def redact_excerpt(text: str, max_chars: int = 512) -> str:
    """Char-boundary-safe truncation of an excerpt.

    Python ``str`` slicing is codepoint-based, so this is always on a character
    boundary. Never stores more than ``max_chars`` characters. (The redaction of
    actual secrets is the caller's responsibility — this only bounds length.)
    """
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


@dataclass
class Verdict:
    """An accumulated judgment about a subject.

    Field order and names mirror the Rust ``Verdict`` struct exactly so the
    JSON payload round-trips between the two implementations.
    """

    id: str
    subject_kind: str          # "action" | "output" | "memory"
    subject_signature: str     # stable id: finding-id, or hash(kind + normalized text)
    subject_excerpt: str       # truncated + redacted (never raw secrets)
    verdict_type: str          # "unsupported_observed" | "poisoned_suspect" | "contradiction" | ...
    decision: str              # "flag" | "warn" | "quarantine" | "allow"
    confidence: float
    rationale: str
    source: str                # "rule" | "llm" | "human" | "recalled"
    refs: list[str] = field(default_factory=list)
    occurrences: int = 1
    first_seen: str = ""
    last_seen: str = ""
    # Provisional verdicts are recorded but not enforced until confirmed by a
    # second independent observation (PE-gated reconsolidation, EWC protection).
    provisional: bool = False

    def to_payload(self) -> dict:
        """Serialize to a plain dict for backend storage (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict) -> "Verdict":
        """Rebuild a Verdict from a stored payload.

        Tolerant of extra keys (backends may attach their own metadata) and of
        missing optional fields (defaults are applied).
        """
        return cls(
            id=payload["id"],
            subject_kind=payload["subject_kind"],
            subject_signature=payload["subject_signature"],
            subject_excerpt=payload.get("subject_excerpt", ""),
            verdict_type=payload["verdict_type"],
            decision=payload["decision"],
            confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.0)))),
            rationale=payload.get("rationale", ""),
            source=payload.get("source", "rule"),
            refs=list(payload.get("refs", []) or []),
            occurrences=int(payload.get("occurrences", 1)),
            first_seen=payload.get("first_seen", ""),
            last_seen=payload.get("last_seen", ""),
            provisional=bool(payload.get("provisional", False)),
        )


def new_verdict(
    *,
    subject_kind: str,
    subject_signature: str,
    subject_excerpt: str,
    verdict_type: str,
    decision: str,
    confidence: float,
    rationale: str,
    source: str,
    refs: list[str] | None = None,
    provisional: bool = False,
) -> Verdict:
    """Factory for a fresh Verdict.

    Sets ``id`` to a uuid4 hex, ``occurrences`` to 1, and both timestamps to the
    current UTC time. The excerpt is run through ``redact_excerpt`` to bound its
    length.
    """
    now = _now_iso()
    return Verdict(
        id=uuid.uuid4().hex,
        subject_kind=subject_kind,
        subject_signature=subject_signature,
        subject_excerpt=redact_excerpt(subject_excerpt),
        verdict_type=verdict_type,
        decision=decision,
        confidence=confidence,
        rationale=rationale,
        source=source,
        refs=list(refs) if refs else [],
        occurrences=1,
        first_seen=now,
        last_seen=now,
        provisional=provisional,
    )

"""memcheck — the shared verdict core for Loci memory self-check and CLI policy.

Record verdicts about subjects (action / output / memory), recall semantically-
similar prior verdicts, and promote a confident, recurring verdict into a
deterministic decision — fail-open.

Importing this package requires only the standard library. The qdrant and
mnemosyne backends are importable as bare classes (their heavy dependencies are
injected at construction time, never imported at module top), so
``import memcheck`` works without qdrant / fastembed / mnemosyne installed.
"""

from .backend import (
    InMemoryBackend,
    ScoredVerdict,
    VerdictBackend,
    cosine_similarity,
)
from .engine import EmlConfig, RecalledDecision, VerdictEngine
from .mnemosyne import MnemosyneBackend
from .qdrant import QdrantBackend
from .verdict import Verdict, make_signature, new_verdict, redact_excerpt

__all__ = [
    "Verdict",
    "new_verdict",
    "make_signature",
    "redact_excerpt",
    "ScoredVerdict",
    "VerdictBackend",
    "InMemoryBackend",
    "QdrantBackend",
    "MnemosyneBackend",
    "VerdictEngine",
    "EmlConfig",
    "RecalledDecision",
    "cosine_similarity",
]

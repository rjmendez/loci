"""The cosine grounding gate of investigation_reason, as one importable rule.

A finding is kept when the cosine of its embedding to the question is at least the
threshold (0.59 by deep-think convention), and the CONTEXT_CAP highest kept findings go
into the prompt, highest first; ties keep input order. investigation_reason calls
``cosine_gate`` and so does the offline D10 replay (scripts/d10_replay.py), so the
replay measures the rule the server runs rather than a copy of it.

An unanswerable cosine (``None``: a dimension mismatch or a zero vector, see
vecmath.py) is dropped. Before this module existed that case raised a TypeError
inside the tool.
"""
from __future__ import annotations

from typing import Optional, Sequence, TypeVar

GROUND_THRESHOLD = 0.59
CONTEXT_CAP = 12

T = TypeVar("T")


def passes(cos: Optional[float], threshold: float = GROUND_THRESHOLD) -> bool:
    """The per-finding test: an answerable cosine at or above the threshold."""
    return cos is not None and cos >= threshold


def cosine_gate(scored: Sequence[tuple[Optional[float], T]], threshold: float = GROUND_THRESHOLD) -> list[T]:
    """The items the gate puts in the prompt, from ``(cosine, item)`` pairs."""
    kept = [(c, f) for c, f in scored if passes(c, threshold)]
    return [f for _, f in sorted(kept, key=lambda x: x[0], reverse=True)[:CONTEXT_CAP]]

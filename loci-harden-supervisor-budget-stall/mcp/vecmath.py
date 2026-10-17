"""Cosine similarity that refuses to compare vectors of different lengths.

Six copies of the same eight lines existed across mcp/ and scripts/. Four
checked the lengths; two did not, and `zip` truncates silently:

    cosine([0.1]*768, [0.1]*384) == 0.707107

Not an error and not a zero — a plausible similarity computed over the first 384
dimensions. This repo mixes 768-float nomic embeddings with the 384-float
hash_embed in mcp/memcheck/vectors.py, so the two meeting is a live possibility
rather than a hypothetical, and glymphatic_sweep uses cosine to decide which
memories to consolidate or discard.

The four that DID check returned 0.0, which is only half right: 0.0 means
"orthogonal — definitely not a match", a confident negative that a ranker will
act on. A dimension mismatch is not a weak match, it is an unanswerable
question, so this returns None and every caller decides what unavailable means
for it.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Cosine similarity, or None when the question cannot be answered.

    None is returned for an empty vector, a length mismatch, or a zero-magnitude
    vector — all cases where no similarity exists rather than a low one.
    """
    if not a or not b:
        return None
    if len(a) != len(b):
        return None
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return None
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def cosine_or(a: Sequence[float], b: Sequence[float], default: float = 0.0) -> float:
    """For call sites that genuinely want a float and have decided what an
    unanswerable comparison should count as. Naming the default at the call site
    is the point: it makes the choice visible instead of buried in the helper."""
    v = cosine(a, b)
    return default if v is None else v

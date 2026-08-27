"""The one definition of the verdict collection's vector contract.

Four call sites write verdicts — the memcheck PreToolUse hook, ``verdict_ops``,
and ``server.py``'s record/forget paths — and ``memory_health`` reports on the
collection. They must agree on name, dimension, vector name and embedder:
whichever writer creates the collection first pins its dimension, and every
other writer then fails on every upsert.

The dimension is 384 (a deterministic hash), not the server's 768-dim model,
because the hook writes this collection on every tool call and may not load a
model, and because nothing reads the vector: every read is an exact uuid5
point-id lookup or a payload scroll.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

__all__ = [
    "COLLECTION",
    "EMBED_DIM",
    "VECTOR_NAME",
    "VerdictDimensionMismatch",
    "hash_embed",
    "dense_dim",
    "ensure_collection",
]

_log = logging.getLogger("memcheck")

COLLECTION = "loci_verdicts"
VECTOR_NAME = "dense"
EMBED_DIM = 384


class VerdictDimensionMismatch(RuntimeError):
    """The live verdict collection was built for a different vector dimension."""


def hash_embed(text: str) -> list[float]:
    """Deterministic text -> ``EMBED_DIM``-float vector in [-1, 1].

    Seeds a byte stream from ``sha256(text)`` and tiles its bytes to
    ``EMBED_DIM`` floats. No model, no network — identical text always yields an
    identical vector, which is all the EXACT-match recall path needs (recall is
    keyed on the stable point id, not on vector proximity). Returning a real
    384-dim vector keeps the qdrant collection's named ``dense`` vector valid.
    """
    if text is None:
        text = ""
    # Expand the 32-byte digest deterministically to >= EMBED_DIM bytes by
    # hashing (digest || counter) repeatedly.
    out: list[float] = []
    counter = 0
    seed = text.encode("utf-8")
    while len(out) < EMBED_DIM:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for b in block:
            if len(out) >= EMBED_DIM:
                break
            # byte 0..255 -> float in [-1, 1]
            out.append((b / 127.5) - 1.0)
        counter += 1
    return out


def dense_dim(info) -> Optional[int]:
    """Size of ``info``'s named ``dense`` vector, or None if not readable.

    ``info`` is a qdrant ``CollectionInfo``. ``config.params.vectors`` is a dict
    of named ``VectorParams`` for a named-vector collection and a bare
    ``VectorParams`` for an unnamed one; both shapes are accepted.
    """
    params = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
    if isinstance(params, dict):
        params = params.get(VECTOR_NAME)
    size = getattr(params, "size", None)
    return int(size) if isinstance(size, int) else None


def ensure_collection(client, collection: str = COLLECTION) -> None:
    """Create the verdict collection if absent; raise if it exists at another dim.

    The sole creator of this collection, so every writer creates the identical
    schema and a dimension can never be pinned by whoever happens to write
    first. An existing collection at the wrong dimension raises
    :class:`VerdictDimensionMismatch` here — at construction, naming both dims —
    rather than surfacing as an opaque 400 on the first upsert.
    """
    from qdrant_client import models as qmodels

    # Only a 404 means absent. Treating every failure as absent turned an
    # unreachable qdrant into get + create, two timeouts on a hook budgeted ~1s.
    try:
        info = client.get_collection(collection)
    except Exception as exc:  # noqa: BLE001
        if getattr(exc, "status_code", None) != 404:
            raise
        info = None

    if info is None:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                VECTOR_NAME: qmodels.VectorParams(
                    size=EMBED_DIM, distance=qmodels.Distance.COSINE
                )
            },
        )
        return

    params = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
    if not isinstance(params, dict) or VECTOR_NAME not in params:
        msg = (
            f"qdrant collection {collection!r} has no named {VECTOR_NAME!r} vector; "
            f"this build writes one and every upsert would fail"
        )
        _log.error("%s", msg)
        raise VerdictDimensionMismatch(msg)

    dim = dense_dim(info)
    if dim != EMBED_DIM:
        msg = (
            f"qdrant collection {collection!r} has {VECTOR_NAME} dim {dim}, but this "
            f"build writes {EMBED_DIM}-dim vectors; every upsert would fail"
        )
        _log.error("%s", msg)
        raise VerdictDimensionMismatch(msg)

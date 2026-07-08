"""Embedding-tier operations for Loci-native workflows — the offload that WORKS today.

Generation on the local GPU is currently unreliable, but the embedding path (Ollama
nomic-embed-text, 768-dim, warm on GPU) is solid. These are the cheap semantic ops that
need only embeddings — no generation model — so they run on local GPU at ~zero token cost:

- dedup(items):      cluster near-duplicate findings so an N-way fan-out doesn't
                     triple-report the same issue (barrier-stage / synthesis dedup).
- relevance(texts):  cosine of each text to a topic — a cheap gate/router that trims
                     what reaches Claude, replacing a classifier agent.

All fail-open: if embeddings are unavailable, dedup returns every item as its own cluster
(i.e. no dedup) and relevance returns null scores, with degraded=True — never an exception.

Embeddings come from OLLAMA_BASE_URL/EMBED_MODEL (same config the Loci server uses). The
embed function is injectable so callers can reuse a warm client and tests can stub it.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

_OLLAMA = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_URL") or ""
_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed via Ollama /api/embed. Returns [] on any failure (fail-open)."""
    texts = [t if isinstance(t, str) else str(t) for t in texts]
    if not texts or not _OLLAMA:
        return []
    try:
        import requests
        r = requests.post(f"{_OLLAMA}/api/embed",
                          json={"model": _EMBED_MODEL, "input": texts}, timeout=60)
        r.raise_for_status()
        embs = r.json().get("embeddings") or []
        return embs if len(embs) == len(texts) else []
    except Exception:
        return []


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _text_of(item, key: Optional[str]) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if key and key in item:
            return str(item[key])
        for k in ("text", "content", "summary", "title"):
            if item.get(k):
                return str(item[k])
    return str(item)


def dedup(items: list, threshold: float = 0.88, key: Optional[str] = None,
          embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None) -> dict:
    """Greedy near-duplicate clustering by cosine >= threshold.

    Returns {clusters:[{rep_index, member_indices, text}], kept:[items], dropped:int,
             degraded:bool}. `kept` is one representative (the first seen) per cluster —
    feed it downstream in place of the raw list. Order-stable. Fail-open: with no
    embeddings, every item is its own cluster (nothing dropped) and degraded=True.
    """
    items = list(items or [])
    n = len(items)
    if n <= 1:
        return {"clusters": [{"rep_index": i, "member_indices": [i], "text": _text_of(items[i], key)}
                             for i in range(n)],
                "kept": items, "dropped": 0, "degraded": False}
    ef = embed_fn or embed_texts
    vecs = ef([_text_of(it, key) for it in items])
    degraded = len(vecs) != n
    clusters: list[dict] = []
    if degraded:
        # no embeddings -> no dedup, each its own cluster
        for i, it in enumerate(items):
            clusters.append({"rep_index": i, "member_indices": [i], "text": _text_of(it, key)})
        return {"clusters": clusters, "kept": items, "dropped": 0, "degraded": True}
    reps: list[int] = []  # representative index per cluster
    for i in range(n):
        placed = False
        for ci, rep in enumerate(reps):
            if _cosine(vecs[i], vecs[rep]) >= threshold:
                clusters[ci]["member_indices"].append(i)
                placed = True
                break
        if not placed:
            reps.append(i)
            clusters.append({"rep_index": i, "member_indices": [i], "text": _text_of(items[i], key)})
    kept = [items[c["rep_index"]] for c in clusters]
    return {"clusters": clusters, "kept": kept, "dropped": n - len(clusters), "degraded": False}


def relevance(texts: list[str], topic: str,
              embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None) -> dict:
    """Cosine of each text to `topic`. Returns {scores:[float|None], degraded:bool}.
    scores align with `texts`; None when embeddings are unavailable (degraded=True)."""
    texts = list(texts or [])
    if not texts or not topic:
        return {"scores": [None] * len(texts), "degraded": not bool(topic)}
    ef = embed_fn or embed_texts
    vecs = ef([topic] + [str(t) for t in texts])
    if len(vecs) != len(texts) + 1:
        return {"scores": [None] * len(texts), "degraded": True}
    tvec = vecs[0]
    return {"scores": [round(_cosine(tvec, v), 4) for v in vecs[1:]], "degraded": False}

"""Pluggable GPU reranker — a reusable, model-pluggable two-stage reranking unit.

This factors the cross-encoder reranking that currently lives inline in server.py
(~588-609, the lazy `_get_cross_encoder` + `_cross_encoder` global) into one
importable primitive. server.py should delegate to `rerank()` here instead of
carrying its own CrossEncoder singleton.

Two backends, chosen by env RERANK_MODEL:
  - 'cross-encoder/ms-marco-MiniLM-L-6-v2'  (DEFAULT — reproduces current server.py
    behavior exactly [rerank]: the 22M-param ms-marco MiniLM cross-encoder.)
  - 'BAAI/bge-reranker-v2-m3'               (opt-in — a stronger reranker that lifts
    retrieval precision [rerank]; heavier, multilingual.)
Any other value is passed through verbatim to CrossEncoder, so operators can point at
an arbitrary sentence-transformers cross-encoder without a code change.

Substrate facts this is built against (session grounding):
  - [hardware] torch in mcp/.venv sees cuda:0 (RTX 4070 Ti). We load the model on
    'cuda' when torch.cuda.is_available(), else 'cpu' — mirroring the intent of the
    server.py cross-encoder path.
  - [rerank] server.py already reranks on GPU with a lazily-loaded, globally-cached
    CrossEncoder. This module mirrors that: lazy-init, cache in a module global, and
    on load failure set the cache to `False` PERMANENTLY so we never re-attempt a
    known-broken load per call.
  - [pattern:fail-open] EVERY op fails open (mirror embed_ops.py / llm_local.py): if
    the model can't load (or no model_fn is injected and the real model is
    unavailable), we return the docs in ORIGINAL order with score=None and set the
    'degraded' marker. We NEVER raise.
  - [pattern:injectable] `model_fn` is injectable — a callable (query, docs) -> list
    of float scores. It defaults to None and is resolved lazily to the real
    CrossEncoder at call time. Tests pass a stub so NO model is downloaded and no GPU
    is touched.

How server.py should delegate (do NOT edit server.py as part of this change):
    from reranker import rerank
    ranked = rerank(query, [hit.text for hit in hits], top_k=k)
    # ranked -> [{"index": i, "score": s or None, "text": t}, ...] sorted desc.
    # 'index' is the position in the INPUT docs list, so callers can re-associate
    # each result with its original hit/payload. On degraded results score is None
    # and order is preserved, so a caller can safely fall back to bi-encoder order.
The existing server.py `_get_cross_encoder()` / `_cross_encoder` global and the inline
`model.predict([[query, doc] ...])` call can then be deleted in favor of this module.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Default backend reproduces current server.py behavior [rerank].
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-global model cache, mirroring server.py's `_cross_encoder` singleton [rerank]:
#   None  -> not yet attempted (lazy)
#   False -> load failed permanently; never retry (fail-open)
#   obj   -> a loaded CrossEncoder
_model = None
_model_lock = threading.Lock()
# Which model name `_model` was loaded for, so a changed RERANK_MODEL forces a reload.
_model_name: Optional[str] = None


def _model_id() -> str:
    """The configured backend, read from env each call so it can be overridden at runtime."""
    return os.environ.get("RERANK_MODEL") or _DEFAULT_MODEL


def _get_model():
    """Lazy-init + globally cache a sentence_transformers.CrossEncoder on GPU if available.

    Mirrors server.py ~588-609: double-checked locking, cache in a module global, and on
    ANY load failure set the global to `False` permanently so we never re-attempt per call.
    Returns the loaded model, or None if unavailable/failed (fail-open — never raises).
    """
    global _model, _model_name
    wanted = _model_id()
    # Fast path: already loaded for the currently-configured model.
    if _model is not None and _model_name == wanted:
        return _model if _model is not False else None
    with _model_lock:
        if _model is not None and _model_name == wanted:  # re-check under lock
            return _model if _model is not False else None
        _model_name = wanted
        try:
            # Resolve device lazily: cuda:0 (RTX 4070 Ti) when torch sees it, else cpu [hardware].
            device = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                device = "cpu"
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(wanted, max_length=512, device=device)
            logger.info("Reranker enabled (%s on %s)", wanted, device)
        except ImportError:
            logger.debug("sentence-transformers not installed — reranking disabled")
            _model = False
        except Exception as exc:
            logger.warning("Reranker init failed — reranking disabled: %s", exc)
            _model = False
    return _model if _model is not False else None


def get_model():
    """Public accessor: the lazily-loaded, globally-cached CrossEncoder for the currently
    configured RERANK_MODEL, or None if unavailable (fail-open). server.py delegates its
    cross-encoder singleton to this so the backend is env-pluggable without touching call
    sites (they keep calling `.predict(pairs)` on the returned model)."""
    return _get_model()


def _real_model_fn(query: str, docs: Sequence[str]) -> Optional[List[float]]:
    """Default model_fn: score each doc with the cached CrossEncoder. None if unavailable."""
    model = _get_model()
    if model is None:
        return None
    try:
        scores = model.predict([[query, d] for d in docs])
        return [float(s) for s in scores]
    except Exception as exc:
        logger.warning("Reranker predict failed — falling back to original order: %s", exc)
        return None


def rerank(query: str,
           docs: Sequence[str],
           top_k: Optional[int] = None,
           model_fn: Optional[Callable[[str, Sequence[str]], Optional[Sequence[float]]]] = None,
           ) -> List[dict]:
    """Rerank `docs` against `query`, best-first. Fail-open — never raises.

    Args:
        query: the query string.
        docs: the candidate documents (strings) to reorder.
        top_k: if set, cap the returned list to the top_k highest-scoring docs.
        model_fn: injectable scorer, `fn(query, docs) -> sequence[float]` (one score per
            doc). Defaults to None -> the lazily-resolved on-GPU CrossEncoder. Tests inject
            a stub so NO model is downloaded [pattern:injectable].

    Returns:
        A list of {"index": int, "score": float|None, "text": str} sorted by score
        descending. `index` is the position in the INPUT `docs` list.

        Fail-open [pattern:fail-open]: if scoring is unavailable (no model_fn and the real
        model can't load, or the model_fn returns None / a wrong-length list / raises), the
        docs are returned in ORIGINAL order with score=None and each dict carries
        "degraded": True. Empty docs -> [].
    """
    items = list(docs)
    n = len(items)
    if n == 0:
        return []

    def _passthrough() -> List[dict]:
        out = [{"index": i, "score": None, "text": items[i], "degraded": True} for i in range(n)]
        if top_k is not None:
            out = out[: max(0, top_k)]
        return out

    scorer = model_fn if model_fn is not None else _real_model_fn

    try:
        scores = scorer(query, items)
    except Exception as exc:
        logger.warning("Reranker model_fn raised — falling back to original order: %s", exc)
        return _passthrough()

    # A None or malformed (wrong-length) score list means "unavailable" -> fail open.
    if scores is None:
        return _passthrough()
    scores = list(scores)
    if len(scores) != n:
        logger.warning("Reranker returned %d scores for %d docs — falling back", len(scores), n)
        return _passthrough()

    ranked = [{"index": i, "score": float(scores[i]), "text": items[i]} for i in range(n)]
    # Sort by score desc; stable sort keeps original order among ties.
    ranked.sort(key=lambda r: r["score"], reverse=True)
    if top_k is not None:
        ranked = ranked[: max(0, top_k)]
    return ranked

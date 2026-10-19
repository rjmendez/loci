"""Pluggable GPU reranker — a reusable, model-pluggable two-stage reranking unit.

This factors the cross-encoder reranking that currently lives inline in server.py
(~588-609, the lazy `_get_cross_encoder` + `_cross_encoder` global) into one
importable primitive. server.py should delegate to `rerank()` here instead of
carrying its own CrossEncoder singleton.

Two backends, chosen by env RERANK_MODEL (default resolved via backends.rerank_model()):
  - 'BAAI/bge-reranker-v2-m3'               (DEFAULT — a stronger reranker that lifts
    retrieval precision [rerank]; heavier (~600MB), multilingual. Flipped in on
    judge-eval evidence: +14% nDCG@10, no regression. See scripts/judge_eval.py.)
  - 'cross-encoder/ms-marco-MiniLM-L-6-v2'  (opt-in — the lighter 22M-param ms-marco
    MiniLM cross-encoder; pin it back on constrained hosts via RERANK_MODEL.)
Any other value is passed through verbatim to CrossEncoder, so operators can point at
an arbitrary sentence-transformers cross-encoder without a code change.

Third backend — RERANK_HTTP_URL: point at a llama.cpp `llama-server --rerank --pooling
rank` endpoint (e.g. serving gpustack/bge-reranker-v2-m3-GGUF) instead of loading
sentence-transformers/torch in this process at all. Takes priority over RERANK_MODEL when
set. Same real cross-encoder model, quantized and served out-of-process — removes the
~900MB torch install and its CUDA-context overhead from this process on hosts where that
matters (e.g. a shared Jetson also running an LLM/VLM stack via llama.cpp already).

Substrate facts this is built against (session grounding):
  - [hardware] We load the model on 'cuda' (cuda:0) when torch.cuda.is_available(), else
    'cpu' — mirroring the intent of the server.py cross-encoder path. Which physical GPU
    cuda:0 maps to is host-specific; see scripts/gpu_placement.md.
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

import contextlib
import contextvars
import logging
import os
import threading
import time
from typing import Callable, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Ultimate fallback if `backends` can't be imported. Flipped to bge on judge-eval
# evidence (+14% nDCG@10); pin MiniLM back via RERANK_MODEL on constrained hosts.
_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# Module-global model cache, mirroring server.py's `_cross_encoder` singleton [rerank]:
#   None  -> not yet attempted (lazy)
#   False -> load failed permanently; never retry (fail-open)
#   obj   -> a loaded CrossEncoder
_model = None
_model_lock = threading.Lock()
# Which model name `_model` was loaded for, so a changed RERANK_MODEL forces a reload.
_model_name: Optional[str] = None
# Where the loaded model runs ("cuda" / "cpu" / "http") and the torch build that decided it.
# Set by _get_model, read by health(): a CPU-only torch wheel on a GPU host is otherwise silent.
_device: Optional[str] = None
_torch_build: Optional[str] = None


def _model_id() -> str:
    """The configured backend, read from env each call so it can be overridden at runtime;
    else via backends (gitignored config), else the MiniLM default."""
    v = os.environ.get("RERANK_MODEL")
    if v:
        return v
    try:
        import backends
        return backends.rerank_model()
    except Exception:
        return _DEFAULT_MODEL


class _HttpCrossEncoder:
    """`.predict(pairs)`-compatible shim calling a llama.cpp `/rerank` endpoint instead
    of an in-process sentence-transformers CrossEncoder. `pairs` are [query, doc]
    two-element lists — matches CrossEncoder.predict()'s input shape. Every call site in
    this module builds pairs as [[query, d] for d in docs], i.e. one shared query per
    call, which this assumes (asserts on mixed queries rather than silently mis-scoring).
    """

    def __init__(self, url: str):
        self._url = url

    def predict(self, pairs):
        import json
        import urllib.request

        if not pairs:
            return []
        query = pairs[0][0]
        assert all(p[0] == query for p in pairs), "_HttpCrossEncoder requires one shared query per call"
        docs = [p[1] for p in pairs]
        body = json.dumps({"query": query, "documents": docs}).encode()
        req = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        scores = [0.0] * len(docs)
        for r in data.get("results", []):
            scores[r["index"]] = float(r["relevance_score"])
        return scores


def _http_url() -> str:
    return os.environ.get("RERANK_HTTP_URL", "")


def _get_model():
    """Lazy-init + globally cache the reranker backend (HTTP shim or local
    sentence_transformers.CrossEncoder on GPU if available).

    Mirrors server.py ~588-609: double-checked locking, cache in a module global, and on
    ANY load failure set the global to `False` permanently so we never re-attempt per call.
    Returns the loaded model, or None if unavailable/failed (fail-open — never raises).
    """
    global _model, _model_name, _device, _torch_build
    http_url = _http_url()
    wanted = f"http:{http_url}" if http_url else _model_id()
    # Fast path: already loaded for the currently-configured model.
    if _model is not None and _model_name == wanted:
        return _model if _model is not False else None
    with _model_lock:
        if _model is not None and _model_name == wanted:  # re-check under lock
            return _model if _model is not False else None
        _model_name = wanted
        if http_url:
            _model = _HttpCrossEncoder(http_url)
            _device, _torch_build = "http", None
            logger.info("Reranker enabled via HTTP backend (%s)", http_url)
            return _model
        try:
            # Resolve device lazily: cuda:0 when torch sees a GPU, else cpu [hardware].
            device = "cpu"
            build = None
            try:
                import torch
                build = str(getattr(torch, "__version__", "")) or None
                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                device = "cpu"
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(wanted, max_length=512, device=device)
            _device, _torch_build = device, build
            if device == "cpu":
                # Loud on purpose: a CPU build of torch on a GPU host once made every search take minutes.
                logger.warning(
                    "Reranker %s runs on CPU (torch %s): expect ~0.1-0.3 s per (query, doc) pair; "
                    "searches are capped by LOCI_RERANK_BUDGET_S", wanted, build)
            else:
                logger.info("Reranker enabled (%s on %s)", wanted, device)
        except ImportError:
            logger.debug("sentence-transformers not installed — reranking disabled")
            _model = False
        except Exception as exc:
            logger.warning("Reranker init failed — reranking disabled: %s", exc)
            _model = False
    return _model if _model is not False else None


# --------------------------------------------------------------------------- #
# Per-search budget: a rerank must never make a search take minutes.
#
# One rag_context_search scores (collections x expanded queries x limit*5) candidates plus a final
# top-20 pass: 60 pairs with expansion off, ~260 with five query variants on one collection and ~510
# on two. At 0.1-0.3 s per pair on a CPU-only torch that is minutes. The budget is a wall-clock
# deadline plus an optional pair cap, shared by every scoring call inside one search.
#
#   LOCI_RERANK_BUDGET_S   wall-clock seconds per search (default 20; 0 disables the deadline)
#   LOCI_RERANK_MAX_PAIRS  pairs scored per search by the candidate stage (default 0 = no cap)
#   LOCI_RERANK_CHUNK      pairs per model.predict call (default 16): the deadline is checked
#                          between chunks, so a search overshoots by at most one chunk
#
# The candidate stage may use at most (1 - FINAL_RESERVE) of the budget; the last quarter is held for
# the final cross-collection pass against the original query so it still runs after a slow
# candidate stage. When the budget runs out the scores obtained are kept, unscored rows follow in
# their original (bi-encoder) order, and the search is marked degraded.
# --------------------------------------------------------------------------- #
DEFAULT_BUDGET_S = 20.0
DEFAULT_CHUNK = 16
FINAL_RESERVE = 0.25


def _env_number(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _now() -> float:
    return time.monotonic()


class RerankBudget:
    """Wall-clock and pair budget shared by every scoring call of one search. `clock` is
    injectable so tests drive the deadline exactly."""

    def __init__(self, budget_s: Optional[float] = None, max_pairs: Optional[int] = None,
                 chunk: Optional[int] = None, clock: Optional[Callable[[], float]] = None):
        clock = _now if clock is None else clock   # resolved per construction so tests can swap _now
        self.budget_s = _env_number("LOCI_RERANK_BUDGET_S", DEFAULT_BUDGET_S) if budget_s is None else float(budget_s)
        self.max_pairs = int(_env_number("LOCI_RERANK_MAX_PAIRS", 0)) if max_pairs is None else int(max_pairs)
        self.chunk = max(1, int(_env_number("LOCI_RERANK_CHUNK", DEFAULT_CHUNK)) if chunk is None else int(chunk))
        self._clock = clock
        self.started = clock()
        self.scored = 0
        self.skipped = 0
        self.stage_scored = 0      # candidate-stage pairs, the ones max_pairs caps
        self.scoring_s = 0.0
        self.reason: Optional[str] = None

    @property
    def degraded(self) -> bool:
        return self.reason is not None

    def _remaining_s(self, final: bool) -> Optional[float]:
        if self.budget_s <= 0:
            return None
        usable = self.budget_s if final else self.budget_s * (1.0 - FINAL_RESERVE)
        return usable - (self._clock() - self.started)

    def allowance(self, final: bool = False) -> tuple[int, Optional[str]]:
        """(pairs that may be scored now, why it is 0 or limited). A huge number means unlimited."""
        left = self._remaining_s(final)
        if left is not None and left <= 0:
            return 0, "deadline"
        if not final and self.max_pairs > 0:
            cap = self.max_pairs - self.stage_scored
            if cap <= 0:
                return 0, "max_pairs"
            return cap, "max_pairs"
        return 1 << 30, None

    def summary(self) -> dict:
        return {
            "degraded": self.degraded,
            "reason": self.reason,
            "pairs_scored": self.scored,
            "pairs_skipped": self.skipped,
            "seconds": round(self.scoring_s, 3),
            "budget_s": self.budget_s,
            "max_pairs": self.max_pairs,
        }


_current_budget: contextvars.ContextVar[Optional[RerankBudget]] = contextvars.ContextVar(
    "loci_rerank_budget", default=None)

_stats_lock = threading.Lock()
_stats: dict = {
    "searches": 0, "degraded_searches": 0, "pairs_scored": 0, "pairs_skipped": 0,
    "last_seconds": None, "last_degraded": None, "last_reason": None, "last_at": None,
}


def _record(b: RerankBudget) -> None:
    """Fold a finished search into the process-wide counters (only searches that reached the scorer)."""
    if b.scored == 0 and b.skipped == 0 and b.reason is None:
        return
    with _stats_lock:
        _stats["searches"] += 1
        _stats["degraded_searches"] += 1 if b.degraded else 0
        _stats["pairs_scored"] += b.scored
        _stats["pairs_skipped"] += b.skipped
        _stats["last_seconds"] = round(b.scoring_s, 3)
        _stats["last_degraded"] = b.degraded
        _stats["last_reason"] = b.reason
        _stats["last_at"] = time.time()


@contextlib.contextmanager
def search_budget(**kwargs) -> Iterator[RerankBudget]:
    """Open the budget for one search. Scoring calls inside it (any depth, same thread) draw on it.
    Re-entrant: a nested scope shares the outer budget, and only the outermost records stats."""
    outer = _current_budget.get()
    if outer is not None:
        yield outer
        return
    b = RerankBudget(**kwargs)
    token = _current_budget.set(b)
    try:
        yield b
    finally:
        _current_budget.reset(token)
        _record(b)


def score_pairs(model, pairs: Sequence, budget: RerankBudget, final: bool = False) -> List[Optional[float]]:
    """Score `pairs` with `model.predict` in chunks until `budget` runs out.

    Returns one entry per pair: the float score, or None for pairs left unscored. A prefix is
    scored, the tail is not. Exceptions from `model.predict` propagate (callers fail open on them);
    nothing is recorded for the failed chunk.
    """
    n = len(pairs)
    scores: List[Optional[float]] = [None] * n
    i = 0
    stopped: Optional[str] = None
    while i < n:
        allowed, why = budget.allowance(final)
        if allowed <= 0:
            stopped = why
            break
        take = min(budget.chunk, allowed, n - i)
        t0 = budget._clock()
        out = list(model.predict(list(pairs[i:i + take])))
        budget.scoring_s += budget._clock() - t0
        if len(out) != take:
            raise ValueError(f"reranker returned {len(out)} scores for {take} pairs")
        scores[i:i + take] = [float(x) for x in out]
        i += take
        budget.scored += take
        if not final:
            budget.stage_scored += take
    if i < n:
        budget.skipped += n - i
        if budget.reason is None:
            budget.reason = stopped or "deadline"
    return scores


def mark_error(budget: RerankBudget, exc: BaseException) -> None:
    """A predict() failure means this rerank did not happen: report the search as degraded."""
    if budget.reason is None:
        budget.reason = "error:" + type(exc).__name__


def stats() -> dict:
    with _stats_lock:
        return dict(_stats)


def health() -> dict:
    """Reranker state for loci_health: model, device, torch build and rerank counters. Never raises
    and never loads the model."""
    try:
        if _model is None:
            state = "not_loaded"
        elif _model is False:
            state = "unavailable"
        else:
            state = "loaded"
        out = {
            "model": _model_id(),
            "state": state,
            "device": _device if state == "loaded" else None,
            "torch": _torch_build,
            "budget_s": _env_number("LOCI_RERANK_BUDGET_S", DEFAULT_BUDGET_S),
            "max_pairs": int(_env_number("LOCI_RERANK_MAX_PAIRS", 0)),
        }
        out.update(stats())
        return out
    except Exception as exc:  # health must report, never raise
        return {"state": "error", "error": f"{type(exc).__name__}: {exc}"}


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

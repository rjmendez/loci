"""Reference selectors for the model pool's shadow hook (``LOCI_MODEL_POOL_SELECTOR=pool_selectors:success_rate``).

A selector is ``callable(role, features) -> model name | [names] | None`` where ``features`` is one dict per
candidate: ``name``, ``rank``, ``effective_rank``, ``resident``, ``eligible``. Its answer is logged beside the
rule's and never used for the real call. ``None`` means "no opinion" (logged as ``abstained``).

``success_rate`` is the baseline any learned selector (a FlyBrain-style brain) has to beat to be worth a place:
among the eligible candidates that have enough logged outcomes, the highest success rate wins, ties go to the
lower median latency, then to the name. It needs outcomes for more than one arm to say anything the rule does not,
which is what ``LOCI_MODEL_POOL_EXPLORE`` is for. Pure numbers from the outcome log; no text is ever read.
"""
from __future__ import annotations

import time

MIN_CALLS = 30
_STATS_TTL_S = 60.0
_cache: dict = {"at": 0.0, "stats": {}}


def _stats() -> dict:
    now = time.monotonic()
    if now - _cache["at"] > _STATS_TTL_S:
        import model_pool
        _cache.update(at=now, stats=model_pool.outcomes_summary())
    return _cache["stats"]


def success_rate(role: str, features: list) -> "str | None":
    stats = _stats()
    best = None
    for f in features:
        if not f.get("eligible"):
            continue
        s = stats.get(f["name"])
        if not s or s["calls"] < MIN_CALLS:
            continue
        key = (-s["ok_rate"], s["p50_ms"], f["name"])
        if best is None or key < best[0]:
            best = (key, f["name"])
    return best[1] if best else None


_gcache: dict = {"at": 0.0, "stats": {}}


def _grades() -> dict:
    now = time.monotonic()
    if now - _gcache["at"] > _STATS_TTL_S:
        import model_pool
        _gcache.update(at=now, stats=model_pool.grades_summary())
    return _gcache["stats"]


def graded_rate(role: str, features: list) -> "str | None":
    """Like ``success_rate`` but ranks by how often the answers were RIGHT (``model_pool.record_grade``), not by
    whether the call returned. Arms with fewer than ``MIN_CALLS`` graded answers are ignored; ties go to the name."""
    stats = _grades()
    best = None
    for f in features:
        if not f.get("eligible"):
            continue
        s = stats.get(f["name"])
        if not s or s["graded"] < MIN_CALLS:
            continue
        key = (-s["correct_rate"], f["name"])
        if best is None or key < best[0]:
            best = (key, f["name"])
    return best[1] if best else None

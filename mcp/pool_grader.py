"""Grade a sample of live classify/compress answers so the pool's log can say which model is RIGHT, not just which one returned.

The pool logs hold no text, so a call can only be graded at the moment it happens, while its prompt and answer are in
memory. ``llm_local.generate`` offers each logged call here; with ``LOCI_MODEL_POOL_GRADE`` set to a probability
(default 0 = off) a sample goes to one background worker that asks a judge model whether the answer was right and
writes the verdict with ``model_pool.record_grade(..., grader="judge")``. Only the verdict is stored.

Rules that keep it honest and cheap:
* the judge is a different model from the one that answered, chosen by the pool (``model_pool.pick_other``, role
  ``LOCI_MODEL_POOL_GRADE_ROLE``; by default ``verify`` first, then ``gen``, so a loaded generation model
  that is not a verifier can still judge) and already loaded: judging never loads a model while traffic
  is being served (a cold judge load stalled a live answer 58 s on 2026-10-06), so with none resident the call is
  skipped as ``no_resident_judge`` (``LOCI_MODEL_POOL_GRADE_COLD=1`` allows cold loads). ``no_other_model`` when the
  pool has none at all and ``same_model`` when the call came back from the answerer anyway are skipped too, never
  self-graded;
* one worker and a short queue: when the judge is busy a sampled call is skipped (``queue_full``), never waited for;
* every skip is written to the grades log with its reason (``model_pool.record_grade_skip``), so the report shows it;
* judge calls carry ``task="judge"`` and are never graded themselves.

Oracle mode (Tripwire-Governor S12) is OFF unless ``LOCI_JUDGE_ORACLE`` is truthy; off, nothing below changes. On, it
stays observe-only (the judge still gates nothing; ``quality_validators.evaluate_gate`` decides when it may):
* the judge's cloud spend is capped at ``JUDGE_DAILY_USD_CAP`` ($0.50/local day) by a ``JudgeSpendBucket`` charged from
  the measured ``cost_usd`` of each judgment (the per-judgment estimate when the result carries none); past it a
  sampled call is skipped as ``oracle_starved``;
* the queue holds 500 and drops the OLDEST item when full (skip ``queue_dropped``, counted), instead of skipping the new one;
* a pool whose only judge candidate is the answerer is skipped as ``same_model`` (``select_judge``), never self-graded.
``oracle_stats()`` reports every counter.

A judge is an opinion, not ground truth: measure it against a gold set before trusting its rate
(``~/.loci/archive/judge_vs_gold.py``). The judge role is ``LOCI_MODEL_POOL_GRADE_ROLE`` (default ``verify``).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import random
import threading
from decimal import Decimal, InvalidOperation
from typing import Callable, Optional

import model_pool
import quality_validators as qv

logger = logging.getLogger(__name__)

GRADE_ENV = "LOCI_MODEL_POOL_GRADE"
COLD_ENV = "LOCI_MODEL_POOL_GRADE_COLD"
DEFAULT_ROLES = ("verify", "gen")
ROLE_ENV = "LOCI_MODEL_POOL_GRADE_ROLE"
TASKS = ("classify", "compress")
QUEUE_MAX = 8
_PROMPT_CHARS = 4000
_ANSWER_CHARS = 2000

_JUDGE_RULES = {
    "classify": "The task asked for ONE label from a list. Correct means the label given is the best fit for the text.",
    "compress": "The task asked for a shorter version of a text. Correct means it keeps the key facts and meaning, "
                "states nothing the text does not say, and respects the length limit.",
}

ORACLE_ENV = "LOCI_JUDGE_ORACLE"
JUDGE_DAILY_USD_CAP = qv.JUDGE_DAILY_USD_CAP
ORACLE_QUEUE_MAX = qv.JUDGE_QUEUE_MAX

_queue: "queue.Queue[dict]" = queue.Queue(maxsize=QUEUE_MAX)
_worker_lock = threading.Lock()
_worker: Optional[threading.Thread] = None
_oracle_lock = threading.Lock()
_oracle_bucket: Optional[qv.JudgeSpendBucket] = None
_oracle_queue: Optional[qv.DropOldestQueue] = None


def oracle_enabled() -> bool:
    return (os.environ.get(ORACLE_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def oracle_bucket() -> qv.JudgeSpendBucket:
    global _oracle_bucket
    with _oracle_lock:
        if _oracle_bucket is None:
            _oracle_bucket = qv.JudgeSpendBucket(JUDGE_DAILY_USD_CAP)
        return _oracle_bucket


def oracle_queue() -> qv.DropOldestQueue:
    global _oracle_queue
    with _oracle_lock:
        if _oracle_queue is None:
            _oracle_queue = qv.DropOldestQueue(ORACLE_QUEUE_MAX)
        return _oracle_queue


def oracle_stats() -> dict:
    """Counters of the oracle path: enabled flag, spend bucket, and the drop-oldest queue (length and drops)."""
    q = oracle_queue()
    return {"enabled": oracle_enabled(), "bucket": oracle_bucket().stats(), "queue_len": q.qsize(),
            "queue_dropped": q.dropped}


def _measured_cost(res: dict):
    """The judgment's measured cost as a Decimal, or None when the result carries none (or garbage)."""
    raw = res.get("cost_usd")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        cost = Decimal(str(raw))
    except InvalidOperation:
        return None
    return cost if cost.is_finite() and cost >= 0 else None


def rate() -> float:
    try:
        return min(1.0, max(0.0, float(os.environ.get(GRADE_ENV, "0") or 0)))
    except ValueError:
        return 0.0


def judge_prompt(task: str, prompt: str, answer: str) -> str:
    return (f"You are grading one answer from another model. {_JUDGE_RULES[task]}\n"
            f"Task given to the model:\n<<<\n{prompt[:_PROMPT_CHARS]}\n>>>\n"
            f"Answer it gave:\n<<<\n{answer[:_ANSWER_CHARS]}\n>>>\n"
            'Reply with JSON only: {"correct": true} or {"correct": false}.')


def _judge(task: str, prompt: str, answer: str, answerer: str) -> dict:
    roles = (os.environ.get(ROLE_ENV),) if os.environ.get(ROLE_ENV) else DEFAULT_ROLES
    cold_ok = (os.environ.get(COLD_ENV) or "").strip().lower() in ("1", "true", "yes", "on")
    if oracle_enabled():
        try:
            pool_names = [c.name for r in roles for c in model_pool.rank_role(r).candidates if c.eligible]
        except Exception as exc:
            logger.debug("pool_grader: rank_role failed: %r", exc)
            pool_names = []
        if qv.select_judge(answerer, pool_names).skip == "same_model":
            return {"ok": False, "why": "same_model"}
    other = next((m for m in (model_pool.pick_other(r, answerer, resident_only=not cold_ok) for r in roles) if m), "")
    if not other:
        anywhere = any(model_pool.pick_other(r, answerer) for r in roles)
        return {"ok": False, "why": "no_resident_judge" if anywhere else "no_other_model"}
    import llm_local
    return llm_local.generate(judge_prompt(task, prompt, answer), model=other, fmt="json", max_tokens=24,
                              task="judge", timeout=60)


def grade_one(item: dict, judge: Optional[Callable[[str, str, str, str], dict]] = None) -> Optional[bool]:
    """Judge one queued answer and record the verdict or the reason it was skipped. Never raises."""
    did, task, model = item["decision_id"], item["task"], item["model"]
    held = None
    if oracle_enabled():
        try:
            held = oracle_bucket().reserve()
        except qv.OracleStarved:
            model_pool.record_grade_skip(did, qv.OracleStarved.reason, model=model, task=task)
            return None
    try:
        res = (judge or _judge)(task, item["prompt"], item["answer"], model)
    except Exception as exc:
        logger.debug("pool_grader: judge raised: %r", exc)
        res = {}
    if held is not None:
        if isinstance(res, dict) and res.get("ok"):
            oracle_bucket().settle(held, _measured_cost(res))
        else:
            oracle_bucket().release(held)
    if not isinstance(res, dict) or not res.get("ok"):
        why = res.get("why") if isinstance(res, dict) else ""
        model_pool.record_grade_skip(
            did, why if why in ("no_other_model", "no_resident_judge") or (why == "same_model" and held is not None)
            else "judge_failed", model=model, task=task)
        return None
    if str(res.get("model") or "") == model:
        model_pool.record_grade_skip(did, "same_model", model=model, task=task)
        return None
    try:
        verdict = json.loads(str(res.get("text") or "")).get("correct")
    except Exception:
        verdict = None
    if not isinstance(verdict, bool):
        model_pool.record_grade_skip(did, "judge_unparseable", model=model, task=task)
        return None
    model_pool.record_grade(did, verdict, model=model, task=task, grader="judge")
    return verdict


def _next_item() -> dict:
    while True:
        try:
            return (oracle_queue() if oracle_enabled() else _queue).get(timeout=1.0)
        except queue.Empty:
            continue


def _run() -> None:
    while True:
        item = _next_item()
        try:
            grade_one(item)
        except Exception as exc:                      # the worker must outlive any one item
            logger.debug("pool_grader: item failed: %r", exc)


def _ensure_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="pool-grader", daemon=True)
            _worker.start()


def maybe_submit(task: str, prompt: str, answer: str, model: str, decision_id: str, *, ok: bool = True) -> bool:
    """Offer one finished call for grading; True when it was queued. Off unless ``LOCI_MODEL_POOL_GRADE`` > 0."""
    if task not in TASKS or not ok or not decision_id or not model or not model_pool._shadow_enabled():
        return False
    p = rate()
    if p <= 0 or random.random() >= p:
        return False
    item = {"task": task, "prompt": prompt or "", "answer": answer or "", "model": model, "decision_id": decision_id}
    if oracle_enabled():
        evicted = oracle_queue().put(item)
        if evicted is not None:
            model_pool.record_grade_skip(evicted["decision_id"], "queue_dropped", model=evicted["model"],
                                         task=evicted["task"])
    else:
        try:
            _queue.put_nowait(item)
        except queue.Full:
            model_pool.record_grade_skip(decision_id, "queue_full", model=model, task=task)
            return False
    _ensure_worker()
    return True

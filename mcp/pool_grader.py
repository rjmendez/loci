"""Grade a sample of live classify/compress answers so the pool's log can say which model is RIGHT, not just which one returned.

The pool logs hold no text, so a call can only be graded at the moment it happens, while its prompt and answer are in
memory. ``llm_local.generate`` offers each logged call here; with ``LOCI_MODEL_POOL_GRADE`` set to a probability
(default 0 = off) a sample goes to one background worker that asks a judge model whether the answer was right and
writes the verdict with ``model_pool.record_grade(..., grader="judge")``. Only the verdict is stored.

Rules that keep it honest and cheap:
* the judge is a different model from the one that answered (``same_model`` is skipped, never self-graded);
* one worker and a short queue: when the judge is busy a sampled call is skipped (``queue_full``), never waited for;
* every skip is written to the grades log with its reason (``model_pool.record_grade_skip``), so the report shows it;
* judge calls carry ``task="judge"`` and are never graded themselves.

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
from typing import Callable, Optional

import model_pool

logger = logging.getLogger(__name__)

GRADE_ENV = "LOCI_MODEL_POOL_GRADE"
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

_queue: "queue.Queue[dict]" = queue.Queue(maxsize=QUEUE_MAX)
_worker_lock = threading.Lock()
_worker: Optional[threading.Thread] = None


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


def _judge(task: str, prompt: str, answer: str) -> dict:
    import llm_local
    return llm_local.generate(judge_prompt(task, prompt, answer), fmt="json", max_tokens=24,
                              role=os.environ.get(ROLE_ENV) or "verify", task="judge", timeout=60)


def grade_one(item: dict, judge: Optional[Callable[[str, str, str], dict]] = None) -> Optional[bool]:
    """Judge one queued answer and record the verdict or the reason it was skipped. Never raises."""
    did, task, model = item["decision_id"], item["task"], item["model"]
    try:
        res = (judge or _judge)(task, item["prompt"], item["answer"])
    except Exception as exc:
        logger.debug("pool_grader: judge raised: %r", exc)
        res = {}
    if not isinstance(res, dict) or not res.get("ok"):
        model_pool.record_grade_skip(did, "judge_failed", model=model, task=task)
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


def _run() -> None:
    while True:
        item = _queue.get()
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
    try:
        _queue.put_nowait(item)
    except queue.Full:
        model_pool.record_grade_skip(decision_id, "queue_full", model=model, task=task)
        return False
    _ensure_worker()
    return True

"""Quality oracle (Tripwire-Governor S12): deterministic validators, the judge gate, the Wilson quality breaker.

Everything here is pure and observe-only. Nothing calls a model, the network or a live store: the judge, the clock
and the spend reader are injected. ``mcp/pool_grader.py`` can use these behind ``LOCI_JUDGE_ORACLE`` (default off).

What is in this module

* Validators over one recorded call outcome (``CallOutcome``): tool-call arguments that are not a JSON object, an
  empty non-stream answer that stopped normally, refusal boilerplate. Verdict ``pass`` / ``fail`` / ``excluded``;
  an empty STREAM is ``empty_suspect_stream`` and is excluded from reward (it is a transport symptom, not a
  quality verdict, so it counts neither for nor against the arm).
* The judge gate (``evaluate_gate``). Agreement is a bad gate (a judge that accepts everything agrees with a gold
  set that is mostly correct), so the gate needs, on gold-labelled items: at least ``MIN_GOLD_ITEMS`` usable
  items, at least ``MIN_HARD_NEGATIVES`` hard negatives, false-accept <= 0.15 overall AND on the hard negatives,
  Cohen's kappa >= 0.6. Anything short of that fails closed, and a judge that was also the answerer on a gold
  item fails the gate. Until it passes the judge gates nothing.
* ``select_judge``: a judge is never the answerer (skip reason ``same_model``) and never the highest-cost arm.
* ``rogan_gladen``: corrects an observed accept rate for the judge's measured sensitivity/specificity.
* ``QualityBreaker``: rolling window of pass/fail results for the downgraded arm and the baseline arm; with
  ``n >= 30`` on both, when the Wilson lower bound of the downgraded arm is more than 0.05 below the baseline's it
  trips and disables the downgrade for 24 h. ``LOCI_QUALITY_BREAKER=enforce`` makes it act; the default only counts
  what it would have done. Judge results feed it only while the judge gate passes.
* ``JudgeSpendBucket`` (<= $0.50/day, Decimal, counted from measured $/judgment) and ``DropOldestQueue`` (500).

Degraded operation is reported, never silent: every counter is in ``stats()``.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------------------------------- constants
PASS = "pass"
FAIL = "fail"
EXCLUDED = "excluded"

R_TOOL_JSON = "invalid_tool_call_json"
R_EMPTY = "empty_text_stop"
R_EMPTY_STREAM = "empty_suspect_stream"
R_REFUSAL = "refusal_boilerplate"

MIN_GOLD_ITEMS = 100
MIN_HARD_NEGATIVES = 30
FALSE_ACCEPT_MAX = Fraction(15, 100)
KAPPA_MIN = Fraction(3, 5)

BREAKER_MIN_N = 30
BREAKER_MARGIN = Decimal("0.05")
BREAKER_DISABLE_S = 24 * 3600.0
BREAKER_WINDOW_S = 24 * 3600.0
BREAKER_MAX_SAMPLES = 1000
WILSON_Z = 1.96
BREAKER_ENV = "LOCI_QUALITY_BREAKER"          # "enforce" acts; anything else only counts

JUDGE_DAILY_USD_CAP = Decimal("0.50")
JUDGE_COST_ESTIMATE_USD = Decimal("0.049")    # measured median for a >20k-token judgment (design 5.2)
JUDGE_QUEUE_MAX = 500

ARM_DOWNGRADED = "downgraded"
ARM_BASELINE = "baseline"


# ---------------------------------------------------------------------------------------------------- validators
@dataclass(frozen=True)
class CallOutcome:
    """The recorded facts of one finished call. ``stream`` is True when the answer was assembled from a stream."""
    text: str = ""
    finish_reason: str = ""
    tool_calls: tuple = ()
    stream: bool = False

    @classmethod
    def from_mapping(cls, row: Mapping) -> "CallOutcome":
        text = row.get("text")
        calls = row.get("tool_calls") or ()
        if not isinstance(calls, (list, tuple)):
            calls = (calls,)                      # a non-list tool_calls value is malformed: it will fail validation
        return cls(text=text if isinstance(text, str) else "",
                   finish_reason=str(row.get("finish_reason") or ""),
                   tool_calls=tuple(calls), stream=bool(row.get("stream")))


@dataclass(frozen=True)
class ValidationResult:
    verdict: str
    reasons: Tuple[str, ...] = ()

    @property
    def passed(self) -> Optional[bool]:
        """True / False for a verdict that counts, None when the call is excluded from reward."""
        return None if self.verdict == EXCLUDED else self.verdict == PASS

    @property
    def reward_eligible(self) -> bool:
        return self.verdict != EXCLUDED


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def _tool_call_ok(call: Any) -> bool:
    if not isinstance(call, Mapping):
        return False
    fn = call.get("function") if isinstance(call.get("function"), Mapping) else call
    name, args = fn.get("name"), fn.get("arguments")
    if not isinstance(name, str) or not name.strip():
        return False
    if isinstance(args, Mapping):
        return True
    if not isinstance(args, str) or not args.strip():
        return False                              # "" is not a JSON object; the standard no-argument form is "{}"
    try:
        return isinstance(json.loads(args, parse_constant=_reject_constant), dict)
    except ValueError:
        return False


_REFUSE_VERB = r"(?:can(?:no|')t|can not|am (?:not able|unable) to|won't|will not)"
_REFUSAL_RES = (
    re.compile(rf"^(?:i(?:'m| am)|we(?:'re| are)) (?:really |very |so )?sorry,? (?:but )?i {_REFUSE_VERB}\b"),
    re.compile(rf"^(?:sorry|i apologi[sz]e),? (?:but )?i {_REFUSE_VERB}\b"),
    re.compile(rf"^i {_REFUSE_VERB} (?:help|assist|comply|provide|fulfil+|do that|answer|continue|create|generate|support)\b"),
    re.compile(rf"^as an? (?:ai|artificial intelligence)(?: language model| assistant)?,? i (?:{_REFUSE_VERB}|do not|don't)\b"),
)
REFUSAL_MAX_CHARS = 500                           # boilerplate is short; a long answer that opens with an apology is not one


def is_refusal(text: str) -> bool:
    t = " ".join(str(text).replace("’", "'").lower().split())
    return 0 < len(t) <= REFUSAL_MAX_CHARS and any(r.match(t) for r in _REFUSAL_RES)


def validate(outcome: CallOutcome) -> ValidationResult:
    """Deterministic checks on one call. Any failure -> ``fail``; otherwise an excluded empty stream -> ``excluded``."""
    fails: List[str] = []
    excluded: List[str] = []
    if any(not _tool_call_ok(c) for c in outcome.tool_calls):
        fails.append(R_TOOL_JSON)
    if not outcome.text.strip() and not outcome.tool_calls:
        if outcome.stream:
            excluded.append(R_EMPTY_STREAM)
        elif outcome.finish_reason == "stop":
            fails.append(R_EMPTY)
    if is_refusal(outcome.text):
        fails.append(R_REFUSAL)
    if fails:
        return ValidationResult(FAIL, tuple(fails))
    if excluded:
        return ValidationResult(EXCLUDED, tuple(excluded))
    return ValidationResult(PASS)


# ---------------------------------------------------------------------------------------------------- judge choice
@dataclass(frozen=True)
class JudgeChoice:
    model: str
    skip: str = ""


def _norm_model(name: Any) -> str:
    n = str(name or "").strip()
    return n[: -len(":latest")] if n.endswith(":latest") else n


def select_judge(answerer: str, candidates: Iterable[str], cost_of: Optional[Callable[[str], Decimal]] = None) -> JudgeChoice:
    """First candidate (ranked order) that is not the answerer and not the highest-cost arm, else a skip reason.

    ``no_other_model``: nothing to choose from. ``same_model``: the only candidate(s) ARE the answerer, so judging
    would be a silent self-grade. ``highest_cost_arm``: every other candidate is the most expensive arm (applies only
    when arm costs differ). ``cost_unknown``: ``cost_of`` failed, so cost cannot be ruled out (fail closed).
    """
    names: List[str] = []
    for c in candidates:
        n = _norm_model(c)
        if n and n not in names:
            names.append(n)
    if not names:
        return JudgeChoice("", "no_other_model")
    me = _norm_model(answerer)
    others = [n for n in names if n != me]
    if not others:
        return JudgeChoice("", "same_model")
    if cost_of is not None:
        try:
            costs = {n: cost_of(n) for n in set(names) | {me}}
            if not all(isinstance(v, Decimal) and v.is_finite() for v in costs.values()):
                raise TypeError("cost must be a finite Decimal")
        except Exception:
            return JudgeChoice("", "cost_unknown")
        hi, lo = max(costs.values()), min(costs.values())
        if hi > lo:
            others = [n for n in others if costs[n] != hi]
            if not others:
                return JudgeChoice("", "highest_cost_arm")
    return JudgeChoice(others[0], "")


# ---------------------------------------------------------------------------------------------------- the judge gate
class GoldFormatError(ValueError):
    pass


@dataclass(frozen=True)
class GoldItem:
    item_id: str
    gold_correct: bool
    judge_accept: Optional[bool]
    hard_negative: bool = False
    bucket: str = ""
    answerer: str = ""
    judge_model: str = ""

    @classmethod
    def from_mapping(cls, row: Mapping) -> "GoldItem":
        gold, judge, hard = row.get("gold_correct"), row.get("judge_accept"), row.get("hard_negative", False)
        if not isinstance(gold, bool):
            raise GoldFormatError("gold_correct must be a bool (an unlabelled item is not gold)")
        if judge is not None and not isinstance(judge, bool):
            raise GoldFormatError("judge_accept must be a bool or null")
        if not isinstance(hard, bool) or (hard and gold):
            raise GoldFormatError("hard_negative must be a bool and only a gold-incorrect item can be one")
        return cls(str(row.get("id", "")), gold, judge, hard, str(row.get("bucket") or ""),
                   str(row.get("answerer") or ""), str(row.get("judge_model") or ""))


def cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> Optional[Fraction]:
    """Exact Cohen's kappa of judge vs gold; None when chance agreement is 1 (kappa undefined)."""
    n = tp + fp + fn + tn
    if n == 0:
        return None
    po = Fraction(tp + tn, n)
    pe = Fraction((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), n * n)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


@dataclass(frozen=True)
class GateReport:
    passed: bool
    reasons: Tuple[str, ...]
    n: int
    n_unusable: int
    n_gold_negative: int
    n_hard_negative: int
    tp: int
    fp: int
    fn: int
    tn: int
    false_accept: Optional[float]
    false_accept_hard: Optional[float]
    kappa: Optional[float]                        # rounded to 3 dp for reports; the gate uses the exact fraction
    agreement: Optional[float]
    sensitivity: Optional[float]
    specificity: Optional[float]
    by_bucket: Dict[str, float] = field(default_factory=dict)   # false-accept per bucket (with gold negatives), informational

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def _frac_or_none(num: int, den: int) -> Optional[Fraction]:
    return Fraction(num, den) if den else None


def evaluate_gate(items: Sequence[GoldItem], *, malformed: int = 0) -> GateReport:
    """Judge-vs-gold gate. PASS only when every requirement is met; every miss is named in ``reasons``."""
    usable = [i for i in items if i.judge_accept is not None]
    tp = sum(i.judge_accept and i.gold_correct for i in usable)
    fp = sum(i.judge_accept and not i.gold_correct for i in usable)
    fn = sum((not i.judge_accept) and i.gold_correct for i in usable)
    tn = sum((not i.judge_accept) and not i.gold_correct for i in usable)
    hard = [i for i in usable if i.hard_negative]
    fp_hard = sum(bool(i.judge_accept) for i in hard)
    fa = _frac_or_none(fp, fp + tn)
    fa_hard = _frac_or_none(fp_hard, len(hard))
    kappa = cohens_kappa(tp, fp, fn, tn)
    reasons: List[str] = []
    if malformed:
        reasons.append("malformed_items")
    if len(usable) < MIN_GOLD_ITEMS:
        reasons.append("insufficient_gold")
    if len(hard) < MIN_HARD_NEGATIVES:
        reasons.append("insufficient_hard_negatives")
    if fa is None or fa > FALSE_ACCEPT_MAX:
        reasons.append("false_accept_too_high")
    if fa_hard is None or fa_hard > FALSE_ACCEPT_MAX:
        reasons.append("false_accept_hard_too_high")
    if kappa is None:
        reasons.append("kappa_undefined")
    elif kappa < KAPPA_MIN:
        reasons.append("kappa_too_low")
    if any(i.judge_model and i.answerer and _norm_model(i.judge_model) == _norm_model(i.answerer) for i in usable):
        reasons.append("same_model_items")
    buckets: Dict[str, float] = {}
    for b in sorted({i.bucket for i in usable}):
        neg = [i for i in usable if i.bucket == b and not i.gold_correct]
        if neg:                                   # a bucket with no gold-incorrect item has no false-accept rate to report
            buckets[b or "(none)"] = float(Fraction(sum(bool(i.judge_accept) for i in neg), len(neg)))
    n = len(usable)
    return GateReport(
        passed=not reasons, reasons=tuple(reasons), n=n, n_unusable=len(items) - n,
        n_gold_negative=fp + tn, n_hard_negative=len(hard), tp=tp, fp=fp, fn=fn, tn=tn,
        false_accept=None if fa is None else float(fa), false_accept_hard=None if fa_hard is None else float(fa_hard),
        kappa=None if kappa is None else round(float(kappa), 3),
        agreement=float(Fraction(tp + tn, n)) if n else None,
        sensitivity=None if not (tp + fn) else float(Fraction(tp, tp + fn)),
        specificity=None if not (tn + fp) else float(Fraction(tn, tn + fp)),
        by_bucket=buckets)


def rogan_gladen(observed: float, sensitivity: float, specificity: float, *, clamp: bool = True) -> float:
    """True accept rate implied by an observed judge accept rate: (p + spec - 1) / (sens + spec - 1).

    A judge with sens + spec <= 1 is no better than a coin and cannot be corrected: ValueError.
    """
    for name, v in (("observed", observed), ("sensitivity", sensitivity), ("specificity", specificity)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be a probability in [0, 1], got {v!r}")
    denom = sensitivity + specificity - 1.0
    if denom <= 0.0:
        raise ValueError("judge is uninformative (sensitivity + specificity <= 1): cannot correct")
    est = (observed + specificity - 1.0) / denom
    return min(1.0, max(0.0, est)) if clamp else est


# ---------------------------------------------------------------------------------------------------- Wilson breaker
def wilson_lower(successes: int, n: int, z: float = WILSON_Z) -> float:
    """Wilson score lower bound of a binomial proportion; 0.0 for n == 0."""
    if n <= 0:
        return 0.0
    p = successes / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - spread) / (1 + z2 / n))


def _dec(x: float) -> Decimal:
    return Decimal(repr(round(float(x), 12)))


def should_trip(lb_downgraded: float, lb_baseline: float, n_downgraded: int, n_baseline: int, *,
                min_n: int = BREAKER_MIN_N, margin: Decimal = BREAKER_MARGIN) -> bool:
    """LB(downgraded) strictly more than ``margin`` below LB(baseline), with n >= min_n on both arms."""
    if n_downgraded < min_n or n_baseline < min_n:
        return False
    return _dec(lb_baseline) - _dec(lb_downgraded) > margin


class QualityBreaker:
    """Rolling-window Wilson breaker over per-call pass/fail results of the downgraded arm vs the baseline arm.

    Observe-only unless ``enforce``: then a trip makes ``downgrade_allowed()`` False for ``disable_s`` and the
    downgraded window restarts from nothing when it ends (fresh evidence is needed to trip again). Excluded calls
    (``None``) are counted, never scored. An internal error fails closed under ``enforce`` and is counted.
    """

    def __init__(self, *, enforce: bool = False, clock: Callable[[], float] = time.time, min_n: int = BREAKER_MIN_N,
                 margin: Decimal = BREAKER_MARGIN, z: float = WILSON_Z, disable_s: float = BREAKER_DISABLE_S,
                 window_s: float = BREAKER_WINDOW_S, max_samples: int = BREAKER_MAX_SAMPLES) -> None:
        self.enforce, self._clock, self.min_n, self.margin, self.z = enforce, clock, min_n, margin, z
        self.disable_s, self.window_s = disable_s, window_s
        self._win: Dict[str, Deque[Tuple[float, bool]]] = {
            ARM_DOWNGRADED: deque(maxlen=max_samples), ARM_BASELINE: deque(maxlen=max_samples)}
        self._lock = threading.Lock()
        self._until: Optional[float] = None
        self.counters: Dict[str, int] = {"recorded": 0, "excluded": 0, "trips": 0, "resets": 0,
                                         "judge_ignored_gate_not_passed": 0, "errors": 0, "blocked_checks": 0}

    @classmethod
    def from_env(cls, **kw: Any) -> "QualityBreaker":
        return cls(enforce=(os.environ.get(BREAKER_ENV) or "").strip().lower() == "enforce", **kw)

    def record(self, arm: str, passed: Optional[bool]) -> None:
        if arm not in self._win:
            raise ValueError(f"unknown arm {arm!r}: use {ARM_DOWNGRADED!r} or {ARM_BASELINE!r}")
        if passed is not None and not isinstance(passed, bool):
            raise ValueError("passed must be True, False or None (excluded)")
        with self._lock:
            if passed is None:
                self.counters["excluded"] += 1
                return
            self._win[arm].append((self._clock(), passed))
            self.counters["recorded"] += 1
            self._evaluate_locked()

    def record_validation(self, arm: str, result: ValidationResult) -> None:
        self.record(arm, result.passed)

    def record_judge(self, arm: str, accepted: Optional[bool], gate_passed: bool) -> None:
        """A judge verdict counts only while the judge gate passes; otherwise it is dropped and counted."""
        if not gate_passed:
            with self._lock:
                self.counters["judge_ignored_gate_not_passed"] += 1
            return
        self.record(arm, accepted)

    def _prune_locked(self, now: float) -> None:
        for w in self._win.values():
            while w and now - w[0][0] >= self.window_s:
                w.popleft()

    def _counts_locked(self, arm: str) -> Tuple[int, int]:
        w = self._win[arm]
        return sum(1 for _, ok in w if ok), len(w)

    def _evaluate_locked(self) -> None:
        now = self._clock()
        self._prune_locked(now)
        if self._until is not None and now >= self._until:
            self._until = None
            self._win[ARM_DOWNGRADED].clear()
            self.counters["resets"] += 1
        if self._until is not None:
            return
        kd, nd = self._counts_locked(ARM_DOWNGRADED)
        kb, nb = self._counts_locked(ARM_BASELINE)
        if should_trip(wilson_lower(kd, nd, self.z), wilson_lower(kb, nb, self.z), nd, nb,
                       min_n=self.min_n, margin=self.margin):
            self._until = now + self.disable_s
            self.counters["trips"] += 1

    def downgrade_allowed(self) -> bool:
        try:
            with self._lock:
                self._evaluate_locked()
                tripped = self._until is not None
                if tripped and self.enforce:
                    self.counters["blocked_checks"] += 1
                return not (tripped and self.enforce)
        except Exception:
            self.counters["errors"] += 1
            return not self.enforce

    def stats(self) -> dict:
        with self._lock:
            now = self._clock()
            self._prune_locked(now)
            kd, nd = self._counts_locked(ARM_DOWNGRADED)
            kb, nb = self._counts_locked(ARM_BASELINE)
            return {"enforce": self.enforce, "tripped": self._until is not None, "disabled_until": self._until,
                    "n_downgraded": nd, "k_downgraded": kd, "n_baseline": nb, "k_baseline": kb,
                    "lb_downgraded": wilson_lower(kd, nd, self.z), "lb_baseline": wilson_lower(kb, nb, self.z),
                    "evaluable": nd >= self.min_n and nb >= self.min_n, **self.counters}


# ---------------------------------------------------------------------------------------------------- judge spend
class OracleStarved(RuntimeError):
    """The judge's daily budget cannot pay for another judgment; the learner pauses, nothing is graded silently."""
    reason = "oracle_starved"


def _money(value: Any, what: str) -> Decimal:
    if isinstance(value, float) or not isinstance(value, Decimal):
        raise TypeError(f"{what} must be a Decimal (money is never a float), got {type(value).__name__}")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{what} must be a finite non-negative amount, got {value}")
    return value


class JudgeSpendBucket:
    """Daily judge budget. ``spend_reader()`` returns today's settled judge spend as a Decimal (its own key's ledger);
    without one the bucket keeps its own per-local-day total of what ``settle`` was told. A reader that fails or
    returns a bad value fails closed: the bucket reports starved and counts ``reader_errors``."""

    def __init__(self, cap: Decimal = JUDGE_DAILY_USD_CAP, *, spend_reader: Optional[Callable[[], Decimal]] = None,
                 estimate: Decimal = JUDGE_COST_ESTIMATE_USD, today: Callable[[], dt.date] = dt.date.today) -> None:
        self.cap, self.estimate = _money(cap, "cap"), _money(estimate, "estimate")
        self._reader, self._today = spend_reader, today
        self._lock = threading.Lock()
        self._day: Optional[dt.date] = None
        self._ledger = Decimal(0)
        self._in_flight = Decimal(0)
        self.counters: Dict[str, int] = {"reserved": 0, "starved": 0, "reader_errors": 0, "settled": 0, "released": 0}

    def _spent_locked(self) -> Decimal:
        if self._reader is not None:
            try:
                return _money(self._reader(), "spend reader value")
            except Exception:
                self.counters["reader_errors"] += 1
                raise OracleStarved("judge spend reader unavailable: failing closed") from None
        day = self._today()
        if day != self._day:
            self._day, self._ledger = day, Decimal(0)
        return self._ledger

    def spent(self) -> Decimal:
        with self._lock:
            return self._spent_locked()

    def reserve(self, cost: Optional[Decimal] = None) -> Decimal:
        """Hold ``cost`` (default: the per-judgment estimate) before a judgment; OracleStarved when it would pass the cap."""
        cost = self.estimate if cost is None else _money(cost, "cost")
        with self._lock:
            try:
                spent = self._spent_locked()
                if spent + self._in_flight + cost > self.cap:
                    raise OracleStarved(f"judge budget {self.cap} exhausted: spent {spent}, in flight {self._in_flight}")
            except OracleStarved:
                self.counters["starved"] += 1
                raise
            self._in_flight += cost
            self.counters["reserved"] += 1
            return cost

    def settle(self, reserved: Decimal, measured: Optional[Decimal] = None) -> None:
        """The judgment ran: free the hold and, with no external reader, add the measured cost (else the hold)."""
        reserved = _money(reserved, "reserved")
        charge = reserved if measured is None else _money(measured, "measured")
        with self._lock:
            self._in_flight = max(Decimal(0), self._in_flight - reserved)
            if self._reader is None:
                self._spent_locked()                     # roll the day over before adding
                self._ledger += charge
            self.counters["settled"] += 1

    def release(self, reserved: Decimal) -> None:
        """The judgment never ran: free the hold, charge nothing."""
        reserved = _money(reserved, "reserved")
        with self._lock:
            self._in_flight = max(Decimal(0), self._in_flight - reserved)
            self.counters["released"] += 1

    def stats(self) -> dict:
        with self._lock:
            return {"cap": self.cap, "in_flight": self._in_flight, **self.counters}


# ---------------------------------------------------------------------------------------------------- bounded queue
class DropOldestQueue:
    """FIFO of at most ``maxsize`` items: a full ``put`` evicts the OLDEST item (returned) and counts it."""

    def __init__(self, maxsize: int = JUDGE_QUEUE_MAX) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self.maxsize, self.dropped = maxsize, 0
        self._items: Deque[Any] = deque()
        self._cv = threading.Condition()

    def put(self, item: Any) -> Any:
        with self._cv:
            evicted = None
            if len(self._items) >= self.maxsize:
                evicted = self._items.popleft()
                self.dropped += 1
            self._items.append(item)
            self._cv.notify()
            return evicted

    def get(self, timeout: Optional[float] = None) -> Any:
        with self._cv:
            if not self._items and not self._cv.wait_for(lambda: bool(self._items), timeout):
                raise queue.Empty
            return self._items.popleft()

    def qsize(self) -> int:
        with self._cv:
            return len(self._items)

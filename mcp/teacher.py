"""Teacher: budget-guarded cloud calls for learning loops (judge, labelers, proposer, verifier).

The guard fails CLOSED (anything doubtful refuses the spend). `teacher_call` never raises: callers
that use it fail OPEN (a refusal or error degrades the feature, not the tool call). Everything is
off by default. Role -> LiteLLM group names and caps come from backends.toml [teacher] or
LOCI_TEACHER_* env, read at call time. Spend is read from the LiteLLM cost ledger (jsonl,
append-only, rotating). LiteLLM enforces no budget here: this guard is the only brake, and the
operator-supplied [teacher.prices] table must be kept in sync with the LiteLLM config.

Public API: ROLES, teacher_config, read_ledger, check, reserve, release, teacher_call, status.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import backends

ROLES = ("judge", "label_a", "label_b", "propose", "adjudicate", "verify")
DEFAULT_LEDGER = "~/.loci/litellm_cost_ledger.jsonl"
LEDGER_CACHE_S = 5.0
DEFAULT_SETTLE_S = 120.0          # how long a finished call's cost stays held while the gateway writes its ledger row
MAX_SIBLINGS = 12                 # ledger.jsonl.1 .. .12
DEFAULT_CHARS_PER_TOKEN = 3.0     # conservative (real text is ~4): more tokens estimated, so more cost
DEFAULT_SAFETY_MARGIN_PCT = 10.0  # usable cap = cap * (100 - margin) / 100
_CHUNK = 1 << 20

Messages = List[Dict[str, Any]]
Transport = Callable[[str, Messages, int, Dict[str, Any]], Dict[str, Any]]

_now: Callable[[], float] = time.time           # seams for tests
_mono: Callable[[], float] = time.monotonic


class LedgerUnreadable(Exception):
    """The ledger exists but cannot be read."""


# ------------------------------------------------------------------ config
def _env(name: str) -> Optional[str]:
    v = os.environ.get(name)
    return v if v is not None and v.strip() != "" else None


def _bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _money(v: Any) -> float:
    """Non-negative finite float; anything else is 0 (= no spend allowed)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) and f > 0 else 0.0


def _settle(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return DEFAULT_SETTLE_S
    return f if math.isfinite(f) and f >= 0 else DEFAULT_SETTLE_S


def _nonneg(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f >= 0 else None


def _margin(v: Any) -> float:
    f = _nonneg(v)
    return f if f is not None and f < 100 else DEFAULT_SAFETY_MARGIN_PCT


def _prices(raw: Any) -> Dict[str, Dict[str, float]]:
    """group -> {in, out} USD per million tokens. Env form: ``group=in:out,group2=in:out``.
    Entries missing either price or with a negative/non-finite one are dropped (= no_price)."""
    items: Dict[str, Any] = {}
    if isinstance(raw, str):
        for part in raw.split(","):
            g, _, pr = part.partition("=")
            i, _, o = pr.partition(":")
            if g.strip():
                items[g.strip()] = {"in_usd_per_mtok": i, "out_usd_per_mtok": o}
    elif isinstance(raw, dict):
        items = {str(g).strip(): v for g, v in raw.items()}
    out: Dict[str, Dict[str, float]] = {}
    for g, v in items.items():
        if not g or not isinstance(v, dict):
            continue
        i, o = _nonneg(v.get("in_usd_per_mtok")), _nonneg(v.get("out_usd_per_mtok"))
        if i is not None and o is not None:
            out[g] = {"in": i, "out": o}
    return out


def teacher_config() -> Dict[str, Any]:
    """Resolved config, read now: env LOCI_TEACHER_* over backends.toml [teacher]."""
    def get(key: str, env: str, default: Any = None) -> Any:
        e = _env(env)
        return e if e is not None else backends._cfg("teacher", key, default)

    cfg_roles = backends._cfg("teacher", "roles", {}) or {}
    roles: Dict[str, str] = {}
    for r in ROLES:
        g = _env("LOCI_TEACHER_ROLE_" + r.upper())
        if g is None and isinstance(cfg_roles, dict):
            g = cfg_roles.get(r)
        if isinstance(g, str) and g.strip():
            roles[r] = g.strip()
    counted = get("groups_counted", "LOCI_TEACHER_GROUPS_COUNTED", None)
    if isinstance(counted, str):
        counted = [c for c in counted.split(",")]
    counted = sorted({str(c).strip() for c in counted if str(c).strip()}) if counted else sorted(set(roles.values()))
    cpt = _nonneg(get("chars_per_token", "LOCI_TEACHER_CHARS_PER_TOKEN", DEFAULT_CHARS_PER_TOKEN))
    try:
        max_bad = int(get("max_bad_lines", "LOCI_TEACHER_MAX_BAD_LINES", 0))
    except (TypeError, ValueError):
        max_bad = 0
    env_prices = _env("LOCI_TEACHER_PRICES")
    return {
        "enabled": _bool(get("enabled", "LOCI_TEACHER_ENABLED", False), False),
        "roles": roles,
        "monthly_usd_cap": _money(get("monthly_usd_cap", "LOCI_TEACHER_MONTHLY_USD_CAP", 0)),
        "daily_usd_cap": _money(get("daily_usd_cap", "LOCI_TEACHER_DAILY_USD_CAP", 0)),
        "max_usd_per_call": _money(get("max_usd_per_call", "LOCI_TEACHER_MAX_USD_PER_CALL", 0)),
        "ledger_file": str(Path(str(get("ledger_file", "LOCI_TEACHER_LEDGER_FILE", DEFAULT_LEDGER))).expanduser()),
        "groups_counted": counted,
        "sanitize": _bool(get("sanitize", "LOCI_TEACHER_SANITIZE", True), True),
        "chars_per_token": cpt if cpt else DEFAULT_CHARS_PER_TOKEN,
        "prices": _prices(env_prices if env_prices is not None else backends._cfg("teacher", "prices", {})),
        "max_bad_lines": max(max_bad, 0),
        "safety_margin_pct": _margin(get("safety_margin_pct", "LOCI_TEACHER_SAFETY_MARGIN_PCT", DEFAULT_SAFETY_MARGIN_PCT)),
        "allow_missing_ledger": _bool(get("allow_missing_ledger", "LOCI_TEACHER_ALLOW_MISSING_LEDGER", False), False),
        "settle_seconds": _settle(get("settle_seconds", "LOCI_TEACHER_SETTLE_SECONDS", DEFAULT_SETTLE_S)),
    }


# ------------------------------------------------------------------ ledger
@dataclass
class LedgerTotals:
    month: float = 0.0
    day: float = 0.0
    by_group_month: Dict[str, float] = field(default_factory=dict)
    by_group_day: Dict[str, float] = field(default_factory=dict)
    missing: bool = False
    bad_lines: int = 0
    rows_counted: int = 0
    rows_over_call_cap: int = 0   # counted rows costing more than max_usd_per_call (a per-call overrun seen in the ledger)
    files: List[str] = field(default_factory=list)


def _reverse_lines(path: str) -> Iterator[bytes]:
    """Lines of a file, last first, in bounded memory."""
    with open(path, "rb") as f:
        pos = f.seek(0, os.SEEK_END)
        tail = b""
        while pos > 0:
            step = min(_CHUNK, pos)
            pos -= step
            f.seek(pos)
            parts = (f.read(step) + tail).split(b"\n")
            tail = parts[0]
            for line in reversed(parts[1:]):
                yield line
        yield tail


def _row(line: bytes) -> Optional[tuple]:
    """(ts, group, cost) or None for a blank/bad line."""
    line = line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
        ts, cost, grp = float(d["ts"]), float(d["cost_usd"]), d["group"]
    except (ValueError, TypeError, KeyError):
        return False  # type: ignore[return-value]
    if not (math.isfinite(ts) and math.isfinite(cost) and cost >= 0 and isinstance(grp, str)):
        return False  # type: ignore[return-value]
    return ts, grp, cost


def _scan(base: str, counted: set, month_start: float, day_start: float, call_cap: float = 0.0) -> LedgerTotals:
    out = LedgerTotals()
    paths = [base] + [f"{base}.{i}" for i in range(1, MAX_SIBLINGS + 1)]
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        out.missing = True
        return out
    for p in existing:
        out.files.append(p)
        crossed = False
        try:
            for line in _reverse_lines(p):
                row = _row(line)
                if row is None:
                    continue
                if row is False:
                    out.bad_lines += 1
                    continue
                ts, grp, cost = row
                if ts < month_start:
                    crossed = True
                    break
                if grp not in counted:
                    continue
                out.rows_counted += 1
                if call_cap > 0 and cost > call_cap:
                    out.rows_over_call_cap += 1
                out.month += cost
                out.by_group_month[grp] = out.by_group_month.get(grp, 0.0) + cost
                if ts >= day_start:
                    out.day += cost
                    out.by_group_day[grp] = out.by_group_day.get(grp, 0.0) + cost
        except OSError as exc:
            raise LedgerUnreadable(f"{p}: {type(exc).__name__}") from exc
        if crossed:
            break
    return out


_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {"key": None, "at": 0.0, "value": None}


def cache_clear() -> None:
    with _cache_lock:
        _cache.update(key=None, at=0.0, value=None)


def read_ledger(cfg: Optional[Dict[str, Any]] = None) -> LedgerTotals:
    """Spend of counted groups for the current local month and day. Raises LedgerUnreadable.

    Unparseable and negative-cost rows are skipped and counted in ``bad_lines``; the guard refuses
    when they exceed ``max_bad_lines``. A missing ledger is zero spend with ``missing=True`` (the guard refuses it unless opted in). Cached for LEDGER_CACHE_S."""
    cfg = cfg or teacher_config()
    now = _now()
    lt = datetime.fromtimestamp(now)
    month_start = lt.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
    day_start = lt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    key = (cfg["ledger_file"], tuple(cfg["groups_counted"]), month_start, day_start, cfg["max_usd_per_call"])
    with _cache_lock:
        if _cache["key"] == key and _mono() - _cache["at"] < LEDGER_CACHE_S:
            return _cache["value"]
    value = _scan(cfg["ledger_file"], set(cfg["groups_counted"]), month_start, day_start, cfg["max_usd_per_call"])
    with _cache_lock:
        _cache.update(key=key, at=_mono(), value=value)
    return value


# ------------------------------------------------------------------ guard
@dataclass
class Decision:
    allowed: bool
    reason: str
    role: str = ""
    group: str = ""
    est_usd: float = 0.0
    month_spent: float = 0.0
    day_spent: float = 0.0
    in_flight: float = 0.0
    caps: Dict[str, float] = field(default_factory=dict)
    ledger_missing: bool = False
    settling: float = 0.0         # finished calls' cost still held while their ledger rows land
    reservation: int = 0          # >0 only for a held reservation (see reserve/release)
    bad_lines: int = 0            # unparseable/negative ledger rows in the scanned month window
    usable_caps: Dict[str, float] = field(default_factory=dict)   # caps after safety_margin_pct

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_res_lock = threading.Lock()
_reservations: Dict[int, float] = {}
_settling: List[tuple] = []     # (expires_mono, usd): spent calls not yet trusted to be in the ledger
_res_seq = 0
_overrun_lock = threading.Lock()
_overruns: Dict[str, Any] = {"count": 0, "excess_usd": 0.0, "last_excess_usd": 0.0}


def overruns_reset() -> None:
    with _overrun_lock:
        _overruns.update(count=0, excess_usd=0.0, last_excess_usd=0.0)


def _record_overrun(actual: float, est: float) -> None:
    """A gateway-reported cost above the pre-call estimate: the price table is out of sync."""
    if actual > est:
        with _overrun_lock:
            _overruns["count"] += 1
            _overruns["excess_usd"] += actual - est
            _overruns["last_excess_usd"] = actual - est


def _decide(role: str, est_usd: float, hold: bool) -> Decision:
    global _res_seq
    cfg = teacher_config()
    caps = {"monthly_usd": cfg["monthly_usd_cap"], "daily_usd": cfg["daily_usd_cap"],
            "per_call_usd": cfg["max_usd_per_call"]}
    group = cfg["roles"].get(role, "")
    margin = cfg["safety_margin_pct"]
    usable = {k: v * (100.0 - margin) / 100.0 for k, v in caps.items()}
    d = Decision(False, "", role=role, group=group, est_usd=est_usd, caps=caps, usable_caps=usable)
    if not cfg["enabled"]:
        d.reason = "disabled"
        return d
    if role not in ROLES:
        d.reason = "unknown_role"
        return d
    if not group:
        d.reason = "role_unmapped"
        return d
    if not (caps["monthly_usd"] > 0 and caps["daily_usd"] > 0 and caps["per_call_usd"] > 0):
        d.reason = "no_cap"
        return d
    if group not in cfg["prices"]:
        d.reason = "no_price"
        return d
    try:
        est = float(est_usd)
    except (TypeError, ValueError):
        est = float("nan")
    if not (math.isfinite(est) and est >= 0):
        d.reason = "bad_estimate"
        return d
    try:
        led = read_ledger(cfg)
    except LedgerUnreadable:
        d.reason = "ledger_unreadable"
        return d
    except Exception:  # noqa: BLE001 - any ledger failure refuses the spend
        d.reason = "ledger_error"
        return d
    d.month_spent, d.day_spent, d.ledger_missing, d.bad_lines = led.month, led.day, led.missing, led.bad_lines
    if led.bad_lines > cfg["max_bad_lines"]:
        d.reason = "ledger_corrupt"
        return d
    if led.missing and not cfg["allow_missing_ledger"]:
        d.reason = "ledger_missing"
        return d
    with _res_lock:
        mono = _mono()
        _settling[:] = [(e, u) for e, u in _settling if e > mono]
        d.settling = sum(u for _, u in _settling)
        held = sum(_reservations.values()) + d.settling
        d.in_flight = held
        if est > usable["per_call_usd"]:
            d.reason = "over_per_call_cap"
        elif d.day_spent + held + est > usable["daily_usd"]:
            d.reason = "over_daily_cap"
        elif d.month_spent + held + est > usable["monthly_usd"]:
            d.reason = "over_monthly_cap"
        else:
            d.allowed, d.reason = True, "ok"
            if hold:
                _res_seq += 1
                _reservations[_res_seq] = est
                d.reservation = _res_seq
    return d


def check(role: str, est_usd: float) -> Decision:
    """Would a call of ``est_usd`` on ``role`` be allowed now? Does not hold budget."""
    return _decide(role, est_usd, hold=False)


def reserve(role: str, est_usd: float) -> Decision:
    """Like check, but an allowed decision holds ``est_usd`` (decision.reservation) until release()."""
    return _decide(role, est_usd, hold=True)


def release(decision: Decision, settle_usd: float = 0.0, settle_s: float = 0.0) -> None:
    """Drop a held reservation (idempotent). The ledger carries the real spend; ``settle_usd`` stays
    counted for ``settle_s`` more seconds because the gateway may write its row after the call returns."""
    with _res_lock:
        held = _reservations.pop(decision.reservation, None)
        if held is not None and settle_usd > 0 and settle_s > 0:
            _settling.append((_mono() + settle_s, settle_usd))
    decision.reservation = 0


# ------------------------------------------------------------------ call
def _default_transport(group: str, messages: Messages, max_tokens: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Through the shared LiteLLM client; returns the raw OpenAI-shaped response dict."""
    from loci.litellm_client import call_litellm_proxy
    return call_litellm_proxy(model=group, messages=messages, max_tokens=max_tokens,
                              temperature=0.0, metadata=metadata).raw


def _to_messages(prompt_or_messages: Union[str, Messages]) -> Messages:
    if isinstance(prompt_or_messages, str):
        return [{"role": "user", "content": prompt_or_messages}]
    return [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in prompt_or_messages]


_PII = (
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[EMAIL]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
    (re.compile(r"(?i)(?:[a-z]:[\\/]+users[\\/]+|/(?:home|users)/)[^\\/\s]+"), "[HOME]"),
)


def _redact_pii(text: str) -> str:
    """Emails, IPv4 addresses and home-directory user names. Not a privacy filter: callers choose what leaves."""
    for rx, sub in _PII:
        text = rx.sub(sub, text)
    return text


def _estimate(messages: Messages, max_tokens: int, cfg: Dict[str, Any], group: str) -> float:
    """Worst-case cost: input tokens (chars / chars_per_token) * in price + max_tokens * out price.
    An unpriced group yields 0.0; the guard refuses it (no_price) before the estimate is used."""
    price = cfg["prices"].get(group)
    if price is None:
        return 0.0
    in_tok = sum(len(m["content"]) for m in messages) / cfg["chars_per_token"]
    return (in_tok * price["in"] + max(int(max_tokens), 0) * price["out"]) / 1e6


def _actual_usd(resp: Dict[str, Any]) -> Optional[float]:
    """Cost the gateway reported in the response, else None."""
    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    hidden = resp.get("_hidden_params") if isinstance(resp.get("_hidden_params"), dict) else {}
    for v in (usage.get("cost"), usage.get("cost_usd"), resp.get("response_cost"), hidden.get("response_cost")):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f >= 0:
            return f
    return None


def _text(resp: Dict[str, Any]) -> str:
    try:
        return str(resp["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        return str(resp.get("text", "")) if isinstance(resp.get("text"), str) else ""


def teacher_call(role: str, prompt_or_messages: Union[str, Messages], max_tokens: int, purpose: str,
                 transport: Optional[Transport] = None) -> Dict[str, Any]:
    """Guarded cloud call. Never raises. Returns {ok, text, usd, reason, role, group, decision}.

    ``usd`` is the cost the gateway reported, else the pre-call estimate (``usd_source`` says which).
    After a request was sent, its cost (reported, else the estimate) stays counted by the guard for
    ``settle_seconds``, so the window before the gateway's ledger row lands is covered. A failure after
    sending (``transport_error``, ``bad_response``) sets ``cost_unknown``: the gateway may have billed."""
    out: Dict[str, Any] = {"ok": False, "text": "", "usd": 0.0, "reason": "", "role": role, "group": "",
                           "decision": None, "usd_source": "none", "cost_unknown": False}
    decision: Optional[Decision] = None
    settle_usd = 0.0
    settle_s = 0.0
    try:
        cfg = teacher_config()
        messages = _to_messages(prompt_or_messages)
        decision = reserve(role, _estimate(messages, max_tokens, cfg, cfg["roles"].get(role, "")))
        out["decision"], out["group"] = decision.to_dict(), decision.group
        if not decision.allowed:
            out["reason"] = decision.reason
            return out
        if cfg["sanitize"]:
            try:
                from llm_local import _sanitize_for_cloud
            except Exception:  # noqa: BLE001
                out["reason"] = "sanitizer_unavailable"
                return out
            messages = [{"role": m["role"],
                         "content": _redact_pii(_sanitize_for_cloud(m["content"], max_len=max(len(m["content"]), 1)))}
                        for m in messages]
        settle_s = cfg["settle_seconds"]
        meta = {"loci_internal_call": True, "loci_teacher_purpose": str(purpose)}
        settle_usd = decision.est_usd     # sent => may be billed, whatever happens next
        try:
            resp = (transport or _default_transport)(decision.group, messages, int(max_tokens), meta)
        except Exception as exc:  # noqa: BLE001
            out["reason"], out["cost_unknown"] = f"transport_error:{type(exc).__name__}", True
            return out
        if not isinstance(resp, dict):
            out["reason"], out["cost_unknown"] = "bad_response", True
            return out
        actual = _actual_usd(resp)
        settle_usd = actual if actual is not None else decision.est_usd
        if actual is not None:
            _record_overrun(actual, decision.est_usd)
        out["usd"], out["usd_source"] = (actual, "gateway") if actual is not None else (decision.est_usd, "estimate")
        out["text"] = _text(resp)
        out["ok"], out["reason"] = True, "ok"
        cache_clear()                 # the next guard read should see the new ledger row
        return out
    except Exception as exc:  # noqa: BLE001 - fail open for the caller
        out["ok"], out["reason"] = False, f"error:{type(exc).__name__}"
        return out
    finally:
        if decision is not None:
            release(decision, settle_usd, settle_s)


# ------------------------------------------------------------------ status / CLI
LIMITS_NOTE = ("Reservations and the settling hold are process-local and the ledger is cached "
               f"{LEDGER_CACHE_S:g}s: N Loci processes can jointly overshoot a cap by up to "
               "N x max_usd_per_call plus the spend inside the cache window. LiteLLM enforces no budget; "
               "this guard is the only brake. Overruns (cost above estimate) mean the price table is stale.")


def status() -> Dict[str, Any]:
    """Caps, spent, remaining, by group. Never raises."""
    cfg = teacher_config()
    out: Dict[str, Any] = {"enabled": cfg["enabled"], "roles": cfg["roles"], "groups_counted": cfg["groups_counted"],
                           "ledger_file": cfg["ledger_file"], "max_bad_lines": cfg["max_bad_lines"],
                           "safety_margin_pct": cfg["safety_margin_pct"], "priced_groups": sorted(cfg["prices"]),
                           "limits": LIMITS_NOTE,
                           "caps": {"monthly_usd": cfg["monthly_usd_cap"], "daily_usd": cfg["daily_usd_cap"],
                                    "per_call_usd": cfg["max_usd_per_call"]}}
    try:
        led = read_ledger(cfg)
    except LedgerUnreadable as exc:
        out.update(ledger_error=str(exc), ledger_missing=False)
        return out
    with _overrun_lock:
        out["overruns"] = {"count": _overruns["count"], "excess_usd": _overruns["excess_usd"],
                           "last_excess_usd": _overruns["last_excess_usd"],
                           "ledger_rows_over_call_cap": led.rows_over_call_cap}
    out.update(ledger_missing=led.missing, bad_lines=led.bad_lines,
               spent={"month_usd": led.month, "day_usd": led.day},
               remaining={"month_usd": max(cfg["monthly_usd_cap"] - led.month, 0.0),
                          "day_usd": max(cfg["daily_usd_cap"] - led.day, 0.0)},
               by_group={"month": led.by_group_month, "day": led.by_group_day})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["status"]:
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0
    if len(argv) == 2 and argv[0] == "check":
        est = _money(teacher_config()["max_usd_per_call"])
        d = check(argv[1], est)
        print(json.dumps(d.to_dict(), indent=2, sort_keys=True))
        return 0 if d.allowed else 1
    print("usage: teacher.py status | check <role>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

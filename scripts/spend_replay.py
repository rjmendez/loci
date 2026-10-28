#!/usr/bin/env python3
"""Replay the LiteLLM cost ledger to calibrate spend caps. Read-only, stdlib only.

Slice S1 of the Tripwire-Governor design. It changes no runtime behaviour: it reads
``~/.loci/litellm_cost_ledger.jsonl`` (and the rotated ``.1`` file), and answers the questions the
caps depend on, with exact Decimal arithmetic:

* what each calendar day cost, on the day boundary you choose (``--tz utc|local|-04:00``);
* the worst rolling 1 h and 24 h spend;
* what a daily hard cap, a rolling-24h cap and an hourly trigger would have blocked or downgraded;
* how much of the spend is Loci's own (``internal=True``) calls;
* how many empty responses are the streaming / zero-token pattern;
* ledger quality (rows read, malformed rows, empty ``key_alias``, rotation).

Spend totals include internal rows, because a cap must bound every dollar; the internal/external split
is reported next to them. ``--exclude-internal`` replays the caps against external traffic only.

    python scripts/spend_replay.py --tz local
    python scripts/spend_replay.py --ledger /path/ledger.jsonl --tz -04:00 --json
    python scripts/spend_replay.py --daily-hard 30,40,50,60 --hour-caps 10,25

Replay semantics: calls are taken in time order; a call that would push its day's cumulative spend past
the cap (``cumulative + cost > cap``, strictly greater) is blocked and does not add to the day's total,
as it would not have been made. Exits non-zero when no row could be read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

UTC = dt.timezone.utc
DEFAULT_LEDGER = Path.home() / ".loci" / "litellm_cost_ledger.jsonl"
HOUR = 3600.0
DAY = 86400.0
CENT = Decimal("0.000001")
# Hosts (host:port, exact match) that are the LiteLLM gateway as Hermes sees it: the lab name behind Caddy and loopback.
# Anything else in Hermes' usage table is a direct route that the gateway ledger and its caps cannot see.
GATEWAY_HOSTS = ("litellm.lab.floppydicks.net:4443", "localhost:4000", "127.0.0.1:4000")

Event = tuple  # (ts: datetime, cost: Decimal, internal: bool, deployment: str, empty: bool, stream: bool, completion_tokens: int, key_alias_empty: bool)


def parse_tz(spec: str) -> Optional[dt.tzinfo]:
    """``utc``, ``local`` (the machine's zone, DST-aware: returns None) or a fixed offset like ``-04:00``."""
    s = spec.strip().lower()
    if s == "utc":
        return UTC
    if s == "local":
        return None
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", spec.strip())
    if not m:
        raise SystemExit(f"spend_replay: bad --tz {spec!r}: use utc, local or an offset such as -04:00")
    delta = dt.timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
    return dt.timezone(delta if m.group(1) == "+" else -delta)


def day_key(ts: dt.datetime, tz: Optional[dt.tzinfo]) -> str:
    """Calendar day of ``ts`` in ``tz`` (None = the machine's local zone)."""
    return (ts.astimezone() if tz is None else ts.astimezone(tz)).date().isoformat()


def _parse_ts(value: Any) -> dt.datetime:
    if isinstance(value, bool):
        raise ValueError("ts is a bool")
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value), UTC)
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def parse_row(line: str) -> Event:
    """One ledger line -> an event tuple. Raises ValueError/KeyError/InvalidOperation on a malformed row."""
    row = json.loads(line)
    if not isinstance(row, dict):
        raise ValueError("row is not an object")
    ts = _parse_ts(row["ts"])
    cost = Decimal(str(row.get("cost_usd") or 0))
    if not cost.is_finite() or cost < 0:
        raise ValueError(f"bad cost_usd {row.get('cost_usd')!r}")
    return (
        ts,
        cost,
        _truthy(row.get("internal")),
        str(row.get("deployment") or row.get("model") or "?"),
        _truthy(row.get("empty_response")),
        _truthy(row.get("stream")),
        int(row.get("completion_tokens") or 0),
        not row.get("key_alias"),
    )


def read_ledger(paths: Iterable[Path]) -> tuple[list[Event], int, list[str]]:
    """Events from every existing path, streamed line by line and sorted by time; the malformed-row count;
    and the paths actually read."""
    events: list[Event] = []
    bad = 0
    read: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        read.append(str(path))
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    events.append(parse_row(line))
                except (ValueError, KeyError, ArithmeticError, TypeError):
                    bad += 1
    events.sort(key=lambda e: e[0])
    return events, bad, read


def would_block(cumulative: Decimal, cost: Decimal, cap: Decimal) -> bool:
    """A call is blocked when it would push the running total strictly past the cap."""
    return cumulative + cost > cap


def day_totals(events: list[Event], tz: Optional[dt.tzinfo]) -> dict[str, dict[str, Any]]:
    days: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": Decimal(0), "external": Decimal(0), "internal": Decimal(0), "calls": 0})
    for ts, cost, internal, *_ in events:
        d = days[day_key(ts, tz)]
        d["total"] += cost
        d["internal" if internal else "external"] += cost
        d["calls"] += 1
    return dict(sorted(days.items()))


def rolling_max(events: list[Event], window_s: float) -> tuple[Decimal, Optional[dt.datetime]]:
    """Largest spend in any window of ``window_s`` seconds ending at a call (half-open: (t - W, t])."""
    best, best_at = Decimal(0), None
    total, left = Decimal(0), 0
    for ev in events:
        total += ev[1]
        while (ev[0] - events[left][0]).total_seconds() >= window_s:
            total -= events[left][1]
            left += 1
        if total > best:
            best, best_at = total, ev[0]
    return best, best_at


def replay_daily_cap(events: list[Event], cap: Decimal, tz: Optional[dt.tzinfo]) -> dict[str, Any]:
    running: dict[str, Decimal] = defaultdict(Decimal)
    blocked_calls, blocked_usd, days_hit = 0, Decimal(0), set()
    for ts, cost, *_ in events:
        key = day_key(ts, tz)
        if would_block(running[key], cost, cap):
            blocked_calls += 1
            blocked_usd += cost
            days_hit.add(key)
        else:
            running[key] += cost
    return {"cap_usd": cap, "blocked_calls": blocked_calls, "blocked_usd": blocked_usd,
            "days_hit": sorted(days_hit)}


def replay_rolling_cap(events: list[Event], cap: Decimal, window_s: float) -> dict[str, Any]:
    """Calls whose window total (including the call) would exceed ``cap``: the downgrade/trigger count."""
    hits, hit_usd = 0, Decimal(0)
    total, left = Decimal(0), 0
    for ev in events:
        total += ev[1]
        while (ev[0] - events[left][0]).total_seconds() >= window_s:
            total -= events[left][1]
            left += 1
        if total > cap:
            hits += 1
            hit_usd += ev[1]
    return {"cap_usd": cap, "calls_over": hits, "usd_over": hit_usd}


def empty_stats(events: list[Event]) -> dict[str, Any]:
    suspect, other = 0, 0
    by_dep: Counter = Counter()
    for _ts, _cost, _internal, dep, empty, stream, ctoks, _ka in events:
        if not empty:
            continue
        by_dep[dep] += 1
        if stream and ctoks == 0:
            suspect += 1
        else:
            other += 1
    return {"empty_total": suspect + other, "empty_suspect_stream": suspect, "empty_other": other,
            "by_deployment": dict(by_dep.most_common())}


def _host(url: Any) -> str:
    return urlparse(str(url or "")).netloc.lower()


def read_hermes_usage(db: Path, gateway_hosts: Iterable[str] = GATEWAY_HOSTS) -> dict[str, Any]:
    """Hermes' own usage table (state.db, session_model_usage), read-only, grouped by provider and route host.

    Tokens only: Hermes' cost columns are not trustworthy (estimated_cost_usd on OpenRouter rows sums to a large
    negative number), so they are never read. A route is "gateway" when its host:port is exactly one of
    ``gateway_hosts``; every other host is "direct" (invisible to the LiteLLM ledger and its caps). An unreadable
    database is reported in ``error``; it is never reported as zero usage."""
    gateway = {h.strip().lower() for h in gateway_hosts}
    out: dict[str, Any] = {"db": str(db), "error": None, "routes": [], "cost_columns": "ignored: unreliable"}
    try:
        con = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, timeout=5)
    except (sqlite3.Error, OSError) as exc:
        out["error"] = f"cannot open: {exc}"
        return out
    try:
        rows = con.execute(
            "SELECT billing_provider, billing_base_url, COUNT(*), SUM(api_call_count), SUM(input_tokens), "
            "SUM(output_tokens), SUM(cache_read_tokens) FROM session_model_usage GROUP BY 1, 2").fetchall()
    except sqlite3.Error as exc:
        out["error"] = f"cannot read session_model_usage: {exc}"
        return out
    finally:
        con.close()
    merged: dict[tuple[str, str], list[int]] = {}
    for provider, base_url, usage_rows, calls, tin, tout, cread in rows:
        acc = merged.setdefault((str(provider or "?"), _host(base_url)), [0, 0, 0, 0, 0])
        for i, v in enumerate((usage_rows, calls, tin, tout, cread)):
            acc[i] += int(v or 0)
    for (provider, host), (usage_rows, calls, tin, tout, cread) in merged.items():
        out["routes"].append({"provider": provider, "host": host, "route": "gateway" if host in gateway else "direct",
                              "usage_rows": usage_rows, "api_calls": calls, "input_tokens": tin,
                              "output_tokens": tout, "cache_read_tokens": cread})
    out["routes"].sort(key=lambda r: (-r["input_tokens"], r["host"]))
    total = sum(r["input_tokens"] for r in out["routes"])
    outside = sum(r["input_tokens"] for r in out["routes"] if r["route"] == "direct")
    out["input_tokens_total"], out["input_tokens_outside_gateway"] = total, outside
    out["outside_gateway_share"] = round(outside / total, 4) if total else 0.0
    return out


def build_report(paths: list[Path], tz_spec: str, daily_caps: list[Decimal], hour_caps: list[Decimal],
                 rolling24_cap: Decimal, exclude_internal: bool = False, hermes_db: Optional[Path] = None,
                 gateway_hosts: Iterable[str] = GATEWAY_HOSTS) -> dict[str, Any]:
    tz = parse_tz(tz_spec)
    all_events, bad, read = read_ledger(paths)
    if not all_events:
        raise SystemExit(f"spend_replay: rows_read=0 (bad_rows={bad}) from {[str(p) for p in paths]}")
    events = [e for e in all_events if not (exclude_internal and e[2])]
    days = day_totals(all_events, tz)
    total = sum((e[1] for e in all_events), Decimal(0))
    internal_usd = sum((e[1] for e in all_events if e[2]), Decimal(0))
    internal_rows = sum(1 for e in all_events if e[2])
    peak_day = max(days.items(), key=lambda kv: kv[1]["total"])
    r1, r1_at = rolling_max(events, HOUR)
    r24, r24_at = rolling_max(events, DAY)
    span_h = (all_events[-1][0] - all_events[0][0]).total_seconds() / HOUR
    report: dict[str, Any] = {
        "ledger": {
            "files_read": read, "rows_read": len(all_events), "bad_rows": bad,
            "first_ts": all_events[0][0].isoformat(), "last_ts": all_events[-1][0].isoformat(),
            "span_hours": round(span_h, 1),
            # True when no rotated <path>.1 was found: history older than the primary file may exist elsewhere
            "ledger_rotated_partial": len(read) < 2,
        },
        "tz": tz_spec,
        "caps_replayed_against": "external only" if exclude_internal else "all rows",
        "spend": {
            "total_usd": total, "internal_usd": internal_usd, "external_usd": total - internal_usd,
            "internal_rows": internal_rows, "internal_row_share": round(internal_rows / len(all_events), 4),
            "internal_usd_share": round(float(internal_usd / total), 4) if total else 0.0,
        },
        "days": {k: {"total_usd": v["total"], "external_usd": v["external"], "internal_usd": v["internal"],
                     "calls": v["calls"]} for k, v in days.items()},
        "peaks": {
            "max_day": {"day": peak_day[0], "usd": peak_day[1]["total"]},
            "max_rolling_1h": {"usd": r1, "ending": r1_at.isoformat() if r1_at else None},
            "max_rolling_24h": {"usd": r24, "ending": r24_at.isoformat() if r24_at else None},
        },
        "daily_hard_cap_replay": [replay_daily_cap(events, c, tz) for c in daily_caps],
        "rolling_24h_cap_replay": replay_rolling_cap(events, rolling24_cap, DAY),
        "hourly_cap_replay": [replay_rolling_cap(events, c, HOUR) for c in hour_caps],
        "empty_response": empty_stats(all_events),
        "ledger_quality": {"key_alias_empty": sum(1 for e in all_events if e[7]), "rows": len(all_events)},
    }
    if hermes_db is not None:
        report["hermes_direct_usage"] = read_hermes_usage(hermes_db, gateway_hosts)
    return report


def _money(value: Decimal) -> str:
    return format(value.quantize(CENT, rounding=ROUND_HALF_EVEN), "f")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return _money(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def render_text(report: dict[str, Any]) -> str:
    led, sp, pk = report["ledger"], report["spend"], report["peaks"]
    out = [
        f"ledger: {led['rows_read']} rows over {led['span_hours']} h ({led['first_ts'][:16]} .. {led['last_ts'][:16]}), "
        f"{led['bad_rows']} malformed, files {len(led['files_read'])}"
        + ("  [no rotated .1 file found: older history may be missing]" if led["ledger_rotated_partial"] else ""),
        f"spend: total ${_money(sp['total_usd'])} = external ${_money(sp['external_usd'])} + internal "
        f"${_money(sp['internal_usd'])} ({sp['internal_row_share']:.0%} of rows, {sp['internal_usd_share']:.0%} of dollars)",
        f"day boundary: {report['tz']}   caps replayed against: {report['caps_replayed_against']}",
        "", "per day (usd, calls):",
    ]
    for day, d in report["days"].items():
        out.append(f"  {day}  ${_money(d['total_usd']):>10}  {d['calls']:>5} calls   (external ${_money(d['external_usd'])})")
    out += [
        "",
        f"peak day: {pk['max_day']['day']} ${_money(pk['max_day']['usd'])}",
        f"max rolling 1h:  ${_money(pk['max_rolling_1h']['usd'])} ending {pk['max_rolling_1h']['ending']}",
        f"max rolling 24h: ${_money(pk['max_rolling_24h']['usd'])} ending {pk['max_rolling_24h']['ending']}",
        "", "daily hard cap, would have blocked:",
    ]
    for r in report["daily_hard_cap_replay"]:
        out.append(f"  cap ${_money(r['cap_usd']):>9}: {r['blocked_calls']:>4} calls, ${_money(r['blocked_usd'])} blocked, days {r['days_hit'] or '-'}")
    r = report["rolling_24h_cap_replay"]
    out.append(f"rolling-24h cap ${_money(r['cap_usd'])}: {r['calls_over']} calls over, ${_money(r['usd_over'])}")
    for r in report["hourly_cap_replay"]:
        out.append(f"hourly cap ${_money(r['cap_usd'])}: {r['calls_over']} calls over, ${_money(r['usd_over'])}")
    e = report["empty_response"]
    out += ["", f"empty responses: {e['empty_total']} (stream + 0 completion tokens: {e['empty_suspect_stream']}, other: {e['empty_other']}) by deployment {e['by_deployment']}",
            f"ledger quality: key_alias empty on {report['ledger_quality']['key_alias_empty']} of {report['ledger_quality']['rows']} rows"]
    h = report.get("hermes_direct_usage")
    if h is not None:
        out += ["", f"Hermes usage ({h['db']}); cost columns {h['cost_columns']}:"]
        if h["error"]:
            out.append(f"  UNREADABLE: {h['error']}")
        else:
            for r in h["routes"]:
                out.append(f"  {r['route']:7} {r['provider']:11} {r['host']:36} {r['api_calls']:>6} calls  in {r['input_tokens']:>11}  out {r['output_tokens']:>9}")
            out.append(f"  input tokens outside the gateway: {h['input_tokens_outside_gateway']} of {h['input_tokens_total']} ({h['outside_gateway_share']:.1%})")
    return "\n".join(out)


def _decimals(csv: str) -> list[Decimal]:
    return [Decimal(x.strip()) for x in csv.split(",") if x.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="primary ledger file; <path>.1 is read too")
    ap.add_argument("--tz", default="local", help="day boundary: utc, local, or a fixed offset such as -04:00")
    ap.add_argument("--daily-hard", default="30,40,50,60", help="comma list of daily hard caps (USD) to replay")
    ap.add_argument("--rolling24", default="60", help="rolling-24h cap (USD) to replay")
    ap.add_argument("--hour-caps", default="10,25", help="comma list of rolling-1h caps (USD) to replay")
    ap.add_argument("--exclude-internal", action="store_true", help="replay caps against external traffic only")
    ap.add_argument("--hermes-db", type=Path, default=None,
                    help="Hermes state.db: also report its usage by route (tokens), flagging routes outside the gateway")
    ap.add_argument("--gateway-host", action="append", default=None,
                    help="host:port counted as the gateway (repeatable; default: the lab name and loopback :4000)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON (money as exact strings)")
    args = ap.parse_args(argv)
    paths = [args.ledger, args.ledger.with_name(args.ledger.name + ".1")]
    report = build_report(paths, args.tz, _decimals(args.daily_hard), _decimals(args.hour_caps),
                          Decimal(args.rolling24), args.exclude_internal, args.hermes_db,
                          args.gateway_host or GATEWAY_HOSTS)
    print(json.dumps(_jsonable(report), indent=1) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""scripts/spend_replay.py: read-only replay of the LiteLLM cost ledger for cap calibration.

Everything runs against fixtures written to tmp_path; the live ledger is never read. Money is compared
as Decimal, and some fixtures use values whose float sums drift (0.1 + 0.2 == 0.30000000000000004).

Mutation log (each applied to scripts/spend_replay.py; the named tests went red; then reverted):
- day_key ignores the zone and buckets in UTC: test_day_boundary_follows_the_chosen_zone, test_local_zone_means_the_machines_zone
- would_block uses >= instead of >: test_cap_boundary_is_strictly_greater, test_replay_at_the_cap_blocks_nothing_and_one_cent_over_blocks_one,
  test_a_blocked_call_does_not_count_toward_the_day
- read_ledger skips the .1 file: test_rotated_file_is_read_and_flagged, test_cli_text_report_names_the_day_and_warns_without_the_rotated_file
- rotation flag inverted: the same two tests
- the grand spend total drops internal rows: test_grand_total_and_split_are_exact_where_float_sums_drift, test_internal_rows_are_counted_and_split,
  test_excluding_internal_replays_the_caps_on_external_traffic_only
- float arithmetic instead of Decimal (per-day sums, rolling-window accumulation, the internal split, the grand total as a naive loop):
  test_totals_are_exact_where_float_sums_drift, test_grand_total_and_split_are_exact_where_float_sums_drift, test_internal_rows_are_counted_and_split
- rolling window made inclusive: test_rolling_window_is_half_open, test_totals_are_exact_where_float_sums_drift
- a blocked call still adds to the day's running total: test_a_blocked_call_does_not_count_toward_the_day
Two fixture lessons from this run: 1.10 + 2.20 + 3.30 sums cleanly in floats (6.6), so that fixture alone could not detect float
arithmetic (hence the 0.1/0.2/0.3 and 0.07 x 3 tests); and sum() over floats is compensated on Python >= 3.12, so the grand-total
mutation has to be a naive accumulation loop to show anything.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "spend_replay.py"


def _load():
    spec = importlib.util.spec_from_file_location("spend_replay_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sr = _load()
D = Decimal
UTC = dt.timezone.utc


def _row(ts, cost, **extra):
    row = {"ts": ts, "cost_usd": cost, "deployment": "cf-test", "internal": False, "call_type": "acompletion"}
    row.update(extra)
    return row


def _write(path: Path, rows, raw_lines=()):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        for line in raw_lines:
            fh.write(line + "\n")


def _report(path, tz="utc", daily=("40",), hour=("10",), r24="60", exclude_internal=False):
    return sr.build_report([path, path.with_name(path.name + ".1")], tz, [D(c) for c in daily],
                           [D(c) for c in hour], D(r24), exclude_internal)


def _at(minutes, base="2026-10-10T12:00:00+00:00"):
    return (dt.datetime.fromisoformat(base) + dt.timedelta(minutes=minutes)).isoformat()


def _events(*pairs):
    """[(minutes_from_base, cost)] -> event tuples the pure functions take."""
    base = dt.datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    return [(base + dt.timedelta(minutes=m), D(str(c)), False, "d", False, False, 0, True) for m, c in pairs]


# ---- per-day totals and the day boundary ------------------------------------------------------------

def test_per_day_totals_are_exact(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row(_at(0), 1.10), _row(_at(10), 2.20), _row(_at(20), 3.30),
                    _row(_at(24 * 60), 4.40), _row(_at(24 * 60 + 10), 5.50), _row(_at(24 * 60 + 20), 6.60)])
    days = _report(ledger)["days"]
    assert list(days) == ["2026-10-10", "2026-10-11"]
    assert days["2026-10-10"]["total_usd"] == D("6.60")
    assert days["2026-10-11"]["total_usd"] == D("16.50")
    assert days["2026-10-10"]["calls"] == 3


def test_totals_are_exact_where_float_sums_drift(tmp_path):
    # 0.1 + 0.2 == 0.30000000000000004 and 0.07 * 3 summed == 0.21000000000000002 in floats; the fixture above
    # (1.10 + 2.20 + 3.30) happens to sum cleanly in floats and so cannot tell exact from approximate.
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row(_at(0), 0.1), _row(_at(1), 0.2),
                    _row(_at(24 * 60), 0.07), _row(_at(24 * 60 + 1), 0.07), _row(_at(24 * 60 + 2), 0.07)])
    rep = _report(ledger)
    assert rep["days"]["2026-10-10"]["total_usd"] == D("0.3")
    assert rep["days"]["2026-10-11"]["total_usd"] == D("0.21")
    assert rep["spend"]["total_usd"] == D("0.51")
    # half-open window: the day-2 calls start exactly 24 h after the first call, so the best window is day 1's 0.1 + 0.2
    assert rep["peaks"]["max_rolling_24h"]["usd"] == D("0.3")


def test_grand_total_and_split_are_exact_where_float_sums_drift(tmp_path):
    # 0.1 + 0.2 + 0.3 == 0.6000000000000001 in floats; the grand total and the internal/external split are summed
    # separately from the per-day totals.
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row(_at(0), 0.1), _row(_at(1), 0.2), _row(_at(2), 0.3, internal=True)])
    sp = _report(ledger)["spend"]
    assert (sp["total_usd"], sp["external_usd"], sp["internal_usd"]) == (D("0.6"), D("0.3"), D("0.3"))


def test_day_boundary_follows_the_chosen_zone(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row("2026-10-17T23:30:00-04:00", 1.00), _row("2026-10-17T12:00:00-04:00", 2.00)])
    in_zone = _report(ledger, tz="-04:00")["days"]
    in_utc = _report(ledger, tz="utc")["days"]
    assert {d: v["total_usd"] for d, v in in_zone.items()} == {"2026-10-17": D("3.00")}
    assert {d: v["total_usd"] for d, v in in_utc.items()} == {"2026-10-17": D("2.00"), "2026-10-18": D("1.00")}


def test_local_zone_means_the_machines_zone():
    ts = dt.datetime(2026, 7, 1, 3, 30, tzinfo=UTC)
    assert sr.day_key(ts, sr.parse_tz("local")) == dt.datetime.fromtimestamp(ts.timestamp()).date().isoformat()
    assert sr.day_key(ts, sr.parse_tz("utc")) == "2026-07-01"


@pytest.mark.parametrize("spec,offset_h", [("utc", 0), ("-04:00", -4), ("+05:30", 5.5), ("-0400", -4)])
def test_tz_specs_parse_to_the_offset_they_name(spec, offset_h):
    tz = sr.parse_tz(spec)
    assert tz.utcoffset(None) == dt.timedelta(hours=offset_h)


def test_bad_tz_is_refused_by_name():
    with pytest.raises(SystemExit, match="bad --tz 'America/New_York'"):
        sr.parse_tz("America/New_York")


def test_timestamp_forms_all_land_on_the_same_day(tmp_path):
    ledger = tmp_path / "l.jsonl"
    epoch = dt.datetime(2026, 10, 10, 12, 0, tzinfo=UTC).timestamp()
    _write(ledger, [_row(epoch, 1.0), _row("2026-10-10T12:00:00Z", 1.0), _row("2026-10-10T12:00:00", 1.0),
                    _row("2026-10-10T08:00:00-04:00", 1.0)])
    days = _report(ledger)["days"]
    assert {d: v["calls"] for d, v in days.items()} == {"2026-10-10": 4}


# ---- cap replay semantics ---------------------------------------------------------------------------

def test_cap_boundary_is_strictly_greater():
    assert sr.would_block(D("39.99"), D("0.02"), D("40.00")) is True
    assert sr.would_block(D("39.98"), D("0.02"), D("40.00")) is False     # exactly at the cap is allowed
    assert sr.would_block(D("39.00"), D("0.02"), D("40.00")) is False     # positive twin, well under


def test_replay_at_the_cap_blocks_nothing_and_one_cent_over_blocks_one():
    day = [(0, "39.98"), (1, "0.02")]
    at_cap = sr.replay_daily_cap(_events(*day), D("40.00"), sr.UTC)
    assert (at_cap["blocked_calls"], at_cap["blocked_usd"], at_cap["days_hit"]) == (0, D(0), [])
    over = sr.replay_daily_cap(_events((0, "39.99"), (1, "0.02")), D("40.00"), sr.UTC)
    assert (over["blocked_calls"], over["blocked_usd"], over["days_hit"]) == (1, D("0.02"), ["2026-10-10"])


def test_a_blocked_call_does_not_count_toward_the_day():
    r = sr.replay_daily_cap(_events((0, "8"), (1, "5"), (2, "2")), D("10"), sr.UTC)
    assert (r["blocked_calls"], r["blocked_usd"]) == (1, D("5"))          # the 2 fits: 8 + 2 == 10, not > 10


def test_the_cap_resets_on_the_next_day():
    r = sr.replay_daily_cap(_events((0, "9"), (24 * 60, "9")), D("10"), sr.UTC)
    assert r["blocked_calls"] == 0


def test_hourly_cap_is_strict_and_counts_the_call_that_crosses_it():
    over = sr.replay_rolling_cap(_events((0, "6"), (10, "5")), D("10"), sr.HOUR)
    assert (over["calls_over"], over["usd_over"]) == (1, D("5"))
    at = sr.replay_rolling_cap(_events((0, "5"), (10, "5")), D("10"), sr.HOUR)
    assert at["calls_over"] == 0


# ---- rolling windows --------------------------------------------------------------------------------

def test_rolling_one_hour_sums_calls_inside_the_hour(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row(_at(0), 4.00), _row(_at(29), 4.00), _row(_at(59), 4.00)])
    assert _report(ledger)["peaks"]["max_rolling_1h"]["usd"] == D("12.00")


def test_calls_61_minutes_apart_never_share_a_window(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row(_at(0), 4.00), _row(_at(61), 4.00), _row(_at(122), 4.00)])
    assert _report(ledger)["peaks"]["max_rolling_1h"]["usd"] == D("4.00")


def test_rolling_window_is_half_open():
    exactly_an_hour, _ = sr.rolling_max(_events((0, "4"), (60, "4")), sr.HOUR)
    assert exactly_an_hour == D("4")
    just_inside, _ = sr.rolling_max(_events((0, "4"), (59, "4")), sr.HOUR)
    assert just_inside == D("8")


def test_rolling_24h_and_its_end_time(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row(_at(0), 3.00), _row(_at(12 * 60), 4.00), _row(_at(23 * 60), 5.00), _row(_at(25 * 60), 1.00)])
    peak = _report(ledger)["peaks"]["max_rolling_24h"]
    assert peak["usd"] == D("12.00")
    assert peak["ending"] == dt.datetime.fromisoformat(_at(23 * 60)).isoformat()


# ---- ledger reading ---------------------------------------------------------------------------------

def test_rotated_file_is_read_and_flagged(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row(_at(24 * 60), 5.50)])
    _write(tmp_path / "l.jsonl.1", [_row(_at(0), 1.10), _row(_at(5), 2.20)])
    both = _report(ledger)
    assert both["spend"]["total_usd"] == D("8.80")
    assert both["ledger"]["ledger_rotated_partial"] is False
    assert both["ledger"]["rows_read"] == 3 and len(both["ledger"]["files_read"]) == 2
    (tmp_path / "l.jsonl.1").unlink()
    only_primary = _report(ledger)
    assert only_primary["spend"]["total_usd"] == D("5.50")
    assert only_primary["ledger"]["ledger_rotated_partial"] is True


def test_zero_rows_exits_nonzero_by_name(tmp_path):
    ledger = tmp_path / "l.jsonl"
    ledger.write_text("")
    with pytest.raises(SystemExit, match=r"rows_read=0 \(bad_rows=0\)"):
        _report(ledger)
    _write(ledger, [_row(_at(0), 1.0)])
    assert _report(ledger)["ledger"]["rows_read"] == 1                    # positive twin


def test_an_all_malformed_ledger_is_also_zero_rows(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [], raw_lines=["{not json", '{"ts": "bad", "cost_usd": 1}'])
    with pytest.raises(SystemExit, match=r"rows_read=0 \(bad_rows=2\)"):
        _report(ledger)


def test_malformed_lines_are_counted_and_good_rows_still_total(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row(_at(0), 1.50), _row(_at(1), 2.50)],
           raw_lines=["{not json", '{"ts": "2026-10-10T12:00:00Z", "cost_usd": -1}', "[1, 2]"])
    rep = _report(ledger)
    assert rep["ledger"]["bad_rows"] == 3
    assert rep["ledger"]["rows_read"] == 2
    assert rep["spend"]["total_usd"] == D("4.00")


def test_a_missing_cost_counts_as_zero_not_a_bad_row(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [{"ts": _at(0)}, _row(_at(1), 2.00)])
    rep = _report(ledger)
    assert (rep["ledger"]["bad_rows"], rep["ledger"]["rows_read"], rep["spend"]["total_usd"]) == (0, 2, D("2.00"))


# ---- internal rows, empties, quality ----------------------------------------------------------------

def test_internal_rows_are_counted_and_split(tmp_path):
    ledger = tmp_path / "l.jsonl"
    rows = [_row(_at(i), 0.10, internal=True) for i in range(3)] + [_row(_at(10), 5.00), _row(_at(20), 5.00)]
    _write(ledger, rows)
    sp = _report(ledger)["spend"]
    assert (sp["total_usd"], sp["internal_usd"], sp["external_usd"]) == (D("10.30"), D("0.30"), D("10.00"))
    assert (sp["internal_rows"], sp["internal_row_share"]) == (3, 0.6)


def test_excluding_internal_replays_the_caps_on_external_traffic_only(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row(_at(0), 0.30, internal=True), _row(_at(1), 5.00), _row(_at(2), 5.00)])
    with_all = _report(ledger)
    external = _report(ledger, exclude_internal=True)
    assert with_all["peaks"]["max_rolling_1h"]["usd"] == D("10.30")
    assert external["peaks"]["max_rolling_1h"]["usd"] == D("10.00")
    assert external["spend"]["total_usd"] == D("10.30")                   # totals still show every dollar
    assert external["caps_replayed_against"] == "external only"


def test_empty_responses_are_split_by_the_streaming_zero_token_pattern(tmp_path):
    ledger = tmp_path / "l.jsonl"
    opus = {"deployment": "cf-claude-opus-5", "empty_response": True, "stream": True, "completion_tokens": 0}
    _write(ledger, [
        _row(_at(0), 0.1, **opus), _row(_at(1), 0.1, **opus), _row(_at(2), 0.1, **opus),
        _row(_at(3), 0.1, deployment="cf-glm", empty_response=True, stream=False, completion_tokens=0),
        _row(_at(4), 0.1, deployment="cf-glm", empty_response=True, stream=True, completion_tokens=5),
        _row(_at(5), 0.1, deployment="cf-claude-opus-5", empty_response=False, stream=True, completion_tokens=0),
    ])
    e = _report(ledger)["empty_response"]
    assert (e["empty_total"], e["empty_suspect_stream"], e["empty_other"]) == (5, 3, 2)
    assert e["by_deployment"] == {"cf-claude-opus-5": 3, "cf-glm": 2}


def test_empty_key_alias_is_counted(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row(_at(0), 1, key_alias=""), _row(_at(1), 1), _row(_at(2), 1, key_alias="hermes")])
    q = _report(ledger)["ledger_quality"]
    assert (q["key_alias_empty"], q["rows"]) == (2, 3)


# ---- the command line -------------------------------------------------------------------------------

def test_cli_json_report_has_exact_money_strings(tmp_path, capsys):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row(_at(0), 1.10), _row(_at(1), 2.20), _row(_at(2), 3.30)])
    assert sr.main(["--ledger", str(ledger), "--tz", "utc", "--json", "--daily-hard", "5,10", "--hour-caps", "2"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["days"]["2026-10-10"]["total_usd"] == "6.600000"
    assert [(c["cap_usd"], c["blocked_calls"]) for c in out["daily_hard_cap_replay"]] == [("5.000000", 1), ("10.000000", 0)]
    assert out["hourly_cap_replay"][0]["calls_over"] == 2


def test_cli_text_report_names_the_day_and_warns_without_the_rotated_file(tmp_path, capsys):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [_row(_at(0), 1.10), _row(_at(1), 2.20), _row(_at(2), 3.30)])
    assert sr.main(["--ledger", str(ledger), "--tz", "utc"]) == 0
    text = capsys.readouterr().out
    assert "2026-10-10" in text and "6.600000" in text
    assert "no rotated .1 file found" in text
    _write(tmp_path / "l.jsonl.1", [_row(_at(-60), 1.0)])
    assert sr.main(["--ledger", str(ledger), "--tz", "utc"]) == 0
    assert "no rotated .1 file found" not in capsys.readouterr().out


def test_cli_exits_by_message_on_an_empty_ledger(tmp_path):
    empty = tmp_path / "e.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit, match="spend_replay: rows_read=0"):
        sr.main(["--ledger", str(empty)])

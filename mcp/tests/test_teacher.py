"""teacher core: config, ledger reader, fail-closed guard, reservations, guarded call, CLI."""
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import backends  # noqa: E402
import teacher as T  # noqa: E402

NOW = datetime(2026, 3, 15, 12, 0, 0).timestamp()
DAY_START = datetime(2026, 3, 15).timestamp()
MONTH_START = datetime(2026, 3, 1).timestamp()
GROUPS = {"judge": "grp-j", "label_a": "grp-a", "label_b": "grp-b",
          "propose": "grp-p", "adjudicate": "grp-d", "verify": "grp-v"}

TOML = """
[teacher]
enabled = true
monthly_usd_cap = 10.0
daily_usd_cap = 2.0
max_usd_per_call = 0.5
safety_margin_pct = 0
ledger_file = "{ledger}"
[teacher.roles]
judge = "grp-j"
label_a = "grp-a"
label_b = "grp-b"
propose = "grp-p"
adjudicate = "grp-d"
verify = "grp-v"
""" + "".join(f'[teacher.prices.{g}]\nin_usd_per_mtok = 50.0\nout_usd_per_mtok = 200.0\n'
              for g in ("grp-j", "grp-a", "grp-b", "grp-p", "grp-d", "grp-v"))


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    for k in list(os.environ):
        if k.startswith("LOCI_TEACHER_"):
            monkeypatch.delenv(k)
    monkeypatch.setattr(T, "_now", lambda: NOW)
    clock = {"t": 1000.0}
    monkeypatch.setattr(T, "_mono", lambda: clock["t"])
    cfg = tmp_path / "backends.toml"
    monkeypatch.setattr(backends, "_CONFIG_PATH", str(cfg))
    backends._config_cache_clear()
    T.cache_clear()
    T._reservations.clear()
    T._settling.clear()
    T.overruns_reset()
    yield {"cfg": cfg, "ledger": tmp_path / "ledger.jsonl", "clock": clock, "tmp": tmp_path}
    T._reservations.clear()
    T._settling.clear()
    T.cache_clear()
    backends._config_cache_clear()


def setup_cfg(env, replace=None, ledger=True):
    text = TOML.format(ledger=str(env["ledger"]).replace("\\", "/"))
    if replace:
        assert replace[0] in text
        text = text.replace(*replace)
    env["cfg"].write_text(text, encoding="utf-8")
    if ledger and not env["ledger"].exists():
        env["ledger"].write_text("", encoding="utf-8")     # the guard refuses a missing ledger
    backends._config_cache_clear()
    T.cache_clear()


def rows(path, items, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for ts, grp, cost in items:
            f.write(json.dumps({"ts": ts, "group": grp, "cost_usd": cost, "ok": True}) + "\n")


# ---------------------------------------------------------------- config
def test_defaults_everything_off():
    c = T.teacher_config()
    assert c["enabled"] is False
    assert c["roles"] == {}
    assert (c["monthly_usd_cap"], c["daily_usd_cap"], c["max_usd_per_call"]) == (0.0, 0.0, 0.0)
    assert c["sanitize"] is True
    assert c["ledger_file"] == str(Path("~/.loci/litellm_cost_ledger.jsonl").expanduser())
    assert c["groups_counted"] == []
    assert (c["max_bad_lines"], c["safety_margin_pct"], c["chars_per_token"], c["prices"]) == (0, 10.0, 3.0, {})


def test_config_from_toml_counts_role_groups_by_default(env):
    setup_cfg(env)
    c = T.teacher_config()
    assert c["enabled"] is True
    assert c["roles"] == GROUPS
    assert c["groups_counted"] == ["grp-a", "grp-b", "grp-d", "grp-j", "grp-p", "grp-v"]
    assert (c["monthly_usd_cap"], c["daily_usd_cap"], c["max_usd_per_call"]) == (10.0, 2.0, 0.5)
    assert c["safety_margin_pct"] == 0.0
    assert c["prices"]["grp-j"] == {"in": 50.0, "out": 200.0} and len(c["prices"]) == 6


def test_prices_from_env_replace_toml_and_drop_incomplete(env, monkeypatch):
    setup_cfg(env)
    monkeypatch.setenv("LOCI_TEACHER_PRICES", "grp-j=1.5:2.5, grp-a=3, grp-b=-1:2, grp-p=x:1")
    assert T.teacher_config()["prices"] == {"grp-j": {"in": 1.5, "out": 2.5}}


def test_chars_per_token_and_bad_lines_from_env(monkeypatch):
    monkeypatch.setenv("LOCI_TEACHER_CHARS_PER_TOKEN", "4")
    monkeypatch.setenv("LOCI_TEACHER_MAX_BAD_LINES", "7")
    monkeypatch.setenv("LOCI_TEACHER_SAFETY_MARGIN_PCT", "25")
    c = T.teacher_config()
    assert (c["chars_per_token"], c["max_bad_lines"], c["safety_margin_pct"]) == (4.0, 7, 25.0)


@pytest.mark.parametrize("bad", ["100", "-5", "abc", "nan"])
def test_unusable_safety_margin_falls_back_to_default(bad, monkeypatch):
    monkeypatch.setenv("LOCI_TEACHER_SAFETY_MARGIN_PCT", bad)
    assert T.teacher_config()["safety_margin_pct"] == 10.0


def test_env_overrides_toml(env, monkeypatch):
    setup_cfg(env)
    monkeypatch.setenv("LOCI_TEACHER_ENABLED", "0")
    monkeypatch.setenv("LOCI_TEACHER_DAILY_USD_CAP", "0.25")
    monkeypatch.setenv("LOCI_TEACHER_ROLE_JUDGE", "env-judge")
    monkeypatch.setenv("LOCI_TEACHER_GROUPS_COUNTED", "x, y")
    monkeypatch.setenv("LOCI_TEACHER_SANITIZE", "false")
    c = T.teacher_config()
    assert c["enabled"] is False
    assert c["daily_usd_cap"] == 0.25
    assert c["roles"]["judge"] == "env-judge" and c["roles"]["verify"] == "grp-v"
    assert c["groups_counted"] == ["x", "y"]
    assert c["sanitize"] is False


@pytest.mark.parametrize("bad", ["abc", "-1", "nan", "inf"])
def test_unusable_cap_means_zero(bad, monkeypatch):
    monkeypatch.setenv("LOCI_TEACHER_MONTHLY_USD_CAP", bad)
    assert T.teacher_config()["monthly_usd_cap"] == 0.0


def test_usable_cap_is_read(monkeypatch):
    monkeypatch.setenv("LOCI_TEACHER_MONTHLY_USD_CAP", "7.5")
    assert T.teacher_config()["monthly_usd_cap"] == 7.5


# ---------------------------------------------------------------- ledger
def test_ledger_sums_counted_groups_by_month_and_day(env):
    setup_cfg(env)
    rows(env["ledger"], [
        (MONTH_START - 1, "grp-j", 100.0),        # last month
        (MONTH_START + 10, "grp-j", 1.0),         # this month, not today
        (DAY_START - 1, "grp-a", 2.0),            # yesterday
        (DAY_START, "grp-a", 0.25),               # today, boundary
        (NOW - 5, "grp-j", 0.5),                  # today
        (NOW - 4, "other-group", 50.0),           # not counted
    ])
    led = T.read_ledger()
    assert (led.month, led.day) == (3.75, 0.75)
    assert led.by_group_month == {"grp-j": 1.5, "grp-a": 2.25}
    assert led.by_group_day == {"grp-a": 0.25, "grp-j": 0.5}
    assert led.missing is False


def test_ledger_tolerates_bad_lines_and_counts_them(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 9, "grp-j", 0.5)])
    with open(env["ledger"], "a", encoding="utf-8") as f:
        f.write("not json\n\n{\"ts\": 1}\n{\"ts\": \"x\", \"group\": \"grp-j\", \"cost_usd\": 1}\n")
        f.write(json.dumps({"ts": NOW - 8, "group": "grp-j", "cost_usd": -3}) + "\n")
    rows(env["ledger"], [(NOW - 7, "grp-j", 0.25)], mode="a")
    led = T.read_ledger()
    assert led.month == 0.75
    assert led.bad_lines == 4


def test_ledger_missing_is_zero_and_reported(env):
    setup_cfg(env, ledger=False)
    led = T.read_ledger()
    assert (led.month, led.day, led.missing) == (0.0, 0.0, True)
    assert T.check("judge", 0.1).ledger_missing is True


def test_guard_refuses_missing_ledger(env):
    setup_cfg(env, ledger=False)
    d = T.check("judge", 0.1)
    assert (d.allowed, d.reason) == (False, "ledger_missing")
    assert T.reserve("judge", 0.1).reservation == 0 and T._reservations == {}


def test_guard_allows_missing_ledger_only_when_opted_in(env, monkeypatch):
    setup_cfg(env, ledger=False)
    assert T.check("judge", 0.1).reason == "ledger_missing"
    monkeypatch.setenv("LOCI_TEACHER_ALLOW_MISSING_LEDGER", "1")
    d = T.check("judge", 0.1)
    assert (d.allowed, d.reason, d.ledger_missing) == (True, "ok", True)


def test_call_with_missing_ledger_never_reaches_transport(env):
    setup_cfg(env, ledger=False)
    rec = Rec(reply())
    out = T.teacher_call("judge", "hello", 50, "p", transport=rec)
    assert rec.calls == []
    assert (out["ok"], out["reason"]) == (False, "ledger_missing")


def test_ledger_present_is_not_reported_missing(env):
    setup_cfg(env)
    rows(env["ledger"], [])
    assert T.check("judge", 0.1).ledger_missing is False


def test_ledger_reads_across_chunk_boundaries(env, monkeypatch):
    setup_cfg(env)
    monkeypatch.setattr(T, "_CHUNK", 16)
    rows(env["ledger"], [(MONTH_START - 5, "grp-j", 9.0)] + [(MONTH_START + i, "grp-j", 0.125) for i in range(1, 41)])
    assert T.read_ledger().month == 5.0


def test_ledger_follows_rotated_siblings(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 1, "grp-j", 0.5)])
    rows(str(env["ledger"]) + ".1", [(MONTH_START + 5, "grp-j", 1.0)])
    rows(str(env["ledger"]) + ".2", [(MONTH_START - 5, "grp-j", 7.0), (MONTH_START + 6, "grp-j", 2.0)])
    assert T.read_ledger().month == 3.5


def test_ledger_stops_before_month_start_and_skips_older_siblings(env):
    setup_cfg(env)
    rows(env["ledger"], [(MONTH_START - 1, "grp-j", 9.0), (NOW - 1, "grp-j", 0.5)])
    (env["tmp"] / "ledger.jsonl.1").mkdir()          # opening it would raise: proves it is not read
    assert T.read_ledger().month == 0.5


def test_ledger_reads_sibling_when_month_not_yet_crossed(env):
    """Twin of the stop test: the same unreadable sibling is reached when the month is not crossed."""
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 1, "grp-j", 0.5)])
    (env["tmp"] / "ledger.jsonl.1").mkdir()
    with pytest.raises(T.LedgerUnreadable, match=r"ledger\.jsonl\.1"):
        T.read_ledger()


def test_ledger_cached_for_a_few_seconds(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 1, "grp-j", 0.5)])
    assert T.read_ledger().month == 0.5
    rows(env["ledger"], [(NOW - 1, "grp-j", 1.0)], mode="a")
    assert T.read_ledger().month == 0.5
    env["clock"]["t"] += T.LEDGER_CACHE_S + 0.1
    assert T.read_ledger().month == 1.5


# ---------------------------------------------------------------- guard
def test_guard_allows_within_caps(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 1, "grp-j", 0.5)])
    d = T.check("judge", 0.25)
    assert (d.allowed, d.reason, d.group) == (True, "ok", "grp-j")
    assert (d.month_spent, d.day_spent) == (0.5, 0.5)
    assert d.caps == {"monthly_usd": 10.0, "daily_usd": 2.0, "per_call_usd": 0.5}


def test_guard_disabled(env):
    setup_cfg(env)
    assert T.check("judge", 0.1).allowed is True
    setup_cfg(env, ("enabled = true", "enabled = false"))
    d = T.check("judge", 0.1)
    assert (d.allowed, d.reason) == (False, "disabled")


def test_guard_unknown_and_unmapped_role(env):
    setup_cfg(env)
    d = T.check("nope", 0.1)
    assert (d.allowed, d.reason) == (False, "unknown_role")
    setup_cfg(env, ('verify = "grp-v"', ""))
    d = T.check("verify", 0.1)
    assert (d.allowed, d.reason) == (False, "role_unmapped")
    assert T.check("judge", 0.1).allowed is True


@pytest.mark.parametrize("key,val", [("monthly_usd_cap", "10.0"), ("daily_usd_cap", "2.0"), ("max_usd_per_call", "0.5")])
def test_guard_missing_or_zero_cap_refuses(env, key, val):
    setup_cfg(env)
    assert T.check("judge", 0.1).allowed is True
    setup_cfg(env, (f"{key} = {val}", f"{key} = 0"))
    d = T.check("judge", 0.1)
    assert (d.allowed, d.reason) == (False, "no_cap")


def test_guard_unreadable_ledger_refuses(env):
    setup_cfg(env)
    assert T.check("judge", 0.1).allowed is True
    env["ledger"].unlink()
    env["ledger"].mkdir()
    T.cache_clear()
    d = T.check("judge", 0.1)
    assert (d.allowed, d.reason) == (False, "ledger_unreadable")


def test_guard_per_call_boundary(env):
    setup_cfg(env)
    assert T.check("judge", 0.4999999).allowed is True
    assert T.check("judge", 0.5).allowed is True
    d = T.check("judge", 0.5000001)
    assert (d.allowed, d.reason) == (False, "over_per_call_cap")


def test_guard_daily_boundary(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 1, "grp-j", 1.75)])
    assert T.check("judge", 0.2499999).allowed is True
    assert T.check("judge", 0.25).allowed is True
    d = T.check("judge", 0.2500001)
    assert (d.allowed, d.reason) == (False, "over_daily_cap")


def test_guard_monthly_boundary(env):
    setup_cfg(env)
    rows(env["ledger"], [(MONTH_START + 1, "grp-j", 9.75)])    # nothing today
    assert T.check("judge", 0.2499999).allowed is True
    assert T.check("judge", 0.25).allowed is True
    d = T.check("judge", 0.2500001)
    assert (d.allowed, d.reason) == (False, "over_monthly_cap")


def test_safety_margin_shrinks_every_usable_cap(env):
    setup_cfg(env, ("safety_margin_pct = 0", "safety_margin_pct = 10"))
    rows(env["ledger"], [(NOW - 1, "grp-j", 1.5)])
    d = T.check("judge", 0.3)                                  # 1.8 == 90% of the 2.0 daily cap
    assert (d.allowed, d.usable_caps) == (True, {"monthly_usd": 9.0, "daily_usd": 1.8, "per_call_usd": 0.45})
    assert d.caps == {"monthly_usd": 10.0, "daily_usd": 2.0, "per_call_usd": 0.5}
    d = T.check("judge", 0.4)                                  # 1.9 fits the raw cap, not the usable one
    assert (d.allowed, d.reason) == (False, "over_daily_cap")
    d = T.check("judge", 0.46)                                 # over 0.45 usable per-call, under 0.5 raw
    assert (d.allowed, d.reason) == (False, "over_per_call_cap")
    setup_cfg(env)                                             # margin 0: the same 0.4 call fits
    assert T.check("judge", 0.4).allowed is True


def test_safety_margin_shrinks_monthly_cap(env):
    setup_cfg(env, ("safety_margin_pct = 0", "safety_margin_pct = 10"))
    rows(env["ledger"], [(MONTH_START + 1, "grp-j", 8.9)])
    assert T.check("judge", 0.1).allowed is True               # 9.0 == usable monthly cap
    d = T.check("judge", 0.2)
    assert (d.allowed, d.reason) == (False, "over_monthly_cap")


def test_guard_no_price_refuses_unpriced_group(env):
    setup_cfg(env)
    assert T.check("judge", 0.1).allowed is True
    env["cfg"].write_text(env["cfg"].read_text(encoding="utf-8").replace("[teacher.prices.grp-j]", "[teacher.prices.other]"),
                          encoding="utf-8")
    backends._config_cache_clear()
    d = T.check("judge", 0.1)
    assert (d.allowed, d.reason) == (False, "no_price")
    assert T.check("verify", 0.1).allowed is True              # priced groups are unaffected


def test_call_no_price_never_reaches_transport(env):
    setup_cfg(env)
    env["cfg"].write_text(env["cfg"].read_text(encoding="utf-8").replace("[teacher.prices.grp-p]", "[teacher.prices.other]"),
                          encoding="utf-8")
    backends._config_cache_clear()
    rec = Rec(reply())
    out = T.teacher_call("propose", "hello", 10, "p", transport=rec)
    assert (out["ok"], out["reason"], rec.calls) == (False, "no_price", [])
    assert T.teacher_call("judge", "hello", 10, "p", transport=rec)["ok"] is True


def test_guard_corrupt_ledger_refuses_by_default(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 9, "grp-j", 0.5)])
    assert T.check("judge", 0.1).allowed is True
    with open(env["ledger"], "a", encoding="utf-8") as f:
        f.write("{\"ts\": 1\n")                               # truncated row
    T.cache_clear()
    d = T.check("judge", 0.1)
    assert (d.allowed, d.reason, d.bad_lines) == (False, "ledger_corrupt", 1)


def test_negative_cost_row_counts_as_corrupt(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 9, "grp-j", -0.5)])
    d = T.check("judge", 0.1)
    assert (d.allowed, d.reason, d.bad_lines) == (False, "ledger_corrupt", 1)


def test_max_bad_lines_is_an_explicit_tolerance(env, monkeypatch):
    setup_cfg(env)
    with open(env["ledger"], "a", encoding="utf-8") as f:
        f.write("junk\nmore junk\n")
    monkeypatch.setenv("LOCI_TEACHER_MAX_BAD_LINES", "1")
    d = T.check("judge", 0.1)
    assert (d.allowed, d.reason, d.bad_lines) == (False, "ledger_corrupt", 2)
    monkeypatch.setenv("LOCI_TEACHER_MAX_BAD_LINES", "2")
    d = T.check("judge", 0.1)
    assert (d.allowed, d.reason, d.bad_lines) == (True, "ok", 2)


def test_bad_lines_before_month_start_are_not_counted(env):
    setup_cfg(env)
    with open(env["ledger"], "w", encoding="utf-8") as f:
        f.write("junk in an old month\n")
        f.write(json.dumps({"ts": MONTH_START - 5, "group": "grp-j", "cost_usd": 1}) + "\n")
    rows(env["ledger"], [(NOW - 9, "grp-j", 0.5)], mode="a")
    d = T.check("judge", 0.1)
    assert (d.allowed, d.bad_lines) == (True, 0)


def test_call_with_corrupt_ledger_never_reaches_transport(env):
    setup_cfg(env)
    with open(env["ledger"], "a", encoding="utf-8") as f:
        f.write("junk\n")
    rec = Rec(reply())
    out = T.teacher_call("judge", "hello", 10, "p", transport=rec)
    assert (out["ok"], out["reason"], rec.calls) == (False, "ledger_corrupt", [])


@pytest.mark.parametrize("bad", [float("nan"), -1.0, "x"])
def test_guard_bad_estimate_refuses(env, bad):
    setup_cfg(env)
    assert T.check("judge", 0.1).allowed is True
    d = T.check("judge", bad)
    assert (d.allowed, d.reason) == (False, "bad_estimate")


def test_check_does_not_hold_budget(env):
    setup_cfg(env)
    for _ in range(5):
        assert T.check("judge", 0.5).allowed is True
    assert T._reservations == {}


def test_reservation_blocks_second_caller_until_release(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 1, "grp-j", 1.2)])           # 0.8 left today
    a = T.reserve("judge", 0.45)
    b = T.reserve("judge", 0.45)
    assert (a.allowed, b.allowed, b.reason) == (True, False, "over_daily_cap")
    assert b.in_flight == 0.45
    T.release(a)
    assert T.reserve("judge", 0.45).allowed is True


def test_concurrent_reserve_admits_exactly_what_fits(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 1, "grp-j", 1.0)])           # 1.0 left: two of 0.4 fit, not three
    barrier = threading.Barrier(6)
    results = []

    def go():
        barrier.wait()
        results.append(T.reserve("judge", 0.4).allowed)

    ts = [threading.Thread(target=go) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(results) == [False] * 4 + [True] * 2


# ---------------------------------------------------------------- teacher_call
class Rec:
    def __init__(self, resp=None, exc=None):
        self.calls, self.resp, self.exc = [], resp, exc

    def __call__(self, group, messages, max_tokens, metadata):
        self.calls.append((group, messages, max_tokens, metadata))
        if self.exc:
            raise self.exc
        return self.resp


def reply(text="hi", cost=None):
    r = {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 3}}
    if cost is not None:
        r["usage"]["cost"] = cost
    return r


def test_call_success_uses_group_metadata_and_gateway_cost(env):
    setup_cfg(env)
    rec = Rec(reply("answer", cost=0.0123))
    out = T.teacher_call("propose", "hello", 100, "unit-purpose", transport=rec)
    assert rec.calls == [("grp-p", [{"role": "user", "content": "hello"}], 100,
                          {"loci_internal_call": True, "loci_teacher_purpose": "unit-purpose"})]
    assert (out["ok"], out["text"], out["usd"], out["usd_source"], out["reason"]) == (True, "answer", 0.0123, "gateway", "ok")
    assert (out["role"], out["group"]) == ("propose", "grp-p")
    assert out["decision"]["allowed"] is True
    assert T._reservations == {}


def test_call_without_reported_cost_charges_estimate(env):
    setup_cfg(env)
    rec = Rec(reply())
    out = T.teacher_call("judge", "x" * 300, 100, "p", transport=rec)
    # 300/3 = 100 in-tokens * 50/M + 100 out-tokens * 200/M
    assert (out["usd"], out["usd_source"]) == (pytest.approx(0.025), "estimate")


def test_call_refused_never_reaches_transport(env):
    setup_cfg(env)
    rec = Rec(reply())
    out = T.teacher_call("nope", "hello", 100, "p", transport=rec)
    assert rec.calls == []
    assert (out["ok"], out["reason"], out["text"], out["usd"]) == (False, "unknown_role", "", 0.0)
    assert T.teacher_call("judge", "hello", 100, "p", transport=rec)["ok"] is True
    assert len(rec.calls) == 1


def test_call_oversize_request_refused_by_per_call_cap(env):
    setup_cfg(env)
    rec = Rec(reply())
    out = T.teacher_call("judge", "x" * 3000, 4000, "p", transport=rec)    # 1000*50/M + 4000*200/M = 0.85 > 0.5
    assert rec.calls == []
    assert out["reason"] == "over_per_call_cap"
    assert out["decision"]["est_usd"] == pytest.approx(0.85)


def test_estimate_follows_group_price_not_request_size(env):
    # A short prompt to an expensive group: the old size-scaled estimate passed this; the price-based one cannot.
    setup_cfg(env, ('[teacher.prices.grp-j]\nin_usd_per_mtok = 50.0\nout_usd_per_mtok = 200.0',
                    '[teacher.prices.grp-j]\nin_usd_per_mtok = 50.0\nout_usd_per_mtok = 10000.0'))
    rec = Rec(reply())
    out = T.teacher_call("judge", "hi", 100, "p", transport=rec)           # 100 out-tokens * 10000/M = 1.0 + 0.0000333
    assert (out["reason"], rec.calls) == ("over_per_call_cap", [])
    assert out["decision"]["est_usd"] == pytest.approx(1.0000333, rel=1e-4)
    assert T.teacher_call("verify", "hi", 100, "p", transport=rec)["ok"] is True   # cheap group, same request


def test_estimate_uses_configured_chars_per_token(env, monkeypatch):
    setup_cfg(env)
    msgs = [{"role": "user", "content": "x" * 400}]
    assert T._estimate(msgs, 0, T.teacher_config(), "grp-j") == pytest.approx(400 / 3.0 * 50 / 1e6)
    monkeypatch.setenv("LOCI_TEACHER_CHARS_PER_TOKEN", "4")
    assert T._estimate(msgs, 0, T.teacher_config(), "grp-j") == pytest.approx(100 * 50 / 1e6)


def test_gateway_cost_above_estimate_counts_an_overrun(env):
    setup_cfg(env)
    T.teacher_call("judge", "x" * 300, 100, "p", transport=Rec(reply(cost=0.125)))   # estimate 0.025
    s = T.status()
    assert s["overruns"]["count"] == 1
    assert s["overruns"]["excess_usd"] == pytest.approx(0.1)
    assert s["overruns"]["last_excess_usd"] == pytest.approx(0.1)


def test_gateway_cost_within_estimate_is_not_an_overrun(env):
    setup_cfg(env)
    T.teacher_call("judge", "x" * 300, 100, "p", transport=Rec(reply(cost=0.025)))   # equal to the estimate
    T.teacher_call("judge", "x" * 300, 100, "p", transport=Rec(reply(cost=0.01)))
    assert T.status()["overruns"]["count"] == 0


def test_ledger_rows_over_per_call_cap_are_surfaced(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 3, "grp-j", 0.5), (NOW - 2, "grp-j", 0.51), (NOW - 1, "other", 9.0)])
    assert T.status()["overruns"]["ledger_rows_over_call_cap"] == 1


def test_call_sanitizes_secrets_by_default_and_not_when_off(env, monkeypatch):
    setup_cfg(env)
    secret = "api_key=SUPERSECRETVALUE and ghp_" + "a" * 24
    rec = Rec(reply())
    T.teacher_call("judge", [{"role": "user", "content": secret}], 50, "p", transport=rec)
    assert rec.calls[0][1][0]["content"] == "api_key=[REDACTED] and [REDACTED]"
    monkeypatch.setenv("LOCI_TEACHER_SANITIZE", "0")
    rec2 = Rec(reply())
    T.teacher_call("judge", secret, 50, "p", transport=rec2)
    assert rec2.calls[0][1][0]["content"] == secret


def test_call_transport_error_is_reported_not_raised_and_releases(env):
    setup_cfg(env)
    rec = Rec(exc=ConnectionError("boom"))
    out = T.teacher_call("judge", "hello", 50, "p", transport=rec)
    assert (out["ok"], out["reason"], out["usd"]) == (False, "transport_error:ConnectionError", 0.0)
    assert len(rec.calls) == 1
    assert T._reservations == {}


def test_call_non_dict_response_is_bad_response(env):
    setup_cfg(env)
    out = T.teacher_call("judge", "hello", 50, "p", transport=Rec("text"))
    assert (out["ok"], out["reason"]) == (False, "bad_response")


def test_call_drops_ledger_cache_so_next_guard_sees_spend(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 1, "grp-j", 0.5)])
    assert T.check("judge", 0.1).day_spent == 0.5
    T.teacher_call("judge", "hello", 50, "p", transport=Rec(reply()))
    rows(env["ledger"], [(NOW, "grp-j", 0.25)], mode="a")
    assert T.check("judge", 0.1).day_spent == 0.75


def test_settling_cost_blocks_next_call_until_window_passes(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 1, "grp-j", 1.2)])           # 0.8 left today
    out = T.teacher_call("judge", "hello", 50, "p", transport=Rec(reply(cost=0.45)))
    assert (out["ok"], out["usd"], out["cost_unknown"]) == (True, 0.45, False)
    assert T._reservations == {}                              # released, but the cost is still held
    d = T.check("judge", 0.45)                                # ledger row has not landed yet
    assert (d.allowed, d.reason, d.settling, d.in_flight) == (False, "over_daily_cap", 0.45, 0.45)
    env["clock"]["t"] += 119.0
    assert T.check("judge", 0.45).allowed is False
    env["clock"]["t"] += 2.0                                  # past settle_seconds (120)
    d = T.check("judge", 0.45)
    assert (d.allowed, d.settling) == (True, 0.0)


def test_settling_uses_estimate_when_gateway_reports_no_cost(env):
    setup_cfg(env)
    T.teacher_call("judge", "x" * 300, 100, "p", transport=Rec(reply()))
    assert T.check("judge", 0.1).settling == pytest.approx(0.025)


def test_settle_seconds_zero_disables_the_hold(env, monkeypatch):
    setup_cfg(env)
    monkeypatch.setenv("LOCI_TEACHER_SETTLE_SECONDS", "0")
    T.teacher_call("judge", "hello", 50, "p", transport=Rec(reply(cost=0.4)))
    assert T.check("judge", 0.1).settling == 0.0


def test_transport_error_holds_estimate_and_flags_cost_unknown(env):
    setup_cfg(env)
    out = T.teacher_call("judge", "x" * 300, 100, "p", transport=Rec(exc=TimeoutError("late")))
    assert (out["ok"], out["reason"], out["cost_unknown"]) == (False, "transport_error:TimeoutError", True)
    assert T.check("judge", 0.1).settling == pytest.approx(0.025)


def test_bad_response_flags_cost_unknown_and_holds(env):
    setup_cfg(env)
    out = T.teacher_call("judge", "x" * 300, 100, "p", transport=Rec("text"))
    assert (out["reason"], out["cost_unknown"]) == ("bad_response", True)
    assert T.check("judge", 0.1).settling == pytest.approx(0.025)


def test_refusal_holds_nothing_and_is_not_cost_unknown(env):
    setup_cfg(env)
    out = T.teacher_call("nope", "hello", 50, "p", transport=Rec(reply()))
    assert (out["reason"], out["cost_unknown"]) == ("unknown_role", False)
    assert T._settling == []


def test_call_redacts_emails_ips_and_home_paths(env):
    setup_cfg(env)
    rec = Rec(reply())
    T.teacher_call("judge", "mail bob@example.org from 10.1.2.3 at /home/amy/y and /users/Bob/z", 80, "p", transport=rec)
    assert rec.calls[0][1][0]["content"] == "mail [EMAIL] from [IP] at [HOME]/y and [HOME]/z"


def test_redaction_leaves_ordinary_text_alone():
    assert T._redact_pii("version 1.2.3 of module a.b") == "version 1.2.3 of module a.b"


def test_default_transport_goes_through_shared_litellm_client(monkeypatch):
    import loci.litellm_client as lc
    seen = []

    class R:
        raw = {"choices": [{"message": {"content": "z"}}]}

    def fake(**kw):
        seen.append(kw)
        return R()

    monkeypatch.setattr(lc, "call_litellm_proxy", fake)
    meta = {"loci_internal_call": True}
    assert T._default_transport("g", [{"role": "user", "content": "q"}], 7, meta) == R.raw
    assert seen == [{"model": "g", "messages": [{"role": "user", "content": "q"}], "max_tokens": 7,
                     "temperature": 0.0, "metadata": meta}]


def test_client_sends_metadata_in_request_body(monkeypatch):
    import loci.litellm_client as lc
    bodies = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"k"}}]}'

    def fake_open(req, timeout):
        bodies.append(json.loads(req.data))
        return Resp()

    monkeypatch.setattr(lc.urllib.request, "urlopen", fake_open)
    lc.call_litellm_proxy("m", [], metadata={"a": 1})
    lc.call_litellm_proxy("m", [])
    assert bodies[0]["metadata"] == {"a": 1}
    assert "metadata" not in bodies[1]


# ---------------------------------------------------------------- status / CLI
def test_status_reports_caps_spent_remaining_by_group(env):
    setup_cfg(env)
    rows(env["ledger"], [(NOW - 1, "grp-j", 0.5), (NOW - 2, "grp-a", 0.25)])
    s = T.status()
    assert s["caps"] == {"monthly_usd": 10.0, "daily_usd": 2.0, "per_call_usd": 0.5}
    assert s["spent"] == {"month_usd": 0.75, "day_usd": 0.75}
    assert s["remaining"] == {"month_usd": 9.25, "day_usd": 1.25}
    assert s["by_group"]["month"] == {"grp-j": 0.5, "grp-a": 0.25}
    assert s["ledger_missing"] is False
    assert (s["safety_margin_pct"], s["max_bad_lines"]) == (0.0, 0)
    assert s["priced_groups"] == ["grp-a", "grp-b", "grp-d", "grp-j", "grp-p", "grp-v"]
    assert "process-local" in s["limits"] and "N x max_usd_per_call" in s["limits"] and "no budget" in s["limits"]


def test_status_reports_unreadable_ledger(env):
    setup_cfg(env, ledger=False)
    env["ledger"].mkdir()
    s = T.status()
    assert "ledger.jsonl" in s["ledger_error"] and "spent" not in s


def test_cli_status_and_check(env, capsys):
    setup_cfg(env)
    assert T.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out)["caps"]["daily_usd"] == 2.0
    assert T.main(["check", "judge"]) == 0
    assert json.loads(capsys.readouterr().out)["reason"] == "ok"
    assert T.main(["check", "nope"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "unknown_role"
    assert T.main(["bogus"]) == 2


def test_cli_check_is_a_readiness_probe_under_default_margin(env, capsys):
    setup_cfg(env, ("safety_margin_pct = 0", "safety_margin_pct = 10"))
    assert T.main(["check", "judge"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["reason"], out["est_usd"], out["usable_caps"]["per_call_usd"]) == ("ok", 0.45, 0.45)
    setup_cfg(env, ("enabled = true", "enabled = false"))      # not ready: still exits 1
    assert T.main(["check", "judge"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "disabled"

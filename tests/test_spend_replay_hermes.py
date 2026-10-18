"""spend_replay --hermes-db: how much of Hermes' traffic goes around the LiteLLM gateway (and so around its caps).

Fixture databases are built in tmp_path with the real table name and columns. Hermes' cost columns are filled with
garbage (negative sums, as the real OpenRouter rows have) to prove they are never read.

Mutation log (each applied to scripts/spend_replay.py; the named tests went red; then reverted):
- every route classified as gateway: test_routes_are_classified_by_exact_host, test_outside_share_is_exact
- host matched by prefix instead of exactly: test_a_lookalike_host_is_direct
- the cost columns summed into the report: test_cost_columns_are_never_read
- an unreadable database reported as zero usage: test_missing_db_is_an_error_not_zero, test_missing_table_is_an_error_not_zero
- the database opened read-write: test_the_database_is_opened_read_only_by_uri (a byte comparison cannot see it)
- same-host rows not merged in Python: test_rows_on_the_same_route_are_merged_with_exact_sums (needs a different PATH on the same host; identical URLs are merged by SQL GROUP BY)
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "spend_replay.py"


def _load():
    spec = importlib.util.spec_from_file_location("spend_replay_hermes_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sr = _load()

# (provider, base_url, api_calls, input_tokens, output_tokens, cache_read_tokens)
ROWS = [
    ("custom", "https://litellm.lab.floppydicks.net:4443/v1", 10, 5000, 500, 100),
    ("custom", "http://localhost:4000/v1", 4, 3000, 300, 50),
    ("custom", "http://localhost:4000/", 1, 200, 20, 0),                        # same host, DIFFERENT path: SQL keeps it apart, Python must merge
    ("openrouter", "https://openrouter.ai/api/v1", 7, 1500, 150, 0),
    ("custom", "https://gateway.ai.cloudflare.com/v1/ACCT/gw/openai", 2, 300, 30, 0),
    ("custom", "http://localhost:40001/v1", 3, 100, 10, 0),                     # lookalike of localhost:4000
]
TOTAL_IN = 5000 + 3000 + 200 + 1500 + 300 + 100          # 10100
OUTSIDE_IN = 1500 + 300 + 100                            # 1900


def _make_db(path: Path, rows=ROWS, with_table=True):
    con = sqlite3.connect(path)
    if with_table:
        con.execute("CREATE TABLE session_model_usage (session_id TEXT, model TEXT, billing_provider TEXT, "
                    "billing_base_url TEXT, api_call_count INTEGER, input_tokens INTEGER, output_tokens INTEGER, "
                    "cache_read_tokens INTEGER, estimated_cost_usd REAL, actual_cost_usd REAL)")
        for i, (prov, url, calls, tin, tout, cread) in enumerate(rows):
            con.execute("INSERT INTO session_model_usage VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (f"s{i}", "m", prov, url, calls, tin, tout, cread, -123456.5, -1.0))
    con.commit()
    con.close()
    return path


def _by_host(report):
    return {r["host"]: r for r in report["routes"]}


def test_routes_are_classified_by_exact_host(tmp_path):
    rep = sr.read_hermes_usage(_make_db(tmp_path / "state.db"))
    routes = _by_host(rep)
    assert rep["error"] is None
    assert {h: r["route"] for h, r in routes.items()} == {
        "litellm.lab.floppydicks.net:4443": "gateway", "localhost:4000": "gateway",
        "openrouter.ai": "direct", "gateway.ai.cloudflare.com": "direct", "localhost:40001": "direct"}


def test_a_lookalike_host_is_direct(tmp_path):
    routes = _by_host(sr.read_hermes_usage(_make_db(tmp_path / "state.db")))
    assert routes["localhost:40001"]["route"] == "direct"         # the real gateway, localhost:4000, is a different row
    assert routes["localhost:4000"]["route"] == "gateway"


def test_rows_on_the_same_route_are_merged_with_exact_sums(tmp_path):
    local = _by_host(sr.read_hermes_usage(_make_db(tmp_path / "state.db")))["localhost:4000"]
    assert (local["usage_rows"], local["api_calls"], local["input_tokens"], local["output_tokens"]) == (2, 5, 3200, 320)
    assert local["cache_read_tokens"] == 50


def test_outside_share_is_exact(tmp_path):
    rep = sr.read_hermes_usage(_make_db(tmp_path / "state.db"))
    assert rep["input_tokens_total"] == TOTAL_IN
    assert rep["input_tokens_outside_gateway"] == OUTSIDE_IN
    assert rep["outside_gateway_share"] == round(OUTSIDE_IN / TOTAL_IN, 4) == 0.1881


def test_a_gateway_only_database_has_zero_outside_share(tmp_path):
    only = ROWS[:3]                                              # the three genuine gateway rows (not the lookalike)
    rep = sr.read_hermes_usage(_make_db(tmp_path / "state.db", only))
    assert (rep["input_tokens_outside_gateway"], rep["outside_gateway_share"]) == (0, 0.0)
    assert rep["input_tokens_total"] == 8200


def test_override_makes_exactly_the_named_host_the_gateway(tmp_path):
    rep = sr.read_hermes_usage(_make_db(tmp_path / "state.db"), gateway_hosts=["openrouter.ai"])
    assert sorted(h for h, r in _by_host(rep).items() if r["route"] == "gateway") == ["openrouter.ai"]
    assert rep["input_tokens_outside_gateway"] == TOTAL_IN - 1500


def test_cost_columns_are_never_read(tmp_path):
    rep = sr.read_hermes_usage(_make_db(tmp_path / "state.db"))
    assert rep["cost_columns"] == "ignored: unreliable"
    assert not any("cost" in k for r in rep["routes"] for k in r)
    assert "-123456" not in json.dumps(rep)


def test_missing_db_is_an_error_not_zero(tmp_path):
    rep = sr.read_hermes_usage(tmp_path / "does_not_exist.db")
    assert rep["error"] == "cannot open: unable to open database file"
    assert rep["routes"] == []
    assert "input_tokens_total" not in rep


def test_missing_table_is_an_error_not_zero(tmp_path):
    rep = sr.read_hermes_usage(_make_db(tmp_path / "state.db", with_table=False))
    assert rep["error"] == "cannot read session_model_usage: no such table: session_model_usage"
    assert "input_tokens_total" not in rep


def test_the_database_is_opened_read_only_by_uri(tmp_path, monkeypatch):
    # sqlite3.connect is the dependency here; record how it is called. (Reading with mode=rw changes no bytes, so only
    # the call itself can pin the read-only intent.)
    db = _make_db(tmp_path / "state.db")                 # built BEFORE the patch: sqlite3 is one shared module
    calls = []
    real = sqlite3.connect

    def recording(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(sr.sqlite3, "connect", recording)
    sr.read_hermes_usage(db)
    assert len(calls) == 1
    (target,), kwargs = calls[0]
    assert target == f"{db.resolve().as_uri()}?mode=ro"
    assert kwargs["uri"] is True


def test_the_database_is_not_modified(tmp_path):
    db = _make_db(tmp_path / "state.db")
    before = (db.stat().st_size, db.read_bytes())
    sr.read_hermes_usage(db)
    assert (db.stat().st_size, db.read_bytes()) == before
    # read-only open: a write through the same URI form must fail (positive twin: a normal connection can write)
    ro = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly database"):
        ro.execute("INSERT INTO session_model_usage (session_id) VALUES ('x')")
    ro.close()
    rw = sqlite3.connect(db)
    rw.execute("INSERT INTO session_model_usage (session_id) VALUES ('x')")
    rw.close()


def test_report_section_appears_only_when_asked(tmp_path):
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(json.dumps({"ts": "2026-10-10T12:00:00Z", "cost_usd": 1.5}) + "\n")
    db = _make_db(tmp_path / "state.db")
    without = sr.build_report([ledger], "utc", [sr.Decimal("40")], [sr.Decimal("10")], sr.Decimal("60"))
    with_db = sr.build_report([ledger], "utc", [sr.Decimal("40")], [sr.Decimal("10")], sr.Decimal("60"), hermes_db=db)
    assert "hermes_direct_usage" not in without
    assert with_db["hermes_direct_usage"]["input_tokens_outside_gateway"] == OUTSIDE_IN


def test_cli_prints_the_outside_share_and_flags_direct_routes(tmp_path, capsys):
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(json.dumps({"ts": "2026-10-10T12:00:00Z", "cost_usd": 1.5}) + "\n")
    db = _make_db(tmp_path / "state.db")
    assert sr.main(["--ledger", str(ledger), "--tz", "utc", "--hermes-db", str(db)]) == 0
    text = capsys.readouterr().out
    assert "input tokens outside the gateway: 1900 of 10100 (18.8%)" in text
    assert "direct  openrouter" in text and "gateway custom" in text
    assert "cost columns ignored: unreliable" in text


def test_cli_reports_an_unreadable_db_loudly_and_still_replays_the_ledger(tmp_path, capsys):
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(json.dumps({"ts": "2026-10-10T12:00:00Z", "cost_usd": 1.5}) + "\n")
    assert sr.main(["--ledger", str(ledger), "--tz", "utc", "--hermes-db", str(tmp_path / "nope.db")]) == 0
    text = capsys.readouterr().out
    assert "UNREADABLE: cannot " in text
    assert "total $1.500000" in text


def test_cli_gateway_host_flag_is_repeatable(tmp_path, capsys):
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(json.dumps({"ts": "2026-10-10T12:00:00Z", "cost_usd": 1.5}) + "\n")
    db = _make_db(tmp_path / "state.db")
    assert sr.main(["--ledger", str(ledger), "--tz", "utc", "--hermes-db", str(db),
                    "--gateway-host", "openrouter.ai", "--gateway-host", "gateway.ai.cloudflare.com"]) == 0
    assert "input tokens outside the gateway: 8300 of 10100 (82.2%)" in capsys.readouterr().out

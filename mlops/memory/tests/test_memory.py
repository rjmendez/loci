"""Characterization tests for mlops/memory/ — decay.py and live_evo.py.

These pin the CURRENT behaviour of the two memory-maintenance jobs, warts included.
Several assertions deliberately lock in behaviour that is arguably wrong (marked
``BUG:``); they exist so that a later refactor cannot change it silently.

No external services are touched. Both modules talk only to a local SQLite file and
the filesystem, so every test builds its own throwaway DB / hook-state directory under
``tmp_path``. An autouse fixture redirects ``live_evo._ADAPTATION_LOG`` away from the
repo so no test can append to ``mlops/memory/live_evo_log.jsonl``.
"""

import io
import json
import math
import sqlite3
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import decay as D  # noqa: E402
import live_evo as L  # noqa: E402


# ======================================================================================
# helpers / fixtures
# ======================================================================================

WM_SCHEMA = (
    "CREATE TABLE working_memory ("
    " id INTEGER PRIMARY KEY,"
    " content TEXT,"
    " importance REAL,"
    " created_at TEXT,"
    " session_id TEXT)"
)


def _make_db(path: Path, rows=()) -> str:
    """rows are (id, content, importance, created_at, session_id) tuples."""
    conn = sqlite3.connect(str(path))
    conn.execute(WM_SCHEMA)
    if rows:
        conn.executemany("INSERT INTO working_memory VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return str(path)


def _read_importance(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    out = dict(conn.execute("SELECT id, importance FROM working_memory").fetchall())
    conn.close()
    return out


def _ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _write_log(path: Path, records) -> Path:
    """records may be dicts (json-encoded) or raw strings (written verbatim)."""
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(autouse=True)
def _isolate_adaptation_log(tmp_path, monkeypatch):
    """Never let a test append to the real mlops/memory/live_evo_log.jsonl."""
    monkeypatch.setattr(L, "_ADAPTATION_LOG", tmp_path / "live_evo_log.jsonl")
    return tmp_path / "live_evo_log.jsonl"


# ======================================================================================
# decay.py — module constants
# ======================================================================================

def test_decay_constants():
    assert D.DEFAULT_LAMBDA == 30.0
    assert D.DEFAULT_K == 0.8
    assert D.DEFAULT_MIN_IMPORTANCE == 0.05
    # DEFAULT_DB is expanduser'd, so it never still contains a literal "~"
    assert "~" not in D.DEFAULT_DB
    assert D.DEFAULT_DB.endswith("mnemosyne.db")


# ======================================================================================
# decay.weibull_retention
# ======================================================================================

def test_weibull_zero_and_negative_age_return_full_retention():
    assert D.weibull_retention(0) == 1.0
    assert D.weibull_retention(-0.0001) == 1.0
    assert D.weibull_retention(-10_000) == 1.0


def test_weibull_known_values():
    # exp(-((age/30)**0.8)) with the module defaults
    assert D.weibull_retention(7) == pytest.approx(0.7318622236806, rel=1e-9)
    assert D.weibull_retention(30) == pytest.approx(math.exp(-1.0), rel=1e-12)
    assert D.weibull_retention(90) == pytest.approx(0.0899748866078, rel=1e-9)


def test_weibull_docstring_7day_claim_is_wrong():
    # BUG (doc): the module docstring advertises "7 days -> 80% retention" but the
    # implementation yields ~73%. Pinned so the numbers cannot drift unnoticed.
    assert D.weibull_retention(7) < 0.75


def test_weibull_is_strictly_decreasing_in_age():
    vals = [D.weibull_retention(a) for a in (1, 5, 15, 30, 60, 120, 365)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert vals[-1] > 0.0  # never underflows to exactly zero at 1 year


def test_weibull_lambda_zero_raises_zero_division():
    # No guard on lambda_days: a caller passing 0 gets an exception, not a fallback.
    with pytest.raises(ZeroDivisionError):
        D.weibull_retention(10, lambda_days=0.0)
    # ...but only for positive ages; age<=0 short-circuits before the division.
    assert D.weibull_retention(0, lambda_days=0.0) == 1.0


def test_weibull_k_zero_is_age_independent():
    # x**0 == 1 for every positive x, so k=0 collapses to a constant exp(-1).
    assert D.weibull_retention(1, k=0.0) == pytest.approx(math.exp(-1.0))
    assert D.weibull_retention(9999, k=0.0) == pytest.approx(math.exp(-1.0))


def test_weibull_custom_lambda_and_k():
    assert D.weibull_retention(10, lambda_days=10.0, k=1.0) == pytest.approx(math.exp(-1.0))
    assert D.weibull_retention(20, lambda_days=10.0, k=2.0) == pytest.approx(math.exp(-4.0))


# ======================================================================================
# decay.apply_decay — failure / degraded paths
# ======================================================================================

def test_apply_decay_missing_db_returns_error_stub(tmp_path):
    res = D.apply_decay(str(tmp_path / "nope.db"))
    assert res == {
        "error": f"db not found: {tmp_path / 'nope.db'}",
        "n_rows": 0,
        "n_decayed": 0,
    }
    # the error stub carries NO mean_retention/min_retention/lambda_days/k/dry_run keys
    assert "mean_retention" not in res
    assert "dry_run" not in res


def test_apply_decay_missing_table_returns_error_stub(tmp_path):
    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()
    res = D.apply_decay(str(db))
    assert res == {"error": "no such table: working_memory", "n_rows": 0, "n_decayed": 0}


def test_apply_decay_empty_table_reports_neutral_retention(tmp_path):
    db = _make_db(tmp_path / "m.db")
    res = D.apply_decay(db)
    assert res["n_rows"] == 0
    assert res["n_decayed"] == 0
    # empty retention list => defaults of 1.0, not 0.0 and not NaN
    assert res["mean_retention"] == 1.0
    assert res["min_retention"] == 1.0


def test_apply_decay_success_dict_shape(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.8, _ago(30), "s1")])
    res = D.apply_decay(db, dry_run=True)
    assert set(res) == {
        "n_rows", "n_decayed", "mean_retention", "min_retention",
        "lambda_days", "k", "dry_run",
        "n_grounding_visible_before", "n_grounding_visible_after",
        "grounding_min_importance",
    }
    assert res["lambda_days"] == 30.0 and res["k"] == 0.8
    assert res["dry_run"] is True


# ======================================================================================
# decay.apply_decay — row selection and timestamp parsing
# ======================================================================================

def test_apply_decay_ignores_null_importance_rows(tmp_path):
    db = _make_db(tmp_path / "m.db", [
        (1, "a", 0.8, _ago(30), "s1"),
        (2, "b", None, _ago(30), "s1"),
    ])
    res = D.apply_decay(db, dry_run=True)
    assert res["n_rows"] == 1  # the WHERE clause drops the NULL row before counting
    assert _read_importance(db)[2] is None


def test_apply_decay_treats_naive_timestamps_as_utc(tmp_path):
    # created_at without a timezone offset is what SQLite CURRENT_TIMESTAMP writes,
    # i.e. every production row. It is naive UTC and must decay like any other row,
    # not raise TypeError into the except and be skipped.
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.9, "2020-01-01T00:00:00", "s1")])
    res = D.apply_decay(db)
    assert res["n_rows"] == 1
    assert res["n_decayed"] == 1
    assert res["mean_retention"] < 0.01
    assert _read_importance(db)[1] == D.DEFAULT_MIN_IMPORTANCE


def test_apply_decay_naive_and_aware_created_at_agree(tmp_path):
    # The sqlite "YYYY-MM-DD HH:MM:SS" form and the same instant written as
    # offset-aware ISO-8601 must produce the same retention.
    aware = (datetime.now(timezone.utc) - timedelta(days=200)).replace(microsecond=0)
    naive_db = _make_db(tmp_path / "naive.db",
                        [(1, "a", 0.9, aware.strftime("%Y-%m-%d %H:%M:%S"), "s1")])
    aware_db = _make_db(tmp_path / "aware.db", [(1, "a", 0.9, aware.isoformat(), "s1")])

    naive_res = D.apply_decay(naive_db, dry_run=True)
    aware_res = D.apply_decay(aware_db, dry_run=True)

    assert naive_res["n_decayed"] == aware_res["n_decayed"] == 1
    assert naive_res["mean_retention"] < 1.0
    assert naive_res["mean_retention"] == pytest.approx(aware_res["mean_retention"], abs=1e-6)


def test_apply_decay_skips_unparseable_and_null_created_at(tmp_path):
    db = _make_db(tmp_path / "m.db", [
        (1, "a", 0.9, "garbage", "s1"),
        (2, "b", 0.9, None, "s1"),
        (3, "c", 0.9, "", "s1"),
    ])
    res = D.apply_decay(db)
    assert res["n_rows"] == 3
    assert res["n_decayed"] == 0
    assert res["mean_retention"] == 1.0 and res["min_retention"] == 1.0
    assert set(_read_importance(db).values()) == {0.9}


def test_apply_decay_accepts_z_suffix_timestamps(tmp_path):
    stamp = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    db = _make_db(tmp_path / "m.db", [(1, "a", 1.0, stamp, "s1")])
    res = D.apply_decay(db)
    assert res["n_decayed"] == 1
    assert _read_importance(db)[1] == pytest.approx(math.exp(-1.0), rel=1e-4)


def test_apply_decay_future_timestamp_is_not_decayed(tmp_path):
    future = (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.7, future, "s1")])
    res = D.apply_decay(db)
    assert res["n_rows"] == 1
    assert res["n_decayed"] == 0          # retention 1.0 => delta below 1e-6
    assert res["mean_retention"] == 1.0   # the row DOES contribute a retention of 1.0
    assert _read_importance(db)[1] == 0.7


# ======================================================================================
# decay.apply_decay — arithmetic and writes
# ======================================================================================

def test_apply_decay_writes_importance_times_retention(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.8, _ago(30), "s1")])
    res = D.apply_decay(db)
    assert res["n_decayed"] == 1
    assert res["dry_run"] is False
    assert _read_importance(db)[1] == pytest.approx(0.8 * math.exp(-1.0), rel=1e-4)


def test_apply_decay_dry_run_computes_but_does_not_write(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.8, _ago(30), "s1")])
    res = D.apply_decay(db, dry_run=True)
    assert res["n_decayed"] == 1  # still reported as if it had been applied
    assert _read_importance(db)[1] == 0.8


def test_apply_decay_zero_importance_is_raised_to_the_floor(tmp_path):
    # BUG: max(min_importance, current * retention) is a floor, not a clamp, so a
    # decay pass *increases* the importance of any row already below min_importance
    # and counts that increase in n_decayed.
    db = _make_db(tmp_path / "m.db", [
        (1, "zero", 0.0, _ago(30), "s1"),
        (2, "tiny", 0.001, _ago(30), "s1"),
    ])
    res = D.apply_decay(db)
    assert res["n_decayed"] == 2
    assert _read_importance(db) == {1: 0.05, 2: 0.05}


def test_apply_decay_stops_at_the_floor(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.05, _ago(365), "s1")])
    res = D.apply_decay(db)
    assert res["n_decayed"] == 0  # already exactly at the floor => no delta
    assert _read_importance(db)[1] == 0.05


def test_apply_decay_custom_min_importance(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.4, _ago(365), "s1")])
    D.apply_decay(db, min_importance=0.3)
    assert _read_importance(db)[1] == 0.3


def test_apply_decay_custom_lambda_and_k_are_echoed_and_used(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "a", 1.0, _ago(10), "s1")])
    res = D.apply_decay(db, lambda_days=10.0, k=1.0)
    assert res["lambda_days"] == 10.0 and res["k"] == 1.0
    assert _read_importance(db)[1] == pytest.approx(math.exp(-1.0), rel=1e-3)


def test_apply_decay_is_idempotent(tmp_path):
    # Was test_apply_decay_is_not_idempotent, which characterised the compounding:
    # retention is a function of absolute age, so recomputing from the *current*
    # value multiplies it again on every run. Decay now reads base_importance.
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.8, _ago(30), "s1")])
    D.apply_decay(db)
    first = _read_importance(db)[1]
    D.apply_decay(db)
    second = _read_importance(db)[1]
    assert second == pytest.approx(first, abs=1e-9)
    assert first == pytest.approx(0.8 * math.exp(-1.0), rel=1e-3)


def _base_importance(db_path: str, rid: int):
    conn = sqlite3.connect(db_path)
    out = conn.execute("SELECT base_importance FROM working_memory WHERE id=?", (rid,)).fetchone()[0]
    conn.close()
    return out


def test_apply_decay_preserves_the_authored_importance(tmp_path):
    """Nothing else stores the authored value, so an in-place write is one-way."""
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.9, _ago(90), "s1")])
    D.apply_decay(db)
    assert _read_importance(db)[1] < 0.2
    assert _base_importance(db, 1) == pytest.approx(0.9), "the authored 0.9 cannot be restored"


def test_apply_decay_does_not_reseed_the_baseline_from_a_decayed_value(tmp_path):
    """Seeding on every run would re-derive the baseline from decayed output --
    idempotent-looking on run two, and still one-way."""
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.9, _ago(90), "s1")])
    D.apply_decay(db)
    D.apply_decay(db)
    assert _base_importance(db, 1) == pytest.approx(0.9)


def test_apply_decay_reports_what_it_costs_the_grounding_hook(tmp_path):
    """A row under HOOK_RECALL_MIN_IMPORTANCE stays in the table and drops out of
    recall, so n_decayed alone does not say what a run costs."""
    db = _make_db(tmp_path / "m.db", [
        (1, "old", 0.9, _ago(90), "s1"),
        (2, "new", 0.9, _ago(1), "s1"),
    ])
    res = D.apply_decay(db, dry_run=True)
    assert (res["n_grounding_visible_before"], res["n_grounding_visible_after"]) == (2, 1)
    assert _read_importance(db) == {1: 0.9, 2: 0.9}, "dry_run wrote to the database"


def test_apply_decay_retention_stats_cover_only_parseable_rows(tmp_path):
    db = _make_db(tmp_path / "m.db", [
        (1, "a", 0.9, _ago(1), "s1"),     # retention ~0.95
        (2, "b", 0.9, _ago(90), "s1"),    # retention ~0.09
        (3, "c", 0.9, "garbage", "s1"),   # skipped entirely
    ])
    res = D.apply_decay(db, dry_run=True)
    assert res["n_rows"] == 3
    assert res["min_retention"] == pytest.approx(D.weibull_retention(90), rel=1e-3)
    expected_mean = (D.weibull_retention(1) + D.weibull_retention(90)) / 2
    assert res["mean_retention"] == pytest.approx(expected_mean, rel=1e-3)


def test_apply_decay_lambda_zero_propagates_zero_division(tmp_path):
    # apply_decay has no try/except around weibull_retention; the DB handle leaks.
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.8, _ago(30), "s1")])
    with pytest.raises(ZeroDivisionError):
        D.apply_decay(db, lambda_days=0.0)


# ======================================================================================
# decay.main
# ======================================================================================

def test_decay_main_prints_summary_and_writes_out_file(tmp_path, monkeypatch):
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.8, _ago(30), "s1")])
    out = tmp_path / "nested" / "stats.json"
    monkeypatch.setattr(sys, "argv", ["decay", "--db", db, "--dry-run", "--out", str(out)])
    buf = io.StringIO()
    with redirect_stdout(buf):
        D.main()
    line = buf.getvalue()
    assert line.startswith("[decay] n_rows=1 n_decayed=1 ")
    assert "lambda=30.0d k=0.8 dry_run=True" in line
    stats = json.loads(out.read_text())          # parent dirs are created for you
    assert stats["n_rows"] == 1 and stats["dry_run"] is True
    assert _read_importance(db)[1] == 0.8        # --dry-run really did not write


def test_decay_main_on_missing_db_reports_zero_retention(tmp_path, monkeypatch):
    # The error stub has no mean_retention key, so the `.get(..., 0)` default prints
    # 0.000 — i.e. "total loss" — for a DB that simply does not exist.
    monkeypatch.setattr(sys, "argv", ["decay", "--db", str(tmp_path / "nope.db")])
    buf = io.StringIO()
    with redirect_stdout(buf):
        D.main()
    assert "n_rows=0 n_decayed=0 mean_retention=0.000 min_retention=0.000" in buf.getvalue()


# ======================================================================================
# live_evo.py — module constants
# ======================================================================================

def test_live_evo_constants():
    assert L.DEFAULT_PENALTY == 0.15
    assert L.DEFAULT_CONFIDENCE_FLOOR == 0.05
    assert L.SIMILARITY_WORDS == 6
    assert "~" not in L._DEFAULT_DB
    assert "~" not in L._DEFAULT_HOOK_STATE


# ======================================================================================
# live_evo._load_guard_failures
# ======================================================================================

def test_load_guard_failures_missing_dir_returns_empty(tmp_path):
    assert L._load_guard_failures(str(tmp_path / "absent")) == []


def test_load_guard_failures_path_that_is_a_file_returns_empty(tmp_path):
    f = tmp_path / "hook-state"
    f.write_text("not a directory")
    assert L._load_guard_failures(str(f)) == []


def test_load_guard_failures_only_matches_guard_bash_glob(tmp_path):
    hs = tmp_path / "hs"
    hs.mkdir()
    _write_log(hs / "guard_bash_a.log", [{"event": "retraction", "session_id": "keep"}])
    _write_log(hs / "guard_other.log", [{"event": "retraction", "session_id": "drop1"}])
    _write_log(hs / "guard_bash_a.jsonl", [{"event": "retraction", "session_id": "drop2"}])
    got = L._load_guard_failures(str(hs))
    assert [f["session_id"] for f in got] == ["keep"]


def test_load_guard_failures_event_filter_and_field_defaults(tmp_path):
    hs = tmp_path / "hs"
    hs.mkdir()
    _write_log(hs / "guard_bash_a.log", [
        {"event": "hallucination_detected", "session_id": "s1", "content": "c1"},
        {"event": "grounding_fail", "text": "from-text"},          # content falls back to text
        {"event": "bleed_detected", "session_id": "s3", "content": ""},  # empty content, no text
        {"event": "retraction", "session_id": "s4", "content": "c4"},
        {"event": "tool_use", "session_id": "s5", "content": "ignored"},  # not a failure event
        {"session_id": "s6", "content": "no event key"},
    ])
    got = L._load_guard_failures(str(hs))
    assert got == [
        {"session_id": "s1", "content": "c1", "event": "hallucination_detected"},
        {"session_id": "", "content": "from-text", "event": "grounding_fail"},
        {"session_id": "s3", "content": "", "event": "bleed_detected"},
        {"session_id": "s4", "content": "c4", "event": "retraction"},
    ]


def test_load_guard_failures_skips_blank_and_malformed_lines(tmp_path):
    hs = tmp_path / "hs"
    hs.mkdir()
    _write_log(hs / "guard_bash_a.log", [
        "",
        "   ",
        "this is not json",
        "{broken",
        {"event": "retraction", "session_id": "survivor", "content": "c"},
    ])
    got = L._load_guard_failures(str(hs))
    assert [f["session_id"] for f in got] == ["survivor"]


def test_load_guard_failures_non_dict_json_line_aborts_rest_of_file(tmp_path):
    # BUG: a syntactically valid but non-object JSON line (a bare number, string or
    # array) makes rec.get() raise AttributeError. That escapes the inner
    # JSONDecodeError handler and is caught by the per-FILE `except Exception`,
    # so every remaining line in that file is discarded.
    hs = tmp_path / "hs"
    hs.mkdir()
    _write_log(hs / "guard_bash_a.log", [
        {"event": "retraction", "session_id": "before", "content": "c"},
        "12345",
        {"event": "retraction", "session_id": "after", "content": "c"},
    ])
    _write_log(hs / "guard_bash_b.log", [
        {"event": "retraction", "session_id": "next-file-ok", "content": "c"},
    ])
    got = L._load_guard_failures(str(hs))
    assert [f["session_id"] for f in got] == ["before", "next-file-ok"]


def test_load_guard_failures_reads_only_the_last_20_files_by_name(tmp_path):
    hs = tmp_path / "hs"
    hs.mkdir()
    for i in range(25):
        _write_log(hs / f"guard_bash_{i:03d}.log",
                   [{"event": "retraction", "session_id": f"s{i:03d}", "content": "c"}])
    got = L._load_guard_failures(str(hs))
    assert len(got) == 20
    # sorted() is lexicographic on the path, and the newest-by-name 20 win
    assert got[0]["session_id"] == "s005"
    assert got[-1]["session_id"] == "s024"


def test_load_guard_failures_tolerates_undecodable_bytes(tmp_path):
    hs = tmp_path / "hs"
    hs.mkdir()
    payload = json.dumps({"event": "retraction", "session_id": "s1", "content": "c"})
    (hs / "guard_bash_a.log").write_bytes(b"\xff\xfe not json\n" + payload.encode() + b"\n")
    got = L._load_guard_failures(str(hs))
    assert [f["session_id"] for f in got] == ["s1"]


# ======================================================================================
# live_evo._word_overlap
# ======================================================================================

def test_word_overlap_threshold_is_six_distinct_words():
    assert L._word_overlap("a b c d e", "a b c d e") is False       # only 5
    assert L._word_overlap("a b c d e f", "a b c d e f") is True    # exactly 6


def test_word_overlap_is_case_insensitive():
    assert L._word_overlap("Alpha Beta Gamma Delta Eps Zeta",
                           "alpha BETA gamma DELTA eps ZETA") is True


def test_word_overlap_counts_distinct_words_not_occurrences():
    assert L._word_overlap("a a a a a a a a", "a a a a a a a a") is False


def test_word_overlap_empty_strings():
    assert L._word_overlap("", "") is False
    assert L._word_overlap("", "a b c d e f g") is False


def test_word_overlap_custom_n():
    assert L._word_overlap("a b", "a b", n=2) is True
    assert L._word_overlap("a b", "a b", n=3) is False
    # n<=0 makes everything a match, including two empty strings
    assert L._word_overlap("", "", n=0) is True


def test_word_overlap_splits_on_whitespace_only_punctuation_sticks():
    # "foo," and "foo" are different tokens — there is no punctuation stripping.
    assert L._word_overlap("a, b, c, d, e, f", "a b c d e f") is False


# ======================================================================================
# live_evo._find_correlated_entries
# ======================================================================================

def _conn(db_path):
    return sqlite3.connect(db_path)


def test_find_correlated_no_failures_short_circuits(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "x", 0.5, _ago(1), "s1")])
    conn = _conn(db)
    try:
        assert L._find_correlated_entries(conn, []) == []
    finally:
        conn.close()


def test_find_correlated_missing_table_returns_empty(tmp_path):
    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()
    conn = _conn(str(db))
    try:
        got = L._find_correlated_entries(
            conn, [{"session_id": "s1", "content": "c", "event": "retraction"}])
        assert got == []
    finally:
        conn.close()


def test_find_correlated_matches_by_session_id(tmp_path):
    # NOTE the positional column order the function relies on:
    # SELECT id, content, importance, session_id  -> row[0..3]
    db = _make_db(tmp_path / "m.db", [
        (1, "totally unrelated text", 0.8, _ago(1), "s1"),
        (2, "totally unrelated text", 0.6, _ago(1), "s2"),
    ])
    conn = _conn(db)
    try:
        got = L._find_correlated_entries(
            conn, [{"session_id": "s1", "content": "", "event": "bleed_detected"}])
    finally:
        conn.close()
    assert got == [(1, 0.8, "bleed_detected")]


def test_find_correlated_ignores_empty_session_ids_on_both_sides(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "x", 0.8, _ago(1), None)])
    conn = _conn(db)
    try:
        got = L._find_correlated_entries(
            conn, [{"session_id": "", "content": "", "event": "retraction"}])
    finally:
        conn.close()
    assert got == []


def test_find_correlated_falls_back_to_word_overlap(tmp_path):
    db = _make_db(tmp_path / "m.db", [
        (1, "alpha beta gamma delta epsilon zeta plus more", 0.5, _ago(1), "other"),
        (2, "alpha beta gamma delta epsilon only five", 0.5, _ago(1), "other"),
    ])
    conn = _conn(db)
    try:
        got = L._find_correlated_entries(conn, [{
            "session_id": "nomatch",
            "content": "alpha beta gamma delta epsilon zeta tail",
            "event": "grounding_fail",
        }])
    finally:
        conn.close()
    assert got == [(1, 0.5, "grounding_fail")]


def test_find_correlated_session_match_wins_over_later_content_match(tmp_path):
    db = _make_db(tmp_path / "m.db", [
        (1, "alpha beta gamma delta epsilon zeta", 0.5, _ago(1), "s1"),
    ])
    failures = [
        {"session_id": "s1", "content": "nothing alike", "event": "retraction"},
        {"session_id": "zz", "content": "alpha beta gamma delta epsilon zeta",
         "event": "hallucination_detected"},
    ]
    conn = _conn(db)
    try:
        got = L._find_correlated_entries(conn, failures)
    finally:
        conn.close()
    # the session branch runs first and the event recorded is the session one
    assert got == [(1, 0.5, "retraction")]


def test_find_correlated_content_match_ignores_the_20_char_filter(tmp_path):
    # BUG (dead code): `fail_texts` filters failures to content longer than 20 chars,
    # but it is never used. The content-overlap loop iterates over ALL failures, so a
    # short failure body that the filter meant to exclude can still trigger a match.
    short = "a b c d e f"           # 11 chars, 6 words
    assert len(short) <= 20
    db = _make_db(tmp_path / "m.db", [(1, "a b c d e f", 0.5, _ago(1), "other")])
    conn = _conn(db)
    try:
        got = L._find_correlated_entries(
            conn, [{"session_id": "zz", "content": short, "event": "retraction"}])
    finally:
        conn.close()
    assert got == [(1, 0.5, "retraction")]


def test_find_correlated_null_content_and_importance_are_coerced(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, None, None, _ago(1), "s1")])
    conn = _conn(db)
    try:
        got = L._find_correlated_entries(
            conn, [{"session_id": "s1", "content": "x", "event": "retraction"}])
    finally:
        conn.close()
    # NULL importance rows are excluded by the WHERE clause, so nothing matches
    assert got == []


def test_find_correlated_zero_importance_row_is_kept(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "x", 0.0, _ago(1), "s1")])
    conn = _conn(db)
    try:
        got = L._find_correlated_entries(
            conn, [{"session_id": "s1", "content": "x", "event": "retraction"}])
    finally:
        conn.close()
    assert got == [(1, 0.0, "retraction")]


# ======================================================================================
# live_evo.adapt
# ======================================================================================

def _hookdir(tmp_path, failures, name="guard_bash_a.log") -> str:
    hs = tmp_path / "hs"
    hs.mkdir(exist_ok=True)
    _write_log(hs / name, failures)
    return str(hs)


def test_adapt_missing_db_returns_two_key_error_stub(tmp_path):
    res = L.adapt(str(tmp_path / "nope.db"), str(tmp_path))
    assert res == {
        "error": f"db not found: {tmp_path / 'nope.db'}",
        "n_failures": 0,
        "n_correlated": 0,
    }
    # unlike every other return path, this one has no n_penalized and no dry_run key
    assert "n_penalized" not in res
    assert "dry_run" not in res


def test_adapt_no_failures_returns_early_without_touching_db(tmp_path, _isolate_adaptation_log):
    db = _make_db(tmp_path / "m.db", [(1, "x", 0.8, _ago(1), "s1")])
    res = L.adapt(db, str(tmp_path / "no-such-hook-dir"), dry_run=False)
    assert res == {"n_failures": 0, "n_correlated": 0, "n_penalized": 0, "dry_run": False}
    assert _read_importance(db)[1] == 0.8
    # the early return also skips the adaptation-log append
    assert not _isolate_adaptation_log.exists()


def test_adapt_applies_penalty_and_returns_counts(tmp_path):
    db = _make_db(tmp_path / "m.db", [
        (1, "hit", 0.8, _ago(1), "s1"),
        (2, "miss", 0.6, _ago(1), "s2"),
    ])
    hs = _hookdir(tmp_path, [{"event": "retraction", "session_id": "s1", "content": ""}])
    res = L.adapt(db, hs)
    assert res == {"n_failures": 1, "n_correlated": 1, "n_penalized": 1, "dry_run": False}
    imp = _read_importance(db)
    assert imp[1] == pytest.approx(0.8 * 0.85)
    assert imp[2] == 0.6


def test_adapt_dry_run_reports_but_does_not_write(tmp_path, _isolate_adaptation_log):
    db = _make_db(tmp_path / "m.db", [(1, "hit", 0.8, _ago(1), "s1")])
    hs = _hookdir(tmp_path, [{"event": "retraction", "session_id": "s1", "content": ""}])
    res = L.adapt(db, hs, dry_run=True)
    assert res["n_penalized"] == 1 and res["dry_run"] is True
    assert _read_importance(db)[1] == 0.8
    assert not _isolate_adaptation_log.exists()


def test_adapt_appends_a_differently_named_record_to_the_log(tmp_path, _isolate_adaptation_log):
    db = _make_db(tmp_path / "m.db", [(1, "hit", 0.8, _ago(1), "s1")])
    hs = _hookdir(tmp_path, [{"event": "retraction", "session_id": "s1", "content": ""}])
    L.adapt(db, hs)
    L.adapt(db, hs)
    lines = _isolate_adaptation_log.read_text().strip().splitlines()
    assert len(lines) == 2  # append-only, one JSON object per run
    rec = json.loads(lines[0])
    # the persisted record uses n_failures_loaded, while adapt() returns n_failures
    assert set(rec) == {"run_at", "n_failures_loaded", "n_correlated", "n_penalized", "dry_run"}
    assert rec["n_failures_loaded"] == 1
    assert rec["dry_run"] is False
    datetime.fromisoformat(rec["run_at"])  # tz-aware ISO timestamp


def test_adapt_zero_importance_entry_is_raised_to_the_floor(tmp_path):
    # BUG: same floor-not-clamp problem as decay — "penalising" an entry whose
    # importance is below the floor RAISES it, and counts as a penalty.
    db = _make_db(tmp_path / "m.db", [
        (1, "zero", 0.0, _ago(1), "s1"),
        (2, "tiny", 0.001, _ago(1), "s1"),
        (3, "atfloor", 0.05, _ago(1), "s1"),
    ])
    hs = _hookdir(tmp_path, [{"event": "retraction", "session_id": "s1", "content": ""}])
    res = L.adapt(db, hs)
    assert res["n_correlated"] == 3
    assert res["n_penalized"] == 2  # id=3 is already at the floor, no delta
    assert _read_importance(db) == {1: 0.05, 2: 0.05, 3: 0.05}


def test_adapt_penalty_zero_correlates_but_penalizes_nothing(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "hit", 0.8, _ago(1), "s1")])
    hs = _hookdir(tmp_path, [{"event": "retraction", "session_id": "s1", "content": ""}])
    res = L.adapt(db, hs, penalty=0.0)
    assert res["n_correlated"] == 1 and res["n_penalized"] == 0
    assert _read_importance(db)[1] == 0.8


def test_adapt_penalty_one_drops_to_the_floor(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "hit", 0.8, _ago(1), "s1")])
    hs = _hookdir(tmp_path, [{"event": "retraction", "session_id": "s1", "content": ""}])
    L.adapt(db, hs, penalty=1.0, importance_floor=0.02)
    assert _read_importance(db)[1] == 0.02


def test_adapt_is_multiplicative_across_runs(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "hit", 0.8, _ago(1), "s1")])
    hs = _hookdir(tmp_path, [{"event": "retraction", "session_id": "s1", "content": ""}])
    L.adapt(db, hs)
    L.adapt(db, hs)
    assert _read_importance(db)[1] == pytest.approx(0.8 * 0.85 * 0.85)


def test_adapt_counts_failures_not_correlations(tmp_path):
    db = _make_db(tmp_path / "m.db", [(1, "hit", 0.8, _ago(1), "s1")])
    hs = _hookdir(tmp_path, [
        {"event": "retraction", "session_id": "s1", "content": ""},
        {"event": "bleed_detected", "session_id": "zzz", "content": ""},
        {"event": "grounding_fail", "session_id": "yyy", "content": ""},
    ])
    res = L.adapt(db, hs, dry_run=True)
    assert res["n_failures"] == 3
    assert res["n_correlated"] == 1


def test_adapt_missing_table_still_logs_a_zero_run(tmp_path, _isolate_adaptation_log):
    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()
    hs = _hookdir(tmp_path, [{"event": "retraction", "session_id": "s1", "content": ""}])
    res = L.adapt(str(db), hs)
    assert res == {"n_failures": 1, "n_correlated": 0, "n_penalized": 0, "dry_run": False}
    assert json.loads(_isolate_adaptation_log.read_text().strip())["n_correlated"] == 0


# ======================================================================================
# live_evo.main
# ======================================================================================

def test_live_evo_main_prints_summary(tmp_path, monkeypatch):
    db = _make_db(tmp_path / "m.db", [(1, "hit", 0.8, _ago(1), "s1")])
    hs = _hookdir(tmp_path, [{"event": "retraction", "session_id": "s1", "content": ""}])
    monkeypatch.setattr(sys, "argv",
                        ["live_evo", "--db", db, "--hook-state", hs, "--dry-run"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        L.main()
    assert buf.getvalue() == "[live_evo] failures=1 correlated=1 penalized=1 dry_run=True\n"
    assert _read_importance(db)[1] == 0.8


def test_live_evo_main_on_missing_db_prints_none_for_penalized(tmp_path, monkeypatch):
    # the missing-db stub has no n_penalized key, so .get() yields None
    monkeypatch.setattr(sys, "argv", ["live_evo", "--db", str(tmp_path / "nope.db")])
    buf = io.StringIO()
    with redirect_stdout(buf):
        L.main()
    assert buf.getvalue() == "[live_evo] failures=0 correlated=0 penalized=None dry_run=False\n"


def test_apply_decay_dry_run_does_not_alter_the_schema(tmp_path):
    """A dry run that adds a column has already written to the database."""
    db = _make_db(tmp_path / "m.db", [(1, "a", 0.9, _ago(90), "s1")])
    D.apply_decay(db, dry_run=True)
    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(working_memory)")}
    conn.close()
    assert "base_importance" not in cols

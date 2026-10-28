"""scripts/check_policy_flags.py: the learned-flag registry check, against fixture registries in tmp_path.

Mutation log (each applied to scripts/check_policy_flags.py; the named tests went red; then reverted):
- expiry "<" -> "<=" (review_by == today fails): test_review_by_today_passes_and_yesterday_fails
- expiry check deleted: test_an_expired_flag_fails_with_its_exact_name, test_review_by_today_passes_and_yesterday_fails
- reason check deleted: test_a_flag_without_a_reason_fails_by_name
- missing registry treated as empty: test_a_missing_registry_is_an_error_not_a_pass
- duplicate check deleted: test_a_flag_registered_twice_fails
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_policy_flags.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_policy_flags_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cpf = _load()
TODAY = "2026-10-18"


def _registry(tmp_path, body):
    path = tmp_path / "policy_flags.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _run(path, capsys, today=TODAY):
    rc = cpf.main(["--registry", str(path), "--today", today])
    return rc, capsys.readouterr()


def _flag(name="learned_router", reason="Stage-0 gate failed; not re-tuned", review_by="2026-11-17"):
    return f'[[flag]]\nname = "{name}"\nreason = "{reason}"\nreview_by = {review_by}\n'


def test_an_empty_registry_is_accepted(tmp_path, capsys):
    rc, out = _run(_registry(tmp_path, "# nothing learned ships default-off yet\n"), capsys)
    assert rc == 0
    assert out.out == "check_policy_flags: 0 learned flag(s) registered, none expired\n"


def test_a_flag_with_a_future_review_date_passes(tmp_path, capsys):
    rc, out = _run(_registry(tmp_path, _flag()), capsys)
    assert (rc, out.err) == (0, "")
    assert "1 learned flag(s) registered" in out.out


def test_an_expired_flag_fails_with_its_exact_name(tmp_path, capsys):
    rc, out = _run(_registry(tmp_path, _flag(review_by="2026-10-01") + _flag("other", review_by="2027-01-01")), capsys)
    assert rc == 1
    assert out.err == ("check_policy_flags: learned_router: review_by 2026-10-01 has expired (today 2026-10-18); "
                       "re-review the flag and move the date, or remove it\n")


def test_review_by_today_passes_and_yesterday_fails(tmp_path, capsys):
    assert _run(_registry(tmp_path, _flag(review_by="2026-10-18")), capsys)[0] == 0
    rc, out = _run(_registry(tmp_path, _flag(review_by="2026-10-17")), capsys)
    assert rc == 1
    assert "learned_router: review_by 2026-10-17 has expired" in out.err


def test_a_flag_without_a_reason_fails_by_name(tmp_path, capsys):
    rc, out = _run(_registry(tmp_path, '[[flag]]\nname = "learned_router"\nreview_by = 2026-11-17\n'), capsys)
    assert rc == 1
    assert out.err == ("check_policy_flags: learned_router: missing 'reason' "
                       "(why is this learned feature default-off?)\n")
    # a blank reason is the same as none
    assert _run(_registry(tmp_path, _flag(reason="  ")), capsys)[0] == 1


def test_a_flag_without_review_by_fails_by_name(tmp_path, capsys):
    rc, out = _run(_registry(tmp_path, '[[flag]]\nname = "learned_router"\nreason = "r"\n'), capsys)
    assert rc == 1
    assert out.err == "check_policy_flags: learned_router: missing 'review_by'\n"


@pytest.mark.parametrize("value, shown", [('"soon"', "'soon'"), ("2026-11-17T09:00:00", "datetime.datetime(2026, 11, 17, 9, 0)"),
                                          ("20261117", "20261117")])
def test_review_by_must_be_a_calendar_date(tmp_path, capsys, value, shown):
    rc, out = _run(_registry(tmp_path, _flag(review_by=value)), capsys)
    assert rc == 1
    assert out.err == f"check_policy_flags: learned_router: 'review_by' must be a date (YYYY-MM-DD), got {shown}\n"


def test_an_iso_date_string_is_also_a_date(tmp_path, capsys):
    assert _run(_registry(tmp_path, _flag(review_by='"2026-11-17"')), capsys)[0] == 0


def test_a_flag_registered_twice_fails(tmp_path, capsys):
    rc, out = _run(_registry(tmp_path, _flag() + _flag()), capsys)
    assert rc == 1
    assert out.err == "check_policy_flags: learned_router: registered twice\n"


def test_a_nameless_entry_fails_and_all_problems_are_listed(tmp_path, capsys):
    body = '[[flag]]\nreason = "r"\nreview_by = 2026-11-17\n' + _flag("a", review_by="2026-01-01") \
        + _flag("b", reason="")
    rc, out = _run(_registry(tmp_path, body), capsys)
    assert rc == 1
    assert out.err.splitlines() == [
        "check_policy_flags: flag #1: missing 'name'",
        "check_policy_flags: a: review_by 2026-01-01 has expired (today 2026-10-18); "
        "re-review the flag and move the date, or remove it",
        "check_policy_flags: b: missing 'reason' (why is this learned feature default-off?)",
    ]


def test_a_missing_registry_is_an_error_not_a_pass(tmp_path, capsys):
    rc, out = _run(tmp_path / "does_not_exist.toml", capsys)
    assert rc == 2
    assert "cannot read registry" in out.err
    assert out.out == ""


def test_an_unparseable_registry_is_an_error(tmp_path, capsys):
    rc, out = _run(_registry(tmp_path, "[[flag\n"), capsys)
    assert rc == 2
    assert "cannot read registry" in out.err


def test_check_is_pure_over_a_given_date():
    flags = [{"name": "x", "reason": "r", "review_by": dt.date(2026, 10, 18)}]
    assert cpf.check(flags, dt.date(2026, 10, 18)) == []
    assert cpf.check(flags, dt.date(2026, 10, 19)) == [
        "x: review_by 2026-10-18 has expired (today 2026-10-19); re-review the flag and move the date, or remove it"]

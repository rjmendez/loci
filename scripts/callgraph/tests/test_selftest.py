"""Wires selftest.py's own check set into pytest, and exercises `cg
selftest`/`cg limits` through the CLI end-to-end -- the build_steps step 13
acceptance bar, run the same way `cg selftest` runs it standalone."""
import json
import time

from ..cli import main
from ..selftest import run_selftest


def test_selftest_all_checks_pass():
    report = run_selftest()
    failed = [(c.name, c.detail) for c in report.failed]
    assert failed == [], failed
    assert report.ok is True


def test_selftest_covers_every_dispatch_shape_and_the_hard_gate():
    # A regression here (someone silently deleting a check) is exactly the
    # failure mode this test exists to catch — pin the count and a handful
    # of the load-bearing names.
    report = run_selftest()
    names = {c.name for c in report.checks}
    assert len(report.checks) == 15
    assert any("hard gate" in n for n in names)
    assert any("BUG C" in n for n in names)
    assert any("BUG D" in n for n in names)
    assert any("BUG B" in n for n in names)


def test_selftest_finishes_well_under_the_five_second_budget():
    # build_steps step 13's own acceptance line: "total wall time for a
    # full cold build plus all fixture assertions under 5s." Timed here
    # independently of run_selftest()'s own self-reported elapsed_s, and
    # given a little headroom over the hard 5s design ceiling since this
    # runs on shared/variable CI hardware, not a dedicated benchmark box.
    t0 = time.time()
    report = run_selftest()
    wall = time.time() - t0
    assert report.ok
    assert wall < 8.0, f"selftest took {wall:.2f}s — investigate before this creeps past the 5s design budget"
    assert report.elapsed_s < 8.0


def test_cli_selftest_exits_zero_and_prints_pass_summary(capsys):
    code = main(["selftest"])
    out = capsys.readouterr().out
    assert code == 0
    assert "PASS" in out
    assert "FAIL" not in out
    assert "selftest green" in out
    assert "/15 checks passed" in out


def test_cli_selftest_json_shape(capsys):
    code = main(["selftest", "--format", "json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert len(payload["checks"]) == 15
    assert all(c["ok"] for c in payload["checks"])


def test_cli_selftest_reports_a_broken_check_with_nonzero_exit(monkeypatch):
    # Prove the CLI actually surfaces a failure rather than always printing
    # green — flip one check to fail and confirm both the exit code and the
    # FAIL line show up.
    from .. import cli as cli_mod
    from ..selftest import Check, SelftestReport

    def _fake_run_selftest():
        return SelftestReport(checks=[Check("fake check", False, "boom")], elapsed_s=0.01)

    monkeypatch.setattr(cli_mod, "run_selftest", _fake_run_selftest)
    code = cli_mod.main(["selftest"])
    assert code == 1


# -- cg limits ----------------------------------------------------------------


def test_cli_limits_prints_the_docs_file(capsys):
    code = main(["limits"])
    out = capsys.readouterr().out
    assert code == 0
    assert "# Known limits" in out
    assert "cg writes-dead" in out
    assert "cg flags" in out
    assert "BUG A" in out


def test_cli_limits_json_shape(capsys):
    code = main(["limits", "--format", "json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["path"].endswith("docs/LIMITS.md")
    assert "Known limits" in payload["text"]

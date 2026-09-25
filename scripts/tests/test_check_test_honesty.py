"""scripts/check_test_honesty.py and check_test_weakening.py reject the hollow-test shapes.

Every rule is tested with a violating snippet (exact rule, test and line reported)
and a compliant twin built from the same fixture (nothing reported), so neither
"flag everything" nor "flag nothing" passes.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_test_honesty as honesty  # noqa: E402
import check_test_weakening as weakening  # noqa: E402


def _cfg(**kw) -> honesty.Config:
    return honesty.Config(**kw)


def _found(tmp_path: Path, src: str, name: str = "test_sample.py", cfg=None, **files):
    for fname, body in files.items():
        (tmp_path / f"{fname}.py").write_text(textwrap.dedent(body))
    path = tmp_path / name
    path.write_text(textwrap.dedent(src).lstrip("\n"))
    return sorted((v.rule, v.test, v.line) for v in honesty.check_file(path, cfg or _cfg()))


# ------------------------------------------------------------------ or-true / assert-true

def test_or_true_is_flagged(tmp_path):
    src = """
    def test_a():
        assert compute() == 3 or True
    """
    assert _found(tmp_path, src) == [("or-true", "test_a", 2)]


def test_or_default_inside_an_assert_is_not_or_true(tmp_path):
    src = """
    import os
    def test_a():
        assert compute() == (os.environ.get("X") or "fallback")
    """
    assert _found(tmp_path, src) == []


def test_top_level_or_with_truthy_literal_is_flagged(tmp_path):
    src = """
    def test_a():
        assert compute() or "anything"
    """
    assert _found(tmp_path, src) == [("or-true", "test_a", 2)]


def test_or_true_nested_inside_a_larger_assert_is_flagged(tmp_path):
    # `a and (b or True)` cannot fail on b: the nested literal True is found anywhere.
    src = """
    def test_a():
        assert compute() == 3 and (other() or True)
    def test_b():
        assert compute() == 3 and (other() or fallback())
    """
    assert _found(tmp_path, src) == [("or-true", "test_a", 2)]


def test_assert_true_constant_is_flagged_but_assert_false_is_a_failure_not_a_tautology(tmp_path):
    src = """
    import unittest
    def test_a():
        assert True
    def test_b():
        if compute():
            assert False, "must not compute"
    class T(unittest.TestCase):
        def test_c(self):
            self.assertTrue(True)
        def test_d(self):
            self.assertTrue(compute())
    """
    assert _found(tmp_path, src) == [("assert-true", "T::test_c", 9), ("assert-true", "test_a", 3)]


# ------------------------------------------------------------------------- broad-raises

def test_broad_raises_without_match_is_flagged_specific_or_matched_is_not(tmp_path):
    src = """
    import pytest
    def test_bare():
        with pytest.raises(Exception):
            f()
    def test_value_error():
        with pytest.raises(ValueError):
            f()
    def test_tuple():
        with pytest.raises((KeyError, OSError)):
            f()
    def test_matched():
        with pytest.raises(ValueError, match="bad port"):
            f()
    def test_specific():
        with pytest.raises(FileNotFoundError):
            f()
    """
    assert _found(tmp_path, src) == [
        ("broad-raises", "test_bare", 3),
        ("broad-raises", "test_tuple", 9),
        ("broad-raises", "test_value_error", 6),
    ]


def test_unittest_assert_raises_broad_is_flagged_regex_form_is_not(tmp_path):
    src = """
    import unittest
    class T(unittest.TestCase):
        def test_broad(self):
            with self.assertRaises(TypeError):
                f()
        def test_regex(self):
            with self.assertRaisesRegex(TypeError, "int"):
                f()
    """
    assert _found(tmp_path, src) == [("broad-raises", "T::test_broad", 4)]


def test_a_match_that_matches_anything_is_not_a_match(tmp_path):
    src = """
    import pytest
    import unittest
    def test_empty():
        with pytest.raises(Exception, match=""):
            f()
    def test_dotstar():
        with pytest.raises(ValueError, match=".*"):
            f()
    def test_real():
        with pytest.raises(ValueError, match="bad port"):
            f()
    class T(unittest.TestCase):
        def test_empty_regex(self):
            with self.assertRaisesRegex(TypeError, ""):
                f()
        def test_regex(self):
            with self.assertRaisesRegex(TypeError, "int"):
                f()
    """
    assert _found(tmp_path, src) == [
        ("broad-raises", "T::test_empty_regex", 14),
        ("broad-raises", "test_dotstar", 7),
        ("broad-raises", "test_empty", 4),
    ]


@pytest.mark.parametrize("exc", ["RuntimeError", "OSError", "AttributeError", "IndexError",
                                 "LookupError", "AssertionError"])
def test_other_broad_builtins_need_a_match_too(tmp_path, exc):
    src = f"""
    import pytest
    def test_bare():
        with pytest.raises({exc}):
            f()
    def test_matched():
        with pytest.raises({exc}, match="the reason"):
            f()
    """
    assert _found(tmp_path, src) == [("broad-raises", "test_bare", 3)]


# --------------------------------------------------------------------- swallowed-assert

def test_assert_in_try_with_swallowing_handler_is_flagged(tmp_path):
    src = """
    def test_swallow():
        try:
            assert f() == 1
        except AssertionError:
            pass
    def test_broad():
        try:
            assert f() == 1
        except Exception:
            print("meh")
    def test_bare():
        try:
            assert f() == 1
        except:
            pass
    """
    assert _found(tmp_path, src) == [
        ("swallowed-assert", "test_bare", 14),
        ("swallowed-assert", "test_broad", 9),
        ("swallowed-assert", "test_swallow", 4),
    ]


def test_reraising_recording_or_narrow_handlers_are_not_swallowing(tmp_path):
    src = """
    import pytest
    def test_reraise():
        try:
            assert f() == 1
        except AssertionError:
            cleanup()
            raise
    def test_fail():
        try:
            assert f() == 1
        except Exception as exc:
            pytest.fail(str(exc))
    def test_thread_records():
        errors = []
        def worker():
            try:
                assert f() == 1
            except Exception as exc:
                errors.append(exc)
        worker()
        assert errors == []
    def test_narrow():
        try:
            assert f() == 1
        except KeyError:
            pass
    """
    assert _found(tmp_path, src) == []


def test_contextlib_suppress_around_an_assert_is_flagged(tmp_path):
    src = """
    import contextlib
    def test_a():
        with contextlib.suppress(AssertionError):
            assert f() == 1
    """
    assert _found(tmp_path, src) == [("swallowed-assert", "test_a", 3)]


# ------------------------------------------------------------------------ no-assertions

def test_test_without_any_assertion_is_flagged(tmp_path):
    src = """
    def test_nothing(monkeypatch):
        monkeypatch.setenv("X", "1")
        run()
    def test_stub_raises_but_test_asserts_nothing():
        def boom():
            raise RuntimeError("x")
        run(boom)
    """
    assert _found(tmp_path, src) == [
        ("no-assertions", "test_nothing", 1),
        ("no-assertions", "test_stub_raises_but_test_asserts_nothing", 4),
    ]


def test_assertions_through_helpers_mocks_and_raises_count(tmp_path):
    src = """
    import pytest
    from .support import check_shape
    def _expect(x):
        assert x == 1
    def _indirect(x):
        _expect(x)
    def test_local_helper():
        _indirect(run())
    def test_sibling_module_helper():
        check_shape(run())
    def test_mock():
        m = run()
        m.assert_called_once_with(1)
    def test_raises():
        with pytest.raises(OSError, match="x"):
            run()
    def test_explicit_assertion_error():
        if run() != 1:
            raise AssertionError("wrong")
    def test_config_helper():
        verify_everything(run())
    """
    support = """
    def check_shape(out):
        assert out == {"a": 1}
    """
    cfg = _cfg(helpers={"verify_everything"})
    assert _found(tmp_path, src, cfg=cfg, support=support) == []


def test_no_assertions_applies_only_to_test_modules(tmp_path):
    src = """
    def test_helper_named_like_a_test():
        return 1
    """
    assert _found(tmp_path, src, name="conftest.py") == []
    assert _found(tmp_path, src, name="test_mod.py") == [
        ("no-assertions", "test_helper_named_like_a_test", 1)]


# --------------------------------------------------------------------- xfail-not-strict

def test_xfail_must_be_strict(tmp_path):
    src = """
    import pytest
    @pytest.mark.xfail(reason="bug 12")
    def test_loose():
        assert f() == 1
    @pytest.mark.xfail
    def test_bare():
        assert f() == 1
    @pytest.mark.xfail(strict=False, reason="bug 12")
    def test_explicit_false():
        assert f() == 1
    @pytest.mark.xfail(strict=True, reason="bug 12")
    def test_strict():
        assert f() == 1
    @pytest.mark.parametrize("x", [1, pytest.param(2, marks=pytest.mark.xfail)])
    def test_param(x):
        assert f(x) == x
    def test_imperative():
        pytest.xfail("later")
        assert f() == 1
    """
    assert _found(tmp_path, src) == [
        ("xfail-not-strict", "test_bare", 5),
        ("xfail-not-strict", "test_explicit_false", 8),
        ("xfail-not-strict", "test_imperative", 18),
        ("xfail-not-strict", "test_loose", 2),
        ("xfail-not-strict", "test_param", 14),
    ]


# -------------------------------------------------------------------------- env-skipif

_ENV_SRC = """
import os
import sys
import unittest
import pytest

_ROOT = os.environ.get("SNAPSHOT_ROOT", "")
_DIR = _ROOT + "/snap" if _ROOT else None

def _live():
    return os.getenv("LOCI_TESTS_LIVE") == "1"

@pytest.mark.skipif(not _DIR, reason="needs snapshot")
def test_indirect():
    assert f() == 1

@pytest.mark.skipif("CI_TOKEN" not in os.environ, reason="token")
def test_membership():
    assert f() == 1

@pytest.mark.skipif(not _live(), reason="live only")
def test_opt_in_via_function():
    assert f() == 1

@pytest.mark.skipif(sys.platform == "win32", reason="posix")
def test_platform():
    assert f() == 1

class T(unittest.TestCase):
    @unittest.skipUnless(os.environ["OTHER"], "other")
    def test_unittest(self):
        self.assertEqual(f(), 1)

def test_imperative():
    if not os.environ.get("SNAPSHOT_ROOT"):
        pytest.skip("no snapshot")
    assert f() == 1
"""


def test_skip_gated_on_undocumented_env_var_is_flagged(tmp_path):
    cfg = _cfg(opt_ins={"LOCI_TESTS_LIVE": "documented"})
    assert _found(tmp_path, _ENV_SRC, cfg=cfg) == [
        ("env-skipif", "T::test_unittest", 29),
        ("env-skipif", "test_imperative", 34),
        ("env-skipif", "test_indirect", 12),
        ("env-skipif", "test_membership", 16),
    ]


def test_documented_opt_ins_are_accepted(tmp_path):
    cfg = _cfg(opt_ins={v: "documented" for v in
                        ("LOCI_TESTS_LIVE", "SNAPSHOT_ROOT", "CI_TOKEN", "OTHER")})
    assert _found(tmp_path, _ENV_SRC, cfg=cfg) == []


def test_undocumented_opt_in_is_not_accepted(tmp_path):
    # Without LOCI_TESTS_LIVE in the opt-ins, the function-mediated skip is flagged too.
    assert ("env-skipif", "test_opt_in_via_function", 20) in _found(tmp_path, _ENV_SRC)


# ------------------------------------------------------------- allowlist and main()

def _repo(tmp_path: Path, monkeypatch, agents: str = "LOCI_TESTS_LIVE") -> Path:
    tmp_path = tmp_path.resolve()
    (tmp_path / "AGENTS.md").write_text(agents)
    monkeypatch.setattr(honesty, "REPO", tmp_path)
    monkeypatch.setattr(honesty, "AGENTS_DOC", tmp_path / "AGENTS.md")
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "test_one.py").write_text("def test_a():\n    assert f() or True\n")
    (tmp_path / "t" / "test_two.py").write_text("def test_b():\n    assert f() == 1\n")
    monkeypatch.setattr(honesty, "discover", lambda: sorted((tmp_path / "t").glob("test_*.py")))
    return tmp_path


def _allow(path: Path, entries: str, opt_ins: str = 'LOCI_TESTS_LIVE = "live smoke"') -> Path:
    path.write_text(f"[live_smoke_opt_ins]\n{opt_ins}\n\n[[allow]]\n{entries}\n")
    return path


def test_main_fails_on_a_new_violation_and_passes_once_allowlisted(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path, monkeypatch)
    empty = _allow(root / "allow.toml", 'file = "t/test_two.py"\nentries = []')
    assert honesty.main(["--allowlist", str(empty)]) == 1
    out = capsys.readouterr().out
    assert "t/test_one.py:2: or-true: test_a:" in out
    assert "1 violations, 0 allowlisted, 1 new, 0 stale" in out

    ok = _allow(root / "ok.toml", 'file = "t/test_one.py"\nentries = [\n'
                '  { test = "test_a", rule = "or-true", reason = "tracked in issue 42" },\n]')
    assert honesty.main(["--allowlist", str(ok), "--fail-stale"]) == 0
    assert "1 violations, 1 allowlisted, 0 new, 0 stale" in capsys.readouterr().out


def test_stale_entry_warns_and_fails_only_with_fail_stale(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path, monkeypatch)
    allow = _allow(root / "allow.toml", 'file = "t/test_one.py"\nentries = [\n'
                   '  { test = "test_a", rule = "or-true", reason = "tracked in issue 42" },\n'
                   '  { test = "test_gone", rule = "no-assertions", reason = "tracked in issue 43" },\n]')
    assert honesty.main(["--allowlist", str(allow)]) == 0
    assert "stale allowlist entry t/test_one.py::test_gone [no-assertions]" in capsys.readouterr().out
    assert honesty.main(["--allowlist", str(allow), "--fail-stale"]) == 1


def test_entries_need_a_reason_and_a_known_rule(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path, monkeypatch)
    allow = _allow(root / "allow.toml", 'file = "t/test_one.py"\nentries = [\n'
                   '  { test = "test_a", rule = "or-true", reason = "" },\n'
                   '  { test = "test_a", rule = "made-up", reason = "tracked in issue 42" },\n]')
    assert honesty.main(["--allowlist", str(allow)]) == 1
    out = capsys.readouterr().out
    assert "t/test_one.py::test_a [or-true]: every entry needs a reason" in out
    assert "t/test_one.py::test_a [made-up]: unknown rule" in out


def test_opt_in_must_be_documented_in_agents_md(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path, monkeypatch, agents="nothing here")
    allow = _allow(root / "allow.toml", 'file = "t/test_one.py"\nentries = [\n'
                   '  { test = "test_a", rule = "or-true", reason = "tracked in issue 42" },\n]')
    assert honesty.main(["--allowlist", str(allow)]) == 1
    assert "live-smoke opt-in LOCI_TESTS_LIVE is not documented in AGENTS.md" in capsys.readouterr().out


def test_write_allowlist_keeps_reasons_and_header(tmp_path, monkeypatch):
    root = _repo(tmp_path, monkeypatch)
    allow = _allow(root / "allow.toml", 'file = "t/test_one.py"\nentries = [\n'
                   '  { test = "test_a", rule = "or-true", reason = "tracked in issue 42" },\n'
                   '  { test = "test_gone", rule = "or-true", reason = "tracked in issue 43" },\n]')
    (root / "t" / "test_two.py").write_text("def test_b():\n    run()\n")
    assert honesty.main(["--allowlist", str(allow), "--write-allowlist", "--reason", "new debt, issue 50"]) == 0
    cfg = honesty.load_config(allow)
    assert cfg.opt_ins == {"LOCI_TESTS_LIVE": "live smoke"}
    assert cfg.allow == {
        ("t/test_one.py", "test_a", "or-true"): "tracked in issue 42",
        ("t/test_two.py", "test_b", "no-assertions"): "new debt, issue 50",
    }


def test_the_repository_itself_passes():
    # The shipped allowlist must cover the current debt: CI runs this same check. Stale
    # entries (debt already paid) only warn here, so fixing a test never breaks this one.
    assert honesty.main([]) == 0


# ------------------------------------------------------------------- weakening check

def test_profile_counts_assertions_compares_and_skips_exactly():
    src = textwrap.dedent("""
    import pytest, unittest
    @pytest.mark.skipif(True, reason="x")
    def test_a():
        assert f() == 1 and g() == 2
        assert h() >= 3
    class T(unittest.TestCase):
        def test_b(self):
            self.assertEqual(f(), 1)
            self.assertIn(1, g())
            m.assert_called_once()
    """)
    assert dict(weakening.profile(src)) == {"asserts": 5, "eq": 3, "loose": 2, "skips": 1}


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_commit_that_changes_source_and_loosens_its_test_is_flagged(tmp_path, monkeypatch):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_NOSYSTEM": "1",
           "HOME": str(tmp_path)}

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, env=env, check=True,
                              capture_output=True, text=True).stdout.strip()

    def commit(msg, files):
        for name, body in files.items():
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text(body)
        git("add", "-A")
        git("commit", "-qm", msg)
        return git("rev-parse", "HEAD")

    git("init", "-q")
    commit("base", {"lib.py": "N = 3\n",
                    "tests/test_lib.py": "def test_n():\n    assert N == 3\n    assert M == 1\n"})
    loosened = commit("change N", {"lib.py": "N = 4\n",
                                   "tests/test_lib.py": "def test_n():\n    assert N >= 3\n    assert M == 1\n"})
    acknowledged = commit("change M\n\nTest-weakened-because: M is now random, see #9",
                          {"lib.py": "N = 5\n", "tests/test_lib.py": "def test_n():\n    assert N >= 3\n"})
    test_only = commit("tidy test", {"tests/test_lib.py": "def test_n():\n    pass\n"})
    strengthened = commit("pin N", {"lib.py": "N = 6\n",
                                    "tests/test_lib.py": "def test_n():\n    assert N == 6\n"})

    monkeypatch.setattr(honesty, "REPO", tmp_path)
    monkeypatch.setattr(weakening, "REPO", tmp_path)
    cfg = honesty.Config()
    assert weakening.check_commit(loosened, cfg) == [
        "tests/test_lib.py: exact compares 2 -> 1 while loose compares 0 -> 1"]
    assert weakening.check_commit(acknowledged, cfg) == []
    assert weakening.check_commit(test_only, cfg) == []
    assert weakening.check_commit(strengthened, cfg) == []
    assert weakening.main(["--range", f"{loosened}^..HEAD", "--strict"]) == 1
    assert weakening.main(["--range", f"{loosened}^..HEAD"]) == 0

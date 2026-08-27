"""Tests for memcheck/code_rules/extended_checks.py — pure stdlib, no mocking.

extended_checks.py is a 1,400-line, 22-rule static checker that had no test
coverage at all, so a dispatch or helper refactor could silently drop rules
without any suite noticing. These tests pin the two structural invariants that
make the module coherent, plus a small set of end-to-end fixtures that exercise
each of the four check arities (tree / tree+name / source / source+name).

Deliberately NOT asserted:

* "every PATTERN_META id has a detector" — PATTERN_META has 58 entries and is a
  taxonomy document; only 24 ids are emitted by a runnable check, and the
  remaining ones record honestly that no static signal exists.
* anything derived from CODE_RULES_COVERAGE.md — coupling a test to markdown
  prose turns a legitimate doc edit into a spurious failure.
"""

import ast
import os
import pathlib
import sys
import tempfile
import unittest

# Resolves whether pytest is launched from mcp/ or from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memcheck.code_rules import extended_checks as ec
from memcheck.code_rules.extended_checks import PATTERN_META, run_extended_checks

# The four dispatch registries, paired with the arity each one is called with.
_REGISTRIES = (
    ("_AST_CHECKS", ec._AST_CHECKS),
    ("_AST_NAME_CHECKS", ec._AST_NAME_CHECKS),
    ("_SOURCE_CHECKS", ec._SOURCE_CHECKS),
    ("_SOURCE_NAME_CHECKS", ec._SOURCE_NAME_CHECKS),
)


def _module_ast() -> ast.Module:
    return ast.parse(pathlib.Path(ec.__file__).read_text(encoding="utf-8"))


def _run_on_source(filename: str, source: str) -> list:
    """Write ``source`` to a temp file named ``filename`` and check it.

    The filename matters: several rules gate on ``_is_test_file`` (which is
    simply ``"test" in filename``), so fixtures that must look like test code
    have to carry 'test' in their name.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / filename
        path.write_text(source, encoding="utf-8")
        return run_extended_checks(path)


def _codes(issues) -> set:
    return {i.code for i in issues}


class TestDispatchRegistryCoverage(unittest.TestCase):
    """Every check_* def is wired into exactly one dispatch registry."""

    def test_every_check_def_is_registered(self):
        defs = {
            node.name
            for node in _module_ast().body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("check_")
        }
        registered = {fn.__name__ for _, reg in _REGISTRIES for fn in reg}

        self.assertEqual(
            defs,
            registered,
            "check_* defs and dispatch registries have drifted apart; "
            f"unregistered={sorted(defs - registered)} "
            f"registered-but-undefined={sorted(registered - defs)}",
        )

    def test_no_check_registered_twice(self):
        names = [fn.__name__ for _, reg in _REGISTRIES for fn in reg]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        self.assertEqual(
            duplicates, [], f"checks registered in more than one registry: {duplicates}"
        )

    def test_registry_sizes(self):
        """Pins the 16 + 1 + 3 + 2 = 22 shape the dispatch loop iterates."""
        sizes = {name: len(reg) for name, reg in _REGISTRIES}
        self.assertEqual(
            sizes,
            {
                "_AST_CHECKS": 16,
                "_AST_NAME_CHECKS": 1,
                "_SOURCE_CHECKS": 3,
                "_SOURCE_NAME_CHECKS": 2,
            },
        )
        self.assertEqual(sum(sizes.values()), 22)

    def test_all_registered_checks_are_callable(self):
        for name, reg in _REGISTRIES:
            for fn in reg:
                self.assertTrue(callable(fn), f"{name} holds a non-callable: {fn!r}")


class TestEmittedCodes(unittest.TestCase):
    """Every code handed to _issue/_src_issue is a real PATTERN_META id."""

    @staticmethod
    def _emitted_codes() -> set:
        codes = set()
        for node in ast.walk(_module_ast()):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("_issue", "_src_issue")
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                codes.add(node.args[0].value)
        return codes

    def test_emitted_codes_are_documented(self):
        undocumented = sorted(self._emitted_codes() - set(PATTERN_META))
        self.assertEqual(
            undocumented,
            [],
            f"codes emitted with no PATTERN_META entry: {undocumented}",
        )

    def test_emitted_code_set_is_pinned(self):
        """A check that stops emitting its code should fail loudly, not quietly.

        Note check_h6_deprecated_api emits AC1 and SB2 in addition to H6, so
        'function name prefix == emitted code' is NOT an invariant here.
        """
        self.assertEqual(
            self._emitted_codes(),
            {
                "AC1", "AC2", "AC4", "H2", "H4", "H5", "H6", "H9",
                "MC1b", "MF2", "MF3", "OG1", "PB1", "SB1", "SB2",
                "SD1", "SD2", "SD3", "SL1", "SL2", "SS3", "SS5",
                "TEC1", "WG1",
            },
        )


class TestRunExtendedChecksFixtures(unittest.TestCase):
    """End-to-end pins, one per dispatch arity."""

    def test_h2_duplicate_metric_literal(self):
        """_AST_CHECKS arm: same literal metric name registered twice."""
        issues = _run_on_source(
            "dup_metric.py",
            "from prometheus_client import Counter\n"
            'A = Counter("jobs_total", "d")\n'
            'B = Counter("jobs_total", "d")\n',
        )
        self.assertEqual(_codes(issues), {"H2"})
        self.assertEqual([i.line for i in issues if i.code == "H2"], [3])

    def test_sd3_and_mc1b_on_labeled_counter(self):
        """_AST_CHECKS arm, two rules on one construct.

        SD3 needs >=3 positional args (or ``labelnames=``) to consider the
        metric labeled; MC1b needs a _GAUGE_WORDS token ('depth') in the name.
        """
        issues = _run_on_source(
            "labeled.py",
            "from prometheus_client import Counter\n"
            'Q = Counter("queue_depth", "doc", ["label"])\n'
            "Q.inc()\n",
        )
        self.assertEqual(_codes(issues), {"SD3", "MC1b"})
        by_code = {i.code: i for i in issues}
        self.assertEqual(by_code["MC1b"].line, 2)
        self.assertEqual(by_code["SD3"].line, 3)
        # MC1b reports the metric name as written, not lowercased for matching.
        self.assertIn("queue_depth", by_code["MC1b"].message)

    def test_h9_async_dunder_init(self):
        """_AST_CHECKS arm: ``async def __init__`` never awaits on construction."""
        issues = _run_on_source(
            "asyncinit.py",
            "class A:\n    async def __init__(self):\n        pass\n",
        )
        self.assertEqual(_codes(issues), {"H9"})
        self.assertEqual([i.line for i in issues], [2])

    def test_h5_hardcoded_float_assert_needs_test_filename(self):
        """_AST_NAME_CHECKS arm: gated on 'test' being in the filename."""
        source = "def test_x():\n    assert compute() == 0.35\n"

        issues = _run_on_source("test_floats.py", source)
        self.assertEqual(_codes(issues), {"H5"})
        self.assertEqual([i.line for i in issues], [2])

        # Same source under a non-test filename: the name-gated rule stays quiet.
        self.assertNotIn("H5", _codes(_run_on_source("floats.py", source)))

    def test_sd2_hardcoded_sql(self):
        """_SOURCE_CHECKS arm: regex over raw text, no AST involvement."""
        issues = _run_on_source(
            "sqlish.py",
            'QUERY = "SELECT id, name FROM findings WHERE investigation_id = ?"\n',
        )
        self.assertIn("SD2", _codes(issues))
        self.assertEqual([i.line for i in issues if i.code == "SD2"], [1])

    def test_clean_source_yields_nothing(self):
        issues = _run_on_source("clean.py", "def add(a, b):\n    return a + b\n")
        self.assertEqual(issues, [])

    def test_results_are_sorted_by_line_col_code(self):
        issues = _run_on_source(
            "labeled.py",
            "from prometheus_client import Counter\n"
            'Q = Counter("queue_depth", "doc", ["label"])\n'
            "Q.inc()\n",
        )
        keys = [(i.line, i.col, i.code) for i in issues]
        self.assertEqual(keys, sorted(keys))


class TestRunExtendedChecksFailSafe(unittest.TestCase):
    """The orchestrator never raises — it degrades to an empty list."""

    def test_syntax_error_returns_empty(self):
        self.assertEqual(_run_on_source("broken.py", "def (:\n"), [])

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run_extended_checks(pathlib.Path(tmp) / "nope.py"), [])

    def test_a_broken_check_does_not_break_the_run(self):
        """One exploding check is logged and skipped; the others still report."""

        def exploding_check(_tree):
            raise RuntimeError("boom")

        original = ec._AST_CHECKS
        ec._AST_CHECKS = (exploding_check,) + original
        try:
            issues = _run_on_source(
                "dup_metric.py",
                "from prometheus_client import Counter\n"
                'A = Counter("jobs_total", "d")\n'
                'B = Counter("jobs_total", "d")\n',
            )
        finally:
            ec._AST_CHECKS = original

        self.assertEqual(_codes(issues), {"H2"})


if __name__ == "__main__":
    unittest.main()

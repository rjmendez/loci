# ===========================================================================
# VENDORED — do not edit here; edit upstream and run
#            scripts/sync_hallucination_rules.sh
#
# Source:  https://github.com/example/llm-code-hallucination-patterns
# File:    rules/ruff_plugin/llm_hallucination_checks.py
# Commit:  b9c6b44b092fc3669dd7197d46c051465f59e193
# License: MIT (Copyright (c) 2026 contributors)
#
# This is a pinned, vendored copy. Local edits will be overwritten by the
# sync script. To update: edit upstream, push, then run
#   bash scripts/sync_hallucination_rules.sh
# ===========================================================================
"""
LLM Hallucination Checks — Ruff Plugin

Ruff AST-based checks for LLM code generation hallucination patterns.
Covers H1, H3, H7, H9 from the taxonomy.

Codes:
  LH001 — Private attribute access on third-party object (H1)
  LH003 — Missing import for name used in call (H3, supplements F821)
  LH007 — Bare comparison in test (vacuous test assertion) (H7)
  LH009 — asyncio.run() inside async function (H9)

Installation:
  pip install ruff
  # In pyproject.toml:
  [tool.ruff.lint]
  select = ["LH"]

  # To use this plugin:
  # ruff check --select=LH path/to/code/

Usage as standalone checker (without ruff plugin API):
  python3 llm_hallucination_checks.py path/to/check.py
"""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Issue:
    code: str
    line: int
    col: int
    message: str


def check_lh001_private_attr(tree: ast.AST) -> list[Issue]:
    """
    LH001: Private attribute access on objects you don't own.
    Flags: obj._private_attr where obj is not 'self'.
    """
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_") and not attr.startswith("__"):
                # Skip self._attr (legitimate private access)
                if isinstance(node.value, ast.Name) and node.value.id == "self":
                    continue
                issues.append(Issue(
                    code="LH001",
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        f"LH001 Private attribute access '_{attr}' on external object. "
                        "This may break across library versions (H1: Private Attr Fabrication). "
                        "Use the public API or read the library source."
                    )
                ))
    return issues


def check_lh007_vacuous_test(tree: ast.AST, filename: str) -> list[Issue]:
    """
    LH007: Bare comparison in test function body (missing assert).
    Only fires in test_*.py files or files containing test_ functions.
    """
    issues = []
    if "test" not in filename:
        return issues

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            for stmt in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Compare):
                    issues.append(Issue(
                        code="LH007",
                        line=stmt.lineno,
                        col=stmt.col_offset,
                        message=(
                            "LH007 Bare comparison with no 'assert' — test is vacuous (H7). "
                            "Add 'assert' keyword or this comparison tests nothing."
                        )
                    ))
    return issues


def check_lh009_asyncio_run_in_async(tree: ast.AST) -> list[Issue]:
    """
    LH009: asyncio.run() called inside an async function.
    This causes RuntimeError: This event loop is already running.
    """
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef,)):
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "run"
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "asyncio"
                ):
                    issues.append(Issue(
                        code="LH009",
                        line=child.lineno,
                        col=child.col_offset,
                        message=(
                            "LH009 asyncio.run() inside async function will raise "
                            "RuntimeError: This event loop is already running (H9). "
                            "Use 'await coro()' or a factory classmethod pattern."
                        )
                    ))
    return issues


def check_file(path: Path) -> list[Issue]:
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [Issue(code="LH000", line=e.lineno or 0, col=0,
                      message=f"LH000 SyntaxError: {e}")]

    issues = []
    issues.extend(check_lh001_private_attr(tree))
    issues.extend(check_lh007_vacuous_test(tree, path.name))
    issues.extend(check_lh009_asyncio_run_in_async(tree))
    return sorted(issues, key=lambda i: (i.line, i.col))


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 llm_hallucination_checks.py path/to/file.py [...]")
        sys.exit(1)

    exit_code = 0
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_dir():
            files = list(path.rglob("*.py"))
        else:
            files = [path]

        for f in files:
            issues = check_file(f)
            for issue in issues:
                print(f"{f}:{issue.line}:{issue.col}: {issue.message}")
                exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

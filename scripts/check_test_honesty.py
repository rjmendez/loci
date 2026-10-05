#!/usr/bin/env python3
"""Fail on test-writing patterns that let a green suite prove nothing.

The 2026-09 adversarial audit of this suite (commit 3a1ad78) mutated the code
under ~330 suspect tests and found 180 that stayed green with the behaviour they
name removed. Most of the damage came from a handful of shapes that are visible
in the syntax tree, so this script rejects them mechanically:

  or-true            ``assert x or True`` -- the assertion cannot fail.
  assert-true        ``assert True`` / ``assertTrue(True)`` -- ditto.
  broad-raises       ``pytest.raises(Exception)`` or another broad built-in class
                     (ValueError, TypeError, KeyError, RuntimeError, OSError, ...;
                     see BROAD_EXC) without a real ``match=`` (and unittest
                     ``assertRaises`` / an empty ``assertRaisesRegex`` pattern). A
                     match that matches anything (``""``, ``".*"``) does not count.
                     Any unrelated error on the way in satisfies it.
  swallowed-assert   an assert inside ``try`` whose handler catches AssertionError
                     (or Exception / bare except) and does not re-raise.
  no-assertions      a test function that asserts nothing, directly or through a
                     helper (same file, a sibling test-helper module, or a name in
                     the allowlist's ``[helpers]``).
  xfail-not-strict   ``xfail`` without ``strict=True``; an unexpected pass must fail.
  env-skipif         a skip condition that reads an environment variable which is
                     not a documented live-smoke opt-in. Such a test never runs in
                     CI, and an always-skipped test is no test.

Justified exceptions live in scripts/test_honesty_allowlist.toml, one entry per
(file, test, rule) with a reason. Entries are keyed by name, not line number, so
unrelated edits do not churn them. The initial list is the pre-existing debt at
3a1ad78; it is meant to shrink. An entry that no longer matches anything is
reported as stale (a failure with --fail-stale).

Usage:
    python3 scripts/check_test_honesty.py                 # check, exit 1 on violations
    python3 scripts/check_test_honesty.py --fail-stale    # also fail on stale entries
    python3 scripts/check_test_honesty.py --write-allowlist  # regenerate (keeps reasons)
    python3 scripts/check_test_honesty.py path/to/test_x.py  # check selected files

Zero dependencies beyond the standard library (tomllib needs Python 3.11+).
"""
from __future__ import annotations

import argparse
import ast
import os
import pathlib
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field

REPO = pathlib.Path(__file__).resolve().parent.parent
ALLOWLIST = REPO / "scripts" / "test_honesty_allowlist.toml"
AGENTS_DOC = REPO / "AGENTS.md"

RULES = (
    "or-true",
    "assert-true",
    "broad-raises",
    "swallowed-assert",
    "no-assertions",
    "xfail-not-strict",
    "env-skipif",
)

# Exception classes so broad that, without match=, "it raised" says nothing about why.
BROAD_EXC = {"Exception", "BaseException", "ValueError", "TypeError", "KeyError",
             "RuntimeError", "OSError", "IOError", "EnvironmentError", "AttributeError",
             "LookupError", "IndexError", "ArithmeticError", "AssertionError"}
# match= / assertRaisesRegex patterns that match any message, so they pin nothing.
TRIVIAL_PATTERNS = {"", ".", ".*", ".+", "(?s).*", "(?s).+", "^", "$", "^.*$", "\\w", "\\S"}
# Handlers that catch a failed assert.
SWALLOWING_EXC = {"AssertionError", "Exception", "BaseException"}
# Calls that are assertions in their own right (besides assert* / *.assert_*).
ASSERTING_CALLS = {"raises", "warns", "deprecated_call", "fail", "approx_equal"}
# Calls that hand a caught exception on to be asserted later.
RECORDING_CALLS = {"append", "put", "put_nowait", "add", "set_exception", "extend"}
EXCLUDED_PARTS = {".git", "node_modules", ".venv", "venv", "__pycache__", "fixtures",
                  "site-packages", ".mypy_cache", ".pytest_cache"}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    rule: str
    test: str
    message: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.path, self.test, self.rule)


@dataclass
class Config:
    opt_ins: dict[str, str] = field(default_factory=dict)
    helpers: set[str] = field(default_factory=set)
    allow: dict[tuple[str, str, str], str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- utils

def _dotted(node: ast.AST) -> str:
    """'pytest.mark.skipif' for the matching Attribute chain, '' if not a plain chain."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _call_name(call: ast.Call) -> str:
    """Last component of the callee: 'raises' for pytest.raises(...)."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _exc_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Tuple):
        out: set[str] = set()
        for elt in node.elts:
            out |= _exc_names(elt)
        return out
    name = _dotted(node)
    return {name.rsplit(".", 1)[-1]} if name else set()


def _real_pattern(node: ast.AST | None) -> bool:
    """False for a missing pattern or a literal one that matches any message."""
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and node.value.strip() not in TRIVIAL_PATTERNS
    return True  # a computed pattern: give it the benefit of the doubt


def _truthy_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and bool(node.value) and node.value is not Ellipsis


def _walk_no_defs(node: ast.AST):
    """ast.walk that does not descend into nested function/class/lambda bodies."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        cur = stack.pop()
        yield cur
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(cur))


def _is_assert_call(call: ast.Call) -> bool:
    name = _call_name(call)
    return name.startswith("assert") or name.startswith("_assert") or name in ASSERTING_CALLS


# ------------------------------------------------------------------ env tracking

def _env_keys_direct(expr: ast.AST) -> set[str]:
    """Environment variable names read by ``expr`` itself ('<dynamic>' if not literal)."""
    found: set[str] = set()

    def key_of(arg: ast.AST | None) -> str:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        return "<dynamic>"

    for node in ast.walk(expr):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in {"os.environ.get", "environ.get", "os.getenv", "getenv",
                        "os.environ.__contains__"}:
                found.add(key_of(node.args[0] if node.args else None))
        elif isinstance(node, ast.Subscript) and _dotted(node.value) in {"os.environ", "environ"}:
            found.add(key_of(node.slice))
        elif isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            if any(_dotted(c) in {"os.environ", "environ"} for c in node.comparators):
                found.add(key_of(node.left))
    return found


class _EnvIndex:
    """Module-level names and functions -> the env vars they (transitively) read."""

    def __init__(self, tree: ast.Module):
        self.names: dict[str, set[str]] = {}
        self.funcs: dict[str, ast.AST] = {}
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs[stmt.name] = stmt
        for stmt in tree.body:
            targets: list[ast.expr] = []
            value = None
            if isinstance(stmt, ast.Assign):
                targets, value = stmt.targets, stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets, value = [stmt.target], stmt.value
            if value is None:
                continue
            keys = self.resolve(value)
            for tgt in targets:
                if isinstance(tgt, ast.Name):
                    self.names[tgt.id] = keys

    def resolve(self, expr: ast.AST, _seen: frozenset[str] = frozenset()) -> set[str]:
        keys = _env_keys_direct(expr)
        for node in ast.walk(expr):
            if isinstance(node, ast.Name):
                keys |= self.names.get(node.id, set())
                fn = self.funcs.get(node.id)
                if fn is not None and node.id not in _seen:
                    keys |= self.resolve(fn, _seen | {node.id})
        return keys


# --------------------------------------------------------------- per-file check

class _HelperIndex:
    """Which functions (in this file and sibling helper modules) contain assertions."""

    def __init__(self, tree: ast.Module, path: pathlib.Path, extra: set[str]):
        self.defs: dict[str, list[ast.AST]] = {}
        self._collect(tree)
        for mod_path in self._sibling_modules(tree, path):
            try:
                self._collect(ast.parse(mod_path.read_text(encoding="utf-8")))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
        self.extra = extra
        self._memo: dict[str, bool] = {}

    def _collect(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs.setdefault(node.name, []).append(node)

    @staticmethod
    def _sibling_modules(tree: ast.Module, path: pathlib.Path) -> list[pathlib.Path]:
        """Modules imported from the test's own directory (helpers.py, _support.py...)."""
        here = path.parent
        out = []
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.level <= 1:
                    names.append(node.module)
            elif isinstance(node, ast.ImportFrom) and node.level == 1:
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            for name in names:
                cand = here.joinpath(*name.split(".")).with_suffix(".py")
                if cand.is_file() and cand != path:
                    out.append(cand)
        return out

    def asserts(self, fn: ast.AST) -> bool:
        return self._asserts(fn, frozenset())

    def _asserts(self, fn: ast.AST, seen: frozenset[str]) -> bool:
        for node in ast.walk(fn):
            if isinstance(node, ast.Assert):
                return True
            if isinstance(node, ast.Raise) and "AssertionError" in _exc_names(
                    node.exc.func if isinstance(node.exc, ast.Call) else node.exc):
                return True
            if isinstance(node, ast.Call):
                if _is_assert_call(node):
                    return True
                name = _call_name(node)
                if name in self.extra:
                    return True
                if name and name not in seen and self._helper_asserts(name, seen):
                    return True
        return False

    def _helper_asserts(self, name: str, seen: frozenset[str]) -> bool:
        if name in self._memo:
            return self._memo[name]
        self._memo[name] = False  # cycle guard
        result = any(self._asserts(d, seen | {name}) for d in self.defs.get(name, ()))
        self._memo[name] = result
        return result


class _Checker(ast.NodeVisitor):
    def __init__(self, rel: str, path: pathlib.Path, tree: ast.Module, cfg: Config,
                 is_test_module: bool):
        self.rel = rel
        self.cfg = cfg
        self.is_test_module = is_test_module
        self.scope: list[str] = []
        self.out: list[Violation] = []
        self.env = _EnvIndex(tree)
        self.helpers = _HelperIndex(tree, path, cfg.helpers)
        self.parents = {child: parent for parent in ast.walk(tree)
                        for child in ast.iter_child_nodes(parent)}

    # scope bookkeeping -------------------------------------------------------
    def _qual(self) -> str:
        return "::".join(self.scope) if self.scope else "<module>"

    def _add(self, node: ast.AST, rule: str, message: str) -> None:
        self.out.append(Violation(self.rel, getattr(node, "lineno", 0), rule, self._qual(), message))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        for dec in node.decorator_list:
            self.visit(dec)
        for stmt in node.body:
            self.visit(stmt)
        self.scope.pop()

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        for dec in node.decorator_list:
            self.visit(dec)
        if self.is_test_module and node.name.startswith("test") and len(self.scope) <= 2 \
                and not self.helpers.asserts(node):
            self._add(node, "no-assertions",
                      "test asserts nothing (no assert, assert* call, pytest.raises, or "
                      "asserting helper)")
        for stmt in node.body:
            self.visit(stmt)
        self.scope.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    # rules -------------------------------------------------------------------
    def visit_Assert(self, node: ast.Assert) -> None:
        if _truthy_constant(node.test):
            self._add(node, "assert-true", "assert of a truthy constant can never fail")
        self._check_or_true(node, node.test)
        self.generic_visit(node)

    def _check_or_true(self, anchor: ast.AST, expr: ast.AST) -> None:
        # `assert x or <truthy literal>` at the top, or a literal True anywhere in an `or`.
        # A nested `env.get(k) or "default"` is an ordinary default, not a tautology.
        top = isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or) \
            and any(_truthy_constant(v) for v in expr.values)
        nested = any(isinstance(sub, ast.BoolOp) and isinstance(sub.op, ast.Or)
                     and any(isinstance(v, ast.Constant) and v.value is True for v in sub.values)
                     for sub in ast.walk(expr))
        if top or nested:
            self._add(anchor, "or-true", "'... or True' makes the assertion unfalsifiable")

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        dotted = _dotted(node.func)
        if name in {"assertTrue", "assert_", "failUnless"} and node.args:
            if _truthy_constant(node.args[0]):
                self._add(node, "assert-true", f"{name}() of a truthy constant can never fail")
            self._check_or_true(node, node.args[0])
        if name == "raises" and dotted.endswith("raises") and node.args:
            broad = _exc_names(node.args[0]) & BROAD_EXC
            match = next((k.value for k in node.keywords if k.arg == "match"), None)
            if broad and not _real_pattern(match):
                self._add(node, "broad-raises",
                          f"raises({', '.join(sorted(broad))}) without a real match=: any "
                          "unrelated error satisfies it")
        if name == "assertRaises" and node.args:
            broad = _exc_names(node.args[0]) & BROAD_EXC
            if broad:
                self._add(node, "broad-raises",
                          f"assertRaises({', '.join(sorted(broad))}): use assertRaisesRegex or a "
                          "specific exception")
        if name == "assertRaisesRegex" and node.args:
            broad = _exc_names(node.args[0]) & BROAD_EXC
            pattern = node.args[1] if len(node.args) > 1 else next(
                (k.value for k in node.keywords if k.arg in {"expected_regex", "expected_regexp"}), None)
            if broad and not _real_pattern(pattern):
                self._add(node, "broad-raises",
                          f"assertRaisesRegex({', '.join(sorted(broad))}) with a pattern that "
                          "matches any message")
        if dotted in {"pytest.xfail"}:
            self._add(node, "xfail-not-strict",
                      "imperative pytest.xfail() cannot be strict; use mark.xfail(strict=True)")
        self._check_marker(node)
        self.generic_visit(node)

    def _check_marker(self, call: ast.Call) -> None:
        """Called xfail / skipif markers: decorator, pytestmark or pytest.param(marks=...).

        An uncalled `pytest.mark.xfail` is caught by visit_Attribute."""
        name = _dotted(call.func)
        last = name.rsplit(".", 1)[-1] if name else ""
        if last == "xfail" and name != "pytest.xfail":
            strict = any(
                k.arg == "strict" and isinstance(k.value, ast.Constant) and k.value.value is True
                for k in call.keywords)
            if not strict:
                self._add(call, "xfail-not-strict", "xfail without strict=True hides an unexpected pass")
        if last in {"skipif", "skipIf", "skipUnless"}:
            cond = call.args[0] if call.args else next(
                (k.value for k in call.keywords if k.arg == "condition"), None)
            if cond is not None:
                self._check_env(call, cond)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # A bare `pytest.mark.xfail` used as a value: pytestmark = ..., pytest.param(marks=...).
        parent = self.parents.get(node)
        is_callee = isinstance(parent, ast.Call) and parent.func is node
        if node.attr == "xfail" and _dotted(node).endswith("mark.xfail") and not is_callee:
            self._add(node, "xfail-not-strict", "xfail without strict=True hides an unexpected pass")
        self.generic_visit(node)

    def _check_env(self, anchor: ast.AST, cond: ast.AST) -> None:
        keys = self.env.resolve(cond)
        bad = sorted(k for k in keys if k not in self.cfg.opt_ins)
        if bad:
            self._add(anchor, "env-skipif",
                      f"skip condition reads {', '.join(bad)}, not a documented live-smoke "
                      "opt-in: the test never runs in CI")

    def visit_If(self, node: ast.If) -> None:
        # `if not os.environ.get("X"): pytest.skip(...)` / self.skipTest(...)
        skips = [n for s in node.body for n in ast.walk(s)
                 if isinstance(n, ast.Call) and (_dotted(n.func) == "pytest.skip"
                                                 or _call_name(n) == "skipTest")]
        if skips:
            self._check_env(node, node.test)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        body_asserts = any(
            isinstance(n, ast.Assert) or (isinstance(n, ast.Call) and _is_assert_call(n)
                                           and _call_name(n) not in {"raises", "warns"})
            for stmt in node.body for n in [stmt, *_walk_no_defs(stmt)])
        if body_asserts:
            for handler in node.handlers:
                caught = _exc_names(handler.type)
                if handler.type is None or caught & SWALLOWING_EXC:
                    body = [n for stmt in handler.body for n in [stmt, *_walk_no_defs(stmt)]]
                    reraises = any(
                        isinstance(n, ast.Raise) or (isinstance(n, ast.Call) and _call_name(n) == "fail")
                        for n in body)
                    # `except Exception as exc: errors.append(exc)` hands the failure to a
                    # later assert (the usual thread pattern); that is not swallowing it.
                    records = handler.name is not None and any(
                        isinstance(n, ast.Call) and _call_name(n) in RECORDING_CALLS
                        and any(isinstance(a, ast.Name) and a.id == handler.name
                                for arg in n.args for a in ast.walk(arg))
                        for n in body)
                    if not reraises and not records:
                        what = "bare except" if handler.type is None else ", ".join(sorted(caught))
                        self._add(handler, "swallowed-assert",
                                  f"assert inside try whose handler ({what}) does not re-raise")
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            ce = item.context_expr
            if isinstance(ce, ast.Call) and _call_name(ce) == "suppress" \
                    and {n for a in ce.args for n in _exc_names(a)} & SWALLOWING_EXC:
                if any(isinstance(n, ast.Assert) for s in node.body for n in [s, *_walk_no_defs(s)]):
                    self._add(node, "swallowed-assert",
                              "assert inside contextlib.suppress(...) that catches AssertionError")
        self.generic_visit(node)


def check_file(path: pathlib.Path, cfg: Config, source: str | None = None) -> list[Violation]:
    """Violations in ``path`` (or in ``source``, parsed as if it were that file)."""
    rel = path.resolve().relative_to(REPO).as_posix() if path.resolve().is_relative_to(REPO) \
        else path.as_posix()
    try:
        text = path.read_text(encoding="utf-8") if source is None else source
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [Violation(rel, exc.lineno or 0, "syntax", "<module>", f"cannot parse: {exc.msg}")]
    is_test = path.name.startswith("test_") or path.name.endswith("_test.py")
    checker = _Checker(rel, path, tree, cfg, is_test)
    checker.visit(tree)
    # A mark may be reached both as a decorator and as a Call node: keep one per site.
    seen: set[tuple] = set()
    unique = []
    for v in checker.out:
        k = (v.line, v.rule, v.test, v.message)
        if k not in seen:
            seen.add(k)
            unique.append(v)
    return unique


# ----------------------------------------------------------------- file finding

def _is_candidate(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in EXCLUDED_PARTS for p in parts[:-1]):
        return False
    if "callgraph/docs/legacy" in rel:
        return False
    name = parts[-1]
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py")
                                     or name == "conftest.py")


def discover() -> list[pathlib.Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others",
                              "--exclude-standard", "*.py"],
                             cwd=REPO, capture_output=True, check=True).stdout
        rels = [r for r in out.decode("utf-8").split("\0") if r]
    except (OSError, subprocess.CalledProcessError):
        rels = [p.relative_to(REPO).as_posix() for p in REPO.rglob("*.py")]
    return [REPO / r for r in sorted(set(rels)) if _is_candidate(r) and (REPO / r).is_file()]


# -------------------------------------------------------------------- allowlist

def load_config(path: pathlib.Path = ALLOWLIST) -> Config:
    cfg = Config()
    if not path.is_file():
        return cfg
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg.opt_ins = {k: str(v) for k, v in data.get("live_smoke_opt_ins", {}).items()}
    cfg.helpers = {str(h) for h in data.get("helpers", {}).get("names", [])}
    for block in data.get("allow", []):
        file = block.get("file", "")
        for entry in block.get("entries", []):
            rule, test, reason = entry.get("rule", ""), entry.get("test", ""), entry.get("reason", "")
            where = f"{path.name}: {file}::{test} [{rule}]"
            if rule not in RULES:
                cfg.errors.append(f"{where}: unknown rule")
            if len(reason.strip()) < 10:
                cfg.errors.append(f"{where}: every entry needs a reason (>= 10 chars)")
            cfg.allow[(file, test, rule)] = reason
    # "Documented" means documented where agents read it, not only in this file.
    doc = AGENTS_DOC.read_text(encoding="utf-8") if AGENTS_DOC.is_file() else ""
    for var in cfg.opt_ins:
        if var not in doc:
            cfg.errors.append(f"live-smoke opt-in {var} is not documented in AGENTS.md")
    return cfg


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_allowlist(violations: list[Violation], cfg: Config, path: pathlib.Path,
                    default_reason: str) -> None:
    head = path.read_text(encoding="utf-8").split("\n[[allow]]", 1)[0].rstrip() + "\n" \
        if path.is_file() else ""
    by_file: dict[str, dict[tuple[str, str], str]] = {}
    for v in violations:
        reason = cfg.allow.get(v.key) or default_reason
        by_file.setdefault(v.path, {})[(v.test, v.rule)] = reason
    lines = [head]
    for file in sorted(by_file):
        lines.append("\n[[allow]]")
        lines.append(f"file = {_toml_str(file)}")
        lines.append("entries = [")
        for (test, rule), reason in sorted(by_file[file].items()):
            lines.append(f"  {{ test = {_toml_str(test)}, rule = {_toml_str(rule)}, "
                         f"reason = {_toml_str(reason)} }},")
        lines.append("]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("paths", nargs="*", help="files to check (default: every test file in the repo)")
    ap.add_argument("--allowlist", type=pathlib.Path, default=ALLOWLIST)
    ap.add_argument("--fail-stale", action="store_true",
                    help="exit 1 when an allowlist entry no longer matches a violation")
    ap.add_argument("--write-allowlist", action="store_true",
                    help="rewrite the allowlist to exactly the current violations, keeping reasons")
    ap.add_argument("--reason", default="pre-existing debt; remove when the test is fixed",
                    help="reason recorded for new entries with --write-allowlist")
    args = ap.parse_args(argv)

    cfg = load_config(args.allowlist)
    files = [pathlib.Path(p) for p in args.paths] if args.paths else discover()
    violations = [v for f in files for v in check_file(f, cfg)]

    if args.write_allowlist:
        write_allowlist(violations, cfg, args.allowlist, args.reason)
        print(f"wrote {len(violations)} entries to {args.allowlist}")
        return 0

    gha = os.environ.get("GITHUB_ACTIONS") == "true"
    new = [v for v in violations if v.key not in cfg.allow]
    for v in new:
        print(f"{v.path}:{v.line}: {v.rule}: {v.test}: {v.message}")
        if gha:
            print(f"::error file={v.path},line={v.line},title=test-honesty {v.rule}::{v.message}")
    for err in cfg.errors:
        print(f"allowlist: {err}")

    stale: list[tuple[str, str, str]] = []
    if not args.paths:  # staleness is only meaningful for a full scan
        live = {v.key for v in violations}
        stale = sorted(k for k in cfg.allow if k not in live)
        for file, test, rule in stale:
            msg = f"stale allowlist entry {file}::{test} [{rule}] -- delete it, the violation is gone"
            print(msg)
            if gha:
                print(f"::warning file=scripts/test_honesty_allowlist.toml,title=stale entry::{msg}")

    print(f"test-honesty: {len(files)} files, {len(violations)} violations, "
          f"{len(violations) - len(new)} allowlisted, {len(new)} new, {len(stale)} stale")
    if new or cfg.errors or (stale and args.fail_stale):
        if new:
            print("See AGENTS.md 'Test-writing rules for agents'. Fix the test; allowlist only "
                  "with a reason.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

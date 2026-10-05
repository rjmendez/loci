"""Shared test helpers: load a fixture .py file as a SourceFile without
going through ingest.load_corpus's repo-root file discovery (fixtures/ is
outside the analyzed corpus by design — config.SELF_PACKAGE_REL excludes
this whole package)."""
from __future__ import annotations

import ast
from pathlib import Path

from ..extract.calls import build_top_level_index, make_call_visitor
from ..extract.defs import extract_module
from ..extract.dispatch import extract_probable_calls
from ..extract.flow import extract_flow
from ..extract.funcrefs import make_funcrefs_visitor
from ..extract.imports import extract_imports
from ..extract.literals import make_literals_visitor
from ..extract.names import emit_non_walk_writes, make_names_visitor
from ..extract.registry import extract_external_roots, extract_injections, extract_registry_module
from ..extract.walk import walk_module
from ..ingest import SourceFile
from ..model import GraphStore
from ..resolve import ResolutionTable

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def load_fixture(rel_path: str) -> SourceFile:
    text = (FIXTURES_DIR / rel_path).read_text()
    return source_file(rel_path, text)


def source_file(rel_path: str, text: str) -> SourceFile:
    import hashlib
    sha1 = hashlib.sha1(text.encode()).hexdigest()
    try:
        tree = ast.parse(text, filename=rel_path)
        return SourceFile(rel_path, text, tree, None, "fixture", sha1)
    except SyntaxError as exc:
        return SourceFile(rel_path, text, None, f"SyntaxError: {exc.msg}", "fixture", sha1)


def build_fixture_store(rel_paths: list[str]):
    """Runs the FULL pipeline (defs -> imports -> calls/names -> registry ->
    injections/roots) over a small, curated set of fixture files — mirrors
    pipeline.build_graph's stage order exactly, without going through
    ingest.load_corpus's repo-root file discovery. Returns (store, table,
    scopes)."""
    sources = [load_fixture(p) for p in rel_paths]
    table = ResolutionTable(sources)
    store = GraphStore()
    scopes = {sf.rel_path: extract_module(store, sf, table) for sf in sources}
    for sf in sources:
        extract_imports(store, sf, scopes[sf.rel_path], table, scopes)
    top_level_index = build_top_level_index(store)
    funcref_flushes = []
    for sf in sources:
        scope = scopes[sf.rel_path]
        emit_non_walk_writes(store, sf, scope)
        call_visit = make_call_visitor(store, sf, scope, table, top_level_index)
        names_visit = make_names_visitor(store, sf, scope)
        literals_visit = make_literals_visitor(store, sf, scope)
        refs_visit, refs_flush = make_funcrefs_visitor(store, sf, scope, top_level_index)
        funcref_flushes.append(refs_flush)

        def combined(node, fi, _c=call_visit, _n=names_visit, _l=literals_visit, _r=refs_visit) -> None:
            _c(node, fi)
            _n(node, fi)
            _l(node, fi)
            _r(node, fi)

        walk_module(scope, combined)
        extract_flow(store, sf, scope)
    for sf in sources:
        extract_registry_module(store, sf, scopes[sf.rel_path], top_level_index)
    extract_injections(store, sources, scopes, top_level_index)
    extract_external_roots(store, sources, top_level_index)
    for flush in funcref_flushes:
        flush()
    extract_probable_calls(store, scopes, top_level_index)
    return store, table, scopes


# ---------------------------------------------------------------------------
# Independent oracles for real-corpus tests.
#
# A real-corpus test that pins a literal count ("43 tools", "line 386") fails
# every time someone adds a tool or a comment, which teaches people to bump the
# number without looking. These helpers compute the expected answer straight
# from the source with plain `ast`, sharing no code with the pipeline under
# test, so the test compares the tool against the code as it is now.
# ---------------------------------------------------------------------------


def source_at(sources, rel_path: str):
    """The SourceFile for `rel_path` from a load_corpus() result."""
    return next(sf for sf in sources if sf.rel_path == rel_path)


def _decorator_target(dec: ast.expr) -> ast.expr:
    return dec.func if isinstance(dec, ast.Call) else dec


def mcp_tool_functions(source: str) -> set[str]:
    """Names of functions decorated `@mcp.tool` / `@mcp.tool(...)`."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = _decorator_target(dec)
            if (isinstance(target, ast.Attribute) and target.attr == "tool"
                    and isinstance(target.value, ast.Name) and target.value.id == "mcp"):
                out.add(node.name)
    return out


def global_write_lines(source: str, enclosing_fn: str, name: str) -> list[int]:
    """Lines where top-level `enclosing_fn` assigns `name` after declaring it
    `global` -- the writes the pipeline records as WRITES_NAME via=global-stmt."""
    tree = ast.parse(source)
    lines: list[int] = []
    for fn in tree.body:
        if not (isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == enclosing_fn):
            continue
        if not any(isinstance(n, ast.Global) and name in n.names for n in ast.walk(fn)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Name) and node.id == name
                    and isinstance(node.ctx, ast.Store)):
                lines.append(node.lineno)
    return sorted(lines)


def called_bare_name_count(source: str, name: str) -> int:
    """How many times `name(...)` is called by bare name anywhere in the module."""
    return sum(
        1 for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    )


def _dotted(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


_DEC_RULES = {
    "mcp.tool": "DEC-tool", "mcp.resource": "DEC-tool",
    "mcp.custom_route": "DEC-mcp-route",
    "app.get": "DEC-route", "app.post": "DEC-route", "app.put": "DEC-route",
    "app.delete": "DEC-route", "app.patch": "DEC-route",
}


def registration_census(sources) -> dict[str, int]:
    """REGISTERS counts per rule, computed with plain `ast` and no pipeline code.

    * DEC-*: functions decorated with the FastMCP / FastAPI registrars above.
    * MAN-LOOP: names in a `for fn in (a, b, ...):` tuple whose body calls
      `mcp.tool()(fn)`, counted when the name is a module-level function.
    * MAN-DICT: bare-name values of a module-level `_SKILL_MAP = {...}` that are
      module-level functions.

    Used to gate HEAD exactly without pinning a number that every new tool
    moves, and itself checked against the hand-validated counts at
    VALIDATED_REV (tests/test_pipeline_real_corpus.py).
    """
    counts: dict[str, int] = {}

    def bump(rule: str, n: int = 1) -> None:
        counts[rule] = counts.get(rule, 0) + n

    for sf in sources:
        if sf.tree is None:
            continue
        tree = ast.parse(sf.source)
        top_level_fns = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    rule = _DEC_RULES.get(_dotted(_decorator_target(dec)) or "")
                    if rule:
                        bump(rule)
                        break
            elif (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                  and isinstance(node.iter, (ast.Tuple, ast.List))
                  and node.iter.elts and all(isinstance(e, ast.Name) for e in node.iter.elts)):
                var = node.target.id
                registers = any(
                    isinstance(c, ast.Call) and isinstance(c.func, ast.Call)
                    and _dotted(c.func.func) == "mcp.tool"
                    and len(c.args) == 1 and isinstance(c.args[0], ast.Name) and c.args[0].id == var
                    for stmt in node.body for c in ast.walk(stmt)
                )
                if registers:
                    bump("MAN-LOOP", sum(1 for e in node.iter.elts if e.id in top_level_fns))
        for stmt in tree.body:
            target = value = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target, value = stmt.targets[0], stmt.value
            elif isinstance(stmt, ast.AnnAssign):
                target, value = stmt.target, stmt.value
            if isinstance(target, ast.Name) and target.id == "_SKILL_MAP" and isinstance(value, ast.Dict):
                bump("MAN-DICT", sum(1 for v in value.values
                                     if isinstance(v, ast.Name) and v.id in top_level_fns))
    return counts

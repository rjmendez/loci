"""Per-module symbol tables, built in one AST pass: module globals,
function/class nesting with qualnames, decorator source text, `global`
declarations, and import statements. Everything downstream (extract/defs.py,
extract/imports.py, and later slices' extract/calls.py etc.) reads a
ModuleScope; nothing re-walks the tree for names.

stdlib only.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

from .ingest import SourceFile

PARAM_KIND_NAMES = {
    "posonly": "POSITIONAL_ONLY",
    "pos_or_kw": "POSITIONAL_OR_KEYWORD",
    "vararg": "VAR_POSITIONAL",
    "kwonly": "KEYWORD_ONLY",
    "kwarg": "VAR_KEYWORD",
}


@dataclass
class ParamInfo:
    name: str
    kind: str
    has_default: bool = False
    default_is_literal: bool = False


@dataclass
class FunctionInfo:
    qualname: str
    node: ast.AST                      # FunctionDef | AsyncFunctionDef | Lambda
    lineno: int
    end_lineno: int
    col: int
    is_async: bool
    is_lambda: bool
    is_method: bool
    is_nested: bool
    params: list[ParamInfo]
    decorators: list[str]              # verbatim source text, outermost first
    docstring_first_line: Optional[str]
    parent_class: Optional[str] = None  # simple class name if is_method
    ambiguous_duplicate: bool = False


@dataclass
class ClassInfo:
    qualname: str
    node: ast.ClassDef
    lineno: int
    end_lineno: int
    bases: list[str]                   # verbatim source text per base
    ambiguous_duplicate: bool = False


@dataclass
class NameInfo:
    ident: str
    binding_kinds: set[str] = field(default_factory=set)   # assign|def|import|global-only
    binding_lines: dict[str, list[int]] = field(default_factory=dict)

    @property
    def is_dunder(self) -> bool:
        return self.ident.startswith("__") and self.ident.endswith("__")

    @property
    def is_private(self) -> bool:
        return self.ident.startswith("_") and not self.is_dunder

    def record(self, kind: str, line: int) -> None:
        self.binding_kinds.add(kind)
        self.binding_lines.setdefault(kind, []).append(line)


@dataclass
class ImportRecord:
    lineno: int
    is_from: bool
    module: Optional[str]              # dotted module (None for bare relative "from . import x")
    level: int                         # 0 for absolute
    names: list[tuple[str, Optional[str]]]  # (imported name, asname)
    scope: str                          # "module-level" | "function-local"
    enclosing_fn: Optional[str]         # qualname of enclosing function, if function-local


@dataclass
class FunctionLocals:
    """A function's own binding surface, computed with the same "linear walk,
    not a CFG" philosophy as the rest of this tool: every assignment target
    reachable in the function's body (without descending into nested
    def/class/lambda scopes) is treated as local for the WHOLE function, and
    `global`/`nonlocal` declarations override that. Used by extract/calls.py
    (to spot `param-call` callsites) and extract/names.py (to tell a real
    module-global read from a same-named local)."""
    params: set[str] = field(default_factory=set)
    assigned: set[str] = field(default_factory=set)
    global_declared: set[str] = field(default_factory=set)
    nonlocal_declared: set[str] = field(default_factory=set)

    @property
    def effective_locals(self) -> set[str]:
        return (self.params | self.assigned) - self.global_declared - self.nonlocal_declared


@dataclass
class ModuleAssign:
    """A module-level `A = B` (or `A = mod.B`) simple aliasing candidate."""
    lineno: int
    target: str
    value_kind: str    # "name" | "attribute" | "other"
    value_name: Optional[str]       # for "name": the RHS identifier
    # For "attribute": the `A = mod.B` halves. These were removed once on the
    # reasoning that imports.py only resolves value_kind == "name", but the call
    # site below was left passing them, so constructing an attribute alias raised
    # TypeError. Nothing noticed because the whole corpus contained exactly one
    # module-level `A = mod.B`, in a test file. Adding two re-exports to
    # mlops/grounding/train.py took the callgraph suite from 3 failures to 19
    # failures and 32 errors.
    value_module: Optional[str] = None   # for "attribute": the `mod` in mod.B
    value_attr: Optional[str] = None     # for "attribute": the `B` in mod.B


class ModuleScope:
    def __init__(self, sf: SourceFile) -> None:
        self.sf = sf
        self._lines = sf.source.splitlines(keepends=True)
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.names: dict[str, NameInfo] = {}
        self.imports: list[ImportRecord] = []
        self.module_assigns: list[ModuleAssign] = []
        self._seen_qualnames: dict[str, int] = {}
        # id(ast-node) -> qualname, so a later whole-module walk (calls.py,
        # names.py, registry.py) can recover "which function am I inside"
        # from the same qualnames this pass already computed (incl. the
        # "#2"-style disambiguation suffix), instead of recomputing a
        # possibly-inconsistent qualname independently.
        self._node_to_qualname: dict[int, str] = {}
        self.function_by_qualname: dict[str, FunctionInfo] = {}
        self._locals_cache: dict[int, FunctionLocals] = {}
        if sf.tree is not None:
            self._build()

    # -- helpers -------------------------------------------------------

    def _name(self, ident: str) -> NameInfo:
        info = self.names.get(ident)
        if info is None:
            info = NameInfo(ident=ident)
            self.names[ident] = info
        return info

    def qualname_for_node(self, node: ast.AST) -> Optional[str]:
        """The qualname this pass assigned to a def/lambda node, or None if
        `node` isn't one of this module's own FunctionDef/AsyncFunctionDef/
        Lambda nodes."""
        return self._node_to_qualname.get(id(node))

    def source_of(self, node: ast.AST) -> str:
        """Public wrapper over the cached-split source-segment extractor,
        for extract/calls.py and extract/registry.py to capture verbatim
        callee/receiver text without each re-splitting the file."""
        return self._src(node)

    def locals_of(self, fi: FunctionInfo) -> FunctionLocals:
        cached = self._locals_cache.get(id(fi.node))
        if cached is not None:
            return cached
        params = {p.name for p in fi.params}
        assigned: set[str] = set()
        global_declared: set[str] = set()
        nonlocal_declared: set[str] = set()

        def add_target(t: ast.expr) -> None:
            if isinstance(t, ast.Name):
                assigned.add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for elt in t.elts:
                    add_target(elt)
            elif isinstance(t, ast.Starred):
                add_target(t.value)

        def walk(node: ast.AST) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)) and node is not fi.node:
                return  # nested scope: its own locals, not this function's
            if isinstance(node, ast.Global):
                global_declared.update(node.names)
            elif isinstance(node, ast.Nonlocal):
                nonlocal_declared.update(node.names)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    add_target(t)
            elif isinstance(node, ast.AnnAssign) and node.target is not None:
                add_target(node.target)
            elif isinstance(node, ast.AugAssign):
                add_target(node.target)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                add_target(node.target)
            elif isinstance(node, ast.withitem):
                if node.optional_vars is not None:
                    add_target(node.optional_vars)
            elif isinstance(node, ast.ExceptHandler):
                if node.name:
                    assigned.add(node.name)
            elif isinstance(node, ast.NamedExpr):
                add_target(node.target)
            for child in ast.iter_child_nodes(node):
                walk(child)

        if isinstance(fi.node, ast.Lambda):
            walk(fi.node.body)
        else:
            for stmt in fi.node.body:
                walk(stmt)
        fl = FunctionLocals(params=params, assigned=assigned,
                             global_declared=global_declared, nonlocal_declared=nonlocal_declared)
        self._locals_cache[id(fi.node)] = fl
        return fl

    def _src(self, node: ast.AST) -> str:
        # ast.get_source_segment() re-splits the WHOLE file into lines on
        # every call; called once per decorator/base that's cheap, but
        # called across a 7,000-line file dozens of times it dominates the
        # build. Cache the split once per module instead.
        end_lineno = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        if end_lineno is None or end_col is None:
            return "<unavailable>"
        lines = self._lines
        lineno = node.lineno - 1
        end_lineno0 = end_lineno - 1
        col = node.col_offset
        if lineno == end_lineno0:
            return lines[lineno].encode()[col:end_col].decode()
        first = lines[lineno].encode()[col:].decode()
        last = lines[end_lineno0].encode()[:end_col].decode()
        mid = lines[lineno + 1:end_lineno0]
        return "".join([first, *mid, last])

    def _unique_qualname(self, qualname: str) -> tuple[str, bool]:
        count = self._seen_qualnames.get(qualname, 0)
        self._seen_qualnames[qualname] = count + 1
        if count == 0:
            return qualname, False
        return f"{qualname}#{count + 1}", True

    # -- main walk -------------------------------------------------------

    def _build(self) -> None:
        tree = self.sf.tree
        assert tree is not None
        self._walk_body(tree.body, stack=[])
        for lam in _find_lambdas_in(tree.body):
            self._visit_lambda(lam, [])
        self._collect_module_names(tree.body)
        self._collect_global_only()

    def _walk_body(self, body: list[ast.stmt], stack: list[tuple[str, str]]) -> None:
        for stmt in body:
            self._walk_stmt(stmt, stack)

    def _walk_stmt(self, stmt: ast.stmt, stack: list[tuple[str, str]]) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._visit_function(stmt, stack)
            return
        if isinstance(stmt, ast.ClassDef):
            self._visit_class(stmt, stack)
            return
        if isinstance(stmt, ast.Import):
            self._visit_import(stmt, stack)
        elif isinstance(stmt, ast.ImportFrom):
            self._visit_import_from(stmt, stack)
        # Recurse into simple nested-statement containers (if/try/with/for/
        # while) so nested `def`/`class`/import/global are still found —
        # this is a linear walk, not a CFG, matching the rest of the tool.
        for field_name, value in ast.iter_fields(stmt):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.stmt):
                        self._walk_stmt(item, stack)
            elif isinstance(value, ast.stmt):
                self._walk_stmt(value, stack)

    def _qualname_for(self, name: str, stack: list[tuple[str, str]]) -> str:
        if not stack:
            return name
        parent_kind, parent_qual = stack[-1]
        if parent_kind == "class":
            return f"{parent_qual}.{name}"
        return f"{parent_qual}.<locals>.{name}"

    def _extract_params(self, args: ast.arguments) -> list[ParamInfo]:
        out: list[ParamInfo] = []

        def add(a: ast.arg, kind: str, default: Optional[ast.expr]) -> None:
            has_default = default is not None
            is_lit = has_default and isinstance(default, ast.Constant)
            out.append(ParamInfo(a.arg, PARAM_KIND_NAMES[kind], has_default, is_lit))

        posonly = list(getattr(args, "posonlyargs", []))
        pos = list(args.args)
        n_pos_defaults = len(args.defaults)
        all_pos = posonly + pos
        pos_defaults: list[Optional[ast.expr]] = [None] * (len(all_pos) - n_pos_defaults) + list(args.defaults)
        for a, d in zip(posonly, pos_defaults[: len(posonly)]):
            add(a, "posonly", d)
        for a, d in zip(pos, pos_defaults[len(posonly):]):
            add(a, "pos_or_kw", d)
        if args.vararg:
            add(args.vararg, "vararg", None)
        for a, d in zip(args.kwonlyargs, args.kw_defaults):
            add(a, "kwonly", d)
        if args.kwarg:
            add(args.kwarg, "kwarg", None)
        return out

    def _visit_function(self, node, stack: list[tuple[str, str]]) -> None:
        is_nested = bool(stack) and stack[-1][0] == "function"
        is_method = bool(stack) and stack[-1][0] == "class"
        qualname_raw = self._qualname_for(node.name, stack)
        qualname, ambiguous = self._unique_qualname(qualname_raw)
        decorators = [self._src(d) for d in node.decorator_list]
        doc = ast.get_docstring(node, clean=True)
        doc_first = doc.splitlines()[0] if doc else None
        fi = FunctionInfo(
            qualname=qualname, node=node, lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno), col=node.col_offset,
            is_async=isinstance(node, ast.AsyncFunctionDef), is_lambda=False,
            is_method=is_method, is_nested=is_nested,
            params=self._extract_params(node.args), decorators=decorators,
            docstring_first_line=doc_first,
            parent_class=stack[-1][1].rsplit(".", 1)[-1] if is_method else None,
            ambiguous_duplicate=ambiguous,
        )
        self.functions.append(fi)
        self._node_to_qualname[id(node)] = qualname
        self.function_by_qualname[qualname] = fi
        if stack and stack[-1][0] == "class":
            # class method table wants simple names, not the module-relative id
            pass
        new_stack = stack + [("function", qualname)]
        self._walk_body(node.body, new_stack)
        for lam in _find_lambdas_in(node.body):
            self._visit_lambda(lam, new_stack)

    def _visit_lambda(self, node: ast.Lambda, stack: list[tuple[str, str]]) -> None:
        raw_name = f"<lambda>@{node.lineno}"
        qualname_raw = self._qualname_for(raw_name, stack)
        qualname, ambiguous = self._unique_qualname(qualname_raw)
        is_method = bool(stack) and stack[-1][0] == "class"
        fi = FunctionInfo(
            qualname=qualname, node=node, lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno), col=node.col_offset,
            is_async=False, is_lambda=True, is_method=is_method, is_nested=True,
            params=self._extract_params(node.args), decorators=[],
            docstring_first_line=None,
            parent_class=stack[-1][1].rsplit(".", 1)[-1] if is_method else None,
            ambiguous_duplicate=ambiguous,
        )
        self.functions.append(fi)
        self._node_to_qualname[id(node)] = qualname
        self.function_by_qualname[qualname] = fi

    def _visit_class(self, node: ast.ClassDef, stack: list[tuple[str, str]]) -> None:
        qualname_raw = self._qualname_for(node.name, stack)
        qualname, ambiguous = self._unique_qualname(qualname_raw)
        bases = [self._src(b) for b in node.bases]
        ci = ClassInfo(qualname=qualname, node=node, lineno=node.lineno,
                        end_lineno=getattr(node, "end_lineno", node.lineno),
                        bases=bases, ambiguous_duplicate=ambiguous)
        self.classes.append(ci)
        new_stack = stack + [("class", qualname)]
        self._walk_body(node.body, new_stack)
        for lam in _find_lambdas_in(node.body):
            self._visit_lambda(lam, new_stack)

    def _visit_import(self, stmt: ast.Import, stack: list[tuple[str, str]]) -> None:
        scope = "module-level" if not stack else "function-local"
        enclosing = stack[-1][1] if stack and stack[-1][0] == "function" else None
        names: list[tuple[str, Optional[str]]] = []
        for alias in stmt.names:
            bound = alias.asname or alias.name.split(".")[0]
            names.append((alias.name, alias.asname))
            if scope == "module-level":
                self._name(bound).record("import", stmt.lineno)
        self.imports.append(ImportRecord(stmt.lineno, False, None, 0, names, scope, enclosing))

    def _visit_import_from(self, stmt: ast.ImportFrom, stack: list[tuple[str, str]]) -> None:
        scope = "module-level" if not stack else "function-local"
        enclosing = stack[-1][1] if stack and stack[-1][0] == "function" else None
        names = [(a.name, a.asname) for a in stmt.names]
        if scope == "module-level":
            for imported, asname in names:
                if imported == "*":
                    continue
                bound = asname or imported
                self._name(bound).record("import", stmt.lineno)
        self.imports.append(ImportRecord(stmt.lineno, True, stmt.module, stmt.level, names, scope, enclosing))

    # -- module-level NAME slots -----------------------------------------

    def _collect_module_names(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._name(stmt.name).record("def", stmt.lineno)
            elif isinstance(stmt, ast.ClassDef):
                self._name(stmt.name).record("def", stmt.lineno)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    self._record_assign_target(target, stmt.lineno)
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    self._record_module_alias(stmt.targets[0].id, stmt.value, stmt.lineno)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                self._name(stmt.target.id).record("assign", stmt.lineno)
                if stmt.value is not None:
                    self._record_module_alias(stmt.target.id, stmt.value, stmt.lineno)
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                self._name(stmt.target.id).record("assign", stmt.lineno)
            elif isinstance(stmt, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                for field_name, value in ast.iter_fields(stmt):
                    if isinstance(value, list):
                        nested = [v for v in value if isinstance(v, ast.stmt)]
                        if nested:
                            self._collect_module_names(nested)

    def _record_assign_target(self, target: ast.expr, lineno: int) -> None:
        if isinstance(target, ast.Name):
            self._name(target.id).record("assign", lineno)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._record_assign_target(elt, lineno)

    def _record_module_alias(self, target_name: str, value: ast.expr, lineno: int) -> None:
        if isinstance(value, ast.Name):
            self.module_assigns.append(ModuleAssign(lineno, target_name, "name", value.id))
        elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            self.module_assigns.append(
                ModuleAssign(lineno, target_name, "attribute", None, value.value.id, value.attr)
            )

    # -- global-only detection -------------------------------------------

    def _collect_global_only(self) -> None:
        if self.sf.tree is None:
            return
        for node in ast.walk(self.sf.tree):
            if isinstance(node, ast.Global):
                for ident in node.names:
                    info = self._name(ident)
                    if "assign" not in info.binding_kinds and "def" not in info.binding_kinds:
                        info.record("global-only", node.lineno)


def _find_lambdas_in(body: list[ast.stmt]):
    """Lambdas that are direct expressions within this body (not inside a
    nested def/class, which get their own scan when that scope is
    visited)."""
    out: list[ast.Lambda] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        if isinstance(node, ast.Lambda):
            out.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in body:
        visit(stmt)
    return out

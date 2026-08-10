"""The module resolution table.

Maps repo-relative path <-> every dotted name a module answers to, modelling
this codebase's actual import reality:

  * The implicit repo-root namespace: `mcp.graph_tools`, `a2a_server.server`
    (works because Python 3 namespace packages don't require __init__.py).
  * Flat sibling imports inside a directory that has been put on sys.path,
    either by a *literal* `sys.path.insert(0, <expr>)` this tool can
    constant-fold, or trivially by virtue of being the importing file's own
    directory (`import qdrant_ops` from mcp/server.py).
  * The 10 `sys.path.insert(0, os.path.join(..., "..", "mcp"))`-shaped
    inserts in scripts/ that make `import graph_tools` &c. work from a
    script that isn't inside mcp/.

Every resolution records WHICH insert made it work (or "implicit repo
root" / "same directory") — that provenance is itself a debugging answer,
and a flat import that only resolves because of an insert written in a
*different* file is exactly the kind of fact this tool exists to surface.

stdlib only.
"""
from __future__ import annotations

import ast
import posixpath
from dataclasses import dataclass, field
from typing import Optional

from . import config
from .ingest import SourceFile

# --------------------------------------------------------------------------
# sys.path.insert / .append literal folding
# --------------------------------------------------------------------------


class _Unresolvable:
    """Sentinel: the expression could not be constant-folded."""


UNRESOLVABLE = _Unresolvable()


@dataclass(frozen=True)
class _Dir:
    """A folded value representing a directory, given as a repo-relative
    POSIX path ("" means the repo root itself)."""
    rel: str


@dataclass(frozen=True)
class _FileSelf:
    """The analyzed file's own path (before any dirname/.parent is applied)."""
    rel_path: str


@dataclass(frozen=True)
class _Lit:
    """A plain string constant with no path semantics attached yet."""
    value: str


_FoldedValue = object  # _Dir | _FileSelf | _Lit | _Unresolvable


def _dirname_of(value: _FoldedValue) -> _FoldedValue:
    if isinstance(value, _FileSelf):
        return _Dir(posixpath.dirname(value.rel_path))
    if isinstance(value, _Dir):
        return _Dir(posixpath.dirname(value.rel))
    return UNRESOLVABLE


def _up_n(value: _FoldedValue, n: int) -> _FoldedValue:
    for _ in range(n):
        value = _dirname_of(value)
        if value is UNRESOLVABLE:
            return UNRESOLVABLE
    return value


def _join(base: _FoldedValue, parts: list[_FoldedValue]) -> _FoldedValue:
    if isinstance(base, _FileSelf):
        base = _Dir(posixpath.dirname(base.rel_path))
    if not isinstance(base, _Dir):
        return UNRESOLVABLE
    segs = [base.rel] if base.rel else []
    for p in parts:
        if not isinstance(p, _Lit):
            return UNRESOLVABLE
        segs.append(p.value)
    joined = posixpath.normpath("/".join(segs)) if segs else "."
    if joined == ".":
        joined = ""
    return _Dir(joined)


class _FoldEnv:
    """Symbol table of simple `NAME = <path-expr>` bindings, built with a
    single linear pass over a statement list (module body, or one
    function's body) — no real control-flow tracking, which matches the
    rest of this tool's "linear walk, not a CFG" philosophy."""

    def __init__(self, rel_path: str, parent: Optional["_FoldEnv"] = None) -> None:
        self.rel_path = rel_path
        self.parent = parent
        self.vars: dict[str, _FoldedValue] = {}
        self.sys_aliases: set[str] = set(parent.sys_aliases) if parent else {"sys"}

    def lookup(self, name: str) -> _FoldedValue:
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return UNRESOLVABLE

    def learn_simple_assigns(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name == "sys":
                        self.sys_aliases.add(alias.asname or "sys")
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                folded = self.fold(stmt.value)
                if folded is not UNRESOLVABLE:
                    self.vars[stmt.targets[0].id] = folded

    def fold(self, node: ast.AST) -> _FoldedValue:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return _Lit(node.value)
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return _FileSelf(self.rel_path)
            return self.lookup(node.id)
        if isinstance(node, ast.Call):
            func = node.func
            fname = _dotted_call_name(func)
            if fname in ("os.path.dirname",) and len(node.args) == 1:
                return _dirname_of(self.fold(node.args[0]))
            if fname in ("os.path.abspath", "os.path.normpath", "os.path.realpath", "str") and len(node.args) == 1:
                return self.fold(node.args[0])
            if fname in ("os.path.join",) and node.args:
                base = self.fold(node.args[0])
                rest = [self.fold(a) for a in node.args[1:]]
                return _join(base, rest)
            if fname in ("Path", "pathlib.Path") and len(node.args) == 1:
                return self.fold(node.args[0])
            # `<expr>.resolve()` / `<expr>.absolute()` as a *call* (the
            # common `Path(__file__).resolve()` shape) — fname is None here
            # because _dotted_call_name gives up on a non-Name base, so this
            # has to be checked explicitly rather than falling through.
            if isinstance(func, ast.Attribute) and func.attr in ("resolve", "absolute") and not node.args:
                return self.fold(func.value)
            return UNRESOLVABLE
        if isinstance(node, ast.Attribute):
            if node.attr == "parent":
                return _up_n(self.fold(node.value), 1)
            return UNRESOLVABLE
        if isinstance(node, ast.Subscript):
            base = node.value
            if isinstance(base, ast.Attribute) and base.attr == "parents":
                idx = node.slice
                if isinstance(idx, ast.Constant) and isinstance(idx.value, int):
                    return _up_n(self.fold(base.value), idx.value + 1)
            return UNRESOLVABLE
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = self.fold(node.left)
            right = self.fold(node.right)
            return _join(left, [right])
        return UNRESOLVABLE


def _dotted_call_name(func: ast.AST) -> Optional[str]:
    """Best-effort dotted name of a Call target, e.g. os.path.dirname."""
    parts: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return None
    return ".".join(reversed(parts))


@dataclass
class SysPathInsert:
    inserting_file: str
    lineno: int
    target_dir: str          # repo-relative POSIX dir, "" for repo root
    scope: str                 # "module-level" | "function-local"


def find_sys_path_inserts(sf: SourceFile) -> list[SysPathInsert]:
    if sf.tree is None:
        return []
    out: list[SysPathInsert] = []
    module_env = _FoldEnv(sf.rel_path)
    module_env.learn_simple_assigns(sf.tree.body)

    def scan(stmts: list[ast.stmt], env: _FoldEnv) -> None:
        for stmt in _iter_all(stmts):
            if isinstance(stmt, ast.Call):
                out_ins = _match_path_insert(stmt, env)
                if out_ins is not None:
                    scope = "module-level" if env is module_env else "function-local"
                    out.append(SysPathInsert(sf.rel_path, stmt.lineno, out_ins, scope))

    def _match_path_insert(call: ast.Call, env: _FoldEnv) -> Optional[str]:
        func = call.func
        if not isinstance(func, ast.Attribute):
            return None
        if func.attr not in ("insert", "append"):
            return None
        path_attr = func.value
        if not isinstance(path_attr, ast.Attribute) or path_attr.attr != "path":
            return None
        base = path_attr.value
        if not isinstance(base, ast.Name) or base.id not in env.sys_aliases:
            return None
        if func.attr == "insert":
            if len(call.args) < 2:
                return None
            expr = call.args[1]
        else:
            if not call.args:
                return None
            expr = call.args[0]
        folded = env.fold(expr)
        if isinstance(folded, _FileSelf):
            folded = _Dir(posixpath.dirname(folded.rel_path))
        if not isinstance(folded, _Dir):
            return None
        if folded.rel.startswith(".."):
            return None  # above repo root: cannot be modelled, not a corpus dir
        return folded.rel

    # Module level (direct statements + inside simple if/try wrappers, which
    # is how every real insert in this corpus is written).
    scan(sf.tree.body, module_env)

    # Function-local (any nesting depth): each function gets its own env
    # seeded from the module env plus its own top-of-body simple assigns.
    for node in ast.walk(sf.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_env = _FoldEnv(sf.rel_path, parent=module_env)
            fn_env.learn_simple_assigns(node.body)
            scan(node.body, fn_env)

    return out


def _iter_all(stmts: list[ast.stmt]):
    """Yield every statement/expression node reachable from a statement
    list, without descending into nested function/class defs (those are
    handled as their own scan() call so scope stays correct)."""
    stack = list(stmts)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue  # nested scope; handled separately (or irrelevant)
        for child in ast.iter_child_nodes(node):
            stack.append(child)


# --------------------------------------------------------------------------
# Module table
# --------------------------------------------------------------------------


@dataclass
class RootProvenance:
    root_dir: str            # "" for repo root
    evidence: str              # "implicit: repo root" | "<file>:<line>" | "implicit: <file> is a __main__ entry point"
    scope: str                  # "module-level" | "function-local" | "implicit"
    source_file: Optional[str] = None   # the file whose insert/guard established this root, if any


@dataclass
class ModuleInfo:
    rel_path: str
    kind: str                                    # flat-dir | package-member | script
    importable_as: dict[str, list[RootProvenance]] = field(default_factory=dict)
    has_main: bool = False


def _module_kind(rel_path: str, all_rel_paths: set[str]) -> str:
    """flat-dir: sibling files reached via a bare filename import (no
    __init__.py in this directory). package-member: this directory (or the
    file itself) is a real package, reached via dotted submodule import."""
    base = posixpath.basename(rel_path)
    if base == "__init__.py":
        return "package-member"
    parent = posixpath.dirname(rel_path)
    init_candidate = f"{parent}/__init__.py" if parent else "__init__.py"
    return "package-member" if init_candidate in all_rel_paths else "flat-dir"


def _has_main_guard(tree: ast.Module) -> bool:
    for stmt in tree.body:
        if isinstance(stmt, ast.If):
            test = stmt.test
            if isinstance(test, ast.Compare) and len(test.comparators) == 1:
                left, right = test.left, test.comparators[0]
                names = {n for n in (left, right) if isinstance(n, ast.Name)}
                lits = {n.value for n in (left, right) if isinstance(n, ast.Constant)}
                if any(n.id == "__name__" for n in names) and "__main__" in lits:
                    return True
    return False


def _dotted_for(rel_path: str, root_dir: str) -> Optional[str]:
    """Dotted module name of rel_path as seen from sys.path root root_dir
    ("" = repo root), or None if rel_path is not under root_dir."""
    if root_dir:
        prefix = root_dir + "/"
        if not rel_path.startswith(prefix):
            return None
        sub = rel_path[len(prefix):]
    else:
        sub = rel_path
    if sub.endswith("/__init__.py"):
        sub = sub[: -len("/__init__.py")]
    elif sub.endswith(".py"):
        sub = sub[: -len(".py")]
    if not sub:
        return None
    return sub.replace("/", ".")


class ResolutionTable:
    """Built once per corpus load. Owns per-module importable_as sets and
    resolves `import`/`from ... import` dotted names to a target module,
    an EXTERNAL classification, or UNRESOLVED."""

    def __init__(self, sources: list[SourceFile]) -> None:
        self.sources_by_path: dict[str, SourceFile] = {sf.rel_path: sf for sf in sources}
        all_rel_paths = set(self.sources_by_path)

        self.inserts: list[SysPathInsert] = []
        for sf in sources:
            self.inserts.extend(find_sys_path_inserts(sf))

        # roots: root_dir -> list of RootProvenance (one per distinct insert
        # site producing that dir; repo root is always present, implicit).
        self.roots: dict[str, list[RootProvenance]] = {"": [RootProvenance("", "implicit: repo root", "implicit")]}
        for ins in self.inserts:
            self.roots.setdefault(ins.target_dir, []).append(
                RootProvenance(ins.target_dir, f"{ins.inserting_file}:{ins.lineno}", ins.scope, ins.inserting_file)
            )
        # A directory that contains at least one `if __name__ == "__main__":`
        # entry point is *also* a usable root for every sibling in that same
        # directory: when Python runs a script directly (`python
        # eval/harness.py`), it auto-adds that script's own directory to
        # sys.path[0] — no explicit insert required. This is exactly how
        # `import harness` / `from tasks import TASKS` resolve from
        # eval/grounding_gate_eval.py: eval/harness.py's own __main__ guard
        # is what puts eval/ on sys.path for every file that runs alongside
        # it, not anything grounding_gate_eval.py itself does.
        main_dirs: dict[str, str] = {}
        for sf in sources:
            if sf.tree is not None and _has_main_guard(sf.tree):
                main_dirs.setdefault(posixpath.dirname(sf.rel_path), sf.rel_path)
        for root_dir, entry_file in main_dirs.items():
            self.roots.setdefault(root_dir, []).append(
                RootProvenance(root_dir, f"implicit: {entry_file} is a __main__ entry point", "implicit", entry_file)
            )

        self.modules: dict[str, ModuleInfo] = {}
        self.by_dotted: dict[str, list[tuple[str, RootProvenance]]] = {}
        for sf in sources:
            info = ModuleInfo(rel_path=sf.rel_path, kind=_module_kind(sf.rel_path, all_rel_paths))
            if sf.tree is not None:
                info.has_main = _has_main_guard(sf.tree)
            for root_dir, provs in self.roots.items():
                dotted = _dotted_for(sf.rel_path, root_dir)
                if dotted is None:
                    continue
                info.importable_as.setdefault(dotted, []).extend(provs)
                for prov in provs:
                    self.by_dotted.setdefault(dotted, []).append((sf.rel_path, prov))
            self.modules[sf.rel_path] = info

    # -- queries -----------------------------------------------------------

    def module_for_path(self, rel_path: str) -> Optional[ModuleInfo]:
        return self.modules.get(rel_path)

    def resolve_dotted(self, dotted: str, importer_rel_path: str) -> "ResolveResult":
        importer_dir = posixpath.dirname(importer_rel_path)

        matches = self.by_dotted.get(dotted, [])
        if matches:
            # Preference order for which provenance to cite when several
            # roots answer to the same dotted name:
            #   1. a root THIS FILE ITSELF established (its own literal
            #      sys.path.insert, or its own __main__ guard) — guaranteed
            #      to have executed immediately before this import runs;
            #   2. same-dir (root_dir == importer's own directory) — always
            #      trivially available;
            #   3. first found — a flat import that only resolves because
            #      of an insert written in a DIFFERENT file, surfaced via
            #      cross_file_root below.
            own_file = [m for m in matches if m[1].source_file == importer_rel_path]
            same_dir = [m for m in matches if m[1].root_dir == importer_dir]
            chosen_path, chosen_prov = (own_file or same_dir or matches)[0]
            resolved_via = self._classify(chosen_prov, importer_dir)
            cross_file = (
                chosen_prov.source_file is not None
                and chosen_prov.source_file != importer_rel_path
                and resolved_via.startswith("sys.path.insert")
            )
            unique = len({m[0] for m in matches}) == 1
            return ResolveResult(
                status="corpus", target_path=chosen_path, resolved_via=resolved_via,
                evidence=chosen_prov.evidence, ambiguous=not unique,
                candidates=sorted({m[0] for m in matches}), cross_file_root=cross_file,
            )

        top = dotted.split(".")[0]
        if top in config.stdlib_module_names():
            return ResolveResult(status="stdlib", target_path=None, resolved_via="stdlib",
                                  evidence="", ambiguous=False, candidates=[], cross_file_root=False)
        if top in config.third_party_top_level_names():
            return ResolveResult(status="third-party", target_path=None, resolved_via="third-party",
                                  evidence="", ambiguous=False, candidates=[], cross_file_root=False)
        return ResolveResult(status="unresolved", target_path=None, resolved_via="unresolved",
                              evidence="", ambiguous=False, candidates=[], cross_file_root=False)

    @staticmethod
    def _classify(prov: RootProvenance, importer_dir: str) -> str:
        if prov.root_dir == importer_dir and prov.root_dir != "":
            return "same-dir"
        if prov.evidence == "implicit: repo root":
            return "package"
        if prov.evidence.startswith("implicit:"):
            return "same-dir (via __main__ entry point)"
        return f"sys.path.insert@{prov.evidence}"

    def resolve_relative(self, level: int, module: Optional[str], importer_rel_path: str) -> "ResolveResult":
        base_dir = posixpath.dirname(importer_rel_path)
        target_dir = base_dir
        for _ in range(max(level - 1, 0)):
            target_dir = posixpath.dirname(target_dir)
        dotted = target_dir.replace("/", ".") if target_dir else ""
        if module:
            dotted = f"{dotted}.{module}" if dotted else module
        if not dotted:
            return ResolveResult(status="unresolved", target_path=None, resolved_via="unresolved",
                                  evidence="", ambiguous=False, candidates=[], cross_file_root=False)
        # Relative imports resolve directly against the filesystem, not
        # through the sys.path root table.
        candidate_pkg = f"{dotted.replace('.', '/')}/__init__.py"
        candidate_mod = f"{dotted.replace('.', '/')}.py"
        for candidate in (candidate_mod, candidate_pkg):
            if candidate in self.sources_by_path:
                return ResolveResult(status="corpus", target_path=candidate, resolved_via="relative",
                                      evidence=f"{importer_rel_path} (relative import, level={level})",
                                      ambiguous=False, candidates=[candidate], cross_file_root=False)
        return ResolveResult(status="unresolved", target_path=None, resolved_via="unresolved",
                              evidence="", ambiguous=False, candidates=[], cross_file_root=False)


@dataclass
class ResolveResult:
    status: str                # "corpus" | "stdlib" | "third-party" | "unresolved"
    target_path: Optional[str]
    resolved_via: str
    evidence: str
    ambiguous: bool
    candidates: list[str]
    cross_file_root: bool

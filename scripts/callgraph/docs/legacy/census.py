#!/usr/bin/env python3
"""
AST-based census of call-graph patterns in mcp/graph/** and mcp/memcheck/**
Counts constructs that a call-graph builder must handle.
"""

import ast
import sys
import subprocess
from collections import defaultdict
from typing import List, Dict, Any

def get_committed_file(file_path: str) -> str:
    """Read file from HEAD commit using git show."""
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{file_path}"],
            cwd="/home/user/loci",
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error reading {file_path}: {e.stderr}", file=sys.stderr)
        return None

def find_target_files() -> List[str]:
    """Find Python files in mcp/graph and mcp/memcheck."""
    files = []
    for root_dir in ["mcp/graph", "mcp/memcheck"]:
        result = subprocess.run(
            ["find", root_dir, "-type", "f", "-name", "*.py"],
            cwd="/home/user/loci",
            capture_output=True,
            text=True
        )
        for line in result.stdout.strip().split("\n"):
            if line and line.endswith(".py"):
                files.append(line)
    return sorted(set(files))

class CallGraphCensus(ast.NodeVisitor):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.lineno_offset = 0

        # Census categories
        self.function_defs = []  # (name, lineno, is_method)
        self.decorated_defs = defaultdict(list)  # decorator_name -> [(func_name, lineno)]
        self.direct_calls = []  # (caller_name, callee_name/attr, lineno, call_type)
        self.bare_refs = []  # (context, name/attr, lineno)
        self.imports = []  # (module, name, lineno, is_local)
        self.global_stmts = []  # (names, lineno)
        self.getattr_calls = []  # (obj, attr_name, lineno)
        self.dict_dispatch = []  # (var_name, lineno)
        self.string_literals = []  # (value, lineno)

        self.current_function = None
        self.current_class = None

    def visit_FunctionDef(self, node):
        parent_func = self.current_function
        self.current_function = node.name

        # Record function def
        is_method = self.current_class is not None
        self.function_defs.append((node.name, node.lineno, is_method))

        # Record decorators
        for dec in node.decorator_list:
            dec_name = self._get_decorator_name(dec)
            if dec_name:
                self.decorated_defs[dec_name].append((node.name, node.lineno))

        self.generic_visit(node)
        self.current_function = parent_func

    def visit_AsyncFunctionDef(self, node):
        # Treat like FunctionDef
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        parent_class = self.current_class
        self.current_class = node.name

        self.generic_visit(node)
        self.current_class = parent_class

    def visit_Call(self, node):
        # Categorize calls
        if isinstance(node.func, ast.Name):
            # Direct function call: foo()
            self.direct_calls.append((
                self.current_function or "<module>",
                node.func.id,
                node.lineno,
                "Name"
            ))
        elif isinstance(node.func, ast.Attribute):
            # Method/attr call: obj.method()
            attr_chain = self._get_attr_chain(node.func)
            self.direct_calls.append((
                self.current_function or "<module>",
                attr_chain,
                node.lineno,
                "Attribute"
            ))

        self.generic_visit(node)

    def visit_Name(self, node):
        # Track bare Name references (for tuple/list/dict literals later)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append((
                alias.name,
                alias.asname or alias.name,
                node.lineno,
                False  # not local
            ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append((
                f"{node.module}.{alias.name}" if node.module else alias.name,
                alias.asname or alias.name,
                node.lineno,
                False
            ))
        self.generic_visit(node)

    def visit_Global(self, node):
        self.global_stmts.append((node.names, node.lineno))
        self.generic_visit(node)

    def visit_Dict(self, node):
        # Track dict literals
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                self.dict_dispatch.append((key.value, node.lineno))
        self.generic_visit(node)

    def visit_Tuple(self, node):
        # Track bare function refs in tuples
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                self.bare_refs.append((
                    self.current_function or "<module>",
                    elt.id,
                    node.lineno
                ))
            elif isinstance(elt, ast.Attribute):
                self.bare_refs.append((
                    self.current_function or "<module>",
                    self._get_attr_chain(elt),
                    node.lineno
                ))
        self.generic_visit(node)

    def visit_List(self, node):
        # Track bare function refs in lists
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                self.bare_refs.append((
                    self.current_function or "<module>",
                    elt.id,
                    node.lineno
                ))
        self.generic_visit(node)

    def visit_Constant(self, node):
        # Track string literals
        if isinstance(node.value, str):
            # Look for path-like or collection-name patterns
            val = node.value
            if "/" in val or val.endswith(".py") or val.startswith("_"):
                self.string_literals.append((val, node.lineno))
        self.generic_visit(node)

    def _get_decorator_name(self, node) -> str:
        """Extract decorator name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attr_chain(node)
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return None

    def _get_attr_chain(self, node) -> str:
        """Get full attribute chain."""
        if isinstance(node, ast.Attribute):
            return self._get_attr_chain(node.value) + "." + node.attr
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return "<expr>"

def analyze_file(filepath: str) -> Dict[str, Any]:
    """Analyze a single Python file."""
    content = get_committed_file(filepath)
    if content is None:
        return None

    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}", file=sys.stderr)
        return None

    census = CallGraphCensus(filepath)
    census.visit(tree)

    return {
        "filepath": filepath,
        "function_defs": census.function_defs,
        "decorated_defs": dict(census.decorated_defs),
        "direct_calls": census.direct_calls,
        "bare_refs": census.bare_refs,
        "imports": census.imports,
        "global_stmts": census.global_stmts,
        "getattr_calls": census.getattr_calls,
        "dict_dispatch": census.dict_dispatch,
        "string_literals": census.string_literals,
    }

def main():
    files = find_target_files()
    print(f"Found {len(files)} target files")
    print("=" * 80)

    all_results = []

    for filepath in files:
        result = analyze_file(filepath)
        if result:
            all_results.append(result)

    # Aggregate stats
    total_functions = sum(len(r["function_defs"]) for r in all_results)
    total_decorated = sum(len(defs) for r in all_results for defs in r["decorated_defs"].values())
    total_calls = sum(len(r["direct_calls"]) for r in all_results)
    total_bare_refs = sum(len(r["bare_refs"]) for r in all_results)
    total_imports = sum(len(r["imports"]) for r in all_results)
    total_globals = sum(len(r["global_stmts"]) for r in all_results)
    total_strings = sum(len(r["string_literals"]) for r in all_results)

    print(f"\n### CENSUS SUMMARY ###")
    print(f"Total files analyzed: {len(all_results)}")
    print(f"Plain function/method defs: {total_functions}")
    print(f"Decorated function defs: {total_decorated}")
    print(f"Direct calls (Name + Attribute): {total_calls}")
    print(f"Bare function references in literals: {total_bare_refs}")
    print(f"Import statements: {total_imports}")
    print(f"Global statements: {total_globals}")
    print(f"String literals (path-like or special): {total_strings}")

    # Decorator breakdown
    print(f"\n### DECORATORS (by type) ###")
    all_decorators = defaultdict(list)
    for r in all_results:
        for dec_name, defs in r["decorated_defs"].items():
            all_decorators[dec_name].extend([(r["filepath"], d) for d in defs])

    for dec_name in sorted(all_decorators.keys()):
        count = len(all_decorators[dec_name])
        print(f"{dec_name}: {count} occurrences")
        for filepath, (func_name, lineno) in all_decorators[dec_name][:3]:
            print(f"  - {filepath}:{lineno} {func_name}")

    # Call types breakdown
    print(f"\n### CALL TYPES ###")
    call_counts = defaultdict(int)
    for r in all_results:
        for caller, callee, lineno, ctype in r["direct_calls"]:
            call_counts[ctype] += 1

    for ctype in sorted(call_counts.keys()):
        print(f"{ctype}: {call_counts[ctype]} calls")

    # Sample direct calls
    print(f"\n### DIRECT CALL EXAMPLES ###")
    call_samples = []
    for r in all_results:
        for caller, callee, lineno, ctype in r["direct_calls"][:5]:
            call_samples.append((r["filepath"], lineno, caller, callee, ctype))

    for filepath, lineno, caller, callee, ctype in call_samples[:10]:
        print(f"{filepath}:{lineno} | {caller}() -> {callee}() [{ctype}]")

    # Sample bare refs
    print(f"\n### BARE FUNCTION REFERENCES ###")
    bare_samples = []
    for r in all_results:
        for context, name, lineno in r["bare_refs"][:3]:
            bare_samples.append((r["filepath"], lineno, context, name))

    for filepath, lineno, context, name in bare_samples[:10]:
        print(f"{filepath}:{lineno} | {context} uses {name} (bare ref)")

    # Import breakdown
    print(f"\n### IMPORTS ###")
    import_types = defaultdict(int)
    for r in all_results:
        for mod, name, lineno, is_local in r["imports"]:
            import_types["total"] += 1

    print(f"Total imports: {import_types['total']}")

    # Sample imports
    print(f"\nSample imports:")
    for r in all_results:
        for mod, name, lineno, is_local in r["imports"][:3]:
            print(f"  {r['filepath']}:{lineno} import {name} (from {mod})")

    # String literals
    print(f"\n### STRING LITERALS (paths/special) ###")
    print(f"Total path-like strings: {total_strings}")

    string_samples = []
    for r in all_results:
        for val, lineno in r["string_literals"][:2]:
            string_samples.append((r["filepath"], lineno, val))

    for filepath, lineno, val in string_samples[:10]:
        print(f"{filepath}:{lineno} | {repr(val)}")

    # Global statements
    print(f"\n### GLOBAL STATEMENTS ###")
    print(f"Total global declarations: {total_globals}")

    global_samples = []
    for r in all_results:
        for names, lineno in r["global_stmts"]:
            global_samples.append((r["filepath"], lineno, names))

    for filepath, lineno, names in global_samples[:5]:
        print(f"{filepath}:{lineno} | global {', '.join(names)}")

if __name__ == "__main__":
    main()

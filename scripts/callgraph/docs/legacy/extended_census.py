#!/usr/bin/env python3
"""
Extended AST census for dispatch patterns across mcp/, scripts/, a2a_server/, mlops/, eval/

Focus on registration roots and cross-module dispatch mechanisms:
  - @mcp.tool() decorators
  - register(mcp, deps) patterns
  - _SKILL_MAP and dict-of-callables
  - FastAPI route decorators
  - lazy/local imports
  - importlib.import_module calls
  - module-level name assignments read by other modules
"""

import ast
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
    except subprocess.CalledProcessError:
        return None

def find_target_files() -> List[str]:
    """Find Python files in mcp/, scripts/, a2a_server/, mlops/, eval/."""
    files = set()
    for root_dir in ["mcp", "scripts", "a2a_server", "mlops", "eval"]:
        result = subprocess.run(
            ["find", root_dir, "-type", "f", "-name", "*.py"],
            cwd="/home/user/loci",
            capture_output=True,
            text=True
        )
        for line in result.stdout.strip().split("\n"):
            if line and line.endswith(".py"):
                files.add(line)
    return sorted(files)

class DispatchCensus(ast.NodeVisitor):
    def __init__(self, filepath: str):
        self.filepath = filepath

        # MCP tool registration
        self.mcp_tool_defs = []  # (func_name, lineno)

        # register() function calls
        self.register_calls = []  # (lineno, args_shape)

        # Dict-of-callables assignments
        self.dict_dispatch_assigns = []  # (var_name, lineno)

        # FastAPI decorators
        self.fastapi_routes = []  # (method, path, lineno)

        # Local imports (inside functions)
        self.local_imports = []  # (module, name, func_context, lineno)

        # importlib usage
        self.importlib_calls = []  # (call_type, module, lineno)

        # Module-level assignments
        self.module_level_assigns = []  # (var_name, value_type, lineno)

        # _SKILL_MAP or similar patterns
        self.skill_maps = []  # (var_name, lineno)

        # Bare function refs in tuples (registration manifests)
        self.tuple_refs = []  # (context, names, lineno)

        self.current_function = None

    def visit_FunctionDef(self, node):
        parent_func = self.current_function
        self.current_function = node.name

        # Check for @mcp.tool() decorator
        for dec in node.decorator_list:
            if self._is_mcp_tool(dec):
                self.mcp_tool_defs.append((node.name, node.lineno))

        self.generic_visit(node)
        self.current_function = parent_func

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Call(self, node):
        # Look for register() calls
        if isinstance(node.func, ast.Name) and node.func.id == "register":
            args_shape = self._analyze_call_args(node)
            self.register_calls.append((node.lineno, args_shape))

        # Look for importlib.import_module()
        elif isinstance(node.func, ast.Attribute):
            chain = self._get_attr_chain(node.func)
            if "importlib" in chain and "import_module" in chain:
                module_arg = self._extract_string_arg(node)
                self.importlib_calls.append(("import_module", module_arg, node.lineno))
            elif "importlib" in chain and "import_" in chain:
                module_arg = self._extract_string_arg(node)
                self.importlib_calls.append((chain.split(".")[-1], module_arg, node.lineno))

        self.generic_visit(node)

    def visit_Assign(self, node):
        # Module-level assignments (potential dict-of-callables)
        if self.current_function is None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id

                    # Check if it's a dict
                    if isinstance(node.value, ast.Dict):
                        self.dict_dispatch_assigns.append((var_name, node.lineno))
                        # Check if this looks like _SKILL_MAP
                        if "_MAP" in var_name or "SKILL" in var_name or "HANDLER" in var_name:
                            self.skill_maps.append((var_name, node.lineno))

                    # Record all module-level assignments
                    value_type = self._get_value_type(node.value)
                    self.module_level_assigns.append((var_name, value_type, node.lineno))

        self.generic_visit(node)

    def visit_Import(self, node):
        # Check if this is a local import (inside a function)
        if self.current_function is not None:
            for alias in node.names:
                self.local_imports.append((
                    alias.name,
                    alias.asname or alias.name,
                    self.current_function,
                    node.lineno
                ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # Check if this is a local import
        if self.current_function is not None:
            for alias in node.names:
                self.local_imports.append((
                    f"{node.module}.{alias.name}" if node.module else alias.name,
                    alias.asname or alias.name,
                    self.current_function,
                    node.lineno
                ))
        self.generic_visit(node)

    def visit_Tuple(self, node):
        # Check for bare function refs (registration manifests)
        names = []
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            elif isinstance(elt, ast.Attribute):
                names.append(self._get_attr_chain(elt))

        if names and len(names) >= 2:  # Multi-item tuple (likely registration)
            self.tuple_refs.append((
                self.current_function or "<module>",
                names,
                node.lineno
            ))

        self.generic_visit(node)

    def _is_mcp_tool(self, node) -> bool:
        """Check if decorator is @mcp.tool()."""
        if isinstance(node, ast.Call):
            node = node.func

        if isinstance(node, ast.Attribute):
            chain = self._get_attr_chain(node)
            return "mcp" in chain and "tool" in chain
        elif isinstance(node, ast.Name):
            return node.id == "tool" or "mcp" in node.id

        return False

    def _analyze_call_args(self, node) -> str:
        """Analyze arguments to a call."""
        arg_types = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                arg_types.append(f"Name:{arg.id}")
            elif isinstance(arg, ast.Tuple):
                arg_types.append(f"Tuple[{len(arg.elts)}]")
            elif isinstance(arg, ast.Dict):
                arg_types.append(f"Dict[{len(arg.keys)}]")
            else:
                arg_types.append(type(arg).__name__)

        for kw in node.keywords:
            if isinstance(kw.value, ast.Name):
                arg_types.append(f"{kw.arg}=Name:{kw.value.id}")
            else:
                arg_types.append(f"{kw.arg}={type(kw.value).__name__}")

        return ", ".join(arg_types) if arg_types else "(no args)"

    def _extract_string_arg(self, node) -> str:
        """Extract first string argument from a call."""
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
        return None

    def _get_attr_chain(self, node) -> str:
        """Get full attribute chain."""
        if isinstance(node, ast.Attribute):
            return self._get_attr_chain(node.value) + "." + node.attr
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return "<expr>"

    def _get_value_type(self, node) -> str:
        """Get the type of a value."""
        if isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Tuple):
            return "tuple"
        elif isinstance(node, ast.Call):
            return "call"
        elif isinstance(node, ast.Name):
            return f"Name:{node.id}"
        elif isinstance(node, ast.Constant):
            return f"const:{type(node.value).__name__}"
        else:
            return type(node).__name__

def analyze_file(filepath: str) -> Dict[str, Any]:
    """Analyze a single Python file."""
    content = get_committed_file(filepath)
    if content is None:
        return None

    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError:
        return None

    census = DispatchCensus(filepath)
    census.visit(tree)

    return {
        "filepath": filepath,
        "mcp_tools": census.mcp_tool_defs,
        "register_calls": census.register_calls,
        "dict_dispatch": census.dict_dispatch_assigns,
        "fastapi_routes": census.fastapi_routes,
        "local_imports": census.local_imports,
        "importlib_calls": census.importlib_calls,
        "module_assigns": census.module_level_assigns,
        "skill_maps": census.skill_maps,
        "tuple_refs": census.tuple_refs,
    }

def main():
    files = find_target_files()
    print(f"Found {len(files)} target files across full corpus")
    print("=" * 80)

    all_results = []

    for filepath in files:
        result = analyze_file(filepath)
        if result:
            all_results.append(result)

    # Aggregate stats
    total_mcp_tools = sum(len(r["mcp_tools"]) for r in all_results)
    total_register = sum(len(r["register_calls"]) for r in all_results)
    total_dict_dispatch = sum(len(r["dict_dispatch"]) for r in all_results)
    total_local_imports = sum(len(r["local_imports"]) for r in all_results)
    total_importlib = sum(len(r["importlib_calls"]) for r in all_results)
    total_skill_maps = sum(len(r["skill_maps"]) for r in all_results)
    total_tuple_refs = sum(len(r["tuple_refs"]) for r in all_results)

    print(f"\n### REGISTRATION & DISPATCH PATTERNS ###")
    print(f"@mcp.tool() decorated functions: {total_mcp_tools}")
    print(f"register() function calls: {total_register}")
    print(f"Dict-of-callables assignments: {total_dict_dispatch}")
    print(f"_SKILL_MAP and similar patterns: {total_skill_maps}")
    print(f"Bare function refs in tuples: {total_tuple_refs}")

    print(f"\n### LAZY / DYNAMIC IMPORT PATTERNS ###")
    print(f"Local imports (inside functions): {total_local_imports}")
    print(f"importlib.import_module() calls: {total_importlib}")

    # Show @mcp.tool() examples
    print(f"\n### @mcp.tool() DECORATORS ###")
    mcp_samples = []
    for r in all_results:
        for func_name, lineno in r["mcp_tools"]:
            mcp_samples.append((r["filepath"], lineno, func_name))

    for filepath, lineno, func_name in mcp_samples[:15]:
        print(f"{filepath}:{lineno} | @mcp.tool() def {func_name}()")

    # Show register() calls
    print(f"\n### register() FUNCTION CALLS ###")
    reg_samples = []
    for r in all_results:
        for lineno, args_shape in r["register_calls"]:
            reg_samples.append((r["filepath"], lineno, args_shape))

    for filepath, lineno, args_shape in reg_samples[:10]:
        print(f"{filepath}:{lineno} | register({args_shape})")

    # Show dict dispatch
    print(f"\n### DICT-OF-CALLABLES ASSIGNMENTS ###")
    dict_samples = []
    for r in all_results:
        for var_name, lineno in r["dict_dispatch"]:
            dict_samples.append((r["filepath"], lineno, var_name))

    for filepath, lineno, var_name in dict_samples[:10]:
        print(f"{filepath}:{lineno} | {var_name} = {{...}}")

    # Show _SKILL_MAP patterns
    print(f"\n### _SKILL_MAP AND SIMILAR PATTERNS ###")
    skill_samples = []
    for r in all_results:
        for var_name, lineno in r["skill_maps"]:
            skill_samples.append((r["filepath"], lineno, var_name))

    for filepath, lineno, var_name in skill_samples[:10]:
        print(f"{filepath}:{lineno} | {var_name} = {{...}}")

    # Show local imports
    print(f"\n### LOCAL IMPORTS (INSIDE FUNCTIONS) ###")
    local_samples = []
    for r in all_results:
        for mod, name, func, lineno in r["local_imports"][:5]:
            local_samples.append((r["filepath"], lineno, func, mod, name))

    for filepath, lineno, func, mod, name in local_samples[:15]:
        print(f"{filepath}:{lineno} | inside {func}(): import {name} (from {mod})")

    # Show importlib usage
    print(f"\n### importlib USAGE ###")
    importlib_samples = []
    for r in all_results:
        for call_type, module, lineno in r["importlib_calls"]:
            importlib_samples.append((r["filepath"], lineno, call_type, module))

    for filepath, lineno, call_type, module in importlib_samples[:10]:
        print(f"{filepath}:{lineno} | importlib.{call_type}('{module}')")

    # Show tuple refs (registration manifests)
    print(f"\n### BARE FUNCTION REFS IN TUPLES (REGISTRATION MANIFESTS) ###")
    tuple_samples = []
    for r in all_results:
        for context, names, lineno in r["tuple_refs"][:5]:
            tuple_samples.append((r["filepath"], lineno, context, names))

    for filepath, lineno, context, names in tuple_samples[:10]:
        names_str = ", ".join(names)
        print(f"{filepath}:{lineno} | {context}: ({names_str})")

    # Count by directory
    print(f"\n### PATTERN COUNTS BY DIRECTORY ###")
    by_dir = defaultdict(lambda: {
        "mcp_tools": 0, "register": 0, "dict": 0, "skill": 0,
        "local_import": 0, "importlib": 0, "tuple_ref": 0
    })

    for r in all_results:
        dir_name = r["filepath"].split("/")[0]
        by_dir[dir_name]["mcp_tools"] += len(r["mcp_tools"])
        by_dir[dir_name]["register"] += len(r["register_calls"])
        by_dir[dir_name]["dict"] += len(r["dict_dispatch"])
        by_dir[dir_name]["skill"] += len(r["skill_maps"])
        by_dir[dir_name]["local_import"] += len(r["local_imports"])
        by_dir[dir_name]["importlib"] += len(r["importlib_calls"])
        by_dir[dir_name]["tuple_ref"] += len(r["tuple_refs"])

    for dir_name in sorted(by_dir.keys()):
        counts = by_dir[dir_name]
        print(f"\n{dir_name}/:")
        if counts["mcp_tools"]:
            print(f"  @mcp.tool(): {counts['mcp_tools']}")
        if counts["register"]:
            print(f"  register(): {counts['register']}")
        if counts["dict"]:
            print(f"  dict-dispatch: {counts['dict']}")
        if counts["skill"]:
            print(f"  skill-maps: {counts['skill']}")
        if counts["local_import"]:
            print(f"  local-imports: {counts['local_import']}")
        if counts["importlib"]:
            print(f"  importlib: {counts['importlib']}")
        if counts["tuple_ref"]:
            print(f"  tuple-refs: {counts['tuple_ref']}")

if __name__ == "__main__":
    main()

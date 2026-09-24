"""swarm_operation_layer's runtime contract must name code that exists.

The module is documentation-shaped data: every ``runtime_hooks`` value and
``runtime_entry`` is a dotted path into scripts/ or mcp/. Nothing executes those
strings, so nothing noticed when two of them named functions that never existed
(``SwarmConfig.decompose`` and ``_cheap_answer``). This resolves each one against
the source with ``ast`` -- no import, so the heavy swarm modules stay unloaded.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "swarm_operation_layer_uut", _REPO / "scripts" / "swarm_operation_layer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _top_level(tree: ast.Module) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _resolves(dotted: str) -> bool:
    """True when ``pkg.module.Name[.member]`` names a def/class in the repo."""
    parts = dotted.split(".")
    for split in range(len(parts) - 1, 0, -1):
        path = _REPO.joinpath(*parts[:split]).with_suffix(".py")
        if path.is_file():
            break
    else:
        return False
    names = parts[split:]
    scope = _top_level(ast.parse(path.read_text(encoding="utf-8")))
    node = scope.get(names[0])
    if node is None:
        return False
    for name in names[1:]:
        if not isinstance(node, ast.ClassDef):
            return False
        members = {
            n.name: n
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        node = members.get(name)
        if node is None:
            return False
    return True


def _contract_paths(contract: dict) -> list[str]:
    paths = list(contract["supervisor"]["runtime_hooks"].values())
    paths += [entry["runtime_entry"] for entry in contract["ensemble"]["composition"]]
    return paths


def test_runtime_contract_exposes_every_layer():
    contract = _load().runtime_contract()
    assert set(contract) == {"supervisor", "ensemble", "prompt_templates", "observability"}


def test_every_runtime_hook_and_entry_names_real_code():
    paths = _contract_paths(_load().runtime_contract())
    assert paths
    missing = [p for p in paths if not _resolves(p)]
    assert not missing, f"runtime contract names code that does not exist: {missing}"


def test_resolver_rejects_a_name_that_does_not_exist():
    # Guards the guard: a resolver that returns True for everything passes above.
    assert _resolves("scripts.swarm_escalate.escalate_findings")
    assert not _resolves("scripts.swarm_escalate._cheap_answer")
    assert not _resolves("scripts.swarm_escalate.SwarmConfig.decompose")

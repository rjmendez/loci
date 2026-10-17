"""A module-level `A = mod.B` must not crash the scope analyser.

ModuleAssign lost its two attribute fields to a cleanup that reasoned imports.py
only resolves value_kind == "name". The call site kept passing them, so every
attribute alias raised TypeError. It never fired because the corpus contained
exactly one such alias, in a test file — adding two re-exports to
mlops/grounding/train.py took this suite from 3 failures to 19 failures and 32
errors, which is how it was found.
"""
import ast
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from callgraph.scopes import ModuleAssign, ModuleScope  # noqa: E402
from callgraph.ingest import SourceFile  # noqa: E402


def test_the_attribute_branch_can_be_constructed_at_all():
    """The regression, in one line. Six positional args against four fields."""
    a = ModuleAssign(1, "alias", "attribute", None, "mod", "attr")
    assert a.value_kind == "attribute"
    assert (a.value_module, a.value_attr) == ("mod", "attr")


def test_the_name_branch_is_unchanged():
    a = ModuleAssign(1, "alias", "name", "other")
    assert a.value_name == "other"
    assert a.value_module is None and a.value_attr is None


def test_the_call_site_and_the_dataclass_agree():
    """The two drifted apart silently. Compare arity rather than trusting either."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "scopes.py").read_text()
    tree = ast.parse(src)
    n_fields = len(dataclasses.fields(ModuleAssign))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "ModuleAssign":
            assert len(node.args) <= n_fields, (
                f"scopes.py:{node.lineno} constructs ModuleAssign with "
                f"{len(node.args)} positional args; the dataclass has {n_fields} fields"
            )


def test_a_real_module_with_an_attribute_alias_parses(tmp_path):
    """End to end: the shape that broke it, through the actual analyser."""
    p = tmp_path / "m.py"
    p.write_text("import os\n\n_j = os.path\n_basename = os.path.basename\n\n"
                 "def f():\n    return _j\n")
    src = p.read_text()
    tree = ast.parse(src)
    sf = SourceFile(rel_path="m.py", source=src, tree=tree, error=None, origin="working tree")
    scope = ModuleScope(sf)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            scope._record_module_alias(node.targets[0].id, node.value, node.lineno)
    kinds = {a.value_kind for a in scope.module_assigns}
    assert "attribute" in kinds, f"no attribute alias recorded: {scope.module_assigns}"
    attr = next(a for a in scope.module_assigns if a.value_kind == "attribute")
    assert (attr.value_module, attr.value_attr) == ("os", "path")

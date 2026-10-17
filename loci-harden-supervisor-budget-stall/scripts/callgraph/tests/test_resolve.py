from .conftest import needs_corpus_deps, needs_git_history  # noqa: F401
"""resolve.py: sys.path.insert constant-folding + the resolution table,
against both synthetic fixtures and the real corpus's known cases."""
from ..resolve import ResolutionTable, find_sys_path_inserts
from ..tests.helpers import load_fixture, source_file


# -- constant-folding of sys.path.insert -------------------------------------

def test_fold_dirname_join_pattern():
    sf = load_fixture("pkg/scripts/entry.py")
    inserts = find_sys_path_inserts(sf)
    assert len(inserts) == 1
    ins = inserts[0]
    assert ins.target_dir == "pkg/mcp"
    assert ins.inserting_file == "pkg/scripts/entry.py"
    assert ins.lineno == 8
    assert ins.scope == "module-level"


def test_fold_this_dir_variable_and_parent_chain():
    text = (
        "from pathlib import Path\n"
        "import sys\n"
        "_THIS_DIR = str(Path(__file__).resolve().parent)\n"
        "if _THIS_DIR not in sys.path:\n"
        "    sys.path.insert(0, _THIS_DIR)\n"
    )
    sf = source_file("mcp/server.py", text)
    inserts = find_sys_path_inserts(sf)
    assert len(inserts) == 1
    assert inserts[0].target_dir == "mcp"


def test_fold_parents_subscript_and_div_operator():
    text = (
        "import sys\n"
        "from pathlib import Path\n"
        "def f():\n"
        "    _scripts = str(Path(__file__).resolve().parent.parent / 'scripts')\n"
        "    sys.path.insert(0, _scripts)\n"
    )
    sf = source_file("mcp/server.py", text)
    inserts = find_sys_path_inserts(sf)
    assert len(inserts) == 1
    assert inserts[0].target_dir == "scripts"
    assert inserts[0].scope == "function-local"


def test_fold_sys_import_alias_is_recognized():
    text = (
        "import sys as _sys\n"
        "import os\n"
        "def f():\n"
        "    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mcp'))\n"
    )
    sf = source_file("scripts/x.py", text)
    inserts = find_sys_path_inserts(sf)
    assert len(inserts) == 1
    assert inserts[0].target_dir == "mcp"


def test_fold_gives_up_on_non_literal_expressions():
    text = (
        "import sys, os\n"
        "sys.path.insert(0, os.environ['SOME_DIR'])\n"
    )
    sf = source_file("scripts/x.py", text)
    assert find_sys_path_inserts(sf) == []


# -- ResolutionTable on a small synthetic corpus -----------------------------

def _synthetic_table() -> ResolutionTable:
    sources = [load_fixture("pkg/scripts/entry.py"), load_fixture("pkg/mcp/sibling.py")]
    return ResolutionTable(sources)


def test_flat_import_resolves_via_named_insert():
    table = _synthetic_table()
    res = table.resolve_dotted("sibling", "pkg/scripts/entry.py")
    assert res.status == "corpus"
    assert res.target_path == "pkg/mcp/sibling.py"
    assert res.resolved_via == "sys.path.insert@pkg/scripts/entry.py:8"
    assert res.ambiguous is False


def test_same_dir_import_needs_no_insert():
    sources = [load_fixture("reexport.py"), load_fixture("reexport_source.py")]
    table = ResolutionTable(sources)
    res = table.resolve_dotted("reexport_source", "reexport.py")
    assert res.status == "corpus"
    assert res.target_path == "reexport_source.py"


def test_stdlib_and_unresolved_classification():
    table = _synthetic_table()
    assert table.resolve_dotted("os", "pkg/scripts/entry.py").status == "stdlib"
    assert table.resolve_dotted("totally_made_up_thing_xyz", "pkg/scripts/entry.py").status == "unresolved"


# -- ResolutionTable against the real corpus: the step-1 acceptance case ----

def test_graph_facts_insert_is_named_at_its_own_line(head_table):
    res = head_table.resolve_dotted("graph.ladybug_store", "scripts/graph_facts.py")
    assert res.status == "corpus"
    assert res.target_path == "mcp/graph/ladybug_store.py"
    assert res.resolved_via == "sys.path.insert@scripts/graph_facts.py:21"


def test_shadow_eval_import_server_resolves_to_mcp_not_a2a(head_table):
    res = head_table.resolve_dotted("server", "scripts/shadow_eval.py")
    assert res.status == "corpus"
    assert res.target_path == "mcp/server.py"


def test_every_flat_mcp_sibling_import_in_server_resolves_same_dir(head_table):
    for name, expected in [
        ("qdrant_ops", "mcp/qdrant_ops.py"),
        ("graph_tools", "mcp/graph_tools.py"),
        ("investigation_tools", "mcp/investigation_tools.py"),
        ("llm_tools", "mcp/llm_tools.py"),
        ("ladybug_ops", "mcp/ladybug_ops.py"),
    ]:
        res = head_table.resolve_dotted(name, "mcp/server.py")
        assert res.status == "corpus" and res.target_path == expected, name
        assert res.resolved_via == "same-dir"


@needs_corpus_deps
def test_real_corpus_import_unresolved_list_is_small_and_only_missing_third_party(head_sources, head_table):
    sources, table = head_sources, head_table
    unresolved = []
    from ..scopes import ModuleScope
    for sf in sources:
        scope = ModuleScope(sf)
        for rec in scope.imports:
            if rec.is_from:
                dotted = rec.module or ""
            else:
                dotted = rec.names[0][0] if rec.names else ""
            if not dotted or rec.level > 0:
                continue
            res = table.resolve_dotted(dotted, sf.rel_path)
            if res.status == "unresolved":
                unresolved.append((sf.rel_path, dotted))
    # The unresolved list must stay readable in one screen; a resolver regression blows past it.
    assert len(unresolved) < 20, unresolved

"""Tests for graph.code_parse — tree-sitter symbol/reference extraction."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.code_parse import (  # noqa: E402
    LANG_BY_EXT,
    detect_lang,
    parse_source,
    parse_path,
)


PY_SNIPPET = b'''
import os
from a.b import c

def helper():
    return 1

class ClassA:
    def method_b(self):
        helper()
        return os.getpid()
'''

JAVA_SNIPPET = b'''
import java.util.List;

class Foo {
    void bar() {
        baz();
        this.qux();
    }
}
'''


def _kinds(result):
    return {s["kind"] for s in result["symbols"]}


def _edge_types(result):
    return {e["type"] for e in result["edges"]}


def test_python_symbols_and_edges():
    res = parse_source("snippet.py", PY_SNIPPET)
    assert res["lang"] == "python"

    kinds = _kinds(res)
    assert "class" in kinds, res["symbols"]
    assert "method" in kinds, res["symbols"]

    names = {s["name"] for s in res["symbols"]}
    assert "ClassA" in names
    assert "method_b" in names

    # method_b qualname should be dotted under the class
    method = next(s for s in res["symbols"] if s["name"] == "method_b")
    assert method["id"].endswith("ClassA.method_b")
    assert method["line"] >= 1

    et = _edge_types(res)
    assert "DEFINES" in et
    assert "CALLS" in et
    assert "IMPORTS" in et

    calls = {e["dst"] for e in res["edges"] if e["type"] == "CALLS"}
    assert "helper" in calls

    imports = set(res["imports"])
    assert any("os" in m or "a.b" in m for m in imports), imports
    imp_edges = {e["dst"] for e in res["edges"] if e["type"] == "IMPORTS"}
    assert imp_edges


def test_java_symbols():
    res = parse_source("Foo.java", JAVA_SNIPPET)
    assert res["lang"] == "java"

    kinds = _kinds(res)
    assert "class" in kinds, res["symbols"]
    assert "method" in kinds, res["symbols"]

    names = {s["name"] for s in res["symbols"]}
    assert "Foo" in names
    assert "bar" in names

    calls = {e["dst"] for e in res["edges"] if e["type"] == "CALLS"}
    assert "baz" in calls


def test_fail_open_garbage_bytes():
    res = parse_source("weird.py", b"\x00\x01\x02 def (((((not python")
    assert res["file"] == "weird.py"
    assert res["lang"] == "python"
    assert isinstance(res["symbols"], list)
    assert isinstance(res["edges"], list)
    assert isinstance(res["imports"], list)


def test_fail_open_unknown_extension():
    res = parse_source("data.unknownext", b"whatever content here")
    assert res["lang"] is None
    assert res["symbols"] == []
    assert res["edges"] == []
    assert res["imports"] == []


def test_detect_lang_and_ext_map():
    assert detect_lang("x.py") == "python"
    assert detect_lang("x.tsx") == "tsx"
    assert detect_lang("x.nope") is None
    assert LANG_BY_EXT[".go"] == "go"
    assert LANG_BY_EXT[".rs"] == "rust"


def test_parse_path_walks_dir(tmp_path):
    (tmp_path / "a.py").write_bytes(PY_SNIPPET)
    (tmp_path / "Foo.java").write_bytes(JAVA_SNIPPET)
    (tmp_path / "notes.txt").write_bytes(b"ignore me")
    skip = tmp_path / "__pycache__"
    skip.mkdir()
    (skip / "b.py").write_bytes(PY_SNIPPET)

    results = parse_path(str(tmp_path))
    langs = {r["lang"] for r in results}
    assert "python" in langs
    assert "java" in langs
    # ignored dir + non-source file excluded
    files = {os.path.basename(r["file"]) for r in results}
    assert "notes.txt" not in files
    assert "b.py" not in files

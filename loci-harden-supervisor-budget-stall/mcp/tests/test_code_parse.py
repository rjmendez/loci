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
import android.util.Log;

class Foo {
    void bar() {
        baz();
        this.qux();
        Log.w(x);
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


def _calls(result):
    return [e for e in result["edges"] if e["type"] == "CALLS"]


def test_java_call_receivers():
    res = parse_source("Foo.java", JAVA_SNIPPET)
    by_dst = {e["dst"]: e for e in _calls(res)}

    # Log.w(x) -> receiver "Log", recv_kind "name"
    assert by_dst["w"]["receiver"] == "Log"
    assert by_dst["w"]["recv_kind"] == "name"

    # this.qux() -> recv_kind "self"
    assert by_dst["qux"]["recv_kind"] == "self"
    assert by_dst["qux"]["receiver"] == "this"

    # bare baz() -> recv_kind "none", receiver None
    assert by_dst["baz"]["recv_kind"] == "none"
    assert by_dst["baz"]["receiver"] is None

    # every CALLS edge carries both keys
    for e in _calls(res):
        assert "receiver" in e
        assert "recv_kind" in e


def test_java_import_map():
    res = parse_source("Foo.java", JAVA_SNIPPET)
    imap = res["import_map"]
    assert imap["Log"] == "android.util.Log"
    assert imap["List"] == "java.util.List"


def test_python_call_receivers_and_import_map():
    res = parse_source("snippet.py", PY_SNIPPET)
    by_dst = {e["dst"]: e for e in _calls(res)}

    # bare helper() -> none
    assert by_dst["helper"]["recv_kind"] == "none"
    assert by_dst["helper"]["receiver"] is None

    # os.getpid() -> receiver "os", recv_kind "name"
    assert by_dst["getpid"]["receiver"] == "os"
    assert by_dst["getpid"]["recv_kind"] == "name"

    imap = res["import_map"]
    assert imap["os"] == "os"
    assert imap["c"] == "a.b.c"


def test_import_map_present_and_fail_open():
    # always a dict, even on garbage / unknown lang
    assert parse_source("weird.py", b"\x00 def (((").get("import_map") == {}
    assert parse_source("data.unknownext", b"x")["import_map"] == {}


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


DECL_JAVA = b'''
class Foo {
    private FooService svc;
    private List<Bar> items;
    void handle(BarClient c) {
        Baz b = new Baz();
    }
}
'''


def test_java_decls_field_param_local():
    res = parse_source("Foo.java", DECL_JAVA)
    decls = res["decls"]
    class_id = next(s["id"] for s in res["symbols"] if s["name"] == "Foo")
    method_id = next(s["id"] for s in res["symbols"] if s["name"] == "handle")

    # every decl has the required shape
    for d in decls:
        assert set(d) == {"name", "type", "scope", "scope_kind"}, d
        assert d["scope_kind"] in ("field", "local", "param")

    by_name = {d["name"]: d for d in decls}

    # field: private FooService svc; -> scoped to the CLASS id
    svc = by_name["svc"]
    assert svc == {
        "name": "svc",
        "type": "FooService",
        "scope": class_id,
        "scope_kind": "field",
    }

    # generics stripped: List<Bar> -> "List"
    assert by_name["items"]["type"] == "List"
    assert by_name["items"]["scope_kind"] == "field"

    # param: void handle(BarClient c) -> type "BarClient", scoped to the METHOD id
    c = by_name["c"]
    assert c["type"] == "BarClient"
    assert c["scope"] == method_id
    assert c["scope_kind"] == "param"

    # local: Baz b = new Baz(); -> type "Baz", scoped to the METHOD id
    b = by_name["b"]
    assert b["type"] == "Baz"
    assert b["scope"] == method_id
    assert b["scope_kind"] == "local"


def test_java_var_local_infers_ctor_type():
    src = b"class C { void m() { var x = new Widget(); } }"
    res = parse_source("C.java", src)
    x = next(d for d in res["decls"] if d["name"] == "x")
    assert x["type"] == "Widget"
    assert x["scope_kind"] == "local"


def test_decls_present_and_fail_open():
    # always a list, even on garbage / unknown lang
    assert parse_source("weird.py", b"\x00 def (((").get("decls") == []
    assert parse_source("data.unknownext", b"x")["decls"] == []
    assert isinstance(parse_source("snippet.py", PY_SNIPPET)["decls"], list)


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


# --------------------------------------------------------------------------- #
# ident_counts — the usage signal behind dead-code detection.
#
# ladybug_store marks a symbol referenced when ident_counts[name] > (times it is
# defined). Counting only tree-sitter's `identifier` node type made a name the
# grammar spells differently score zero occurrences, which reads as "never used"
# and is returned as a delete-this candidate. Each case below is a method that is
# both defined AND called in the same snippet.
# --------------------------------------------------------------------------- #

_USED_METHOD_SNIPPETS = {
    # (file, source, method name, times the name is defined in the snippet)
    "kotlin": ("F.kt", b"class Foo {\n"
                       b"    fun greetUser(x: Int): Int { return x + 1 }\n"
                       b"    fun other() { greetUser(1) }\n"
                       b"}\n", "greetUser"),
    "typescript": ("f.ts", b"class Foo {\n"
                           b"  greetUser(x: number): number { return x + 1; }\n"
                           b"}\n"
                           b"const y = new Foo().greetUser(1);\n", "greetUser"),
    "javascript": ("f.js", b"class Foo { greetUser(x) { return x + 1; } }\n"
                           b"new Foo().greetUser(1);\n", "greetUser"),
    "go": ("f.go", b"package main\n"
                   b"type F struct{}\n"
                   b"func (f *F) GreetUser(x int) int { return x + 1 }\n"
                   b"func Run() { f := &F{}; x := f.GreetUser(1); _ = x }\n", "GreetUser"),
    "rust": ("f.rs", b"struct S;\n"
                     b"impl S { fn greet_user(&self, x: i32) -> i32 { x + 1 } }\n"
                     b"fn run() { let s = S; let _ = s.greet_user(1); }\n", "greet_user"),
    "java": ("Foo.java", b"class Foo {\n"
                         b"  int greetUser(int x) { return x + 1; }\n"
                         b"  void other() { greetUser(1); }\n"
                         b"}\n", "greetUser"),
    "python": ("f.py", b"class Foo:\n"
                       b"    def greet_user(self, x):\n"
                       b"        return x + 1\n"
                       b"    def other(self):\n"
                       b"        return self.greet_user(1)\n", "greet_user"),
}


def test_ident_counts_see_a_called_method_in_every_language():
    """A method defined once and called once must out-count its definitions.

    This is exactly the comparison ladybug_store makes to set `referenced`, so a
    language missing here is a language whose every method is reported dead.
    """
    for lang, (fname, src, method) in _USED_METHOD_SNIPPETS.items():
        res = parse_source(fname, src)
        counts = res["ident_counts"]
        defs = sum(1 for s in res["symbols"] if s["name"] == method)
        assert defs == 1, f"{lang}: expected one definition of {method}, got {defs}"
        assert counts.get(method, 0) > defs, (
            f"{lang}: {method} is called but scores "
            f"{counts.get(method, 0)} occurrences vs {defs} definitions "
            f"-> dead-code detection would flag it. ident_counts={counts}"
        )


def test_ident_counts_non_empty_for_every_supported_language():
    """An empty ident_counts is a silent zero for every symbol in the file."""
    for lang, (fname, src, _method) in _USED_METHOD_SNIPPETS.items():
        res = parse_source(fname, src)
        assert res["ident_counts"], f"{lang}: ident_counts is empty for {fname}"


def test_ident_counts_still_flags_a_genuinely_uncalled_method():
    """The broadened count must not make everything look used."""
    _fname, src, _m = _USED_METHOD_SNIPPETS["kotlin"]
    res = parse_source("F.kt", src)
    # `other` is defined but never called anywhere in the snippet.
    assert res["ident_counts"].get("other", 0) == 1
    assert sum(1 for s in res["symbols"] if s["name"] == "other") == 1

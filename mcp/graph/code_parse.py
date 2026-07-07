"""Code symbol / reference graph extraction via tree-sitter.

Self-contained module for the Loci MCP server. Extracts, per source file, a set
of symbols (functions / classes / methods / structs / enums / interfaces /
traits / impls) and edges between them (DEFINES / CALLS / IMPORTS / REFERENCES).

Design notes / hard constraints (verified on this machine):
  * tree_sitter 0.26.0 + tree-sitter-language-pack 1.12.2
  * Use ``tree_sitter.Parser(language)`` (constructor takes the Language).
    Do NOT use ``get_parser`` from the language pack (bytes/str bug in this combo).
  * ``parser.parse`` requires *bytes*.
  * ``Language.query(str).captures(node)`` returns a dict {name: [nodes]} in 0.26,
    but we normalise defensively for the list-of-(node, name) shape too.

Everything is FAIL-OPEN: any unknown/None language, missing grammar, or parse
exception yields an empty (but well-formed) result dict. Nothing here raises.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import tree_sitter as ts

try:  # language grammars
    from tree_sitter_language_pack import get_language as _get_language
except Exception:  # pragma: no cover - grammar pack missing entirely
    _get_language = None  # type: ignore


# --------------------------------------------------------------------------- #
# Public: extension -> tree-sitter language name
# --------------------------------------------------------------------------- #
LANG_BY_EXT: Dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".rs": "rust",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}

_MAX_FILE_BYTES = 1_500_000  # 1.5 MB


def detect_lang(path: str) -> Optional[str]:
    """Return the tree-sitter language name for ``path`` by extension, or None."""
    try:
        _, ext = os.path.splitext(path)
        return LANG_BY_EXT.get(ext.lower())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Per-language configuration
# --------------------------------------------------------------------------- #
# Definition node types -> default symbol kind. Used both to (a) enumerate the
# definition captures from the query and (b) walk parents to build dotted
# qualnames and locate the enclosing symbol of a call.
KINDS: Dict[str, Dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
    },
    "java": {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "record_declaration": "class",
        "method_declaration": "method",
        "constructor_declaration": "method",
    },
    "kotlin": {
        "class_declaration": "class",
        "object_declaration": "class",
        "function_declaration": "function",
    },
    "rust": {
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
        "impl_item": "impl",
        "function_item": "function",
        "function_signature_item": "function",
    },
    "javascript": {
        "class_declaration": "class",
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "method_definition": "method",
    },
    "typescript": {
        "class_declaration": "class",
        "abstract_class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "function_declaration": "function",
        "method_definition": "method",
        "method_signature": "method",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_spec": "struct",  # refined by the query capture name
    },
    "c": {
        "function_definition": "function",
        "struct_specifier": "struct",
        "enum_specifier": "enum",
    },
    "cpp": {
        "function_definition": "function",
        "struct_specifier": "struct",
        "class_specifier": "class",
        "enum_specifier": "enum",
    },
}
KINDS["tsx"] = KINDS["typescript"]

# Ancestor node types that turn a bare "function" into a "method".
CLASS_LIKE: Dict[str, set] = {
    "python": {"class_definition"},
    "kotlin": {"class_declaration", "object_declaration"},
    "rust": {"impl_item", "trait_item"},
    "cpp": {"class_specifier", "struct_specifier"},
}

# Tree-sitter queries. Definition captures are named ``d.<kind>`` so the kind is
# recovered from the capture-name suffix. Call captures are ``call``; import
# captures are ``import``. Callee names and import module strings are extracted
# from the captured nodes in Python (grammar-specific but robust).
QUERIES: Dict[str, str] = {
    "python": """
        (function_definition) @d.function
        (class_definition) @d.class
        (call) @call
        (import_statement) @import
        (import_from_statement) @import
    """,
    "java": """
        (class_declaration) @d.class
        (interface_declaration) @d.interface
        (enum_declaration) @d.enum
        (record_declaration) @d.class
        (method_declaration) @d.method
        (constructor_declaration) @d.method
        (method_invocation) @call
        (import_declaration) @import
    """,
    "kotlin": """
        (class_declaration) @d.class
        (object_declaration) @d.class
        (function_declaration) @d.function
        (call_expression) @call
        (import_header) @import
    """,
    "rust": """
        (struct_item) @d.struct
        (enum_item) @d.enum
        (trait_item) @d.trait
        (impl_item) @d.impl
        (function_item) @d.function
        (function_signature_item) @d.function
        (call_expression) @call
        (macro_invocation) @call
        (use_declaration) @import
    """,
    "javascript": """
        (class_declaration) @d.class
        (function_declaration) @d.function
        (generator_function_declaration) @d.function
        (method_definition) @d.method
        (call_expression) @call
        (import_statement) @import
    """,
    "typescript": """
        (class_declaration) @d.class
        (abstract_class_declaration) @d.class
        (interface_declaration) @d.interface
        (enum_declaration) @d.enum
        (function_declaration) @d.function
        (method_definition) @d.method
        (method_signature) @d.method
        (call_expression) @call
        (import_statement) @import
    """,
    "go": """
        (function_declaration) @d.function
        (method_declaration) @d.method
        (type_spec name: (type_identifier) type: (struct_type)) @d.struct
        (type_spec name: (type_identifier) type: (interface_type)) @d.interface
        (call_expression) @call
        (import_declaration) @import
    """,
}
QUERIES["tsx"] = QUERIES["typescript"]

# Leaf node types that carry a bare name/identifier.
_NAME_LEAVES = {
    "identifier",
    "type_identifier",
    "simple_identifier",
    "property_identifier",
    "field_identifier",
    "scoped_type_identifier",
}

# Argument-container node types to skip when finding a call's callee.
_ARG_NODES = {"argument_list", "arguments", "call_suffix", "value_arguments"}


# --------------------------------------------------------------------------- #
# Small tree helpers
# --------------------------------------------------------------------------- #
def _text(node) -> str:
    try:
        return node.text.decode("utf-8", "replace")
    except Exception:
        return ""


def _node_name(node, lang: str) -> Optional[str]:
    """Best-effort name of a definition node."""
    try:
        n = node.child_by_field_name("name")
        if n is not None:
            return _text(n)
        if node.type == "impl_item":
            t = node.child_by_field_name("type")
            if t is not None:
                return _text(t)
        for c in node.named_children:
            if c.type in _NAME_LEAVES:
                return _text(c)
    except Exception:
        pass
    return None


def _rightmost_name(node) -> Optional[str]:
    """Rightmost identifier-like leaf under ``node`` (for dotted callees)."""
    found = [None]

    def walk(n):
        if n is None:
            return
        if n.child_count == 0:
            if n.type in _NAME_LEAVES:
                found[0] = _text(n)
            return
        for c in n.children:
            walk(c)

    try:
        walk(node)
    except Exception:
        return None
    return found[0]


def _call_receiver(call_node, lang: str):
    """Return ``(receiver_text_or_None, recv_kind)`` for a call/invocation node.

    recv_kind is one of:
      "self" -> receiver is this/self
      "name" -> receiver is a plain identifier ("Log", "obj")
      "expr" -> receiver is a complex expression (e.g. ``a.b().c``)
      "none" -> bare call with no receiver (``foo()``)
    """
    try:
        if lang == "java":
            obj = call_node.child_by_field_name("object")
            if obj is None:
                return (None, "none")
            if obj.type == "this":
                return ("this", "self")
            if obj.child_count == 0 and obj.type in _NAME_LEAVES:
                return (_text(obj), "name")
            return (_text(obj), "expr")

        if lang == "python":
            fn = call_node.child_by_field_name("function")
            if fn is None:
                return (None, "none")
            if fn.type == "attribute":
                obj = fn.child_by_field_name("object")
                if obj is None:
                    return (None, "expr")
                if obj.type == "identifier" and obj.child_count == 0:
                    txt = _text(obj)
                    return (txt, "self" if txt == "self" else "name")
                return (_text(obj), "expr")
            return (None, "none")

        if lang == "kotlin":
            callee = None
            for c in call_node.named_children:
                if c.type not in _ARG_NODES:
                    callee = c
                    break
            if callee is None:
                return (None, "none")
            if callee.type == "navigation_expression":
                recv = None
                for c in callee.named_children:
                    if c.type == "navigation_suffix":
                        break
                    recv = c
                if recv is None:
                    return (None, "none")
                if recv.type in ("this_expression", "this"):
                    return ("this", "self")
                if recv.type in _NAME_LEAVES:
                    return (_text(recv), "name")
                return (_text(recv), "expr")
            return (None, "none")
    except Exception:
        return (None, "none")
    return (None, "none")


def _callee_name(call_node) -> Optional[str]:
    """Best-effort callee NAME (string) from a call/invocation node."""
    try:
        target = None
        for field in ("function", "name"):
            c = call_node.child_by_field_name(field)
            if c is not None:
                target = c
                break
        if target is None:
            for c in call_node.named_children:
                if c.type not in _ARG_NODES:
                    target = c
                    break
        if target is None:
            return None
        if target.child_count == 0 and target.type in _NAME_LEAVES:
            return _text(target)
        return _rightmost_name(target)
    except Exception:
        return None


def _norm_type(s: Optional[str]) -> Optional[str]:
    """Normalise a raw type string to a simple (rightmost) type name.

    Strips generics (``List<Foo>`` -> ``List``), arrays (``Foo[]`` -> ``Foo``),
    trailing nullability ``?``, and package/scope qualifiers. Returns None when
    nothing usable remains.
    """
    if not s:
        return None
    try:
        s = s.strip()
        if "<" in s:  # strip generic args
            s = s.split("<", 1)[0]
        s = s.replace("[]", "")  # array dims
        s = s.replace("[", "").replace("]", "")
        s = s.strip().rstrip("?").strip()
        # rightmost of dotted / scoped names
        for sep in ("::", "."):
            if sep in s:
                s = s.split(sep)[-1]
        s = s.strip()
        if not s or not any(ch.isalnum() or ch == "_" for ch in s):
            return None
        return s
    except Exception:
        return None


def _infer_java_ctor_type(value_node):
    """Infer a type name from a java initializer (``new Foo()`` -> ``Foo``)."""
    try:
        stack = [value_node]
        while stack:
            n = stack.pop()
            if n is None:
                continue
            if n.type == "object_creation_expression":
                t = n.child_by_field_name("type")
                if t is not None:
                    return _norm_type(_text(t))
            stack.extend(n.children)
    except Exception:
        return None
    return None


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'`" and s[-1] == s[0]:
        return s[1:-1]
    return s


def _import_modules(node, lang: str) -> List[str]:
    """Extract module/path string(s) from an import-capture node."""
    out: List[str] = []
    try:
        t = node.type
        if lang in ("javascript", "typescript", "tsx"):
            src = node.child_by_field_name("source")
            if src is not None:
                out.append(_strip_quotes(_text(src)))
        elif lang == "python":
            if t == "import_from_statement":
                m = node.child_by_field_name("module_name")
                if m is not None:
                    out.append(_text(m))
                else:
                    for c in node.named_children:
                        if c.type in ("dotted_name", "relative_import"):
                            out.append(_text(c))
                            break
            else:  # import_statement
                for c in node.named_children:
                    if c.type in ("dotted_name", "aliased_import"):
                        nm = c.child_by_field_name("name") if c.type == "aliased_import" else c
                        out.append(_text(nm if nm is not None else c))
        elif lang == "java":
            for c in node.named_children:
                if c.type in ("scoped_identifier", "identifier"):
                    out.append(_text(c))
        elif lang == "kotlin":
            for c in node.named_children:
                if c.type == "identifier":
                    out.append(_text(c))
        elif lang == "rust":
            for c in node.named_children:
                out.append(_text(c))
        elif lang == "go":
            # descend to interpreted_string_literal leaves
            stack = [node]
            while stack:
                n = stack.pop()
                if n.type == "interpreted_string_literal":
                    out.append(_strip_quotes(_text(n)))
                else:
                    stack.extend(n.children)
    except Exception:
        return [m for m in out if m]
    return [m for m in out if m]


def _import_map_entries(node, lang: str):
    """Best-effort ``(simple_name, fqn)`` pairs from an import-capture node.

    java  ``import android.util.Log;``      -> ("Log", "android.util.Log")
    kotlin ``import a.b.C`` / ``... as D``   -> ("C", "a.b.C") / ("D", "a.b.C")
    python ``import os``                     -> ("os", "os")
           ``import numpy as np``            -> ("np", "numpy")
           ``from a.b import C``             -> ("C", "a.b.C")
           ``from a.b import c as d``        -> ("d", "a.b.c")
    """
    out = []
    try:
        if lang == "java":
            fqn = None
            for c in node.named_children:
                if c.type in ("scoped_identifier", "identifier"):
                    fqn = _text(c)
                    break
            if fqn:
                simple = fqn.split(".")[-1]
                if simple and simple != "*":
                    out.append((simple, fqn))

        elif lang == "kotlin":
            fqn = None
            alias = None
            for c in node.named_children:
                if c.type == "identifier":
                    fqn = _text(c)
                elif c.type == "import_alias":
                    an = c.child_by_field_name("name")
                    if an is None:
                        for gc in c.named_children:
                            if gc.type in _NAME_LEAVES:
                                an = gc
                                break
                    alias = _text(an) if an is not None else None
            if fqn:
                simple = alias or fqn.split(".")[-1]
                if simple and simple != "*":
                    out.append((simple, fqn))

        elif lang == "python":
            if node.type == "import_from_statement":
                mod = node.child_by_field_name("module_name")
                mod_txt = _text(mod) if mod is not None else ""
                mod_span = (mod.start_byte, mod.end_byte) if mod is not None else None
                for c in node.named_children:
                    if mod_span is not None and (c.start_byte, c.end_byte) == mod_span:
                        continue
                    if c.type == "aliased_import":
                        nm = c.child_by_field_name("name")
                        al = c.child_by_field_name("alias")
                        base = _text(nm) if nm is not None else ""
                        key = _text(al) if al is not None else base.split(".")[-1]
                        # Relative imports ("from . import queries") bind the module
                        # by its simple name — don't prefix the dotted package path.
                        fqn = base if (not mod_txt or set(mod_txt) <= {"."}) else f"{mod_txt}.{base}"
                        if key:
                            out.append((key, fqn))
                    elif c.type in ("dotted_name", "identifier"):
                        base = _text(c)
                        key = base.split(".")[-1]
                        # Relative imports ("from . import queries") bind the module
                        # by its simple name — don't prefix the dotted package path.
                        fqn = base if (not mod_txt or set(mod_txt) <= {"."}) else f"{mod_txt}.{base}"
                        if key:
                            out.append((key, fqn))
            else:  # import_statement
                for c in node.named_children:
                    if c.type == "aliased_import":
                        nm = c.child_by_field_name("name")
                        al = c.child_by_field_name("alias")
                        base = _text(nm) if nm is not None else ""
                        key = _text(al) if al is not None else base.split(".")[0]
                        if key:
                            out.append((key, base))
                    elif c.type == "dotted_name":
                        base = _text(c)
                        key = base.split(".")[0]  # ``import a.b`` binds ``a``
                        if key:
                            out.append((key, base))
    except Exception:
        return out
    return out


# --------------------------------------------------------------------------- #
# Query capture normalisation
# --------------------------------------------------------------------------- #
def _run_query(lang_obj, query_str: str, root) -> Dict[str, List[Any]]:
    """Return {capture_name: [nodes]}, tolerant across tree-sitter API variants."""
    out: Dict[str, List[Any]] = {}
    caps = None
    # tree_sitter 0.25+/0.26: ts.Query(lang, str) + ts.QueryCursor(query).captures(root)
    try:
        q = ts.Query(lang_obj, query_str)
        try:
            cursor = ts.QueryCursor(q)
            caps = cursor.captures(root)
        except Exception:
            caps = q.captures(root)  # some builds expose captures on Query
    except Exception:
        caps = None
    # Older API: Language.query(str).captures(root)
    if caps is None:
        try:
            q = lang_obj.query(query_str)
            caps = q.captures(root)
        except Exception:
            return out
    try:
        if isinstance(caps, dict):
            for name, nodes in caps.items():
                out.setdefault(name, []).extend(nodes)
        else:  # list of (node, name) tuples (older/alt API)
            for item in caps:
                try:
                    node, name = item
                except Exception:
                    continue
                out.setdefault(name, []).append(node)
    except Exception:
        return {}
    return out


# --------------------------------------------------------------------------- #
# Declaration capture (fields / params / locals) for receiver-type inference
# --------------------------------------------------------------------------- #
def _first_child_type(node, types):
    """First direct named child whose type is in ``types`` (or None)."""
    try:
        for c in node.named_children:
            if c.type in types:
                return c
    except Exception:
        return None
    return None


def _kotlin_property_name_type(prop_node):
    """(name, type_str_or_None) for a kotlin property/local declaration node."""
    name = None
    type_ = None
    vd = _first_child_type(prop_node, {"variable_declaration"})
    if vd is not None:
        for c in vd.named_children:
            if c.type == "simple_identifier" and name is None:
                name = _text(c)
            elif c.type == "user_type" and type_ is None:
                ut = _first_child_type(c, {"type_identifier"})
                type_ = _norm_type(_text(ut)) if ut is not None else _norm_type(_text(c))
    if type_ is None:  # infer from ``val x = Foo()`` constructor call
        call = _first_child_type(prop_node, {"call_expression"})
        if call is not None:
            type_ = _norm_type(_callee_name(call))
    return name, type_


def _collect_decls(root, lang, def_types, enclosing_symbol_src, in_method, add_decl):
    """Walk the whole tree once, emitting field/param/local declarations.

    FAIL-OPEN: individual node handling is wrapped; nothing raises out.
    """
    stack = [root]
    while stack:
        node = stack.pop()
        try:
            t = node.type

            if lang == "java":
                if t == "field_declaration":
                    tnode = node.child_by_field_name("type")
                    type_ = _norm_type(_text(tnode)) if tnode is not None else None
                    scope = enclosing_symbol_src(node)
                    for c in node.named_children:
                        if c.type == "variable_declarator":
                            nm = c.child_by_field_name("name")
                            if nm is not None:
                                add_decl(_text(nm), type_, scope, "field")
                elif t == "formal_parameter":
                    tnode = node.child_by_field_name("type")
                    nm = node.child_by_field_name("name")
                    if nm is not None:
                        type_ = _norm_type(_text(tnode)) if tnode is not None else None
                        add_decl(_text(nm), type_, enclosing_symbol_src(node), "param")
                elif t == "local_variable_declaration":
                    tnode = node.child_by_field_name("type")
                    type_ = _norm_type(_text(tnode)) if tnode is not None else None
                    scope = enclosing_symbol_src(node)
                    for c in node.named_children:
                        if c.type == "variable_declarator":
                            nm = c.child_by_field_name("name")
                            if nm is None:
                                continue
                            dtype = type_
                            if dtype == "var" or dtype is None:
                                val = c.child_by_field_name("value")
                                inferred = _infer_java_ctor_type(val) if val is not None else None
                                dtype = inferred if inferred else (None if dtype == "var" else dtype)
                            add_decl(_text(nm), dtype, scope, "local")

            elif lang == "kotlin":
                if t == "property_declaration":
                    name, type_ = _kotlin_property_name_type(node)
                    if name is not None:
                        kind = "local" if in_method(node) else "field"
                        add_decl(name, type_, enclosing_symbol_src(node), kind)
                elif t in ("parameter", "class_parameter"):
                    nm = _first_child_type(node, {"simple_identifier"})
                    ut = _first_child_type(node, {"user_type"})
                    type_ = None
                    if ut is not None:
                        ti = _first_child_type(ut, {"type_identifier"})
                        type_ = _norm_type(_text(ti)) if ti is not None else _norm_type(_text(ut))
                    if nm is not None:
                        add_decl(_text(nm), type_, enclosing_symbol_src(node), "param")

            elif lang == "python":
                if t == "assignment":
                    left = node.child_by_field_name("left")
                    if left is not None and left.type == "identifier":
                        edn_method = in_method(node)
                        # only class-body (field) or function-body (local) assigns
                        if edn_method or _python_is_class_field(node, def_types):
                            tnode = node.child_by_field_name("type")
                            type_ = None
                            if tnode is not None:
                                ti = _first_child_type(tnode, _NAME_LEAVES) or tnode
                                type_ = _norm_type(_text(ti))
                            if type_ is None:
                                val = node.child_by_field_name("right")
                                if val is not None and val.type == "call":
                                    type_ = _norm_type(_callee_name(val))
                            kind = "local" if edn_method else "field"
                            add_decl(_text(left), type_, enclosing_symbol_src(node), kind)
                elif t in ("typed_parameter", "typed_default_parameter"):
                    nm = _first_child_type(node, {"identifier"})
                    tnode = node.child_by_field_name("type")
                    type_ = None
                    if tnode is not None:
                        ti = _first_child_type(tnode, _NAME_LEAVES) or tnode
                        type_ = _norm_type(_text(ti))
                    if nm is not None:
                        add_decl(_text(nm), type_, enclosing_symbol_src(node), "param")
                elif t == "parameters":
                    for c in node.named_children:
                        if c.type == "identifier":
                            add_decl(_text(c), None, enclosing_symbol_src(node), "param")
                        elif c.type == "default_parameter":
                            nm = c.child_by_field_name("name")
                            if nm is None:
                                nm = _first_child_type(c, {"identifier"})
                            if nm is not None:
                                add_decl(_text(nm), None, enclosing_symbol_src(node), "param")
        except Exception:
            pass

        try:
            stack.extend(node.children)
        except Exception:
            pass


def _python_is_class_field(assign_node, def_types) -> bool:
    """True when the nearest enclosing definition of a python assignment is a class."""
    try:
        cur = assign_node.parent
        while cur is not None:
            if cur.type in def_types:
                return cur.type == "class_definition"
            cur = cur.parent
    except Exception:
        return False
    return False


# --------------------------------------------------------------------------- #
# Core per-file parse
# --------------------------------------------------------------------------- #
def _empty(file: str, lang: Optional[str]) -> Dict[str, Any]:
    return {
        "file": file,
        "lang": lang,
        "symbols": [],
        "edges": [],
        "imports": [],
        "import_map": {},
        "decls": [],
    }


def parse_source(file: str, source: bytes, lang: Optional[str] = None) -> Dict[str, Any]:
    """Parse ``source`` (bytes) for ``file`` and return the symbol/edge graph.

    FAIL-OPEN: never raises; returns an empty well-formed dict on any problem.
    """
    if lang is None:
        lang = detect_lang(file)

    result = _empty(file, lang)

    try:
        if not lang or _get_language is None:
            return result
        if not isinstance(source, (bytes, bytearray)):
            source = str(source).encode("utf-8", "replace")
        else:
            source = bytes(source)

        try:
            lang_obj = _get_language(lang)
        except Exception:
            return _empty(file, lang)

        parser = ts.Parser(lang_obj)
        tree = parser.parse(source)
        root = tree.root_node

        query_str = QUERIES.get(lang)
        if not query_str:
            # Grammar available but no query defined: file node only.
            return result

        def_types = KINDS.get(lang, {})
        class_like = CLASS_LIKE.get(lang, set())

        def qualname(node) -> str:
            parts: List[str] = []
            cur = node
            while cur is not None:
                if cur.type in def_types:
                    nm = _node_name(cur, lang)
                    if nm:
                        parts.append(nm)
                cur = cur.parent
            return ".".join(reversed(parts))

        def enclosing_symbol_src(node) -> str:
            cur = node.parent
            while cur is not None:
                if cur.type in def_types:
                    qn = qualname(cur)
                    if qn:
                        return f"{file}::{qn}"
                cur = cur.parent
            return file

        caps = _run_query(lang_obj, query_str, root)

        symbols: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        imports: List[str] = []
        import_map: Dict[str, str] = {}
        seen_syms: set = set()
        seen_edges: set = set()

        def add_edge(src, dst, etype, **extra):
            if not dst:
                return
            key = (src, dst, etype, extra.get("receiver"), extra.get("recv_kind"))
            if key in seen_edges:
                return
            seen_edges.add(key)
            edge = {"src": src, "dst": dst, "type": etype}
            edge.update(extra)
            edges.append(edge)

        # ---- definitions (symbols + DEFINES) ----
        for cap_name, nodes in caps.items():
            if not cap_name.startswith("d."):
                continue
            base_kind = cap_name[2:]
            for node in nodes:
                name = _node_name(node, lang)
                if not name:
                    continue
                qn = qualname(node) or name
                sid = f"{file}::{qn}"
                if sid in seen_syms:
                    continue
                seen_syms.add(sid)
                kind = base_kind
                if kind == "function" and class_like:
                    cur = node.parent
                    while cur is not None:
                        if cur.type in class_like:
                            kind = "method"
                            break
                        cur = cur.parent
                symbols.append(
                    {
                        "id": sid,
                        "name": name,
                        "kind": kind,
                        "line": node.start_point[0] + 1,
                        "lang": lang,
                        "file": file,
                    }
                )
                add_edge(file, sid, "DEFINES")

        # ---- calls (CALLS) ----
        for node in caps.get("call", []):
            callee = _callee_name(node)
            if not callee:
                continue
            src = enclosing_symbol_src(node)
            recv, recv_kind = _call_receiver(node, lang)
            add_edge(src, callee, "CALLS", receiver=recv, recv_kind=recv_kind)

        # ---- imports (IMPORTS + import_map) ----
        for node in caps.get("import", []):
            for mod in _import_modules(node, lang):
                imports.append(mod)
                add_edge(file, mod, "IMPORTS")
            for simple, fqn in _import_map_entries(node, lang):
                if simple and simple not in import_map:
                    import_map[simple] = fqn

        # ---- declarations (fields / params / locals) ----
        method_kinds = {"method", "function"}

        def enclosing_def_node(node):
            cur = node.parent
            while cur is not None:
                if cur.type in def_types:
                    return cur
                cur = cur.parent
            return None

        decls: List[Dict[str, Any]] = []
        seen_decls: set = set()

        def add_decl(name, type_, scope, scope_kind):
            if not name or not scope_kind:
                return
            key = (name, type_, scope, scope_kind)
            if key in seen_decls:
                return
            seen_decls.add(key)
            decls.append(
                {
                    "name": name,
                    "type": type_,
                    "scope": scope,
                    "scope_kind": scope_kind,
                }
            )

        def _in_method(node) -> bool:
            edn = enclosing_def_node(node)
            return edn is not None and def_types.get(edn.type) in method_kinds

        try:
            _collect_decls(
                root, lang, def_types, enclosing_symbol_src, _in_method, add_decl
            )
        except Exception:
            pass

        result["symbols"] = symbols
        result["edges"] = edges
        result["imports"] = imports
        result["import_map"] = import_map
        result["decls"] = decls
        return result
    except Exception:
        return _empty(file, lang)


# --------------------------------------------------------------------------- #
# Directory walk
# --------------------------------------------------------------------------- #
def parse_path(
    root: str,
    *,
    max_files: Optional[int] = None,
    ignore=(".git", ".venv", "node_modules", "__pycache__", "build", "dist", "target"),
) -> List[Dict[str, Any]]:
    """Walk ``root`` and parse every file with a detectable language.

    Skips ``ignore`` directories and binary / oversized (>1.5 MB) files.
    FAIL-OPEN per file; never raises.
    """
    results: List[Dict[str, Any]] = []
    ignore_set = set(ignore or ())
    try:
        if os.path.isfile(root):
            lang = detect_lang(root)
            if lang:
                data = _read_and_parse(root, lang)
                if data is not None:
                    results.append(data)
            return results

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ignore_set]
            for fn in filenames:
                if max_files is not None and len(results) >= max_files:
                    return results
                lang = detect_lang(fn)
                if not lang:
                    continue
                full = os.path.join(dirpath, fn)
                data = _read_and_parse(full, lang)
                if data is not None:
                    results.append(data)
    except Exception:
        return results
    return results


def _read_and_parse(full: str, lang: str) -> Optional[Dict[str, Any]]:
    try:
        st = os.stat(full)
        if not os.path.isfile(full) or st.st_size > _MAX_FILE_BYTES:
            return None
        with open(full, "rb") as fh:
            source = fh.read()
        if b"\x00" in source[:4096]:  # crude binary guard
            return None
        return parse_source(full, source, lang)
    except Exception:
        return _empty(full, lang)

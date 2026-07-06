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
# Core per-file parse
# --------------------------------------------------------------------------- #
def _empty(file: str, lang: Optional[str]) -> Dict[str, Any]:
    return {"file": file, "lang": lang, "symbols": [], "edges": [], "imports": []}


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
        seen_syms: set = set()
        seen_edges: set = set()

        def add_edge(src, dst, etype):
            if not dst:
                return
            key = (src, dst, etype)
            if key in seen_edges:
                return
            seen_edges.add(key)
            edges.append({"src": src, "dst": dst, "type": etype})

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
            add_edge(src, callee, "CALLS")

        # ---- imports (IMPORTS) ----
        for node in caps.get("import", []):
            for mod in _import_modules(node, lang):
                imports.append(mod)
                add_edge(file, mod, "IMPORTS")

        result["symbols"] = symbols
        result["edges"] = edges
        result["imports"] = imports
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

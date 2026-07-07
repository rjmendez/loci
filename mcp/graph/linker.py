"""Finding -> CodeSymbol linker: populate ``REFERENCES`` edges.

A **precision-focused** engine that scans ``Finding`` text for references to
``CodeSymbol`` nodes and records each as a ``REFERENCES`` edge in the embedded
Kuzu graph (:class:`graph.kuzu_store.KuzuStore`).

The whole point is *precision, not recall* — it mirrors the hard-won call-graph
lesson: never resolve on a bare, common, ambiguous word. A finding that says
"we should get the value" must link to **nothing**, even though ``get`` is a
real method name somewhere; a finding that names ``DeviceMetricsPoller`` links
to exactly that type. The classifier only fires on evidence that is distinctive
enough to be almost-certainly an intentional code reference:

Confidence rules (see :func:`extract_symbol_refs`):

* **high** — a CamelCase *type* name as a whole-word token; an explicit
  ``symbol:NAME`` token; a ``Name.member`` dotted reference where ``Name`` is a
  known type; or a ``SomeFile.java`` / ``.kt`` mention matching a symbol's file
  basename.
* **medium** — a distinctive ``camelCase`` / ``UPPER_CASE`` identifier token of
  length >= 8 that is **globally unique** in the symbol index.
* **never** — short, lowercase or common words (``get``, ``run``, ``put``,
  ``update``, ``handle``, ``data``, ``text`` …); or an ambiguous multi-id plain
  name (unless it is a type).

All public functions are **fail-open**: on any error the writers return ``0`` /
an empty result rather than propagating, exactly like the rest of the store.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger("loci-mcp.linker")

__all__ = [
    "build_symbol_index",
    "extract_symbol_refs",
    "link_findings",
    "relink_all",
]

# Symbol kinds that count as a "distinctive type" — a whole-word match on one of
# these is strong enough to be a HIGH-confidence reference on its own. Mirrors
# ingest_code's _TYPE_KINDS.
_TYPE_KINDS = {"class", "interface", "enum", "struct", "trait"}

# Minimum length for a bare identifier to be eligible for a MEDIUM match. Short
# tokens are almost never distinctive enough to be safe.
_MIN_IDENT_LEN = 8

# Common programming words that are NEVER, on their own, a code reference — even
# if a symbol happens to share the name. Guards the MEDIUM path (the HIGH paths
# require a type / explicit marker and so are already safe).
_COMMON_WORDS = frozenset({
    "get", "set", "run", "put", "add", "new", "old", "map", "key", "val",
    "value", "data", "text", "name", "type", "kind", "size", "list", "item",
    "update", "handle", "process", "result", "state", "status", "count",
    "index", "start", "stop", "close", "open", "read", "write", "send",
    "recv", "init", "main", "test", "check", "build", "parse", "load",
    "save", "call", "next", "prev", "node", "edge", "path", "file", "line",
    "true", "false", "none", "null", "self", "this", "return", "should",
})

# File-extension mentions we treat as a source-file reference.
_SOURCE_EXTS = (
    "java", "kt", "kts", "py", "js", "ts", "jsx", "tsx", "go", "rs", "rb",
    "swift", "c", "cc", "cpp", "cxx", "h", "hpp", "hxx", "cs", "m", "mm",
    "scala", "php", "dart",
)

# --- token/pattern regexes ---------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SYMBOL_TOKEN_RE = re.compile(r"\bsymbol:([A-Za-z_][A-Za-z0-9_.]*)")
_DOTTED_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")
_FILE_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*\.(?:" + "|".join(_SOURCE_EXTS) + r"))\b"
)


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
def build_symbol_index(symbols: list[dict]) -> dict:
    """Build a fast lookup index over ``symbols`` for reference extraction.

    ``symbols`` — an iterable of dicts shaped like ``{id, name, kind, file}``
    (``file`` optional). Symbol ids are ``"file::Qualname"``.

    Returns a dict with these keys (all sets/dicts are safe to share read-only):

    * ``by_name``       — ``name -> list[symbol_id]`` (every symbol, dup ids
                          collapsed).
    * ``type_names``    — ``set[str]`` of names whose kind is a distinctive type
                          (class/interface/enum/struct/trait).
    * ``type_ids``      — ``name -> list[symbol_id]`` restricted to type symbols.
    * ``unique_names``  — ``set[str]`` of names that resolve to exactly one
                          symbol id (eligible for MEDIUM matching).
    * ``file_basenames``— ``basename -> list[symbol_id]`` for ``SomeFile.java``
                          style references.
    """
    by_name: dict[str, list[str]] = {}
    type_names: set[str] = set()
    type_ids: dict[str, list[str]] = {}
    file_basenames: dict[str, list[str]] = {}
    # track seen (name,id) / (base,id) to collapse duplicate rows
    _seen_name: set[tuple[str, str]] = set()
    _seen_base: set[tuple[str, str]] = set()

    for s in (symbols or []):
        if not isinstance(s, dict):
            continue
        sid = s.get("id")
        if not sid:
            continue
        sid = str(sid)
        name = s.get("name")
        name = str(name) if name else ""
        kind = str(s.get("kind") or "").lower()
        if name:
            if (name, sid) not in _seen_name:
                _seen_name.add((name, sid))
                by_name.setdefault(name, []).append(sid)
            if kind in _TYPE_KINDS:
                type_names.add(name)
                if sid not in type_ids.setdefault(name, []):
                    type_ids[name].append(sid)
        f = s.get("file")
        if f:
            base = os.path.basename(str(f))
            if base and (base, sid) not in _seen_base:
                _seen_base.add((base, sid))
                file_basenames.setdefault(base, []).append(sid)

    unique_names = {n for n, ids in by_name.items() if len(ids) == 1}

    return {
        "by_name": by_name,
        "type_names": type_names,
        "type_ids": type_ids,
        "unique_names": unique_names,
        "file_basenames": file_basenames,
    }


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _enclosing_class(sid: str) -> Optional[str]:
    """Simple enclosing-class name of a symbol id ("file::A.B.m" -> "B")."""
    qual = sid.split("::", 1)[1] if "::" in sid else sid
    segs = qual.split(".")
    return segs[-2] if len(segs) >= 2 else None


def _is_distinctive_ident(tok: str) -> bool:
    """True for a ``camelCase`` / ``PascalCase`` / ``UPPER_CASE`` token >= 8 chars."""
    if len(tok) < _MIN_IDENT_LEN:
        return False
    has_lower = any(c.islower() for c in tok)
    has_upper = any(c.isupper() for c in tok)
    if has_lower and has_upper:
        return True  # camelCase / PascalCase — mixed case is distinctive
    if has_upper and not has_lower:
        # ALL-CAPS identifier (e.g. SIGMA_GPS_LEARNED); require a real letter.
        return any(c.isalpha() for c in tok)
    return False


def extract_symbol_refs(text: str, index: dict) -> list[tuple[str, str]]:
    """Extract precision-filtered symbol references from ``text``.

    Returns ``[(symbol_id, confidence)]`` with ``confidence`` in
    ``{"high", "medium"}``, deduped so each symbol id keeps its best confidence.
    See the module docstring for the precision rules.
    """
    if not text or not isinstance(index, dict):
        return []

    by_name: dict = index.get("by_name") or {}
    type_names: set = index.get("type_names") or set()
    type_ids: dict = index.get("type_ids") or {}
    unique_names: set = index.get("unique_names") or set()
    file_basenames: dict = index.get("file_basenames") or {}

    best: dict[str, str] = {}

    def _add(sid: Optional[str], conf: str) -> None:
        if not sid:
            return
        prev = best.get(sid)
        if prev == "high":
            return  # already best
        if prev is None or conf == "high":
            best[sid] = conf

    # 1) Explicit "symbol:NAME" markers — intentional, so link every id.
    for m in _SYMBOL_TOKEN_RE.finditer(text):
        raw = m.group(1)
        # allow a dotted "symbol:Type.member" form
        if "." in raw:
            head, _, tail = raw.rpartition(".")
            for cid in by_name.get(tail, []):
                if _enclosing_class(cid) == head:
                    _add(cid, "high")
            for cid in by_name.get(raw, []):
                _add(cid, "high")
        else:
            for cid in by_name.get(raw, []):
                _add(cid, "high")

    # 2) Source-file mentions ("DeviceMetricsPoller.java") -> the file's PRIMARY TYPE
    #    (the class named after the file), NOT every symbol defined in that file.
    #    Linking all members over-links badly (a 251-method file -> 251 spurious refs).
    for m in _FILE_RE.finditer(text):
        base = m.group(1)
        stem = base.rsplit(".", 1)[0]  # "MainActivity.java" -> "MainActivity"
        for cid in type_ids.get(stem, []):
            _add(cid, "high")

    # 3) Dotted "Name.member" where Name is a known type -> the member symbol.
    for m in _DOTTED_RE.finditer(text):
        head, member = m.group(1), m.group(2)
        if head not in type_names:
            continue
        matched = False
        for cid in by_name.get(member, []):
            if _enclosing_class(cid) == head:
                _add(cid, "high")
                matched = True
        if not matched:
            # fall back to linking the type itself (handled again in step 4).
            for cid in type_ids.get(head, []):
                _add(cid, "high")

    # 4) Whole-word tokens: CamelCase type names (HIGH) and unique distinctive
    #    identifiers (MEDIUM).
    for m in _WORD_RE.finditer(text):
        tok = m.group(0)
        # HIGH: a type name, whole-word, that looks like a type (Uppercase lead).
        if tok in type_names and tok[:1].isupper():
            for cid in type_ids.get(tok, []):
                _add(cid, "high")
            continue
        # MEDIUM: a globally-unique, distinctive, non-common identifier.
        if (
            tok in unique_names
            and tok.lower() not in _COMMON_WORDS
            and _is_distinctive_ident(tok)
        ):
            ids = by_name.get(tok) or []
            if len(ids) == 1:
                _add(ids[0], "medium")

    return sorted(best.items())


# --------------------------------------------------------------------------- #
# Writers (fail-open, batched, idempotent via MERGE)
# --------------------------------------------------------------------------- #
def _index_from_store(ks) -> dict:
    """Build a symbol index by querying every CodeSymbol in the graph."""
    rows = ks._rows("MATCH (s:CodeSymbol) RETURN s.id, s.name, s.kind, s.file")
    symbols = [
        {"id": r[0], "name": r[1], "kind": r[2], "file": r[3]} for r in rows
    ]
    return build_symbol_index(symbols)


def link_findings(ks, finding_rows: list[dict], index=None) -> int:
    """Create ``REFERENCES`` edges for ``finding_rows`` and return the count.

    ``finding_rows`` — dicts with ``{id, text}``. When ``index`` is ``None`` it
    is built from every ``CodeSymbol`` in the graph. Edges are created in one or
    a few batched ``UNWIND`` MERGE statements (chunked via ``ks._chunks``), so
    the operation is idempotent. Fail-open: returns ``0`` on any error.
    """
    if ks is None:
        return 0
    try:
        if not ks.available():
            return 0
        if index is None:
            index = _index_from_store(ks)

        pairs: list[dict] = []
        for fr in (finding_rows or []):
            if not isinstance(fr, dict):
                continue
            fid = fr.get("id")
            text = fr.get("text")
            if not fid or not text:
                continue
            fid = str(fid)
            for sid, _conf in extract_symbol_refs(str(text), index):
                pairs.append({"f": fid, "s": sid})

        if not pairs:
            return 0

        created = 0
        for chunk in ks._chunks(pairs):
            ks._exec(
                "UNWIND $rows AS r "
                "MATCH (f:Finding {id:r.f}), (s:CodeSymbol {id:r.s}) "
                "MERGE (f)-[:REFERENCES]->(s)",
                {"rows": chunk},
            )
            created += len(chunk)
        return created
    except Exception as exc:
        logger.debug("link_findings failed: %s", exc)
        return 0


def relink_all(ks) -> dict:
    """Rebuild ``REFERENCES`` for every ``Finding`` in the graph.

    Reads all findings and all symbols, builds the index once, then links.
    Idempotent (MERGE). Fail-open: returns zeros on any error.

    Returns ``{"findings_scanned": int, "links_created": int}``.
    """
    empty = {"findings_scanned": 0, "links_created": 0}
    if ks is None:
        return empty
    try:
        if not ks.available():
            return empty
        frows = ks._rows("MATCH (f:Finding) RETURN f.id, f.text")
        findings = [
            {"id": r[0], "text": r[1]} for r in frows if r and r[0]
        ]
        index = _index_from_store(ks)
        created = link_findings(ks, findings, index)
        return {"findings_scanned": len(findings), "links_created": created}
    except Exception as exc:
        logger.debug("relink_all failed: %s", exc)
        return empty

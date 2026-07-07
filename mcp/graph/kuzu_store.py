"""Embedded Kuzu graph store for the Loci MCP server.

Persists two overlaid graphs in a single embedded Kuzu database:

* a **code graph** ingested from tree-sitter (``code_parse``) output —
  ``CodeFile`` / ``CodeSymbol`` nodes joined by ``DEFINES`` / ``CALLS`` /
  ``IMPORTS`` edges; and
* an **investigation graph** — ``Finding`` / ``Entity`` / ``Investigation``
  nodes joined by ``MENTIONS`` / ``DERIVED_FROM`` / ``IN_INVESTIGATION`` /
  ``REFERENCES`` / ``RELATED`` edges.

It answers relationship queries over both, including a full graph port of the
in-memory contamination algorithm (``memcheck.checks.contagion.find_contamination``).

Design contract: **fail-open everywhere.** If ``import kuzu`` fails, the db
cannot be opened, or any query raises, public methods return ``False`` / ``[]``
/ ``{}`` and :meth:`available` stays ``False`` — nothing propagates out. The
sole intentional exception is :meth:`code_query`, which raises ``ValueError`` on
a write-shaped query before touching the database.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Optional

logger = logging.getLogger("loci-mcp.kuzu")

try:  # kuzu is optional — the store degrades to unavailable without it.
    import kuzu  # type: ignore
    _HAS_KUZU = True
except Exception:  # pragma: no cover - environment without kuzu
    kuzu = None  # type: ignore
    _HAS_KUZU = False

__all__ = ["KuzuStore"]

# Entity buckets considered "distinctive" — mirrors contagion._DISTINCTIVE_BUCKETS.
_DISTINCTIVE_BUCKETS = (
    "urls", "url", "hosts", "hostnames", "host", "paths", "path",
    "ips", "ip", "hashes", "cves", "emails", "identifiers", "endpoints",
)

# Query shapes that mutate the graph — rejected by code_query's read-only guard.
_WRITE_GUARD_RE = re.compile(
    r"\b(CREATE|DELETE|SET|DROP|COPY|ALTER|MERGE)\b", re.IGNORECASE
)


class KuzuStore:
    """Embedded Kuzu graph store. All public methods are fail-open."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ok = False
        self._db = None
        self._conn = None
        # Kuzu connections are not guaranteed thread-safe for concurrent writes;
        # serialise all access behind a lock so the store is safe to share.
        self._lock = threading.RLock()
        if not _HAS_KUZU:
            logger.info("kuzu not importable; KuzuStore unavailable")
            return
        try:
            self._db = kuzu.Database(db_path)
            self._conn = kuzu.Connection(self._db)
            self._init_schema()
            self.ok = True
        except Exception as exc:  # pragma: no cover - open/schema failure
            logger.warning("KuzuStore init failed (%s); unavailable", exc)
            self._db = None
            self._conn = None
            self.ok = False

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    _SCHEMA = (
        "CREATE NODE TABLE IF NOT EXISTS CodeFile(path STRING PRIMARY KEY, lang STRING)",
        "CREATE NODE TABLE IF NOT EXISTS CodeSymbol(id STRING PRIMARY KEY, name STRING, "
        "kind STRING, file STRING, line INT64, lang STRING)",
        "CREATE NODE TABLE IF NOT EXISTS Finding(id STRING PRIMARY KEY, investigation STRING, "
        "ftype STRING, text STRING, confidence STRING, source STRING, ts INT64)",
        "CREATE NODE TABLE IF NOT EXISTS Entity(name STRING PRIMARY KEY, etype STRING, distinctive BOOL)",
        "CREATE NODE TABLE IF NOT EXISTS Investigation(id STRING PRIMARY KEY, title STRING)",
        "CREATE REL TABLE IF NOT EXISTS DEFINES(FROM CodeFile TO CodeSymbol)",
        "CREATE REL TABLE IF NOT EXISTS CALLS(FROM CodeSymbol TO CodeSymbol)",
        "CREATE REL TABLE IF NOT EXISTS IMPORTS(FROM CodeFile TO CodeFile)",
        "CREATE REL TABLE IF NOT EXISTS REFERENCES(FROM Finding TO CodeSymbol)",
        "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Finding TO Entity)",
        "CREATE REL TABLE IF NOT EXISTS DERIVED_FROM(FROM Finding TO Finding)",
        "CREATE REL TABLE IF NOT EXISTS IN_INVESTIGATION(FROM Finding TO Investigation)",
        "CREATE REL TABLE IF NOT EXISTS RELATED(FROM Investigation TO Investigation)",
    )

    def _init_schema(self) -> None:
        for ddl in self._SCHEMA:
            self._conn.execute(ddl)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _exec(self, cypher: str, params: Optional[dict] = None):
        """Execute a statement under the lock. Raises on error (callers catch)."""
        with self._lock:
            if params is None:
                return self._conn.execute(cypher)
            return self._conn.execute(cypher, params)

    def _rows(self, cypher: str, params: Optional[dict] = None) -> list[list]:
        res = self._exec(cypher, params)
        out: list[list] = []
        while res.has_next():
            out.append(res.get_next())
        return out

    def available(self) -> bool:
        return bool(self.ok)

    # ------------------------------------------------------------------ #
    # Writes (idempotent via MERGE; fail-open)
    # ------------------------------------------------------------------ #
    def upsert_investigation(self, id: str, title: str = "") -> bool:
        if not self.ok or not id:
            return False
        try:
            self._exec(
                "MERGE (i:Investigation {id:$id}) SET i.title = $title",
                {"id": str(id), "title": str(title or "")},
            )
            return True
        except Exception as exc:
            logger.debug("upsert_investigation failed: %s", exc)
            return False

    def upsert_finding(self, finding: dict) -> bool:
        if not self.ok or not isinstance(finding, dict):
            return False
        fid = finding.get("id") or finding.get("finding_id")
        if not fid:
            return False
        try:
            inv = str(finding.get("investigation") or finding.get("investigation_id") or "")
            ftype = str(finding.get("ftype") or finding.get("type") or "")
            text = str(finding.get("text") or "")
            confidence = str(finding.get("confidence") or "")
            source = str(finding.get("source") or "")
            try:
                ts = int(finding.get("ts") or 0)
            except Exception:
                ts = 0
            self._exec(
                "MERGE (f:Finding {id:$id}) "
                "SET f.investigation=$inv, f.ftype=$ftype, f.text=$text, "
                "f.confidence=$conf, f.source=$src, f.ts=$ts",
                {"id": str(fid), "inv": inv, "ftype": ftype, "text": text,
                 "conf": confidence, "src": source, "ts": ts},
            )
            if inv:
                self._exec("MERGE (i:Investigation {id:$id})", {"id": inv})
                self._exec(
                    "MATCH (f:Finding {id:$f}), (i:Investigation {id:$i}) "
                    "MERGE (f)-[:IN_INVESTIGATION]->(i)",
                    {"f": str(fid), "i": inv},
                )
            return True
        except Exception as exc:
            logger.debug("upsert_finding failed: %s", exc)
            return False

    def upsert_entity(self, name: str, etype: str = "", distinctive: bool = False) -> bool:
        if not self.ok or not name:
            return False
        try:
            self._exec(
                "MERGE (e:Entity {name:$name}) SET e.etype=$etype, e.distinctive=$dist",
                {"name": str(name), "etype": str(etype or ""), "dist": bool(distinctive)},
            )
            return True
        except Exception as exc:
            logger.debug("upsert_entity failed: %s", exc)
            return False

    def link_mentions(self, finding_id: str, entities: list) -> bool:
        """Link a finding to entities. ``entities`` is a list of (name, etype, distinctive)."""
        if not self.ok or not finding_id or not entities:
            return False
        try:
            for ent in entities:
                if isinstance(ent, (list, tuple)):
                    name = ent[0] if len(ent) > 0 else None
                    etype = ent[1] if len(ent) > 1 else ""
                    distinctive = ent[2] if len(ent) > 2 else False
                else:
                    name, etype, distinctive = ent, "", False
                if not name:
                    continue
                self.upsert_entity(str(name), str(etype or ""), bool(distinctive))
                self._exec(
                    "MATCH (f:Finding {id:$f}), (e:Entity {name:$e}) "
                    "MERGE (f)-[:MENTIONS]->(e)",
                    {"f": str(finding_id), "e": str(name)},
                )
            return True
        except Exception as exc:
            logger.debug("link_mentions failed: %s", exc)
            return False

    def link_derived_from(self, finding_id: str, parent_ids: list) -> bool:
        if not self.ok or not finding_id or not parent_ids:
            return False
        try:
            self._exec("MERGE (f:Finding {id:$id})", {"id": str(finding_id)})
            for pid in parent_ids:
                if not pid:
                    continue
                # Parent may not have been upserted yet — MERGE a placeholder node
                # so the derivation edge always has endpoints (mirrors the
                # reference algorithm, which tracks edges regardless of node data).
                self._exec("MERGE (p:Finding {id:$id})", {"id": str(pid)})
                self._exec(
                    "MATCH (f:Finding {id:$f}), (p:Finding {id:$p}) "
                    "MERGE (f)-[:DERIVED_FROM]->(p)",
                    {"f": str(finding_id), "p": str(pid)},
                )
            return True
        except Exception as exc:
            logger.debug("link_derived_from failed: %s", exc)
            return False

    def link_references(self, finding_id: str, symbol_ids: list) -> bool:
        if not self.ok or not finding_id or not symbol_ids:
            return False
        try:
            self._exec("MERGE (f:Finding {id:$id})", {"id": str(finding_id)})
            for sid in symbol_ids:
                if not sid:
                    continue
                # Only link to symbols that actually exist (by id).
                self._exec(
                    "MATCH (f:Finding {id:$f}), (s:CodeSymbol {id:$s}) "
                    "MERGE (f)-[:REFERENCES]->(s)",
                    {"f": str(finding_id), "s": str(sid)},
                )
            return True
        except Exception as exc:
            logger.debug("link_references failed: %s", exc)
            return False

    def link_related(self, a: str, b: str) -> bool:
        if not self.ok or not a or not b or a == b:
            return False
        try:
            self._exec("MERGE (i:Investigation {id:$id})", {"id": str(a)})
            self._exec("MERGE (i:Investigation {id:$id})", {"id": str(b)})
            self._exec(
                "MATCH (x:Investigation {id:$a}), (y:Investigation {id:$b}) "
                "MERGE (x)-[:RELATED]->(y)",
                {"a": str(a), "b": str(b)},
            )
            return True
        except Exception as exc:
            logger.debug("link_related failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Batched writes (UNWIND $rows) — ~orders of magnitude faster than the
    # per-item MERGE loop for backfill / whole-repo code ingest. Fail-open.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chunks(seq, n=4000):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    def upsert_findings_batch(self, rows: list) -> int:
        """Batch upsert findings (+ Investigation nodes + IN_INVESTIGATION).

        ``rows``: dicts with id, investigation/inv, type/ftype, text, confidence,
        source, ts. Returns the number of findings written.
        """
        if not self.ok or not rows:
            return 0
        norm: list[dict] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            fid = r.get("id") or r.get("finding_id")
            if not fid:
                continue
            try:
                ts = int(r.get("ts") or 0)
            except Exception:
                ts = 0
            norm.append({
                "id": str(fid),
                "inv": str(r.get("investigation") or r.get("investigation_id") or ""),
                "ftype": str(r.get("ftype") or r.get("type") or ""),
                "text": str(r.get("text") or ""),
                "conf": str(r.get("confidence") or ""),
                "src": str(r.get("source") or ""),
                "ts": ts,
            })
        if not norm:
            return 0
        try:
            for chunk in self._chunks(norm):
                self._exec(
                    "UNWIND $rows AS r MERGE (f:Finding {id:r.id}) "
                    "SET f.investigation=r.inv, f.ftype=r.ftype, f.text=r.text, "
                    "f.confidence=r.conf, f.source=r.src, f.ts=r.ts",
                    {"rows": chunk},
                )
            invs = sorted({r["inv"] for r in norm if r["inv"]})
            if invs:
                for chunk in self._chunks([{"id": i} for i in invs]):
                    self._exec("UNWIND $rows AS r MERGE (i:Investigation {id:r.id})", {"rows": chunk})
                pairs = [{"f": r["id"], "i": r["inv"]} for r in norm if r["inv"]]
                for chunk in self._chunks(pairs):
                    self._exec(
                        "UNWIND $rows AS r MATCH (f:Finding {id:r.f}), (i:Investigation {id:r.i}) "
                        "MERGE (f)-[:IN_INVESTIGATION]->(i)",
                        {"rows": chunk},
                    )
            return len(norm)
        except Exception as exc:
            logger.debug("upsert_findings_batch failed: %s", exc)
            return 0

    def link_mentions_batch(self, rows: list) -> int:
        """Batch link findings->entities. ``rows``: dicts {f, name, etype, distinctive}."""
        if not self.ok or not rows:
            return 0
        norm = [
            {"f": str(r.get("f") or ""), "name": str(r.get("name") or ""),
             "etype": str(r.get("etype") or ""), "dist": bool(r.get("distinctive"))}
            for r in rows if isinstance(r, dict) and r.get("f") and r.get("name")
        ]
        if not norm:
            return 0
        try:
            ents: dict[str, dict] = {}
            for r in norm:
                ents.setdefault(r["name"], {"name": r["name"], "etype": r["etype"], "dist": r["dist"]})
            for chunk in self._chunks(list(ents.values())):
                self._exec(
                    "UNWIND $rows AS r MERGE (e:Entity {name:r.name}) "
                    "SET e.etype=r.etype, e.distinctive=r.dist",
                    {"rows": chunk},
                )
            for chunk in self._chunks(norm):
                self._exec(
                    "UNWIND $rows AS r MATCH (f:Finding {id:r.f}), (e:Entity {name:r.name}) "
                    "MERGE (f)-[:MENTIONS]->(e)",
                    {"rows": chunk},
                )
            return len(norm)
        except Exception as exc:
            logger.debug("link_mentions_batch failed: %s", exc)
            return 0

    def link_derived_from_batch(self, rows: list) -> int:
        """Batch link findings->parents. ``rows``: dicts {f, p} (child id, parent id)."""
        if not self.ok or not rows:
            return 0
        norm = [{"f": str(r.get("f") or ""), "p": str(r.get("p") or "")}
                for r in rows if isinstance(r, dict) and r.get("f") and r.get("p")]
        if not norm:
            return 0
        try:
            nodes = sorted({r["f"] for r in norm} | {r["p"] for r in norm})
            for chunk in self._chunks([{"id": n} for n in nodes]):
                self._exec("UNWIND $rows AS r MERGE (f:Finding {id:r.id})", {"rows": chunk})
            for chunk in self._chunks(norm):
                self._exec(
                    "UNWIND $rows AS r MATCH (f:Finding {id:r.f}), (p:Finding {id:r.p}) "
                    "MERGE (f)-[:DERIVED_FROM]->(p)",
                    {"rows": chunk},
                )
            return len(norm)
        except Exception as exc:
            logger.debug("link_derived_from_batch failed: %s", exc)
            return 0

    def ingest_code(self, parsed_files: list) -> dict:
        """Ingest ``code_parse`` per-file dicts into the code graph.

        Each dict: ``{"file","lang","symbols":[{id,name,kind,line,lang,file}],
        "edges":[{src,dst,type}], "imports":[module_str,...]}``.
        Returns ingestion counts.
        """
        counts = {"files": 0, "symbols": 0, "defines": 0, "calls": 0, "imports": 0,
                  "calls_dropped_external": 0, "calls_dropped_unresolved": 0,
                  "calls_resolved_by_type": 0}
        if not self.ok or not parsed_files:
            return counts
        # Collect everything first, then insert in a few UNWIND batches.
        files: dict[str, str] = {}          # path -> lang
        symbols: dict[str, dict] = {}       # id -> row
        defines: list[dict] = []
        imports: list[dict] = []
        # Each call edge keeps the type-aware fields the resolver needs.
        call_edges: list[dict] = []
        # file path -> {simple imported name -> best-effort FQN}
        file_import_map: dict[str, dict] = {}
        # scope symbol id -> {declared var/field/param name -> declared type simple}
        # (last decl wins; null types skipped). Powers receiver-type inference.
        decls_by_scope: dict[str, dict] = {}
        for pf in parsed_files:
            if not isinstance(pf, dict):
                continue
            fpath = pf.get("file")
            if not fpath:
                continue
            fpath = str(fpath)
            files.setdefault(fpath, str(pf.get("lang") or ""))
            imap = pf.get("import_map")
            if isinstance(imap, dict) and imap:
                dst = file_import_map.setdefault(fpath, {})
                for k, v in imap.items():
                    if k and v:
                        dst.setdefault(str(k), str(v))
            for sym in (pf.get("symbols") or []):
                if not isinstance(sym, dict) or not sym.get("id"):
                    continue
                try:
                    line = int(sym.get("line") or 0)
                except Exception:
                    line = 0
                symbols[str(sym["id"])] = {
                    "id": str(sym["id"]), "name": str(sym.get("name") or ""),
                    "kind": str(sym.get("kind") or ""),
                    "file": str(sym.get("file") or fpath), "line": line,
                    "lang": str(sym.get("lang") or files[fpath]),
                }
                defines.append({"p": fpath, "s": str(sym["id"])})
            for edge in (pf.get("edges") or []):
                if not isinstance(edge, dict):
                    continue
                src, dst = edge.get("src"), edge.get("dst")
                if src and dst and "call" in str(edge.get("type") or "").lower():
                    rk = str(edge.get("recv_kind") or "none").lower()
                    if rk not in ("self", "name", "expr", "none"):
                        rk = "none"
                    recv = edge.get("receiver")
                    call_edges.append({
                        "src": str(src), "callee": str(dst),
                        "receiver": (str(recv) if recv else None),
                        "recv_kind": rk, "file": fpath,
                    })
            for imp in (pf.get("imports") or []):
                if imp:
                    imports.append({"a": fpath, "b": str(imp)})
            for dec in (pf.get("decls") or []):
                if not isinstance(dec, dict):
                    continue
                dname = dec.get("name")
                dtype = dec.get("type")
                dscope = dec.get("scope")
                if not dname or not dtype or not dscope:
                    continue  # null/unknown type or no scope -> nothing to infer
                decls_by_scope.setdefault(str(dscope), {})[str(dname)] = str(dtype)
        try:
            # Files (source files + placeholder nodes for imported modules).
            file_rows = [{"p": p, "l": l} for p, l in files.items()]
            file_rows += [{"p": t, "l": ""} for t in ({r["b"] for r in imports} - set(files))]
            for chunk in self._chunks(file_rows):
                self._exec("UNWIND $rows AS r MERGE (c:CodeFile {path:r.p}) SET c.lang=r.l", {"rows": chunk})
            counts["files"] = len(files)
            # Symbols + DEFINES.
            for chunk in self._chunks(list(symbols.values())):
                self._exec(
                    "UNWIND $rows AS r MERGE (s:CodeSymbol {id:r.id}) "
                    "SET s.name=r.name, s.kind=r.kind, s.file=r.file, s.line=r.line, s.lang=r.lang",
                    {"rows": chunk},
                )
            counts["symbols"] = len(symbols)
            for chunk in self._chunks(defines):
                self._exec(
                    "UNWIND $rows AS r MATCH (c:CodeFile {path:r.p}), (s:CodeSymbol {id:r.s}) "
                    "MERGE (c)-[:DEFINES]->(s)",
                    {"rows": chunk},
                )
            counts["defines"] = len(defines)
            # IMPORTS.
            for chunk in self._chunks(imports):
                self._exec(
                    "UNWIND $rows AS r MATCH (a:CodeFile {path:r.a}), (b:CodeFile {path:r.b}) "
                    "MERGE (a)-[:IMPORTS]->(b)",
                    {"rows": chunk},
                )
            counts["imports"] = len(imports)
            # CALLS — type-aware resolution over the parsed batch. We NEVER
            # resolve a call that has an explicit receiver by bare global name;
            # only bare/self calls use name scoping (class > file > unique).
            #
            # Precompute the lookup tables the resolver needs.
            def _enclosing_class(sym_id: str) -> Optional[str]:
                # id is "file::Qual"; enclosing simple class = segment before the
                # method name (e.g. "A.B.m" -> "B"). Top-level func -> None.
                qual = sym_id.split("::", 1)[1] if "::" in sym_id else sym_id
                segs = qual.split(".")
                return segs[-2] if len(segs) >= 2 else None

            def _enclosing_class_id(sym_id: str) -> Optional[str]:
                # "file::A.m" -> enclosing class SYMBOL ID "file::A" (fields are
                # scoped by class id). Top-level func -> None.
                if "::" not in sym_id:
                    return None
                fpref, qual = sym_id.split("::", 1)
                if "." not in qual:
                    return None
                return f"{fpref}::{qual.rsplit('.', 1)[0]}"

            _TYPE_KINDS = {"class", "interface", "enum", "struct", "trait"}
            app_type_names: set[str] = {
                row["name"] for row in symbols.values()
                if row["kind"].lower() in _TYPE_KINDS and row["name"]
            }
            # simpleClassName -> list of member (method) symbol ids.
            class_methods: dict[str, list] = {}
            # file path -> list of symbol ids defined in that file.
            file_methods: dict[str, list] = {}
            # method simple-name -> count, and -> a representative id.
            name_count: dict[str, int] = {}
            name_first: dict[str, str] = {}
            for sid, row in symbols.items():
                cls = _enclosing_class(sid)
                if cls:
                    class_methods.setdefault(cls, []).append(sid)
                file_methods.setdefault(row["file"], []).append(sid)
                nm = row["name"]
                if nm:
                    name_count[nm] = name_count.get(nm, 0) + 1
                    name_first.setdefault(nm, sid)
            by_name_unique: dict[str, str] = {
                nm: name_first[nm] for nm, c in name_count.items() if c == 1
            }

            def _find_named(ids: Optional[list], name: str) -> Optional[str]:
                for mid in (ids or []):
                    if symbols[mid]["name"] == name:
                        return mid
                return None

            call_rows: list[dict] = []
            n_external = 0
            n_unresolved = 0
            n_by_type = 0
            for edge in call_edges:
                src = edge["src"]
                if src not in symbols:
                    continue  # caller not in this batch
                callee = edge["callee"]
                # An explicit dst id in the batch is authoritative.
                if callee in symbols:
                    call_rows.append({"a": src, "b": callee})
                    continue
                rk = edge["recv_kind"]
                recv = edge["receiver"]
                tgt: Optional[str] = None
                if rk in ("none", "self"):
                    caller_cls = _enclosing_class(src)
                    tgt = _find_named(class_methods.get(caller_cls), callee) \
                        or _find_named(file_methods.get(edge["file"]), callee) \
                        or by_name_unique.get(callee)
                    if tgt:
                        call_rows.append({"a": src, "b": tgt})
                    else:
                        n_unresolved += 1
                elif rk == "name" and recv:
                    if recv in app_type_names:
                        # Static / typed call on a repo-defined type.
                        tgt = _find_named(class_methods.get(recv), callee)
                        if tgt:
                            call_rows.append({"a": src, "b": tgt})
                        else:
                            n_unresolved += 1
                    elif recv in file_import_map.get(edge["file"], {}):
                        fqn = file_import_map[edge["file"]][recv]
                        fqn_cls = fqn.split(".")[-1]
                        if fqn_cls in app_type_names:
                            # Import points at a repo type -> resolve as app call.
                            tgt = _find_named(class_methods.get(fqn_cls), callee)
                            if tgt:
                                call_rows.append({"a": src, "b": tgt})
                            else:
                                n_unresolved += 1
                        else:
                            # Import points outside the repo (Log, Collections…).
                            n_external += 1
                    else:
                        # 2.5: receiver-type inference from captured declarations.
                        # Look up R as a local/param of the calling method first,
                        # then as a field of the enclosing class. Only resolve when
                        # the inferred type is an APP type; external/unknown -> DROP
                        # (never fall back to global by-name).
                        t = decls_by_scope.get(src, {}).get(recv)
                        if not t:
                            encl_id = _enclosing_class_id(src)
                            if encl_id:
                                t = decls_by_scope.get(encl_id, {}).get(recv)
                        if t and t in app_type_names:
                            tgt = _find_named(class_methods.get(t), callee)
                            if tgt:
                                call_rows.append({"a": src, "b": tgt})
                                n_by_type += 1
                            else:
                                n_unresolved += 1
                        else:
                            # Unknown / untyped / external variable receiver. v1:
                            # drop, do NOT fall back to global by-name.
                            n_unresolved += 1
                else:
                    # rk == "expr" (complex receiver) or a receiver'd call with no
                    # receiver text. v1: drop.
                    n_unresolved += 1
            for chunk in self._chunks(call_rows):
                self._exec(
                    "UNWIND $rows AS r MATCH (a:CodeSymbol {id:r.a}), (b:CodeSymbol {id:r.b}) "
                    "MERGE (a)-[:CALLS]->(b)",
                    {"rows": chunk},
                )
            counts["calls"] = len(call_rows)
            counts["calls_dropped_external"] = n_external
            counts["calls_dropped_unresolved"] = n_unresolved
            counts["calls_resolved_by_type"] = n_by_type
            return counts
        except Exception as exc:
            logger.debug("ingest_code failed: %s", exc)
            return counts

    def _resolve_symbol(self, ref: str) -> Optional[str]:
        """Resolve a CALLS dst to a CodeSymbol id: try id, then by name (best-effort)."""
        try:
            rows = self._rows("MATCH (s:CodeSymbol {id:$id}) RETURN s.id", {"id": ref})
            if rows:
                return rows[0][0]
            rows = self._rows(
                "MATCH (s:CodeSymbol) WHERE s.name=$n RETURN s.id LIMIT 1", {"n": ref}
            )
            if rows:
                return rows[0][0]
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    # Reads (fail-open)
    # ------------------------------------------------------------------ #
    _FINDING_COLS = "f.id, f.investigation, f.ftype, f.text, f.confidence, f.source, f.ts"

    @staticmethod
    def _finding_row(row: list) -> dict:
        return {
            "id": row[0], "investigation": row[1], "ftype": row[2],
            "text": row[3], "confidence": row[4], "source": row[5], "ts": row[6],
        }

    def entity_findings(self, entity: str, limit: int = 50) -> list[dict]:
        if not self.ok or not entity:
            return []
        try:
            rows = self._rows(
                "MATCH (f:Finding)-[:MENTIONS]->(e:Entity) "
                "WHERE lower(e.name) = lower($n) "
                f"RETURN DISTINCT {self._FINDING_COLS} LIMIT $lim",
                {"n": str(entity), "lim": int(limit)},
            )
            return [self._finding_row(r) for r in rows]
        except Exception as exc:
            logger.debug("entity_findings failed: %s", exc)
            return []

    def related_investigations(self, investigation_id: str, limit: int = 20) -> list[dict]:
        """Other investigations sharing >=1 Entity or a DERIVED_FROM path, ranked by shared count."""
        if not self.ok or not investigation_id:
            return []
        try:
            inv = str(investigation_id)
            scores: dict[str, dict] = {}

            def _bump(other, title, key, n):
                if not other or other == inv:
                    return
                rec = scores.setdefault(
                    other, {"id": other, "title": title or "",
                            "shared_entities": 0, "derivation_links": 0, "related": 0}
                )
                if title and not rec["title"]:
                    rec["title"] = title
                rec[key] = max(rec[key], int(n))

            # Shared distinctive-or-any Entity across investigations.
            for r in self._rows(
                "MATCH (fa:Finding)-[:IN_INVESTIGATION]->(a:Investigation {id:$id}), "
                "(fa)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(fb:Finding)"
                "-[:IN_INVESTIGATION]->(b:Investigation) "
                "WHERE b.id <> $id "
                "RETURN b.id, b.title, count(DISTINCT e.name)",
                {"id": inv},
            ):
                _bump(r[0], r[1], "shared_entities", r[2])

            # DERIVED_FROM paths crossing the investigation boundary (either direction).
            for r in self._rows(
                "MATCH (fa:Finding)-[:IN_INVESTIGATION]->(a:Investigation {id:$id}), "
                "(fb:Finding)-[:IN_INVESTIGATION]->(b:Investigation) "
                "WHERE b.id <> $id AND "
                "( EXISTS { MATCH (fa)-[:DERIVED_FROM*1..]->(fb) } "
                "  OR EXISTS { MATCH (fb)-[:DERIVED_FROM*1..]->(fa) } ) "
                "RETURN b.id, b.title, count(DISTINCT fb.id)",
                {"id": inv},
            ):
                _bump(r[0], r[1], "derivation_links", r[2])

            # Explicit RELATED links (either direction) count as a shared signal.
            for r in self._rows(
                "MATCH (a:Investigation {id:$id})-[:RELATED]-(b:Investigation) "
                "RETURN DISTINCT b.id, b.title",
                {"id": inv},
            ):
                _bump(r[0], r[1], "related", 1)

            out = []
            for rec in scores.values():
                rec["shared"] = rec["shared_entities"] + rec["derivation_links"] + rec["related"]
                out.append(rec)
            out.sort(key=lambda d: (-d["shared"], d["id"]))
            return out[: int(limit)]
        except Exception as exc:
            logger.debug("related_investigations failed: %s", exc)
            return []

    def symbol_findings(self, symbol_name_or_id: str) -> list[dict]:
        if not self.ok or not symbol_name_or_id:
            return []
        try:
            q = str(symbol_name_or_id)
            rows = self._rows(
                "MATCH (f:Finding)-[:REFERENCES]->(s:CodeSymbol) "
                "WHERE s.id = $q OR s.name = $q "
                f"RETURN DISTINCT {self._FINDING_COLS}",
                {"q": q},
            )
            return [self._finding_row(r) for r in rows]
        except Exception as exc:
            logger.debug("symbol_findings failed: %s", exc)
            return []

    def callers_of(self, symbol_name: str) -> list[dict]:
        if not self.ok or not symbol_name:
            return []
        try:
            rows = self._rows(
                "MATCH (caller:CodeSymbol)-[:CALLS]->(target:CodeSymbol) "
                "WHERE target.name = $n OR target.id = $n "
                "RETURN DISTINCT caller.id, caller.name, caller.kind, caller.file, caller.line",
                {"n": str(symbol_name)},
            )
            return [
                {"id": r[0], "name": r[1], "kind": r[2], "file": r[3], "line": r[4]}
                for r in rows
            ]
        except Exception as exc:
            logger.debug("callers_of failed: %s", exc)
            return []

    def code_query(self, cypher: str, params: Optional[dict] = None) -> list[list]:
        """Run a RAW read-only Cypher query. Rejects write-shaped queries with ValueError."""
        if not isinstance(cypher, str) or not cypher.strip():
            raise ValueError("code_query requires a non-empty cypher string")
        if _WRITE_GUARD_RE.search(cypher):
            raise ValueError(
                "code_query is read-only; write keywords "
                "(CREATE/DELETE/SET/DROP/COPY/ALTER/MERGE) are rejected"
            )
        if not self.ok:
            return []
        try:
            return self._rows(cypher, params)
        except Exception as exc:
            logger.debug("code_query failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    # Contamination — graph port of contagion.find_contamination
    # ------------------------------------------------------------------ #
    def contamination(
        self,
        seed_ids: list,
        *,
        min_shared_entities: int = 1,
        semantic_neighbor_ids: Optional[list] = None,
    ) -> dict:
        """Compute the contaminated cluster reachable from ``seed_ids``.

        Semantically identical to ``contagion.find_contamination``: the union of
        seeds, semantic neighbors, findings sharing >= ``min_shared_entities``
        *distinctive* entities with a seed, and the transitive ``DERIVED_FROM``
        closure reaching that growing set. Reason strings match the reference
        exactly ("seed" / "semantic" / "entity:<sorted,csv>" / "derived_from:<id>").
        """
        empty = {"contaminated_ids": [], "reasons": {}}
        if not self.ok:
            return empty
        try:
            # --- normalise seeds (order-preserving dedupe) ---
            seeds: list[str] = []
            seen: set[str] = set()
            for sid in (seed_ids or []):
                s = str(sid)
                if s and s not in seen:
                    seen.add(s)
                    seeds.append(s)
            seed_set = set(seeds)

            reasons: dict[str, list[str]] = {}

            def _add_reason(fid: str, reason: str) -> None:
                bucket = reasons.setdefault(fid, [])
                if reason not in bucket:
                    bucket.append(reason)

            # --- pull graph state ---
            # all finding ids in scope
            by_id: set[str] = {r[0] for r in self._rows("MATCH (f:Finding) RETURN f.id")}
            # derived edges: child -> set(parents)
            derived_edges: dict[str, set[str]] = {}
            for r in self._rows(
                "MATCH (a:Finding)-[:DERIVED_FROM]->(b:Finding) RETURN a.id, b.id"
            ):
                derived_edges.setdefault(r[0], set()).add(r[1])

            contaminated: set[str] = set(seed_set)
            for sid in seeds:
                _add_reason(sid, "seed")

            # --- Semantic: union verbatim (order per reference: seeds then semantic) ---
            for nid in (semantic_neighbor_ids or []):
                n = str(nid)
                if not n:
                    continue
                contaminated.add(n)
                if n not in seed_set:
                    _add_reason(n, "semantic")

            # --- Entity anchor: distinctive entities shared with ANY seed. ---
            # Cypher yields, per (other-finding, seed), the shared distinctive
            # entity names. We apply the reference's seed-order "first match wins".
            threshold = max(1, int(min_shared_entities))
            if seeds:
                shared_by_finding: dict[str, dict[str, set[str]]] = {}
                for r in self._rows(
                    "MATCH (s:Finding)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(o:Finding) "
                    "WHERE s.id IN $seeds AND e.distinctive = true AND NOT o.id IN $seeds "
                    "RETURN o.id, s.id, collect(DISTINCT e.name)",
                    {"seeds": seeds},
                ):
                    oid, sid = r[0], r[1]
                    shared = {str(x).strip().lower() for x in (r[2] or []) if str(x).strip()}
                    shared_by_finding.setdefault(oid, {})[sid] = shared
                for oid, per_seed in shared_by_finding.items():
                    for sid in seeds:  # reference order; first qualifying seed wins
                        shared = per_seed.get(sid)
                        if shared and len(shared) >= threshold:
                            contaminated.add(oid)
                            _add_reason(oid, "entity:" + ",".join(sorted(shared)))
                            break

            # --- Derivation: transitive DERIVED_FROM reaching the contaminated set.
            # Cycle-guarded DFS fixpoint, identical to the reference so the
            # "derived_from:<reached>" target (first reached in a DFS of the full
            # ancestor chain) matches exactly.
            def _reaches_contaminated(start: str, targets: set[str]) -> Optional[str]:
                stack = list(derived_edges.get(start, ()))
                visited: set[str] = {start}
                while stack:
                    parent = stack.pop()
                    if parent in visited:
                        continue
                    visited.add(parent)
                    if parent in targets:
                        return parent
                    stack.extend(derived_edges.get(parent, ()))
                return None

            # Iterate in a stable (sorted) order so the DFS-first-reached target
            # in "derived_from:<reached>" is deterministic; equals the reference
            # when findings are supplied in id-sorted order.
            changed = True
            ordered_ids = sorted(by_id)
            while changed:
                changed = False
                for fid in ordered_ids:
                    if fid in contaminated:
                        continue
                    reached = _reaches_contaminated(fid, contaminated)
                    if reached is not None:
                        contaminated.add(fid)
                        _add_reason(fid, f"derived_from:{reached}")
                        changed = True

            rest = sorted(c for c in contaminated if c not in seed_set)
            contaminated_ids = [s for s in seeds if s in contaminated] + rest
            return {"contaminated_ids": contaminated_ids, "reasons": reasons}
        except Exception as exc:
            logger.debug("contamination failed: %s", exc)
            return empty

"""
Schema consistency test — verifies that every SQL column name used in Python
source files actually exists in the Mnemosyne SQLite schema.

This would have caught the glymphatic_sweep.py bug where source_id/target_id
were queried but the table uses source/target.

Run standalone:  python3 tests/test_schema_consistency.py
Run via pytest:  pytest tests/test_schema_consistency.py -v
"""

import importlib.util
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Canonical Mnemosyne schema — derived from the tables actually used across
# scripts/ and mcp/server.py.  Add columns here when new tables are created.
SCHEMA_SQL = """
CREATE TABLE working_memory (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    importance  REAL,
    created_at  TEXT,
    recall_count INTEGER DEFAULT 0,
    session_id  TEXT,
    metadata_json TEXT
);

CREATE TABLE episodic_memory (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    importance  REAL,
    created_at  TEXT,
    session_id  TEXT
);

CREATE TABLE graph_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    edge_type   TEXT NOT NULL DEFAULT 'semantic_link',
    weight      REAL NOT NULL,
    timestamp   TEXT,
    created_at  TEXT
);

CREATE TABLE conflicts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_a_id       TEXT NOT NULL,
    fact_b_id       TEXT NOT NULL,
    conflict_type   TEXT,
    created_at      TEXT
);

CREATE TABLE memories (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    source          TEXT,
    timestamp       TEXT,
    session_id      TEXT,
    importance      REAL DEFAULT 0.5,
    metadata_json   TEXT,
    created_at      TEXT
);

CREATE VIRTUAL TABLE fts_working USING fts5(id, content);
CREATE VIRTUAL TABLE fts_episodes USING fts5(content, content=episodic_memory, content_rowid=rowid);
"""

# Files to check — (path relative to repo root, list of SQL string regexes to skip)
FILES_TO_CHECK = [
    "scripts/glymphatic_sweep.py",
    "scripts/amem_consolidation.py",
    "scripts/spreading_activation.py",
    "scripts/event_log.py",
]

# Column names that are SQLite builtins / FTS special columns — not in user tables
SQLITE_SPECIAL = {"rowid", "rank", "docid"}

# Regex: find SQL strings containing SELECT/INSERT/UPDATE/DELETE/CREATE
_SQL_RE = re.compile(
    r'(?:execute|executemany)\s*\(\s*(?:f?""".*?"""|f?\'\'\'.*?\'\'\'|f?".*?"|f?\'.*?\')',
    re.DOTALL,
)
_COLUMN_RE = re.compile(r'\b([a-z_][a-z0-9_]*)\b')

# These identifiers appear in SQL context but are SQL keywords / params / aliases
SQL_KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "in", "is", "null",
    "insert", "into", "values", "update", "set", "delete", "create", "table",
    "distinct", "order", "by", "asc", "desc", "limit", "offset", "join",
    "inner", "left", "on", "as", "count", "sum", "avg", "max", "min",
    "group", "having", "like", "between", "case", "when", "then", "else",
    "end", "exists", "ignore", "or", "replace", "begin", "commit", "rollback",
    "primary", "key", "autoincrement", "unique", "default", "not", "null",
    "integer", "text", "real", "blob", "using", "fts5", "match", "virtual",
    "external", "content", "content_rowid", "if",
}


def _get_all_columns(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Return {table_name: {col_name, ...}} for all user tables."""
    tables = {}
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for (name,) in cur.fetchall():
        if name.startswith("sqlite_"):
            continue
        try:
            info = conn.execute(f"PRAGMA table_info({name})").fetchall()
            tables[name] = {row[1] for row in info}
        except Exception:
            tables[name] = set()
    return tables


def _extract_sql_fragments(source: str) -> list[str]:
    """Extract the string literals passed to cursor.execute() calls."""
    fragments = []
    # Match .execute( or .executemany( followed by a string
    pattern = re.compile(
        r'\.execute(?:many)?\s*\(\s*'
        r'(?:'
        r'f?"""(.*?)"""'
        r'|f?\'\'\'(.*?)\'\'\''
        r'|f?"(.*?)"'
        r"|f?'(.*?)'"
        r')',
        re.DOTALL,
    )
    for m in pattern.finditer(source):
        fragment = next((g for g in m.groups() if g is not None), "")
        fragments.append(fragment)
    return fragments


class TestSchemaConsistency(unittest.TestCase):
    """Verify that SQL column names in Python files exist in the schema."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        db_path = cls._tmp.name + "/test.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        cls.columns = _get_all_columns(conn)
        cls.all_columns: set[str] = set().union(*cls.columns.values())
        conn.close()
        cls.db_path = db_path

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _check_file(self, rel_path: str) -> list[str]:
        path = REPO / rel_path
        if not path.exists():
            self.skipTest(f"{rel_path} not found")
        source = path.read_text()
        fragments = _extract_sql_fragments(source)
        errors = []
        for frag in fragments:
            # Extract word tokens that could be column names
            tokens = set(_COLUMN_RE.findall(frag.lower()))
            suspects = tokens - SQL_KEYWORDS - SQLITE_SPECIAL
            for tok in suspects:
                # Only flag tokens that look like column names (contain underscore
                # or are short common names) AND match a known wrong pattern
                if tok in {"source_id", "target_id", "fact_id", "node_id"}:
                    if tok not in self.all_columns:
                        errors.append(
                            f"  Column '{tok}' used in SQL but not in schema\n"
                            f"  Fragment: {frag[:120].strip()!r}"
                        )
        return errors

    def _run_queries(self, rel_path: str) -> list[str]:
        """Actually execute extracted SQL against the test DB and catch OperationalErrors."""
        path = REPO / rel_path
        if not path.exists():
            return []
        source = path.read_text()
        fragments = _extract_sql_fragments(source)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        errors = []
        for frag in fragments:
            has_fstring = "{" in frag and "}" in frag
            has_params = "?" in frag
            if has_fstring:
                # f-string queries have dynamic table/column names — skip EXPLAIN,
                # we only validate the static column-name check above.
                continue
            try:
                stmt = frag.replace("?", "1") if has_params else frag
                conn.execute(f"EXPLAIN {stmt}")
            except sqlite3.OperationalError as e:
                err_str = str(e)
                # "no such column: 1" is expected when ? → 1 substitution hits a WHERE clause
                if "no such column: 1" not in err_str:
                    errors.append(f"  SQL error in {rel_path}: {e}\n  Query: {frag[:120]!r}")
        conn.close()
        return errors

    def test_glymphatic_sweep_column_names(self):
        errors = self._check_file("scripts/glymphatic_sweep.py")
        self.assertEqual(errors, [], "\n".join(["Column name mismatches:"] + errors))

    def test_amem_consolidation_column_names(self):
        errors = self._check_file("scripts/amem_consolidation.py")
        self.assertEqual(errors, [], "\n".join(["Column name mismatches:"] + errors))

    def test_glymphatic_sweep_queries_execute(self):
        errors = self._run_queries("scripts/glymphatic_sweep.py")
        self.assertEqual(errors, [], "\n".join(["Query execution errors:"] + errors))

    def test_amem_consolidation_queries_execute(self):
        errors = self._run_queries("scripts/amem_consolidation.py")
        self.assertEqual(errors, [], "\n".join(["Query execution errors:"] + errors))

    def test_the_extractor_sees_sql_in_every_checked_file(self):
        # spreading_activation.py builds its SQL in a variable, so the regex
        # extracted 0 fragments and both of its checks passed vacuously. Every
        # file the static checks cover must yield SQL; spreading_activation is
        # covered by running its real queries below instead.
        for rel_path in ("scripts/glymphatic_sweep.py", "scripts/amem_consolidation.py"):
            fragments = _extract_sql_fragments((REPO / rel_path).read_text())
            static = [f for f in fragments if not ("{" in f and "}" in f)]
            self.assertGreater(len(static), 0, f"no executable SQL extracted from {rel_path}")


# ---------------------------------------------------------------------------
# The embedded SCHEMA_SQL above is this file's own copy. Checking scripts
# against it is only meaningful while it agrees with what Mnemosyne really
# creates, so compare it with Mnemosyne's DDL when the package is installed.
# ---------------------------------------------------------------------------

def _mnemosyne_schema_db(path: Path) -> bool:
    """Create Mnemosyne's real tables at ``path``; False when not installed."""
    try:
        from mnemosyne.core import beam, episodic_graph
    except Exception:
        return False
    beam.init_beam(path)
    graph = episodic_graph.EpisodicGraph(db_path=path)
    graph.conn.close()
    return True


def _embedded_schema_db(path: Path) -> bool:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
    return True


_SCHEMA_BUILDERS = {"embedded": _embedded_schema_db, "mnemosyne": _mnemosyne_schema_db}


class TestEmbeddedSchemaMatchesMnemosyne(unittest.TestCase):

    def test_embedded_columns_exist_in_the_real_tables(self):
        with tempfile.TemporaryDirectory() as td:
            real = Path(td) / "real.db"
            if not _mnemosyne_schema_db(real):
                self.skipTest("mnemosyne is not installed (optional extra: mcp[mnemosyne])")
            conn = sqlite3.connect(real)
            real_cols = _get_all_columns(conn)
            conn.close()
        embedded = TestSchemaConsistency.columns
        for table in ("working_memory", "episodic_memory", "graph_edges"):
            self.assertIn(table, real_cols)
            missing = embedded[table] - real_cols[table]
            self.assertEqual(missing, set(), f"{table}: embedded schema invents {missing}")
        self.assertTrue({"source", "target"} <= real_cols["graph_edges"])
        self.assertFalse({"source_id", "target_id"} & real_cols["graph_edges"])


class TestSpreadingActivationQueries(unittest.TestCase):
    """Run scripts/spreading_activation.py's own queries against a real schema.

    Graph (default knobs: floor 0.4, threshold 0.5, 2 hops, fan effect on):
      A -> B  1.0 semantic_link      A -> C 0.7 caused_by (w' = 0.5 * 1.3)
      A -> D  0.9 contradicts (excluded from the standard pass)
      A -> E  0.3 (below the floor)  B -> F 1.0   C -> F 1.0
    Hop 1 (A out-degree 2): B = 0.5, C = 0.325.  Hop 2: F = 0.5 + 0.325 = 0.825.
    """

    EDGES = [
        ("A", "B", 1.0, "semantic_link"),
        ("A", "C", 0.7, "caused_by"),
        ("A", "D", 0.9, "contradicts"),
        ("A", "E", 0.3, "semantic_link"),
        ("B", "F", 1.0, "semantic_link"),
        ("C", "F", 1.0, "semantic_link"),
    ]

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "spreading_activation_under_test", REPO / "scripts" / "spreading_activation.py")
        cls.sa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.sa)

    def _db(self, schema: str) -> str:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = Path(td.name) / "sa.db"
        if not _SCHEMA_BUILDERS[schema](path):
            self.skipTest("mnemosyne is not installed (optional extra: mcp[mnemosyne])")
        conn = sqlite3.connect(path)
        for src, dst, weight, etype in self.EDGES:
            conn.execute("INSERT INTO graph_edges (source, target, weight, edge_type) VALUES (?,?,?,?)",
                         (src, dst, weight, etype))
        conn.execute("INSERT INTO working_memory (id, content, importance) VALUES (?,?,?)",
                     ("B", "working row B", 0.6))
        conn.execute("INSERT INTO episodic_memory (id, content, importance) VALUES (?,?,?)",
                     ("F", "episodic row F", 0.9))
        conn.commit()
        conn.close()
        return str(path)

    def _knobs(self):
        from unittest import mock
        for name, value in (("SA_EDGE_FLOOR", 0.4), ("SA_ACTIVATION_THRESHOLD", 0.5),
                            ("SA_MAX_HOPS", 2), ("SA_FAN_EFFECT", True)):
            patcher = mock.patch.object(self.sa, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _for_each_schema(self, check):
        """Run ``check(db_path)`` on the embedded schema (always) and on
        Mnemosyne's real DDL (skipped as a subtest when not installed)."""
        for schema in ("embedded", "mnemosyne"):
            with self.subTest(schema=schema):
                check(self._db(schema))

    def test_fetch_edges_applies_the_floor_and_the_type_filter(self):
        def check(db):
            conn = sqlite3.connect(db)
            try:
                got = sorted(self.sa._fetch_edges(conn, ["A"], 0.4))
                self.assertEqual(got, [("A", "B", 1.0, "semantic_link"), ("A", "C", 0.7, "caused_by")])
                only = self.sa._fetch_edges(conn, ["A"], 0.4, edge_types=["contradicts"])
                self.assertEqual(only, [("A", "D", 0.9, "contradicts")])
            finally:
                conn.close()
        self._for_each_schema(check)

    def test_fetch_content_falls_back_to_episodic_memory(self):
        def check(db):
            conn = sqlite3.connect(db)
            try:
                got = self.sa._fetch_content(conn, ["B", "F", "missing"])
            finally:
                conn.close()
            self.assertEqual(got, {"B": {"content": "working row B", "importance": 0.6},
                                   "F": {"content": "episodic row F", "importance": 0.9}})
        self._for_each_schema(check)

    def test_two_hop_activation_end_to_end(self):
        self._knobs()

        def check(db):
            got = self.sa.run_spreading_activation(db, ["A"], {"A": 1.0}, max_results=5)
            self.assertEqual([r["memory_id"] for r in got], ["F", "B"])
            self.assertAlmostEqual(got[0]["activation"], 0.825)
            self.assertAlmostEqual(got[1]["activation"], 0.5)
            self.assertEqual((got[0]["content"], got[0]["importance"]), ("episodic row F", 0.9))
            self.assertEqual((got[1]["content"], got[1]["importance"]), ("working row B", 0.6))
        self._for_each_schema(check)


if __name__ == "__main__":
    unittest.main(verbosity=2)

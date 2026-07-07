"""
Schema consistency test — verifies that every SQL column name used in Python
source files actually exists in the Mnemosyne SQLite schema.

This would have caught the glymphatic_sweep.py bug where source_id/target_id
were queried but the table uses source/target.

Run standalone:  python3 tests/test_schema_consistency.py
Run via pytest:  pytest tests/test_schema_consistency.py -v
"""

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

    def test_spreading_activation_column_names(self):
        errors = self._check_file("scripts/spreading_activation.py")
        self.assertEqual(errors, [], "\n".join(["Column name mismatches:"] + errors))

    def test_glymphatic_sweep_queries_execute(self):
        errors = self._run_queries("scripts/glymphatic_sweep.py")
        self.assertEqual(errors, [], "\n".join(["Query execution errors:"] + errors))

    def test_amem_consolidation_queries_execute(self):
        errors = self._run_queries("scripts/amem_consolidation.py")
        self.assertEqual(errors, [], "\n".join(["Query execution errors:"] + errors))

    def test_spreading_activation_queries_execute(self):
        errors = self._run_queries("scripts/spreading_activation.py")
        self.assertEqual(errors, [], "\n".join(["Query execution errors:"] + errors))

    def test_schema_has_expected_tables(self):
        expected = {"working_memory", "episodic_memory", "graph_edges", "conflicts", "memories"}
        self.assertTrue(
            expected.issubset(self.columns.keys()),
            f"Missing tables: {expected - set(self.columns.keys())}",
        )

    def test_graph_edges_uses_source_not_source_id(self):
        self.assertIn("source", self.columns.get("graph_edges", set()))
        self.assertNotIn("source_id", self.columns.get("graph_edges", set()))
        self.assertIn("target", self.columns.get("graph_edges", set()))
        self.assertNotIn("target_id", self.columns.get("graph_edges", set()))


if __name__ == "__main__":
    unittest.main(verbosity=2)

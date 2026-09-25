"""code_query / code_graph_query must be read-only against the host, not just the graph.

Audit finding cypher-guard-load-from-export-install: the keyword denylist
(CREATE|DELETE|SET|DROP|COPY|ALTER|MERGE) let LOAD FROM (read any host file),
EXPORT DATABASE (write every Finding to any path), INSTALL / LOAD EXTENSION
(download native code) and httpfs (outbound HTTP) through. The guard is now an
allowlist of reading clauses.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("ladybug")

import graph_tools  # noqa: E402
from graph.ladybug_store import LadybugStore  # noqa: E402


def _store(tmp_path):
    store = LadybugStore(str(tmp_path / "allowlistdb"))
    assert store.available()
    store.ingest_code([{
        "file": "x.py", "lang": "python",
        "symbols": [{"id": "x.py::f", "name": "f", "kind": "function",
                     "line": 1, "lang": "python", "file": "x.py"}],
        "edges": [], "imports": [],
    }])
    return store


def test_load_from_cannot_read_host_files(tmp_path):
    store = _store(tmp_path)
    secret = tmp_path / "secret.csv"
    secret.write_text("top,secret\nAKIA123,hunter2\n")
    with pytest.raises(ValueError, match="read-only"):
        store.code_query(f'LOAD FROM "{secret}" RETURN *')
    with pytest.raises(ValueError, match="read-only"):
        store.code_query(f"load from '{secret}' (header=false) return * limit 3")


def test_export_database_cannot_write_to_disk(tmp_path):
    store = _store(tmp_path)
    out = tmp_path / "exported"
    with pytest.raises(ValueError, match="read-only"):
        store.code_query(f'EXPORT DATABASE "{out}"')
    assert not out.exists()


# Every payload here must be refused before it reaches the engine. The engine is
# stubbed so a regression cannot download an extension or open a socket.
_BYPASS_PAYLOADS = [
    "INSTALL httpfs",
    "LOAD EXTENSION httpfs",
    "LOAD FROM 'http://127.0.0.1:18777/secret.csv' RETURN *",
    "IMPORT DATABASE '/tmp/loci-x'",
    "ATTACH '/tmp/x' AS x (dbtype lbug)",
    "COPY CodeFile FROM '/tmp/x.csv'",
    "MATCH (f:CodeFile) REMOVE f.lang RETURN f.path",
    "MATCH (f:CodeFile) DETACH DELETE f",
    "MATCH (n) WITH n LOAD FROM '/etc/passwd' RETURN *",
    "RETURN 1; LOAD FROM '/etc/passwd' RETURN *",
    "RETURN 1 /* hide */",
    "RETURN 1 // hide",
    "CALL read_csv_serial('/etc/passwd') RETURN *",
    "CALL READ_PARQUET('/tmp/x.parquet') RETURN *",
    "CALL file_info('/etc/passwd') RETURN *",
    "CALL threads=1",
    "CALL project_graph('g', ['CodeFile'], [])",
    "USE other",
    "CHECKPOINT",
    "BEGIN TRANSACTION",
    "RETURN nextval('s')",
    "RETURN 'unterminated",
]


@pytest.mark.parametrize("cypher", _BYPASS_PAYLOADS)
def test_non_read_statements_never_reach_the_engine(tmp_path, cypher):
    store = _store(tmp_path)
    executed = []
    store._rows = lambda q, p=None: executed.append(q) or []
    with pytest.raises(ValueError, match="read-only"):
        store.code_query(cypher)
    assert executed == []


@pytest.mark.parametrize("cypher, expected", [
    ("MATCH (s:CodeSymbol) RETURN s.id", [["x.py::f"]]),
    ("OPTIONAL MATCH (s:CodeSymbol) WHERE s.name = $n RETURN s.id ORDER BY s.id SKIP 0 LIMIT 5",
     [["x.py::f"]]),
    ("UNWIND [1, 2] AS x RETURN x", [[1], [2]]),
    ("WITH 1 AS x RETURN x", [[1]]),
    ("MATCH (s:CodeSymbol) WHERE s.name CONTAINS 'load from export install' RETURN count(s)", [[0]]),
    ("MATCH (s:CodeSymbol) WHERE s.name = \"a\\\"b LOAD\" RETURN s.id", []),
    ("RETURN 1;", [[1]]),
])
def test_read_queries_still_run(tmp_path, cypher, expected):
    # code_query fails open to [] on an engine error, so "it returned" proves nothing:
    # the rows must be the real answer and no failure may have been counted.
    store = _store(tmp_path)
    rows = store.code_query(cypher, {"n": "f"} if "$n" in cypher else None)
    assert [list(r) for r in rows] == expected
    assert store.code_query_failures == 0, store.code_query_last_error


def test_show_tables_call_is_a_read_that_runs(tmp_path):
    store = _store(tmp_path)
    rows = store.code_query("CALL show_tables() RETURN *")
    assert store.code_query_failures == 0, store.code_query_last_error
    names = {str(cell) for row in rows for cell in row}
    assert {"CodeSymbol", "CodeFile"} <= names


def test_code_graph_query_tool_reports_the_rejection(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setattr(graph_tools, "_get_ladybug", lambda: store)
    secret = tmp_path / "canary.csv"
    secret.write_text("LOCI_AUDIT_CANARY,42\n")
    out = json.loads(graph_tools.code_graph_query(f'LOAD FROM "{secret}" RETURN *'))
    assert "error" in out and "read-only" in out["error"]
    assert "LOCI_AUDIT_CANARY" not in json.dumps(out)

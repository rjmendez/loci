"""ladybug_ops: a graph read that FAILED must not be reported as a graph that is EMPTY.

Both paths here fail open on purpose; what is under test is that the failure leaves
a trace instead of a clean, wrong-looking success.
"""
import json
import logging

import pytest

import ladybug_ops as L


class FakeStore:
    """Stands in for LadybugStore: code_query fails open to [] the way the real one does."""

    def __init__(self, count_rows, write_returns=(0, 0, 0)):
        self.count_rows = count_rows
        self.write_returns = write_returns
        self.investigations = []

    def code_query(self, cypher, params=None):
        return self.count_rows

    def _rows(self, cypher, params=None):
        return [["a.py::foo", "foo", "function", "a.py"]]

    def upsert_investigation(self, iv, title=""):
        self.investigations.append(iv)

    def upsert_findings_batch(self, rows):
        return self.write_returns[0]

    def link_mentions_batch(self, rows):
        return self.write_returns[1]

    def link_derived_from_batch(self, rows):
        return self.write_returns[2]


@pytest.fixture(autouse=True)
def _clean_index_cache():
    L.invalidate_symbol_index()
    yield
    L.invalidate_symbol_index()


# --------------------------------------------------------------------------- #
# _get_symbol_index
# --------------------------------------------------------------------------- #

def test_failed_count_query_keeps_the_warm_index_instead_of_evicting_it(caplog):
    """A count query ALWAYS returns one row, so [] is the query failing.

    Evicting here drops every subsequent finding's REFERENCES edges on a transient
    store error, and _autolink then reports it as "no code graph ingested yet".
    """
    L._symbol_index_cache, L._symbol_index_count = {"warm": ["a.py::foo"]}, 7
    with caplog.at_level(logging.WARNING, logger="loci-mcp"):
        index = L._get_symbol_index(FakeStore(count_rows=[]))

    assert index == {"warm": ["a.py::foo"]}
    assert L._symbol_index_cache == {"warm": ["a.py::foo"]}
    assert L._symbol_index_count == 7
    assert "count query failed" in caplog.text


def test_a_genuinely_empty_graph_still_clears_the_index():
    L._symbol_index_cache, L._symbol_index_count = {"warm": ["a.py::foo"]}, 7
    assert L._get_symbol_index(FakeStore(count_rows=[[0]])) is None
    assert L._symbol_index_cache is None
    assert L._symbol_index_count == 0


# --------------------------------------------------------------------------- #
# _ladybug_backfill_if_empty
# --------------------------------------------------------------------------- #

def _seed_memory(tmp_path):
    inv = tmp_path / "inv1"
    inv.mkdir()
    (inv / "findings.jsonl").write_text(
        json.dumps({"id": "f1", "text": "t", "entities": {"file": ["a.py"]},
                    "derived_from": ["f0"]}) + "\n"
    )
    return tmp_path


def test_backfill_logs_what_was_written_not_what_was_offered(tmp_path, caplog, monkeypatch):
    """Under a contended write lease every batch writes nothing and returns 0."""
    monkeypatch.setattr(L, "_get_memory_dir", lambda: _seed_memory(tmp_path))
    with caplog.at_level(logging.INFO, logger="loci-mcp"):
        L._ladybug_backfill_if_empty(FakeStore(count_rows=[[0]], write_returns=(0, 0, 0)))

    assert "mirrored 0 findings, 0 mentions, 0 derivations" in caplog.text
    assert "short-wrote" in caplog.text


def test_backfill_is_quiet_when_every_batch_wrote_its_input(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(L, "_get_memory_dir", lambda: _seed_memory(tmp_path))
    with caplog.at_level(logging.INFO, logger="loci-mcp"):
        L._ladybug_backfill_if_empty(FakeStore(count_rows=[[0]], write_returns=(1, 1, 1)))

    assert "mirrored 1 findings, 1 mentions, 1 derivations" in caplog.text
    assert "short-wrote" not in caplog.text

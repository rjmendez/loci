"""Tests for the context bridge's echo suppression and source selection.

The bridge relays memories to mesh peers. Three separate defects made scheduling it unsafe:

1. It read `working_memory` and only fell back to `memories` on OperationalError - i.e. only
   if the table were missing. It exists, and it is the small staging tier, so the real corpus
   was never bridged.
2. Nothing excluded memories that had themselves arrived through the mesh, and the push
   passed the ORIGINAL source through, so a relayed memory reached the peer looking
   locally-authored and was relayed straight back.
3. created_at was compared as a raw string, so space-separated timestamps sort below every
   T-separated one and get skipped forever.

These pin all three. They matter more than most tests here: the failure mode is unbounded
growth on both nodes, not a wrong answer.
"""

import importlib.util
import os
import pathlib
import sqlite3
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("LOCI_ENV_FILE", "/nonexistent-env-file-for-tests")
os.environ.setdefault("LOCI_A2A_TOKEN", "t")

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "bridge_impl", _SCRIPTS / "a2a_context_bridge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bridge = _load()

SCHEMA = ("CREATE TABLE {t} (id TEXT, content TEXT, importance REAL, "
          "created_at TEXT, source TEXT)")


class _DB:
    def __init__(self, memories=(), working=()):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        con = sqlite3.connect(self.path)
        for t in ("memories", "working_memory"):
            con.execute(SCHEMA.format(t=t))
        for t, rows in (("memories", memories), ("working_memory", working)):
            con.executemany(f"INSERT INTO {t} VALUES (?,?,?,?,?)", rows)
        con.commit()
        con.close()


def row(i, created="2026-07-27T12:00:00", imp=0.9, source="local-work"):
    return (i, f"content {i}", imp, created, source)


class TestEchoSuppression(unittest.TestCase):
    def _fetch(self, memories, since="2026-07-01T00:00:00"):
        db = _DB(memories=memories)
        self.addCleanup(os.unlink, db.path)
        with mock.patch.object(bridge, "MNEMOSYNE_DB", db.path):
            return bridge._fetch_recent_memories(since, 0.5, 20)

    def test_locally_authored_memory_is_bridged(self):
        got = self._fetch([row("a")])
        self.assertEqual([m["id"] for m in got], ["a"])

    def test_peer_broadcast_is_not_re_bridged(self):
        got = self._fetch([row("a"), row("echo", source="broadcast:oxalis-mrpink")])
        self.assertEqual([m["id"] for m in got], ["a"])

    def test_our_own_bridge_output_is_not_re_bridged(self):
        got = self._fetch([row("a"), row("mine", source="bridge:loci")])
        self.assertEqual([m["id"] for m in got], ["a"])

    def test_context_broadcast_already_fanned_out_is_not_re_bridged(self):
        got = self._fetch([row("a"), row("cb", source="context_broadcast")])
        self.assertEqual([m["id"] for m in got], ["a"])

    def test_null_source_is_not_bridged(self):
        # A null source cannot be shown to be locally authored; excluding it is the safe
        # direction, since the cost of a false exclude is one un-propagated memory and the
        # cost of a false include is a loop.
        got = self._fetch([row("a"), row("nul", source=None)])
        self.assertEqual([m["id"] for m in got], ["a"])

    def test_all_echo_rows_yields_nothing(self):
        got = self._fetch([row("x", source="broadcast:p"), row("y", source="bridge:q")])
        self.assertEqual(got, [])


class TestSourceTables(unittest.TestCase):
    def _fetch(self, memories, working, since="2026-07-01T00:00:00"):
        db = _DB(memories=memories, working=working)
        self.addCleanup(os.unlink, db.path)
        with mock.patch.object(bridge, "MNEMOSYNE_DB", db.path):
            return bridge._fetch_recent_memories(since, 0.5, 20)

    def test_reads_the_memories_table_even_when_working_memory_exists(self):
        # The regression: working_memory exists, so the old fallback never fired.
        got = self._fetch([row("real")], [row("staged")])
        self.assertIn("real", [m["id"] for m in got])

    def test_reads_both_tiers(self):
        got = self._fetch([row("real")], [row("staged")])
        self.assertEqual({m["id"] for m in got}, {"real", "staged"})

    def test_id_in_both_tiers_is_not_duplicated(self):
        got = self._fetch([row("dup")], [row("dup")])
        self.assertEqual(len(got), 1)


class TestTimestampNormalisation(unittest.TestCase):
    def _fetch(self, memories, since):
        db = _DB(memories=memories)
        self.addCleanup(os.unlink, db.path)
        with mock.patch.object(bridge, "MNEMOSYNE_DB", db.path):
            return bridge._fetch_recent_memories(since, 0.5, 20)

    def test_space_separated_row_is_not_lost(self):
        # ' ' (0x20) sorts below 'T' (0x54), so a raw string compare drops this row.
        got = self._fetch([row("space", created="2026-07-27 12:00:00")],
                          since="2026-07-26T00:00:00")
        self.assertEqual([m["id"] for m in got], ["space"])

    def test_space_separated_since_still_filters(self):
        got = self._fetch([row("old", created="2026-07-25T00:00:00")],
                          since="2026-07-26 00:00:00")
        self.assertEqual(got, [])

    def test_older_rows_are_still_excluded(self):
        got = self._fetch([row("old", created="2026-07-01T00:00:00"),
                           row("new", created="2026-07-27T00:00:00")],
                          since="2026-07-26T00:00:00")
        self.assertEqual([m["id"] for m in got], ["new"])


class TestImportanceAndLimit(unittest.TestCase):
    def _fetch(self, memories, min_imp=0.5, max_items=20):
        db = _DB(memories=memories)
        self.addCleanup(os.unlink, db.path)
        with mock.patch.object(bridge, "MNEMOSYNE_DB", db.path):
            return bridge._fetch_recent_memories("2026-07-01T00:00:00", min_imp, max_items)

    def test_below_threshold_is_excluded(self):
        got = self._fetch([row("lo", imp=0.2), row("hi", imp=0.9)])
        self.assertEqual([m["id"] for m in got], ["hi"])

    def test_max_items_is_honoured_across_both_tiers(self):
        got = self._fetch([row(f"m{i}", created=f"2026-07-27T12:00:{i:02d}")
                           for i in range(10)], max_items=3)
        self.assertEqual(len(got), 3)

    def test_newest_first(self):
        got = self._fetch([row("old", created="2026-07-27T10:00:00"),
                           row("new", created="2026-07-27T18:00:00")])
        self.assertEqual([m["id"] for m in got], ["new", "old"])


if __name__ == "__main__":
    unittest.main()

"""read_recent_failures()'s fallback connection must close even when the
fallback query itself raises (the same condition that triggered the
fallback in the first place).
"""
import importlib.util
import pathlib
import sqlite3
import unittest
from unittest import mock

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "_exif_skill_discovery", _SCRIPTS / "exif_skill_discovery.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeConn:
    """Stand-in sqlite3 connection: every execute() raises OperationalError,
    same as a locked/busy db. Records whether close() was ever called."""

    def __init__(self, registry):
        self.closed = False
        registry.append(self)

    def execute(self, *args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    def close(self):
        self.closed = True


class TestFallbackConnectionCloses(unittest.TestCase):
    def test_fallback_conn_closed_even_when_fallback_query_raises(self):
        mod = _load()
        created = []

        with mock.patch.object(mod.os.path, "exists", return_value=True), \
             mock.patch.object(
                 mod.sqlite3, "connect", side_effect=lambda *a, **k: _FakeConn(created)
             ):
            result = mod.read_recent_failures()

        self.assertEqual(result, [])
        # Primary attempt opens one connection, fallback opens a second.
        self.assertEqual(len(created), 2)
        self.assertTrue(
            all(c.closed for c in created),
            "every sqlite3 connection opened by read_recent_failures() must be closed",
        )


if __name__ == "__main__":
    unittest.main()

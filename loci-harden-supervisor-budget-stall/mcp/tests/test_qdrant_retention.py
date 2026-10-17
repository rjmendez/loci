"""The startup TTL purge deletes data, so its window has to be configurable.

_get_qdrant() calls _purge_old_records on the first client construction of every
process, and findings carry created_at_ts — so anything past the window is gone.
The window used to be a literal at both the definition and the call site, which
left no way to turn it off.
"""
import os
import unittest
from unittest import mock

import qdrant_ops


class _Count:
    def __init__(self, count):
        self.count = count


class _Client:
    def __init__(self, count=5):
        self._count = count
        self.deletes = []
        self.counts = []

    def count(self, **kwargs):
        self.counts.append(kwargs)
        return _Count(self._count)

    def delete(self, **kwargs):
        self.deletes.append(kwargs)


def _purge(env, count=5):
    client = _Client(count)
    with mock.patch.dict(os.environ, env, clear=False):
        qdrant_ops._purge_old_records(client, "loci_memory")
    return client


class TestRetentionWindow(unittest.TestCase):
    def test_default_is_disabled_not_a_window(self):
        """An unset variable must not delete anything.

        This defaulted to 30 and the four live MCP servers all ran with the
        variable unset. Measured consequence: the index held exactly the 912
        findings younger than 30 days and zero of the 1,919 older ones — the
        index boundary was the purge window. Re-indexing fixed it until the next
        process start.
        """
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOCI_QDRANT_RETENTION_DAYS", None)
            with mock.patch.object(qdrant_ops, "logger"):
                # no backends.toml value in the test env -> the floor is 0
                with mock.patch.dict("sys.modules", {"backends": None}):
                    self.assertEqual(qdrant_ops._retention_days(), 0)

    def test_zero_disables_the_purge(self):
        client = _purge({"LOCI_QDRANT_RETENTION_DAYS": "0"})
        self.assertEqual(client.deletes, [])
        self.assertEqual(client.counts, [])

    def test_a_negative_window_is_treated_as_disabled(self):
        client = _purge({"LOCI_QDRANT_RETENTION_DAYS": "-1"})
        self.assertEqual(client.deletes, [])

    def test_garbage_disables_rather_than_guessing_a_window(self):
        """Guessing a number here is guessing how much of the corpus to delete."""
        with mock.patch.dict(os.environ, {"LOCI_QDRANT_RETENTION_DAYS": "banana"}):
            self.assertEqual(qdrant_ops._retention_days(), 0)

    def test_a_custom_window_is_honoured(self):
        client = _purge({"LOCI_QDRANT_RETENTION_DAYS": "90"})
        self.assertEqual(len(client.deletes), 1)
        rng = client.counts[0]["count_filter"].must[0].range
        self.assertAlmostEqual(rng.lt, int(__import__("time").time()) - 90 * 86400, delta=5)

    def test_nothing_stale_means_no_delete_call(self):
        client = _purge({"LOCI_QDRANT_RETENTION_DAYS": "30"}, count=0)
        self.assertEqual(client.deletes, [])

    def test_it_counts_before_it_deletes(self):
        client = _purge({"LOCI_QDRANT_RETENTION_DAYS": "30"}, count=7)
        self.assertEqual(len(client.counts), 1)
        self.assertTrue(client.counts[0]["exact"])
        self.assertEqual(len(client.deletes), 1)


class TestCallSiteDoesNotPinTheWindow(unittest.TestCase):
    def test_get_qdrant_passes_no_literal(self):
        import inspect
        src = inspect.getsource(qdrant_ops._get_qdrant)
        self.assertIn("_purge_old_records(client, col)", src)
        self.assertNotIn("retention_days=30", src)


if __name__ == "__main__":
    unittest.main()


class TestRetentionResolutionOrder(unittest.TestCase):
    """The env var only reaches a process whose launcher sets it.

    All four live MCP servers ran with it unset, which is how the destructive
    default went unnoticed. backends.toml is the durable floor: stdlib tomllib,
    no third-party import, readable by anything that can read the file.
    """

    def test_backends_toml_supplies_the_window_when_env_is_unset(self):
        fake = mock.Mock()
        fake._cfg.return_value = 45
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOCI_QDRANT_RETENTION_DAYS", None)
            with mock.patch.dict("sys.modules", {"backends": fake}):
                self.assertEqual(qdrant_ops._retention_days(), 45)

    def test_env_wins_over_backends_toml(self):
        fake = mock.Mock()
        fake._cfg.return_value = 45
        with mock.patch.dict(os.environ, {"LOCI_QDRANT_RETENTION_DAYS": "7"}):
            with mock.patch.dict("sys.modules", {"backends": fake}):
                self.assertEqual(qdrant_ops._retention_days(), 7)

    def test_an_unreadable_backends_file_disables_rather_than_purges(self):
        fake = mock.Mock()
        fake._cfg.side_effect = OSError("no such file")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOCI_QDRANT_RETENTION_DAYS", None)
            with mock.patch.dict("sys.modules", {"backends": fake}):
                self.assertEqual(qdrant_ops._retention_days(), 0)

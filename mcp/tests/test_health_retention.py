"""loci_health reports whether the corpus is being deleted.

The 30-day purge default removed every indexed finding older than a month on
each process start. Nothing surfaced it: coverage looked correct immediately
after a re-index and collapsed at the next restart, so the condition was only
visible as a set-difference between Qdrant and disk. These fields make it
answerable from the tool that is already called to diagnose the server.
"""
import json
import unittest
from unittest import mock

import server


class HealthRetentionTest(unittest.TestCase):

    def _health(self, days):
        with mock.patch("qdrant_ops._retention_days", return_value=days):
            return json.loads(server.loci_health())

    def test_disabled_purge_reports_safe(self):
        h = self._health(0)
        self.assertEqual(h["retention_days"], 0)
        self.assertFalse(h["purge_active"])
        self.assertNotIn("purge_warning", h)

    def test_active_purge_is_reported_with_a_warning(self):
        h = self._health(30)
        self.assertEqual(h["retention_days"], 30)
        self.assertTrue(h["purge_active"])
        self.assertIn("30 days", h["purge_warning"])

    def test_a_failing_retention_probe_does_not_break_health(self):
        """Every loci_health probe is fail-open; this one is no exception."""
        with mock.patch("qdrant_ops._retention_days", side_effect=RuntimeError("boom")):
            h = json.loads(server.loci_health())
        self.assertIn("code_version", h)
        self.assertNotIn("retention_days", h)


if __name__ == "__main__":
    unittest.main()

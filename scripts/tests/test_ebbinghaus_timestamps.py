"""days_since() must parse the timestamps this script itself writes back.

now_iso() emits datetime.now(timezone.utc).isoformat(), which carries microseconds
('2026-09-01T13:56:27.745013+00:00'). The old three-format strptime menu had no
format with fractional seconds, so every such row fell through to the -1.0
sentinel; retention() maps that to 0.0, which sorts ahead of every genuinely
computed retention and pins the same rows at the head of every batch. 246 of the
2756 rows in the live mnemosyne DB carry that shape.
"""
import importlib.util
import pathlib
import unittest
from datetime import datetime, timedelta, timezone

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "_ebb_ts", _SCRIPTS / "ebbinghaus_consolidation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DaysSinceTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load()

    def _one_day_ago(self):
        return datetime.now(timezone.utc) - timedelta(days=1)

    def test_parses_the_timestamp_now_iso_writes(self):
        ts = self.mod.now_iso()
        self.assertIn(".", ts.split("+")[0], "now_iso should carry microseconds")
        self.assertAlmostEqual(self.mod.days_since(ts), 0.0, places=3)

    def test_parses_fractional_seconds_with_offset(self):
        ts = self._one_day_ago().isoformat()
        self.assertAlmostEqual(self.mod.days_since(ts), 1.0, places=3)

    def test_parses_naive_fractional_seconds_as_utc(self):
        ts = self._one_day_ago().replace(tzinfo=None).isoformat()
        self.assertAlmostEqual(self.mod.days_since(ts), 1.0, places=3)

    def test_still_parses_the_three_shapes_that_already_worked(self):
        dt = self._one_day_ago().replace(microsecond=0)
        for ts in (dt.isoformat(),
                   dt.strftime("%Y-%m-%d %H:%M:%S"),
                   dt.replace(tzinfo=None).isoformat()):
            with self.subTest(ts=ts):
                self.assertAlmostEqual(self.mod.days_since(ts), 1.0, places=3)

    def test_parses_a_trailing_z_as_utc(self):
        dt = self._one_day_ago().replace(microsecond=0, tzinfo=None)
        self.assertAlmostEqual(self.mod.days_since(dt.isoformat() + "Z"), 1.0, places=3)

    def test_keeps_the_sentinel_for_genuinely_unparseable_input(self):
        self.assertEqual(self.mod.days_since("last tuesday"), -1.0)

    def test_empty_is_still_zero_not_the_sentinel(self):
        self.assertEqual(self.mod.days_since(""), 0.0)

    def test_retention_no_longer_pins_a_fractional_timestamp_at_zero(self):
        """retention() forces 0.0 on the sentinel so the row is eligible for
        consolidation; a fresh microsecond timestamp must not land there."""
        self.assertGreater(self.mod.retention(1, self.mod.now_iso(), "", 5.0), 0.9)


if __name__ == "__main__":
    unittest.main()

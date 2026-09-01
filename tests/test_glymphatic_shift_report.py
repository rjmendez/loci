"""main(--check-shift) must survive a drift_score of None.

check_content_shift documents and returns ``drift_score=None`` on four paths
(db_not_found, table_not_found, too_few_rows, embed_failed), each paired with
``should_sweep=False`` — which is exactly the branch main() reports on. Formatting
that None with ``:.4f`` raised TypeError out of main(), so the one condition
--check-shift exists to detect cheaply (embeddings unavailable) aborted the run
before the mutex was taken and no sweep ran at all.
"""
import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "glymphatic_sweep.py"


def _load():
    spec = importlib.util.spec_from_file_location("glymphatic_sweep_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class ShiftReportTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load()

    def _run_main(self, shift_result):
        self.mod.check_content_shift = lambda **_kw: shift_result
        for step in ("sweep_verdicts", "sweep_orphans", "sweep_edges", "sweep_duplicates"):
            setattr(self.mod, step, lambda *_a, **_kw: self.fail(f"{step} ran despite should_sweep=False"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.mod.main(check_shift=True)
        return buf.getvalue()

    def test_none_drift_score_reports_reason_instead_of_crashing(self):
        for reason in ("db_not_found", "table_not_found", "too_few_rows", "embed_failed"):
            with self.subTest(reason=reason):
                out = self._run_main(
                    {"drift_score": None, "should_sweep": False, "reason": reason}
                )
                self.assertIn("skipping sweep", out)
                self.assertIn(reason, out)

    def test_none_drift_score_without_reason_key(self):
        out = self._run_main({"drift_score": None, "should_sweep": False})
        self.assertIn("skipping sweep", out)

    def test_real_drift_score_still_formatted(self):
        out = self._run_main({"drift_score": 0.0123456, "should_sweep": False, "n_sampled": 50})
        self.assertIn("0.0123", out)
        self.assertIn("skipping sweep", out)


if __name__ == "__main__":
    unittest.main()

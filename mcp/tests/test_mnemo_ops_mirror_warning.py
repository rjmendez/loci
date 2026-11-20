"""The loci->mnemo mirror must fail LOUD, not silent.

When `import mnemosyne` fails the mirror goes inert (remember/recall disabled, Qdrant-only). That must be a
WARNING with an actionable remediation in the log — not an INFO line that hides a whole-session outage (the
2026-10-30 incident). Positive twin: when mnemosyne imports, no warning and the funcs are returned.

Mutation run to prove the negative test bites: reverted mnemo_ops.py:36 to `logger.info(...)` ->
test_import_failure_warns_loudly_with_remediation FAILED (assertLogs found no WARNING record). Restored -> green.

Run: pytest mcp/tests/test_mnemo_ops_mirror_warning.py -v
"""
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import mnemo_ops  # noqa: E402


class MnemoMirrorWarningTest(unittest.TestCase):
    def setUp(self) -> None:
        # the resolver caches its result in module globals; clear them so each test re-resolves
        mnemo_ops._mnemo_remember_fn = None
        mnemo_ops._mnemo_recall_fn = None
        self.addCleanup(setattr, mnemo_ops, "_mnemo_remember_fn", None)
        self.addCleanup(setattr, mnemo_ops, "_mnemo_recall_fn", None)

    def test_import_failure_warns_loudly_with_remediation(self) -> None:
        # make `import mnemosyne` raise ImportError
        with mock.patch.dict(sys.modules, {"mnemosyne": None}):
            with self.assertLogs("loci.mnemo" if mnemo_ops.logger.name == "loci.mnemo" else mnemo_ops.logger.name,
                                 level="WARNING") as cm:
                remember, recall = mnemo_ops._get_mnemo_funcs()
        self.assertIsNone(remember)
        self.assertIsNone(recall)
        joined = "\n".join(cm.output)
        self.assertIn("WARNING", joined)
        self.assertIn("INERT", joined)
        self.assertIn("pip install mnemosyne-memory", joined)

    def test_successful_import_returns_funcs_and_does_not_warn(self) -> None:
        # positive twin: a module exposing remember/recall resolves cleanly and logs no warning
        fake = types.ModuleType("mnemosyne")
        fake.remember = lambda *a, **k: True          # noqa: E731
        fake.recall = lambda *a, **k: []              # noqa: E731
        with mock.patch.dict(sys.modules, {"mnemosyne": fake}):
            logger_name = mnemo_ops.logger.name
            with self.assertLogs(logger_name, level="DEBUG") as cm:
                mnemo_ops.logger.debug("probe")       # ensure the logger emits at least one record
                remember, recall = mnemo_ops._get_mnemo_funcs()
        self.assertIs(remember, fake.remember)
        self.assertIs(recall, fake.recall)
        self.assertFalse([r for r in cm.records if r.levelname == "WARNING"])


if __name__ == "__main__":
    unittest.main()

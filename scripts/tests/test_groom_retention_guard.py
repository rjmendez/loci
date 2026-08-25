"""The groomer must not be able to quietly undo its own work.

pass_index exists to repair an index the startup purge emptied. _get_qdrant()
runs that purge on its first call in a process — so merely CONNECTING with
retention unset destroys what the pass is there to restore. Measured before the
guard: with python-dotenv unimportable, load_env() reported success, resolved
QDRANT_URL from backends.toml, and left retention at 30.
"""
import importlib.util
import os
import pathlib
import sys
import unittest
from unittest import mock

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("loci_groom_rg", _SCRIPTS / "loci_groom.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


groom = _load()


class TestPassIndexRefusesToPurge(unittest.TestCase):
    def _run(self, retention):
        touched = []
        fake = mock.Mock()
        fake._retention_days.return_value = retention
        fake._get_qdrant.side_effect = lambda: touched.append("connected") or (None, None)
        with mock.patch.dict(sys.modules, {"qdrant_ops": fake}):
            return groom.pass_index(), touched

    def test_a_nonzero_retention_refuses_before_connecting(self):
        report, touched = self._run(30)
        self.assertEqual(report["status"], "refused")
        self.assertIn("30", report["detail"])
        self.assertEqual(touched, [], "must not connect — connecting IS the purge")

    def test_the_refusal_names_where_to_set_it(self):
        report, _ = self._run(7)
        self.assertIn("backends.toml", report["detail"])
        self.assertIn("LOCI_QDRANT_RETENTION_DAYS", report["detail"])

    def test_zero_retention_proceeds(self):
        report, touched = self._run(0)
        self.assertNotEqual(report["status"], "refused")
        self.assertEqual(touched, ["connected"])


class TestLoadEnvIsNotSilent(unittest.TestCase):
    def test_retention_resolves_from_backends_when_dotenv_is_missing(self):
        # the durable channel is stdlib tomllib, so it must not depend on dotenv
        fake_backends = mock.Mock()
        fake_backends.qdrant.return_value = ("http://q", "")
        fake_backends.ollama_url.return_value = ""
        fake_backends._cfg.side_effect = lambda s, k, d=None: 0 if (s, k) == ("qdrant", "retention_days") else d
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("LOCI_QDRANT_RETENTION_DAYS", "QDRANT_URL", "OLLAMA_BASE_URL")}
        with mock.patch.dict(os.environ, clean, clear=True), \
             mock.patch.dict(sys.modules, {"backends": fake_backends, "dotenv": None}):
            resolved = groom.load_env()
        self.assertEqual(resolved.get("LOCI_QDRANT_RETENTION_DAYS"), "0")

    def test_a_missing_dotenv_is_logged_not_swallowed(self):
        clean = {k: v for k, v in os.environ.items() if k != "LOCI_QDRANT_RETENTION_DAYS"}
        with mock.patch.dict(os.environ, clean, clear=True), \
             mock.patch.dict(sys.modules, {"dotenv": None}), \
             self.assertLogs("loci-groom", level="WARNING") as cm:
            groom.load_env()
        self.assertTrue(any("LOCI_QDRANT_RETENTION_DAYS" in m for m in cm.output),
                        "the warning must name the setting that silently lapsed")


if __name__ == "__main__":
    unittest.main()

"""ua-watch's stamp has to attest to the graph, not to the commit.

run_ingest returns True on `returncode == 0`, and ua-ingest.py exits 0 after upserting
zero points (flush_batch is a no-op on an empty batch, so an empty final-graph.json
prints "Done. 0 points upserted" and exits clean). ua-watch then writes {git_hash,
fg_mtime} for that project.

The git gate used to be evaluated first and on its own, so once that hash was stamped
the project was skipped at that commit no matter what happened to the graph afterwards.
Regenerating the graph — the one repair that fixes an empty ingest — could not bring the
project back into the collection until someone committed to the repo. These pin that a
changed graph always re-ingests, while everything that was legitimately skipped before
is still skipped.
"""

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "ua_watch_impl", _SCRIPTS / "ua-watch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ua = _load()
HASH = "a" * 40
OTHER = "b" * 40


class TestSkipGate(unittest.TestCase):
    def test_regenerated_graph_re_ingests_at_the_same_commit(self):
        # The repair path: the graph was rewritten after a 0-point ingest.
        self.assertEqual(ua.skip_reason(HASH, HASH, fg_mtime=200, last_mtime=100), "")

    def test_nothing_changed_is_skipped(self):
        self.assertNotEqual(ua.skip_reason(HASH, HASH, fg_mtime=100, last_mtime=100), "")

    def test_new_commit_re_ingests_even_with_an_older_graph(self):
        self.assertEqual(ua.skip_reason(OTHER, HASH, fg_mtime=100, last_mtime=100), "")

    def test_never_ingested_is_not_skipped(self):
        self.assertEqual(ua.skip_reason(HASH, "", fg_mtime=100, last_mtime=0), "")

    def test_non_git_project_still_uses_the_graph_mtime(self):
        self.assertNotEqual(ua.skip_reason("", "", fg_mtime=100, last_mtime=100), "")
        self.assertEqual(ua.skip_reason("", "", fg_mtime=200, last_mtime=100), "")

    def test_skip_reason_names_the_gate_that_fired(self):
        self.assertIn(HASH[:8], ua.skip_reason(HASH, HASH, 100, 100))
        self.assertEqual(ua.skip_reason("", "", 100, 100), "graph unchanged")


class TestMainReIngests(unittest.TestCase):
    """End-to-end through main(), which is where the two gates are actually applied."""

    def _project(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name) / "proj"
        fg = root / ".understand-anything" / "intermediate" / "final-graph.json"
        fg.parent.mkdir(parents=True)
        fg.write_text(json.dumps({"nodes": [], "edges": []}))
        return root, fg

    def _run(self, root, ingested, git_hash=HASH):
        state_file = pathlib.Path(self.state_dir.name) / "state.json"
        with mock.patch.object(ua, "STATE_FILE", state_file), \
                mock.patch.object(ua, "get_git_hash", lambda p: git_hash), \
                mock.patch.object(ua, "run_ingest",
                                  lambda p: (ingested.append(str(p)), True)[1]), \
                mock.patch.object(sys, "argv", ["ua-watch.py", str(root)]):
            ua.main()

    def setUp(self):
        self.state_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.state_dir.cleanup)

    def test_regenerated_graph_is_ingested_again(self):
        # The 0-point ingest is stamped as done (ua-ingest exits 0 either way), so the
        # only repair is to regenerate the graph. That has to be enough on its own.
        root, fg = self._project()
        ingested = []
        self._run(root, ingested)
        self.assertEqual(len(ingested), 1)

        newer = fg.stat().st_mtime + 60
        os.utime(fg, (newer, newer))
        self._run(root, ingested)
        self.assertEqual(len(ingested), 2)

    def test_untouched_graph_is_not_ingested_again(self):
        root, _ = self._project()
        ingested = []
        self._run(root, ingested)
        self._run(root, ingested)
        self.assertEqual(len(ingested), 1)


if __name__ == "__main__":
    unittest.main()

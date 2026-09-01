"""Tests that skill_docker_status / skill_ua_search surface a non-zero
subprocess exit as an error field, instead of silently returning an
empty result indistinguishable from a genuine "nothing found" answer.
"""

import asyncio
import importlib.util
import os
import pathlib
import subprocess
import unittest
from unittest import mock

os.environ.setdefault("LOCI_A2A_TOKEN", "test-token-abc123")
os.environ.setdefault("LOCI_A2A_URL", "http://localhost:8201")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("MNEMOSYNE_EMBEDDING_API_URL", "http://localhost:11434/v1")

_server_path = pathlib.Path(__file__).parent.parent / "server.py"
_spec = importlib.util.spec_from_file_location("a2a_server_impl", _server_path)
a2a_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a_server)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestDockerStatusSurfacesFailure(unittest.TestCase):
    def test_docker_daemon_unreachable_is_an_error_not_an_empty_list(self):
        # docker exits non-zero with empty stdout when the daemon is unreachable;
        # kubectl succeeds with no pods, to isolate the docker branch.
        def fake_run(args, **kwargs):
            if args[0] == 'docker':
                return _completed(1, stdout="", stderr="Cannot connect to the Docker daemon")
            return _completed(0, stdout="", stderr="")

        with mock.patch('subprocess.run', side_effect=fake_run):
            result = _run(a2a_server.skill_docker_status({'input': {}}))

        self.assertIn('docker_error', result)
        self.assertNotIn('docker', result)

    def test_kubectl_no_context_is_an_error_not_an_empty_list(self):
        def fake_run(args, **kwargs):
            if args[0] == 'kubectl':
                return _completed(1, stdout="", stderr="error: no context set")
            return _completed(0, stdout="", stderr="")

        with mock.patch('subprocess.run', side_effect=fake_run):
            result = _run(a2a_server.skill_docker_status({'input': {}}))

        self.assertIn('k3s_error', result)
        self.assertNotIn('k3s_pods', result)

    def test_success_still_parses_normally(self):
        def fake_run(args, **kwargs):
            if args[0] == 'docker':
                return _completed(0, stdout="web\tUp\tnginx\n", stderr="")
            return _completed(0, stdout="", stderr="")

        with mock.patch('subprocess.run', side_effect=fake_run):
            result = _run(a2a_server.skill_docker_status({'input': {}}))

        self.assertNotIn('docker_error', result)
        self.assertEqual(result['docker'], [{'name': 'web', 'status': 'Up', 'image': 'nginx'}])


class TestUaSearchSurfacesFailure(unittest.TestCase):
    def test_script_crash_before_stdout_is_an_error_not_a_zero_hit_search(self):
        with mock.patch('subprocess.run', return_value=_completed(1, stdout="", stderr="Traceback: ImportError")):
            with mock.patch.object(os.path, 'exists', return_value=True):
                with mock.patch.dict(os.environ, {'UA_SEARCH_SCRIPT': '/fake/ua_search.py'}):
                    result = _run(a2a_server.skill_ua_search({'input': {'query': 'foo'}}))

        self.assertIn('error', result)
        self.assertNotIn('results', result)


if __name__ == '__main__':
    unittest.main()

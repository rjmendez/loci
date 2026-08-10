"""The QDRANT_API_KEY fallback must read whichever name the MCP server is registered under.

Three scripts fall back to ``~/.claude/settings.json`` when QDRANT_API_KEY is unset in the
environment, and each indexes ``mcpServers`` by the server's registration name. That name is
now ``loci`` (what the checked-in ``.mcp.json`` and the READMEs use); it used to be
``hermes_memory``. A lookup pinned to one name fails soft on the other -- the scripts swallow
the KeyError and connect with no api-key, which surfaces as an auth error from Qdrant far from
the actual cause, or as a silent unauthenticated connection against an open instance.

These assert both names resolve, and that a settings file with neither still fails soft.
"""

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPTS = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    """Import a scripts/ module under a private name, fresh each call.

    qdrant_payload_indexes resolves the key at module scope, so it has to be re-imported
    per case rather than reused.
    """
    spec = importlib.util.spec_from_file_location(f"_t_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeHome:
    """Point ~ at a temp dir holding a .claude/settings.json with the given mcpServers block."""

    def __init__(self, servers):
        self._servers = servers

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        home = pathlib.Path(self._tmp.name)
        claude = home / ".claude"
        claude.mkdir()
        if self._servers is not None:
            (claude / "settings.json").write_text(json.dumps({"mcpServers": self._servers}))
        self._env = mock.patch.dict(os.environ, {"HOME": str(home)}, clear=False)
        self._env.start()
        # os.path.expanduser consults USERPROFILE first on Windows and caches nothing,
        # but it also honours an explicit HOME everywhere pytest runs here.
        return home

    def __exit__(self, *exc):
        self._env.stop()
        self._tmp.cleanup()
        return False


def _entry(key):
    return {"env": {"QDRANT_API_KEY": key}}


class TestEbbinghausKeyLookup(unittest.TestCase):
    def test_reads_loci_registration(self):
        with _FakeHome({"loci": _entry("key-loci")}):
            mod = _load("ebbinghaus_consolidation")
            self.assertEqual(mod._load_qdrant_api_key(), "key-loci")

    def test_reads_legacy_hermes_memory_registration(self):
        with _FakeHome({"hermes_memory": _entry("key-legacy")}):
            mod = _load("ebbinghaus_consolidation")
            self.assertEqual(mod._load_qdrant_api_key(), "key-legacy")

    def test_loci_wins_when_both_are_present(self):
        with _FakeHome({"loci": _entry("key-new"), "hermes_memory": _entry("key-old")}):
            mod = _load("ebbinghaus_consolidation")
            self.assertEqual(mod._load_qdrant_api_key(), "key-new")

    def test_returns_empty_when_neither_name_is_registered(self):
        with _FakeHome({"something_else": _entry("nope")}):
            mod = _load("ebbinghaus_consolidation")
            self.assertEqual(mod._load_qdrant_api_key(), "")


class TestPayloadIndexesKeyLookup(unittest.TestCase):
    """This one resolves at import time, so the assertion is on the module global."""

    def _key_with(self, servers):
        with mock.patch.dict(os.environ, {"QDRANT_API_KEY": ""}, clear=False):
            with _FakeHome(servers):
                return _load("qdrant_payload_indexes").QDRANT_KEY

    def test_reads_loci_registration(self):
        self.assertEqual(self._key_with({"loci": _entry("key-loci")}), "key-loci")

    def test_reads_legacy_hermes_memory_registration(self):
        self.assertEqual(
            self._key_with({"hermes_memory": _entry("key-legacy")}), "key-legacy")

    def test_returns_empty_when_neither_name_is_registered(self):
        self.assertEqual(self._key_with({"something_else": _entry("nope")}), "")


class TestReembedDaemonKeyLookup(unittest.TestCase):
    """_resolve_qdrant builds a real QdrantClient; stub the constructor and read api_key back."""

    def _api_key_with(self, servers):
        captured = {}

        class _Client:
            def __init__(self, url=None, api_key=None):
                captured["url"] = url
                captured["api_key"] = api_key

        stub = types.ModuleType("qdrant_client")
        stub.QdrantClient = _Client
        env = {"QDRANT_URL": "http://qdrant.invalid:6333", "QDRANT_API_KEY": ""}
        with mock.patch.dict(sys.modules, {"qdrant_client": stub}):
            with mock.patch.dict(os.environ, env, clear=False):
                with _FakeHome(servers):
                    _load("reembed_daemon")._resolve_qdrant()
        return captured["api_key"]

    def test_reads_loci_registration(self):
        self.assertEqual(self._api_key_with({"loci": _entry("key-loci")}), "key-loci")

    def test_reads_legacy_hermes_memory_registration(self):
        self.assertEqual(
            self._api_key_with({"hermes_memory": _entry("key-legacy")}), "key-legacy")

    def test_passes_none_when_neither_name_is_registered(self):
        # No key found -> `api_key=key or None`, i.e. an unauthenticated client.
        self.assertIsNone(self._api_key_with({"something_else": _entry("nope")}))


if __name__ == "__main__":
    unittest.main()

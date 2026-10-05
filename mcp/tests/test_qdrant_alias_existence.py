"""A collection name that is an alias already exists.

The LOCI_* rename points loci_memory and loci_sessions at the old collections
with Qdrant aliases, which is the migration path docs/hermes-name-audit.md
documents. get_collections() lists real collections only, so an aliased name
read as absent, _get_qdrant tried to create it, and Qdrant answered:

    400 Wrong input: Can't create collection with name loci_memory.
    Alias with the same name already exists

which surfaced as "qdrant unreachable" and took the every-6h index pass down.

These drive the real ``qdrant_ops._get_qdrant`` against a recording fake client
(only the network client is replaced), so the existence check under test is the
shipped one, not a copy of it.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import qdrant_client  # noqa: E402
import qdrant_ops  # noqa: E402


class _FakeClient:
    """Records every create/update; never raises into _get_qdrant's fail-open except."""

    def __init__(self, collections, aliases, aliases_supported=True):
        self._collections = list(collections)
        self._aliases = list(aliases)
        self._aliases_supported = aliases_supported
        self.created: list[str] = []
        self.updated: list[str] = []

    def get_collections(self):
        return types.SimpleNamespace(
            collections=[types.SimpleNamespace(name=n) for n in self._collections])

    def get_aliases(self):
        if not self._aliases_supported:
            raise RuntimeError("aliases unsupported")
        return types.SimpleNamespace(
            aliases=[types.SimpleNamespace(alias_name=a, collection_name=c)
                     for a, c in self._aliases])

    def create_collection(self, name, *a, **kw):
        self.created.append(name)

    def update_collection(self, name, *a, **kw):
        self.updated.append(name)

    def create_payload_index(self, **kw):
        pass


@pytest.fixture
def get_qdrant(monkeypatch):
    """Run the real _get_qdrant against a given fake client; returns its result."""
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.invalid:6333")
    monkeypatch.setattr(qdrant_ops, "_qdrant_client", None)
    monkeypatch.setattr(qdrant_ops, "_qdrant_failed_at", None)
    monkeypatch.setattr(qdrant_ops, "_purge_old_records", lambda client, col, *a, **k: None)

    def run(client):
        monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: client)
        return qdrant_ops._get_qdrant()
    return run


COL = qdrant_ops.QDRANT_COLLECTION_PREFIX


def test_an_aliased_name_counts_as_existing(get_qdrant):
    client = _FakeClient(["hermes_memory"], [(COL, "hermes_memory")])
    assert get_qdrant(client) == (client, COL)
    assert client.created == [], "must not try to create over an alias"
    assert client.updated == [COL]           # the existing-collection branch ran


def test_a_real_collection_still_counts_as_existing(get_qdrant):
    client = _FakeClient([COL], [])
    assert get_qdrant(client) == (client, COL)
    assert (client.created, client.updated) == ([], [COL])


def test_an_unknown_name_is_still_absent_and_gets_created(get_qdrant):
    # Positive twin: the check must not call everything "existing".
    client = _FakeClient(["something_else"], [("other_alias", "x")])
    assert get_qdrant(client) == (client, COL)
    assert (client.created, client.updated) == ([COL], [])


def test_a_server_without_alias_support_does_not_break_the_check(get_qdrant):
    client = _FakeClient([COL], [], aliases_supported=False)
    assert get_qdrant(client) == (client, COL)
    assert client.created == []


def test_a_server_without_alias_support_still_creates_a_missing_collection(get_qdrant):
    client = _FakeClient(["something_else"], [], aliases_supported=False)
    assert get_qdrant(client) == (client, COL)
    assert client.created == [COL]

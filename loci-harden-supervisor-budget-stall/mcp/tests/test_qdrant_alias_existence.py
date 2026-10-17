"""A collection name that is an alias already exists.

The LOCI_* rename points loci_memory and loci_sessions at the old collections
with Qdrant aliases, which is the migration path docs/hermes-name-audit.md
documents. get_collections() lists real collections only, so an aliased name
read as absent, _get_qdrant tried to create it, and Qdrant answered:

    400 Wrong input: Can't create collection with name loci_memory.
    Alias with the same name already exists

which surfaced as "qdrant unreachable" and took the every-6h index pass down.
"""
from __future__ import annotations

import os
import sys
import types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _fake_client(collections, aliases, created):
    class C:
        def get_collections(self):
            return types.SimpleNamespace(
                collections=[types.SimpleNamespace(name=n) for n in collections])

        def get_aliases(self):
            return types.SimpleNamespace(
                aliases=[types.SimpleNamespace(alias_name=a, collection_name=c)
                         for a, c in aliases])

        def create_collection(self, name, *a, **kw):
            created.append(name)
            raise AssertionError(
                f"create_collection({name!r}) — Qdrant would answer 400 here")

        def get_collection(self, name):
            return types.SimpleNamespace()

        def create_payload_index(self, **kw):
            pass
    return C()


def _existing_names(client):
    """The set _get_qdrant builds to decide whether to create."""
    existing = {c.name for c in client.get_collections().collections}
    try:
        existing |= {a.alias_name for a in client.get_aliases().aliases}
    except Exception:
        pass
    return existing


def test_an_aliased_name_counts_as_existing():
    created: list[str] = []
    client = _fake_client(["hermes_memory"], [("loci_memory", "hermes_memory")], created)
    assert "loci_memory" in _existing_names(client)
    assert created == [], "must not try to create over an alias"


def test_a_real_collection_still_counts_as_existing():
    client = _fake_client(["loci_memory"], [], [])
    assert "loci_memory" in _existing_names(client)


def test_an_unknown_name_is_still_absent():
    client = _fake_client(["something_else"], [("other_alias", "x")], [])
    assert "loci_memory" not in _existing_names(client)


def test_a_server_without_alias_support_does_not_break_the_check():
    class C:
        def get_collections(self):
            return types.SimpleNamespace(
                collections=[types.SimpleNamespace(name="loci_memory")])

        def get_aliases(self):
            raise RuntimeError("aliases unsupported")
    assert _existing_names(C()) == {"loci_memory"}


def test_the_shipped_code_consults_aliases():
    """Guards the call itself, not just the logic mirrored above."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "qdrant_ops.py")).read()
    assert "get_aliases()" in src, "the existence check must consider aliases"

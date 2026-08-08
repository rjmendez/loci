"""Regression tests for the injected investigation-store memory root.

The store lives in inv_store.py, but its root is injected as a lambda closing over
``server.MEMORY_DIR`` rather than copied at import time. That indirection is what
lets tests (and only tests) redirect every write to a tmpdir. If it ever regresses
to a captured value or a module-level default, these tests fail loudly — instead of
the store silently writing into the operator's real ~/.hermes/memory-sessions while
the fail-open read paths keep the rest of the suite green.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import inv_store  # noqa: E402 — must follow the path setup above
import server  # noqa: E402


def test_rebinding_server_memory_dir_redirects_inv_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    d = server._inv_dir("case-x")
    assert d == tmp_path / "case-x"
    assert d.is_dir()


def test_rebinding_is_seen_by_inv_store_itself(tmp_path, monkeypatch):
    """The redirect must reach the module that actually writes, not just the re-export."""
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    assert inv_store._inv_dir("case-y") == tmp_path / "case-y"
    assert inv_store._root() == tmp_path


def test_store_writes_land_under_the_rebound_root(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    path = server._inv_dir("case-z") / "findings.jsonl"
    server._append_jsonl(path, {"id": "f1", "text": "hello"})
    assert path.exists()
    assert server._read_jsonl(path) == [{"id": "f1", "text": "hello"}]

    server._save_manifest({"id": "case-z", "title": "t"})
    manifest_path = tmp_path / "case-z" / "manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text())["id"] == "case-z"


def test_inv_store_holds_no_memory_root_of_its_own():
    """A module-level root in inv_store would shadow the injected one."""
    assert not [n for n in vars(inv_store) if "MEMORY_DIR" in n]

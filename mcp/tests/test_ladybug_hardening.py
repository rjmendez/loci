from __future__ import annotations

import pytest

from graph.ladybug_store import LadybugCorruptColumnError, LadybugStore


def test_risky_finding_text_sigsegv_is_contained(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_LADYBUG_PROBE_FORCE_SIGSEGV", "Finding.text")
    store = LadybugStore(str(tmp_path / "graphdb"))

    with pytest.raises(
        LadybugCorruptColumnError,
        match=r"corrupt column Finding\.text:.*SIGSEGV",
    ):
        store._rows("MATCH (f:Finding) RETURN f.id, f.text")


def test_risky_finding_node_sigsegv_is_contained(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_LADYBUG_PROBE_FORCE_SIGSEGV", "Finding.text")
    store = LadybugStore(str(tmp_path / "graphdb"))

    with pytest.raises(LadybugCorruptColumnError, match=r"corrupt column Finding\.text"):
        store._rows("MATCH (f:Finding) RETURN f LIMIT 1")


def test_code_query_fails_closed_on_corrupt_finding_text(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_LADYBUG_PROBE_FORCE_SIGSEGV", "Finding.text")
    store = LadybugStore(str(tmp_path / "graphdb"))

    with pytest.raises(LadybugCorruptColumnError, match=r"corrupt column Finding\.text"):
        store.code_query("MATCH (f:Finding) RETURN f.id, f.text")


def test_healthy_finding_text_probe_is_latched(tmp_path, monkeypatch):
    pytest.importorskip("ladybug")
    store = LadybugStore(str(tmp_path / "graphdb"))
    assert store.upsert_finding({"id": "f1", "text": "healthy text", "investigation": "i"})

    calls: list[tuple[str, str]] = []
    original = LadybugStore._probe_ladybug_column_in_child

    def counted(self: LadybugStore, label: str, column: str):
        calls.append((label, column))
        return original(self, label, column)

    monkeypatch.setattr(LadybugStore, "_probe_ladybug_column_in_child", counted)

    assert store._rows("MATCH (f:Finding) RETURN f.id, f.text") == [["f1", "healthy text"]]
    assert store._rows("MATCH (f:Finding) RETURN f.id, f.text") == [["f1", "healthy text"]]
    assert calls == [("Finding", "text")]


def test_session_body_exception_is_not_masked(tmp_path):
    pytest.importorskip("ladybug")
    store = LadybugStore(str(tmp_path / "graphdb"))
    assert store.upsert_investigation("i", "title")

    with pytest.raises(ValueError, match="body boom"):
        with store._session(write=False):
            raise ValueError("body boom")

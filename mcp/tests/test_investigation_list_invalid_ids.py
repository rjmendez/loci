"""investigation_list must survive legacy dirs whose names no longer validate.

A directory literally named 'undefined' (created before investigation ids were
validated, by an orchestrator that substituted an unset JS variable) made every
investigation_list page fail with "Invalid investigation_id: 'undefined'". The
listing now skips and reports such dirs, and investigation_start rejects the
sentinel ids up front without creating anything on disk.

Hermetic: runs against a temp MEMORY_DIR with Qdrant, Mnemosyne and graph side
effects stubbed out.
"""
import json
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import inv_store  # noqa: E402
import investigation_tools  # noqa: E402
import mnemo_ops  # noqa: E402
import server  # noqa: E402


def _j(s: str) -> dict:
    return json.loads(s)


@pytest.fixture
def store(tmp_path, monkeypatch):
    mem = tmp_path / "mem"
    mem.mkdir()
    monkeypatch.setattr(server, "MEMORY_DIR", mem)
    monkeypatch.setenv("LOCI_CODE_ROOT", str(tmp_path / "code"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    inv_store._manifest_cache.clear()
    no = lambda *a, **k: None  # noqa: E731
    for name, value in {
        "_get_qdrant": lambda *a, **k: (None, None),
        "_qdrant_upsert": no,
        "_mnemo_remember": lambda *a, **k: False,
        "_event_log_append": no,
        "_mirror_finding_to_ladybug": no,
        "_autolink_finding_to_ladybug": no,
        "_update_entities_jsonl": no,
    }.items():
        monkeypatch.setattr(server, name, value, raising=False)
    monkeypatch.setattr(investigation_tools, "_ladybug_upsert_investigation", no)
    no_mnemo = lambda: (None, None)  # noqa: E731
    monkeypatch.setattr(server, "_get_mnemo_funcs", no_mnemo)
    monkeypatch.setattr(mnemo_ops, "_get_mnemo_funcs", no_mnemo)
    yield mem
    inv_store._manifest_cache.clear()


def _legacy_dir(mem: Path, name: str) -> Path:
    d = mem / name
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({
        "id": name, "title": "legacy", "status": "active", "updated_at": "9999",
        "finding_counts": {"observed": 1},
    }))
    return d


def test_list_skips_and_reports_undefined_dir(store):
    assert _j(server.investigation_start(investigation_id="good-1", title="t1"))["status"] == "created"
    assert _j(server.investigation_start(investigation_id="good-2", title="t2"))["status"] == "created"
    _legacy_dir(store, "undefined")

    out = _j(server.investigation_list())
    assert "error" not in out
    assert sorted(r["id"] for r in out["investigations"]) == ["good-1", "good-2"]
    assert out["total"] == 2
    assert [s["investigation_id"] for s in out["skipped_investigations"]] == ["undefined"]
    assert "missing-value sentinel" in out["skipped_investigations"][0]["reason"]
    # The legacy dir is reported, never modified or removed.
    assert (store / "undefined" / "manifest.json").exists()


@pytest.mark.parametrize("summary", [True, False])
def test_every_page_survives_the_bad_dir(store, summary):
    for i in range(5):
        server.investigation_start(investigation_id=f"inv-{i}", title=f"t{i}")
    _legacy_dir(store, "undefined")
    seen = []
    for offset in range(0, 5, 2):
        out = _j(server.investigation_list(limit=2, offset=offset, summary=summary))
        assert out["total"] == 5
        seen += [r["id"] for r in out["investigations"]]
    assert sorted(seen) == [f"inv-{i}" for i in range(5)]


def test_list_skips_unparseable_and_non_object_manifests(store):
    server.investigation_start(investigation_id="ok", title="fine")
    bad_json = store / "broken"
    bad_json.mkdir()
    (bad_json / "manifest.json").write_text("{not json")
    not_obj = store / "listy"
    not_obj.mkdir()
    (not_obj / "manifest.json").write_text("[1, 2]")

    out = _j(server.investigation_list(summary=False))
    assert [r["id"] for r in out["investigations"]] == ["ok"]
    assert sorted(s["investigation_id"] for s in out["skipped_investigations"]) == ["broken", "listy"]


def test_list_tolerates_manifest_missing_fields(store):
    d = store / "sparse"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"title": "only a title"}))
    out = _j(server.investigation_list(summary=False))
    assert out["investigations"][0]["id"] == "sparse"
    assert out["investigations"][0]["open_questions_count"] == 0
    assert out["skipped_investigations"] == []


def test_empty_root_reports_no_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path / "absent")
    out = _j(server.investigation_list())
    assert out["investigations"] == [] and out["skipped_investigations"] == []


@pytest.mark.parametrize("bad_id", ["undefined", "null", "None", "", "  ", "a/b"])
def test_start_rejects_invalid_ids_without_creating_anything(store, bad_id):
    out = _j(server.investigation_start(investigation_id=bad_id, title="should not exist"))
    assert "Invalid investigation_id" in out["error"]
    assert list(store.iterdir()) == []


def test_start_trims_surrounding_whitespace_in_stored_id(store):
    out = _j(server.investigation_start(investigation_id="  case-x  ", title="t"))
    assert out["manifest"]["id"] == "case-x"
    assert (store / "case-x" / "manifest.json").exists()

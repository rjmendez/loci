import json
from pathlib import Path

import server


def _json(result):
    return json.loads(result)


def test_queue_happy_path_and_lease_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-happy"
    server.investigation_start(investigation_id=inv_id, title="Queue happy path")

    enq = _json(server.investigation_queue_enqueue(
        investigation_id=inv_id,
        item_json=json.dumps({
            "id": "task-1",
            "scope_kind": "finding",
            "scope_targets": ["finding-42"],
            "notes": "Check source graph",
            "dependencies": ["task-0"],
        }),
    ))
    assert "error" not in enq
    assert enq["item"]["state"] == "queued"

    claim = _json(server.investigation_queue_claim(
        investigation_id=inv_id, item_id="task-1", owner_session="session-a", lease_seconds=30
    ))
    assert claim["item"]["state"] == "claimed"
    assert claim["item"]["owner_session"] == "session-a"
    assert claim["item"]["lease_expires_at"]

    status = _json(server.investigation_queue_status(investigation_id=inv_id))
    assert status["item_count"] == 1
    assert status["queue"][0]["id"] == "task-1"

    done = _json(server.investigation_queue_complete(
        investigation_id=inv_id, item_id="task-1", owner_session="session-a", state="done", notes="Validated"
    ))
    assert done["item"]["state"] == "done"
    assert done["item"]["notes"] == "Validated"


def test_queue_duplicate_ids_and_conflicts(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-dup"
    server.investigation_start(investigation_id=inv_id, title="Queue duplicates")
    first = _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="dup-task"))
    assert "error" not in first

    second = _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="dup-task"))
    assert "error" in second
    assert "already exists" in second["error"].lower()

    claim_a = _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="dup-task", owner_session="session-a", lease_seconds=60))
    assert "error" not in claim_a

    conflict = _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="dup-task", owner_session="session-b", lease_seconds=30))
    assert "error" in conflict
    assert "already claimed" in conflict["error"].lower()


def test_queue_expired_lease_allows_reclaim_and_owner_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-expired"
    server.investigation_start(investigation_id=inv_id, title="Queue expired lease")
    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="lease-task", scope_kind="session", scope_targets=["session-a"]))
    _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="lease-task", owner_session="session-a", lease_seconds=1))

    manifest_path = tmp_path / inv_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["coordination"]["items"][0]["lease_expires_at"] = "2000-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    server._manifest_cache.clear()

    reclaim = _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="lease-task", owner_session="session-b", lease_seconds=10))
    assert "error" not in reclaim
    assert reclaim["item"]["owner_session"] == "session-b"

    bad_complete = _json(server.investigation_queue_complete(investigation_id=inv_id, item_id="lease-task", owner_session="session-a", state="done"))
    assert "error" in bad_complete
    assert "owned by session" in bad_complete["error"].lower()


def test_queue_invalid_json_and_legacy_manifest_migration(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-json"
    server.investigation_start(investigation_id=inv_id, title="JSON and migration")

    bad = _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_json="{invalid-json"))
    assert "error" in bad
    assert "invalid json" in bad["error"].lower()

    legacy_id = "legacy-queue"
    legacy_dir = tmp_path / legacy_id
    legacy_dir.mkdir()
    legacy_manifest = {
        "id": legacy_id,
        "title": "Legacy queue",
        "status": "active",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "owner": "",
        "acl": [],
        "summary_l1": [],
        "summary_l2": "",
        "open_questions": [],
        "checked_sources": {},
        "finding_counts": {"observed": 0, "inferred": 0, "assumed": 0, "gap": 0},
    }
    (legacy_dir / "manifest.json").write_text(json.dumps(legacy_manifest))

    loaded = _json(server.investigation_load(investigation_id=legacy_id))
    assert "coordination" in loaded["manifest"]
    assert loaded["manifest"]["coordination"]["items"] == []

    queued = _json(server.investigation_queue_status(investigation_id=legacy_id))
    assert queued["item_count"] == 0


def test_queue_release_alias_and_status_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-release"
    server.investigation_start(investigation_id=inv_id, title="Queue release alias")
    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="release-task", scope_kind="investigation", scope_targets=[inv_id]))
    _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="release-task", owner_session="session-z", lease_seconds=20))

    released = _json(server.investigation_queue_release(investigation_id=inv_id, item_id="release-task", owner_session="session-z", notes="blocked by dependency"))
    assert released["item"]["state"] == "blocked"

    filtered = _json(server.investigation_queue_status(investigation_id=inv_id, state="blocked"))
    assert filtered["item_count"] == 1
    assert filtered["queue"][0]["id"] == "release-task"

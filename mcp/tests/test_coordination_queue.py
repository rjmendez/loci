import json
from pathlib import Path

import server


def _json(result):
    return json.loads(result)


def test_queue_runtime_smoke_flow_is_deterministic_and_machine_friendly(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)

    tick = {"value": 0}

    def _fake_now():
        tick["value"] += 1
        return f"2026-01-01T00:00:{tick['value']:02d}+00:00"

    def _fake_now_plus(seconds):
        ttl = int(float(seconds))
        return f"2030-01-01T00:00:{ttl:02d}+00:00"

    queue_globals = server.investigation_queue_claim.__globals__
    monkeypatch.setitem(queue_globals, "_now", _fake_now)
    monkeypatch.setitem(queue_globals, "_coordination_now_plus", _fake_now_plus)

    inv_id = "q-smoke-flow"
    started = _json(server.investigation_start(investigation_id=inv_id, title="Queue smoke flow"))
    assert started["status"] == "created"

    discovered = _json(server.investigation_queue_status(investigation_id=inv_id))
    assert discovered == {
        "investigation_id": inv_id,
        "queue": [],
        "item_count": 0,
    }

    enqueued = _json(server.investigation_queue_enqueue(
        investigation_id=inv_id,
        item_json=json.dumps({
            "id": "flow-1",
            "scope_kind": "finding",
            "scope_targets": ["finding-99"],
            "dependencies": ["flow-0"],
            "notes": "initial enqueue",
        }),
    ))
    assert "error" not in enqueued
    assert set(enqueued) == {"queued", "item"}
    assert enqueued["queued"] is True
    enqueued_item = enqueued["item"]
    assert set(enqueued_item) == {
        "id", "scope_kind", "scope_targets", "state", "owner_session",
        "lease_expires_at", "dependencies", "notes", "created_at", "updated_at",
    }
    assert enqueued_item["id"] == "flow-1"
    assert enqueued_item["scope_kind"] == "finding"
    assert enqueued_item["scope_targets"] == ["finding-99"]
    assert enqueued_item["state"] == "queued"
    assert enqueued_item["owner_session"] is None
    assert enqueued_item["lease_expires_at"] is None
    assert enqueued_item["dependencies"] == ["flow-0"]
    assert enqueued_item["notes"] == "initial enqueue"
    assert enqueued_item["created_at"].startswith("2026-01-01T00:00:")
    assert enqueued_item["updated_at"].startswith("2026-01-01T00:00:")
    created_second = int(enqueued_item["created_at"][17:19])
    updated_second = int(enqueued_item["updated_at"][17:19])
    assert updated_second == created_second + 1

    # flow-1 depends on flow-0, which does not exist yet: the claim must wait for it.
    gated = _json(server.investigation_queue_claim(
        investigation_id=inv_id,
        item_id="flow-1",
        owner_session="session-smoke",
        lease_seconds=30,
    ))
    assert "unmet dependencies" in gated["error"]
    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="flow-0"))
    _json(server.investigation_queue_complete(investigation_id=inv_id, item_id="flow-0", state="done"))

    claimed = _json(server.investigation_queue_claim(
        investigation_id=inv_id,
        item_id="flow-1",
        owner_session="session-smoke",
        lease_seconds=30,
    ))
    assert claimed["claimed"] is True
    assert claimed["item"]["state"] == "claimed"
    assert claimed["item"]["owner_session"] == "session-smoke"
    assert claimed["item"]["lease_expires_at"] == "2030-01-01T00:00:30+00:00"

    renewed = _json(server.investigation_queue_claim(
        investigation_id=inv_id,
        item_id="flow-1",
        owner_session="session-smoke",
        lease_seconds=45,
    ))
    assert renewed["claimed"] is True
    assert renewed["item"]["state"] == "claimed"
    assert renewed["item"]["owner_session"] == "session-smoke"
    assert renewed["item"]["lease_expires_at"] == "2030-01-01T00:00:45+00:00"
    assert renewed["item"]["updated_at"] != claimed["item"]["updated_at"]

    completed = _json(server.investigation_queue_complete(
        investigation_id=inv_id,
        item_id="flow-1",
        owner_session="session-smoke",
        state="done",
        notes="flow completed",
    ))
    assert completed["updated"] is True
    assert completed["item"]["state"] == "done"
    assert completed["item"]["owner_session"] == "session-smoke"
    assert completed["item"]["lease_expires_at"] is None
    assert completed["item"]["notes"] == "flow completed"

    done_only = _json(server.investigation_queue_status(investigation_id=inv_id, state="done"))
    assert done_only["item_count"] == 2
    assert [item["id"] for item in done_only["queue"]] == ["flow-1", "flow-0"]
    assert [item["state"] for item in done_only["queue"]] == ["done", "done"]

    listed = _json(server.investigation_queue_list(investigation_id=inv_id, state="done"))
    assert listed == done_only

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
    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="task-0"))
    _json(server.investigation_queue_complete(investigation_id=inv_id, item_id="task-0", state="done"))

    claim = _json(server.investigation_queue_claim(
        investigation_id=inv_id, item_id="task-1", owner_session="session-a", lease_seconds=30
    ))
    assert claim["item"]["state"] == "claimed"
    assert claim["item"]["owner_session"] == "session-a"
    assert claim["item"]["lease_expires_at"]

    status = _json(server.investigation_queue_status(investigation_id=inv_id, item_id="task-1"))
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


def test_queue_release_returns_item_to_queue_and_status_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-release"
    server.investigation_start(investigation_id=inv_id, title="Queue release")
    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="release-task", scope_kind="investigation", scope_targets=[inv_id]))
    _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="release-task", owner_session="session-z", lease_seconds=20))

    released = _json(server.investigation_queue_release(investigation_id=inv_id, item_id="release-task", owner_session="session-z", notes="giving this up"))
    assert released["released"] is True
    assert released["updated"] is True  # kept for callers of the old release response
    assert released["item"]["state"] == "queued"
    assert released["item"]["owner_session"] is None
    assert released["item"]["lease_expires_at"] is None
    assert released["item"]["notes"] == "giving this up"

    filtered = _json(server.investigation_queue_status(investigation_id=inv_id, state="queued"))
    assert filtered["item_count"] == 1
    assert filtered["queue"][0]["id"] == "release-task"
    assert filtered["queue"][0]["available"] is True

    reclaimed = _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="release-task", owner_session="session-y", lease_seconds=20))
    assert reclaimed["claimed"] is True
    assert reclaimed["item"]["owner_session"] == "session-y"

    # Stop-the-line is still available, explicitly, through complete(state="blocked").
    blocked = _json(server.investigation_queue_complete(investigation_id=inv_id, item_id="release-task", owner_session="session-y", state="blocked"))
    assert blocked["item"]["state"] == "blocked"
    filtered = _json(server.investigation_queue_status(investigation_id=inv_id, state="blocked"))
    assert filtered["item_count"] == 1
    assert filtered["queue"][0]["available"] is False
    final_release = _json(server.investigation_queue_release(investigation_id=inv_id, item_id="release-task", owner_session="session-y"))
    assert "already final" in final_release["error"]
def test_queue_rejects_invalid_payload_values(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-invalid"
    server.investigation_start(investigation_id=inv_id, title="Queue invalid payloads")

    bad_cases = [
        ("malformed_json", lambda: server.investigation_queue_enqueue(investigation_id=inv_id, item_json="{invalid-json"), "invalid json"),
        ("empty_id", lambda: server.investigation_queue_enqueue(investigation_id=inv_id, item_json=json.dumps({"id": "   ", "scope_kind": "session", "scope_targets": ["x"]})), "required"),
        ("bad_scope_kind", lambda: server.investigation_queue_enqueue(investigation_id=inv_id, item_json=json.dumps({"id": "bad-kind", "scope_kind": "bad kind"})), "scope_kind"),
        ("bad_scope_targets", lambda: server.investigation_queue_enqueue(investigation_id=inv_id, item_json=json.dumps({"id": "bad-targets", "scope_targets": {"not": "a-list"}})), "scope_targets"),
        ("bad_dependencies", lambda: server.investigation_queue_enqueue(investigation_id=inv_id, item_json=json.dumps({"id": "bad-deps", "dependencies": {"child": "item-1"}})), "dependencies"),
        ("bad_state", lambda: server.investigation_queue_enqueue(investigation_id=inv_id, item_json=json.dumps({"id": "bad-state", "state": 123})), "state"),
        ("bad_lease_expires_at", lambda: server.investigation_queue_enqueue(investigation_id=inv_id, item_json=json.dumps({"id": "lease-bad", "lease_expires_at": 123})), "lease_expires_at"),
        ("bad_lease_seconds", lambda: server.investigation_queue_claim(investigation_id=inv_id, item_id="missing", owner_session="session-a", lease_seconds=float("nan")), "lease_seconds"),
    ]

    for _, call, needle in bad_cases:
        result = _json(call())
        assert "error" in result
        assert needle in result["error"].lower()


def test_queue_requires_owner_to_complete_active_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-owner-required"
    server.investigation_start(investigation_id=inv_id, title="Queue owner requirement")
    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="owned-task"))
    _json(server.investigation_queue_claim(
        investigation_id=inv_id,
        item_id="owned-task",
        owner_session="session-owner",
        lease_seconds=120,
    ))

    missing_owner = _json(server.investigation_queue_complete(
        investigation_id=inv_id,
        item_id="owned-task",
        state="done",
    ))
    assert "error" in missing_owner
    assert "requires owner_session" in missing_owner["error"].lower()

    wrong_owner = _json(server.investigation_queue_complete(
        investigation_id=inv_id,
        item_id="owned-task",
        owner_session="session-other",
        state="done",
    ))
    assert "error" in wrong_owner
    assert "cannot be completed" in wrong_owner["error"].lower()


def test_queue_interleaving_takeover_after_expiry(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-interleave"
    server.investigation_start(investigation_id=inv_id, title="Queue interleaving")
    _json(server.investigation_queue_enqueue(
        investigation_id=inv_id,
        item_json=json.dumps({
            "id": "task-a",
            "scope_kind": "file",
            "scope_targets": ["mcp/investigation_tools.py"],
        }),
    ))
    _json(server.investigation_queue_claim(
        investigation_id=inv_id,
        item_id="task-a",
        owner_session="session-a",
        lease_seconds=60,
    ))

    blocked_claim = _json(server.investigation_queue_claim(
        investigation_id=inv_id,
        item_id="task-a",
        owner_session="session-b",
        lease_seconds=10,
    ))
    assert "error" in blocked_claim
    assert "already claimed" in blocked_claim["error"].lower()

    manifest_path = tmp_path / inv_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["coordination"]["items"][0]["lease_expires_at"] = "2000-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    server._manifest_cache.clear()

    takeover = _json(server.investigation_queue_claim(
        investigation_id=inv_id,
        item_id="task-a",
        owner_session="session-b",
        lease_seconds=90,
    ))
    assert "error" not in takeover
    assert takeover["item"]["owner_session"] == "session-b"

    done = _json(server.investigation_queue_complete(
        investigation_id=inv_id,
        item_id="task-a",
        owner_session="session-b",
        state="done",
    ))
    assert done["updated"] is True
    assert done["item"]["state"] == "done"


def test_queue_expiry_boundary_requires_current_owner_for_completion_and_release(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-expiry-boundary"
    server.investigation_start(investigation_id=inv_id, title="Queue expiry boundary")
    _json(server.investigation_queue_enqueue(
        investigation_id=inv_id,
        item_id="boundary-task",
        scope_kind="session",
        scope_targets=["session-a"],
    ))
    _json(server.investigation_queue_claim(
        investigation_id=inv_id,
        item_id="boundary-task",
        owner_session="session-a",
        lease_seconds=30,
    ))

    manifest_path = tmp_path / inv_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["coordination"]["items"][0]["lease_expires_at"] = "2999-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    server._manifest_cache.clear()

    blocked_claim = _json(server.investigation_queue_claim(
        investigation_id=inv_id,
        item_id="boundary-task",
        owner_session="session-b",
        lease_seconds=10,
    ))
    assert "error" in blocked_claim
    assert "already claimed" in blocked_claim["error"].lower()

    manifest = json.loads(manifest_path.read_text())
    manifest["coordination"]["items"][0]["lease_expires_at"] = "2000-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    server._manifest_cache.clear()

    missing_owner = _json(server.investigation_queue_complete(
        investigation_id=inv_id,
        item_id="boundary-task",
        state="done",
    ))
    assert "error" in missing_owner
    assert "requires owner_session" in missing_owner["error"].lower()

    stale_owner = _json(server.investigation_queue_complete(
        investigation_id=inv_id,
        item_id="boundary-task",
        owner_session="session-b",
        state="done",
    ))
    assert "error" in stale_owner
    assert "cannot be completed" in stale_owner["error"].lower()

    released = _json(server.investigation_queue_release(
        investigation_id=inv_id,
        item_id="boundary-task",
        owner_session="session-b",
        notes="blocked by stale owner",
    ))
    assert "error" in released
    assert "cannot be released" in released["error"].lower()


def _expire_lease(tmp_path, inv_id, item_id):
    manifest_path = tmp_path / inv_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["coordination"]["items"]:
        if item["id"] == item_id:
            item["lease_expires_at"] = "2000-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    server._manifest_cache.clear()


def test_queue_claimed_enqueue_always_has_a_lease(tmp_path, monkeypatch):
    # Audit repro: enqueue(state='claimed', owner_session=S) stored no lease, and a missing
    # lease counted as never expiring, so the item was locked by S forever.
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-ghost"
    server.investigation_start(investigation_id=inv_id, title="Queue ghost claim")
    ghost = _json(server.investigation_queue_enqueue(
        investigation_id=inv_id, item_id="ghost", state="claimed", owner_session="crashed-session",
    ))
    assert ghost["item"]["state"] == "claimed"
    assert ghost["item"]["lease_expires_at"]

    ownerless = _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="nobody", state="claimed"))
    assert "requires owner_session" in ownerless["error"]

    _expire_lease(tmp_path, inv_id, "ghost")
    taken = _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="ghost", owner_session="B", lease_seconds=1))
    assert taken["claimed"] is True
    assert taken["item"]["owner_session"] == "B"


def test_queue_claim_without_lease_is_reclaimable(tmp_path, monkeypatch):
    # Legacy/imported manifests can carry a claimed item with an owner but no lease.
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-no-lease"
    server.investigation_start(investigation_id=inv_id, title="Queue lease-less claim")
    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="stuck"))
    manifest_path = tmp_path / inv_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["coordination"]["items"][0].update(state="claimed", owner_session="gone", lease_expires_at=None)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    server._manifest_cache.clear()

    status = _json(server.investigation_queue_status(investigation_id=inv_id, item_id="stuck"))
    assert status["queue"][0]["lease_expired"] is True
    assert status["queue"][0]["available"] is True
    taken = _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="stuck", owner_session="B", lease_seconds=30))
    assert taken["claimed"] is True


def test_queue_claim_enforces_dependencies(tmp_path, monkeypatch):
    # Audit repro: a child whose dependencies ['parent', 'does-not-exist'] were unmet was claimed.
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-deps"
    server.investigation_start(investigation_id=inv_id, title="Queue dependencies")
    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="parent"))
    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="child", dependencies=["parent", "does-not-exist"]))

    blocked = _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="child", owner_session="B", lease_seconds=60))
    assert "unmet dependencies" in blocked["error"]
    assert "parent" in blocked["error"] and "does-not-exist" in blocked["error"]
    # Missing ids are reported apart from existing-but-not-done ones.
    assert blocked["unmet_dependencies"] == ["parent"]
    assert blocked["unknown_dependencies"] == ["does-not-exist"]
    status = _json(server.investigation_queue_status(investigation_id=inv_id, item_id="child"))
    assert status["queue"][0]["state"] == "queued"
    assert status["queue"][0]["available"] is False

    _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="parent", owner_session="A", lease_seconds=60))
    _json(server.investigation_queue_complete(investigation_id=inv_id, item_id="parent", owner_session="A", state="done"))
    still_blocked = _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="child", owner_session="B", lease_seconds=60))
    assert still_blocked["error"].endswith("does-not-exist")

    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="does-not-exist"))
    _json(server.investigation_queue_complete(investigation_id=inv_id, item_id="does-not-exist", state="cancelled"))
    # A cancelled dependency is not a satisfied one.
    assert "error" in _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="child", owner_session="B", lease_seconds=60))


def test_queue_release_is_not_terminal(tmp_path, monkeypatch):
    # Audit repro: release(parent) set 'blocked', and a reclaim by C failed with 'already final'.
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-release-reclaim"
    server.investigation_start(investigation_id=inv_id, title="Queue release reclaim")
    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="parent"))
    _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="parent", owner_session="A", lease_seconds=60))
    released = _json(server.investigation_queue_release(investigation_id=inv_id, item_id="parent", owner_session="A"))
    assert released["item"]["state"] == "queued"
    reclaimed = _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="parent", owner_session="C", lease_seconds=60))
    assert reclaimed["claimed"] is True
    assert reclaimed["item"]["owner_session"] == "C"


def test_queue_status_reports_expired_lease(tmp_path, monkeypatch):
    # Audit repro: an item whose lease had expired was reported as plainly 'claimed' by D.
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    inv_id = "q-status-expired"
    server.investigation_start(investigation_id=inv_id, title="Queue status expiry")
    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="short"))
    _json(server.investigation_queue_claim(investigation_id=inv_id, item_id="short", owner_session="D", lease_seconds=60))

    live = _json(server.investigation_queue_status(investigation_id=inv_id, item_id="short"))["queue"][0]
    assert live["state"] == "claimed"
    assert live["lease_expired"] is False
    assert live["available"] is False

    _expire_lease(tmp_path, inv_id, "short")
    expired = _json(server.investigation_queue_list(investigation_id=inv_id, item_id="short"))["queue"][0]
    assert expired["state"] == "claimed"
    assert expired["owner_session"] == "D"
    assert expired["lease_expired"] is True
    assert expired["available"] is True

    # Derived fields are a view only; they must not leak into the stored manifest
    # (status reads the cached manifest, and the next mutation saves it).
    _json(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="other"))
    stored =json.loads((tmp_path / inv_id / "manifest.json").read_text())
    assert "lease_expired" not in stored["coordination"]["items"][0]
    assert "available" not in stored["coordination"]["items"][0]

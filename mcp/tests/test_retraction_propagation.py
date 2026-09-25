"""#383 follow-up (c): a retraction reaches the index stores, softly and reversibly.

Before: memory_retract only appended to retractions.jsonl. The finding's own
Qdrant point and its Mnemosyne rows stayed live, so anything that read those
stores directly (the A2A rag_search skill, Mnemosyne's own recall, a filter
whose log could not be read) still served the retracted text.

Now: retract sets ``retracted: true`` on the Qdrant point payload and stamps
Mnemosyne's ``valid_until`` on matching rows; restore reverses both. Nothing is
deleted. Recall filters honour the payload flag on its own.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

import inv_store
import mnemo_ops
import server


class FakeQdrant:
    def __init__(self, points=(), fail_with=None):
        self.payloads = {str(p): {} for p in points}
        self.calls = []
        self.deleted = []
        self.fail_with = fail_with

    def set_payload(self, collection_name, payload, points, wait=True, **kw):
        self.calls.append((collection_name, dict(payload), list(points)))
        if self.fail_with:
            raise self.fail_with
        for p in points:
            if str(p) not in self.payloads:
                raise RuntimeError("Unexpected Response: 404 (Not Found) No point with id")
            self.payloads[str(p)].update(payload)

    def delete(self, *a, **k):  # must never be called by retract/restore
        self.deleted.append((a, k))


@pytest.fixture
def store(tmp_path, monkeypatch):
    mem = tmp_path / "mem"
    mem.mkdir()
    monkeypatch.setattr(server, "MEMORY_DIR", mem)
    monkeypatch.setenv("LOCI_RAG_EXPAND", "0")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "mnemo"))
    monkeypatch.delenv("HERMES_AGENT_ID", raising=False)
    inv_store._manifest_cache.clear()
    no = lambda *a, **k: None  # noqa: E731
    for name, value in {
        "_get_qdrant": lambda *a, **k: (None, None),
        "_qdrant_upsert": lambda *a, **k: False,
        "_mnemo_remember": lambda *a, **k: False,
        "_event_log_append": no,
        "_mirror_finding_to_ladybug": no,
        "_autolink_finding_to_ladybug": no,
        "_retract_quarantine_verdict": lambda *a, **k: False,
        "_forget_finding_verdicts": lambda *a, **k: 0,
        "_semantic_neighbor_ids": lambda *a, **k: [],
        "_get_cross_encoder": lambda *a, **k: None,
        "docs_search": lambda *a, **k: json.dumps({"results": []}),
    }.items():
        monkeypatch.setattr(server, name, value, raising=False)
    no_mnemo = lambda: (None, None)  # noqa: E731
    monkeypatch.setattr(server, "_get_mnemo_funcs", no_mnemo)
    monkeypatch.setattr(mnemo_ops, "_get_mnemo_funcs", no_mnemo)
    yield tmp_path
    inv_store._manifest_cache.clear()


def _case(inv="rp-case"):
    server.investigation_start(investigation_id=inv, title="propagation")
    bad = json.loads(server.investigation_store(inv, "observed", "Endpoint /v9/debug exists on alpha", "t"))["finding_id"]
    good = json.loads(server.investigation_store(inv, "observed", "Service beta answers on 443", "t"))["finding_id"]
    return inv, bad, good


def _retract(inv, fid):
    r = json.loads(server.memory_retract(inv, fid, reason="hallucination", dry_run=False, scope_semantic=False))
    assert r.get("applied") is True, r
    return r


def _mnemo_db(tmp_path, rows):
    d = tmp_path / "mnemo"
    d.mkdir(exist_ok=True)
    conn = sqlite3.connect(d / "mnemosyne.db")
    for table in ("working_memory", "episodic_memory"):
        conn.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, content TEXT, metadata_json TEXT, valid_until TEXT)")
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, metadata_json TEXT)")
    for table, mid, meta, vu in rows:
        if table == "memories":
            conn.execute("INSERT INTO memories VALUES (?,?,?)", (mid, "x", meta))
        else:
            conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?)", (mid, "x", meta, vu))
    conn.commit()
    conn.close()
    return d / "mnemosyne.db"


def _valid_until(db, table, mid):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(f"SELECT valid_until FROM {table} WHERE id=?", (mid,)).fetchone()[0]
    finally:
        conn.close()


def test_retract_flags_the_qdrant_point_and_restore_unflags_it(store, monkeypatch):
    inv, bad, good = _case()
    q = FakeQdrant(points=[bad, good])
    monkeypatch.setattr(server, "_get_qdrant", lambda *a, **k: (q, "col"))
    r = _retract(inv, bad)
    assert r["propagation"]["qdrant"] == {"status": "ok", "updated": 1, "missing": 0, "failed": 0}
    assert q.payloads[bad]["retracted"] is True and "retracted_at" in q.payloads[bad]
    assert q.payloads[good] == {}
    assert q.deleted == []  # soft: never a delete

    back = json.loads(server.memory_restore(inv, finding_id=bad))
    assert back["restored"] is True
    assert back["propagation"]["qdrant"]["updated"] == 1
    assert q.payloads[bad]["retracted"] is False and "restored_at" in q.payloads[bad]
    assert q.deleted == []


def test_cold_finding_without_a_point_is_missing_not_failed(store, monkeypatch):
    inv, bad, _ = _case()
    q = FakeQdrant(points=[])
    monkeypatch.setattr(server, "_get_qdrant", lambda *a, **k: (q, "col"))
    r = _retract(inv, bad)
    assert r["propagation"]["qdrant"]["status"] == "ok"
    assert r["propagation"]["qdrant"]["missing"] == 1


def test_propagation_failure_is_reported_and_does_not_undo_the_tombstone(store, monkeypatch):
    inv, bad, _ = _case()
    q = FakeQdrant(points=[bad], fail_with=ConnectionError("connection refused"))
    monkeypatch.setattr(server, "_get_qdrant", lambda *a, **k: (q, "col"))
    r = _retract(inv, bad)
    assert r["propagation"]["qdrant"]["status"] == "failed"
    assert r["propagation"]["qdrant"]["failed"] == 1
    assert bad in server._load_retracted_ids(inv)
    audit = server._read_jsonl(server._inv_dir(inv) / "retraction_audit.jsonl")
    assert audit[-1]["propagation"]["qdrant"]["status"] == "failed"


def test_retract_stamps_mnemosyne_valid_until_and_restore_clears_only_its_stamp(store):
    inv, bad, good = _case()
    meta = lambda fid: json.dumps({"finding_id": fid, "investigation_id": inv})  # noqa: E731
    db = _mnemo_db(store, [
        ("working_memory", "w-bad", meta(bad), None),
        ("episodic_memory", "e-bad", meta(bad), None),
        ("working_memory", "w-good", meta(good), None),
        # Mnemosyne invalidated this one itself: retract/restore must leave it alone.
        ("episodic_memory", "e-bad-old", meta(bad), "2020-01-01T00:00:00"),
        ("working_memory", "w-junk", "not json", None),
        ("memories", "m-bad", meta(bad), None),
    ])
    r = _retract(inv, bad)
    m = r["propagation"]["mnemosyne"]
    assert m["status"] == "ok" and m["updated"] == 2 and m["legacy_unflagged"] == 1, m
    assert _valid_until(db, "working_memory", "w-bad") == m["stamp"]
    assert _valid_until(db, "episodic_memory", "e-bad") == m["stamp"]
    assert _valid_until(db, "working_memory", "w-good") is None
    assert _valid_until(db, "episodic_memory", "e-bad-old") == "2020-01-01T00:00:00"

    back = json.loads(server.memory_restore(inv, finding_id=bad))
    assert back["propagation"]["mnemosyne"]["updated"] == 2
    assert _valid_until(db, "working_memory", "w-bad") is None
    assert _valid_until(db, "episodic_memory", "e-bad") is None
    assert _valid_until(db, "episodic_memory", "e-bad-old") == "2020-01-01T00:00:00"
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 3  # nothing deleted
    conn.close()


def test_missing_mnemosyne_db_is_reported_and_never_created(store):
    inv, bad, _ = _case()
    r = _retract(inv, bad)
    assert r["propagation"]["mnemosyne"]["status"] == "unavailable"
    assert not (store / "mnemo" / "mnemosyne.db").exists()


def test_recall_filter_honours_the_payload_flag_on_its_own(store, monkeypatch):
    """A hit flagged retracted in Qdrant is dropped even with no tombstone in view."""
    inv, bad, good = _case()
    hits = [
        {"id": bad, "investigation_id": inv, "text": "Endpoint /v9/debug exists on alpha", "score": 0.9,
         "origin": server.QDRANT_COLLECTION_PREFIX, "retracted": True},
        {"id": good, "investigation_id": inv, "text": "Service beta answers on 443", "score": 0.8,
         "origin": server.QDRANT_COLLECTION_PREFIX, "retracted": False},
    ]
    monkeypatch.setattr(server, "_get_qdrant", lambda *a, **k: (object(), server.QDRANT_COLLECTION_PREFIX))
    monkeypatch.setattr(server, "_qdrant_search_collection", lambda *a, **k: [dict(h) for h in hits])
    r = json.loads(server.rag_context_search("endpoint debug", expand_query=False))
    assert "/v9/debug" not in r["context"] and "beta answers" in r["context"]
    assert r["excluded_retracted"] == 1
    route = json.loads(server.memory_route("endpoint debug"))
    assert [row.get("finding_id") or row.get("id") for row in route["routed"]] == [good]
    assert route["excluded_retracted"] == 1


def test_propagation_can_be_disabled(store, monkeypatch):
    inv, bad, _ = _case()
    q = FakeQdrant(points=[bad])
    monkeypatch.setattr(server, "_get_qdrant", lambda *a, **k: (q, "col"))
    monkeypatch.setenv("LOCI_RETRACT_PROPAGATE", "0")
    r = _retract(inv, bad)
    assert r["propagation"]["qdrant"]["status"] == "disabled" and q.calls == []

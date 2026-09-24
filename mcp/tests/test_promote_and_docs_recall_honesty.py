"""#383 follow-ups (a) and (e): no success or score that the store did not earn.

(a) memory_promote returned ok:true after _qdrant_upsert swallowed a failure,
    although hot/warm are documented as "Qdrant indexed".
(e) docs_recall reported every hit with a fixed score of 0.95, so a doc that
    shared one incidental token read as a near-certain match.
"""
from __future__ import annotations

import json
import os

import pytest

import server


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    server._manifest_cache.clear()
    for name in ("_event_log_append", "_mirror_finding_to_ladybug", "_autolink_finding_to_ladybug"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_mnemo_remember", lambda *a, **k: False)
    monkeypatch.setattr(server, "_get_qdrant", lambda *a, **k: (None, None))
    yield tmp_path
    server._manifest_cache.clear()


def _store_cold(inv):
    server.investigation_start(investigation_id=inv, title="promote honesty")
    r = json.loads(server.investigation_store(inv, "observed", "Port 8443 answers TLS", "unit-test", tier="cold"))
    assert r.get("stored"), r
    return r["finding_id"]


def test_promote_with_failed_upsert_is_not_ok(store, monkeypatch):
    fid = _store_cold("promote-fail")
    monkeypatch.setattr(server, "_qdrant_upsert", lambda *a, **k: False)
    r = json.loads(server.memory_promote("promote-fail", fid, "warm"))
    assert r["ok"] is False, r
    assert r["qdrant_indexed"] is False
    assert r["degraded"] is True and r["retryable"] is True
    assert "not indexed" in r["error"]


def test_promote_retry_after_failure_indexes_and_reports_ok(store, monkeypatch):
    fid = _store_cold("promote-retry")
    monkeypatch.setattr(server, "_qdrant_upsert", lambda *a, **k: False)
    assert json.loads(server.memory_promote("promote-retry", fid, "warm"))["ok"] is False
    calls = []
    monkeypatch.setattr(server, "_qdrant_upsert", lambda *a, **k: calls.append(a) or True)
    # Same tier as now recorded: must still attempt the index write, not short-circuit to ok.
    r = json.loads(server.memory_promote("promote-retry", fid, "warm"))
    assert r["ok"] is True and r["qdrant_indexed"] is True, r
    assert len(calls) == 1


def test_promote_with_landed_upsert_is_ok(store, monkeypatch):
    fid = _store_cold("promote-ok")
    monkeypatch.setattr(server, "_qdrant_upsert", lambda *a, **k: True)
    r = json.loads(server.memory_promote("promote-ok", fid, "hot"))
    assert r == {"finding_id": fid, "old_tier": "cold", "new_tier": "hot", "ok": True, "qdrant_indexed": True}


def test_demote_to_cold_with_qdrant_down_does_not_claim_removal(store):
    server.investigation_start(investigation_id="demote-unknown", title="t")
    fid = json.loads(server.investigation_store("demote-unknown", "observed", "x y z", "unit-test"))["finding_id"]
    r = json.loads(server.memory_demote("demote-unknown", fid, "cold"))
    assert r["qdrant_removed"] is None and r["degraded"] is True, r


@pytest.fixture
def docs(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path / "mem")
    server._manifest_cache.clear()
    monkeypatch.setitem(os.environ, "LOCI_DOCS_ROOTS", str(tmp_path))
    strong = tmp_path / "STRONG.md"
    strong.write_text("# Strong\n\nClaims must carry dataset version scope.\n", encoding="utf-8")
    weak = tmp_path / "WEAK.md"
    weak.write_text("# Weak\n\nThe dataset is large.\n", encoding="utf-8")
    server.docs_ingest_indexer(str(weak), investigation_id="docs-score")
    server.docs_ingest_indexer(str(strong), investigation_id="docs-score")
    yield {"strong": str(strong), "weak": str(weak)}
    server._manifest_cache.clear()


def test_docs_recall_reports_the_real_lexical_score(docs):
    r = json.loads(server.docs_recall("dataset version scope", investigation_id="docs-score"))
    by_path = {hit["path"]: hit for hit in r["results"]}
    assert by_path[docs["strong"]]["score"] == 1.0
    # One of three query tokens: a weak lexical hit, not 0.95.
    assert by_path[docs["weak"]]["score"] == pytest.approx(1 / 3, abs=1e-3)
    assert all(hit["score"] != 0.95 for hit in r["results"])
    assert all(hit["score_kind"] == "lexical" and hit["origin"] == "docs_search" for hit in r["results"])


def test_docs_search_ranks_the_better_match_first(docs):
    r = json.loads(server.docs_search("dataset version scope", investigation_id="docs-score"))
    assert [h["path"] for h in r["results"]] == [docs["strong"], docs["weak"]]
    assert r["results"][0]["score"] >= r["results"][1]["score"]

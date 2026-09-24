"""_search_qdrant_claim_evidence driven against a real in-memory Qdrant.

test_semantic_gate_corroboration.py monkeypatches this function away and feeds the
gate hand-built refs, so the fields the gate reads (pool_size, pool_median, margin,
lexical_overlap) were never computed by the code under test: replacing all four with
constants passed the suite. Here only the dependencies are faked -- the Qdrant client
is qdrant_client's local ``:memory:`` engine and the embedder returns a fixed vector --
and every field of every ref is pinned to a hand-computed value.
"""
from __future__ import annotations

import json
import math
import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import server

COL = "loci_memory_claim_evidence_test"
QUERY = [1.0, 0.0, 0.0]
CLAIM = "The rotation daemon leaks refresh tokens into syslog"
# tokenize() drops stopwords ("the", "into"): 6 content tokens.
assert server.tokenize(CLAIM) == {"rotation", "daemon", "leaks", "refresh", "tokens", "syslog"}


def _vec(cos: float) -> list[float]:
    """A unit vector whose cosine with QUERY is exactly ``cos``."""
    return [cos, math.sqrt(1.0 - cos * cos), 0.0]


# (payload id, investigation, cosine with the claim, payload extras)
_POINTS = [
    ("f-1", "inv-a", 0.90, {"text": "rotation daemon leaks refresh tokens", "record_type": "observed",
                            "source": "edr", "ts": "2026-09-01T00:00:00Z",
                            "evidence_provenance_tier": "tool_verified"}),
    ("f-ret", "inv-a", 0.85, {"text": "rotation daemon leaks refresh tokens into syslog", "source": "edr"}),
    ("f-2", "inv-a", 0.80, {"output": "syslog noise\nonly", "type": "hypothesis", "tool": "grep",
                            "evidence_provenance_tier": "model_asserted"}),
    ("f-3", "inv-a", 0.70, {"text": "unrelated disk quota warning"}),
    ("f-4", "inv-a", 0.60, {"text": "daemon restarted", "record_type": "observed", "source": "systemd"}),
    ("f-5", "inv-a", 0.50, {"text": "rotation daemon leaks refresh tokens into syslog"}),  # below the top 5
    ("f-other", "inv-b", 0.99, {"text": "rotation daemon leaks refresh tokens into syslog"}),  # other case
]


@pytest.fixture
def qdrant(monkeypatch, tmp_path):
    client = QdrantClient(location=":memory:")
    client.create_collection(COL, vectors_config={"dense": VectorParams(size=3, distance=Distance.COSINE)})
    client.upsert(COL, points=[
        PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, pid)), vector={"dense": _vec(cos)},
                    payload={"id": pid, "investigation_id": inv, **extra})
        for pid, inv, cos, extra in _POINTS
    ])
    embedded = []

    def _embed(text):
        embedded.append(text)
        return list(QUERY)

    monkeypatch.setattr(server, "_get_qdrant", lambda: (client, COL))
    monkeypatch.setattr(server, "_embed", _embed)
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    (tmp_path / "inv-a").mkdir()
    (tmp_path / "inv-a" / "retractions.jsonl").write_text(
        json.dumps({"finding_id": "f-ret", "ts": "2026-09-02T00:00:00Z", "active": True}) + "\n")
    yield embedded
    client.close()


def _ref(pid, score, overlap, text, record_type="unknown", source="", ts=None,
         tier="tool_verified", defaulted=True):
    # Pool for inv-a is all six of its points (the retracted one included):
    # scores 0.9 0.85 0.8 0.7 0.6 0.5 -> median (0.8 + 0.7) / 2 = 0.75.
    return {
        "evidence_id": pid, "record_type": record_type, "source": source, "ts": ts,
        "origin": "qdrant", "score": score, "text": text, "lexical_overlap": overlap,
        "pool_median": 0.75, "pool_size": 6, "margin": round(score - 0.75, 4),
        "snippet": text.replace("\n", " ").strip(),
        "evidence_provenance_tier": tier, "provenance_defaulted": defaulted,
    }


def test_claim_evidence_fields_are_computed_from_the_real_neighbourhood(qdrant):
    refs, status = server._search_qdrant_claim_evidence(CLAIM, "inv-a")

    assert qdrant == [CLAIM]
    assert status == {"enabled": True, "available": True, "query_attempted": True, "error": None,
                      "excluded_retracted": 1}
    # Top 5 of inv-a by score, minus the retracted f-ret; f-other (inv-b) never enters.
    assert refs == [
        _ref("f-1", 0.9, round(5 / 6, 4), "rotation daemon leaks refresh tokens",
             record_type="observed", source="edr", ts="2026-09-01T00:00:00Z", defaulted=False),
        _ref("f-2", 0.8, round(1 / 6, 4), "syslog noise\nonly", record_type="hypothesis", source="grep",
             tier="model_asserted", defaulted=False),
        _ref("f-3", 0.7, 0.0, "unrelated disk quota warning"),
        _ref("f-4", 0.6, round(1 / 6, 4), "daemon restarted", record_type="observed", source="systemd"),
    ]
    assert [r["margin"] for r in refs] == [0.15, 0.05, -0.05, -0.15]


def test_claim_evidence_limit_bounds_the_surfaced_refs_not_the_pool(qdrant):
    refs, status = server._search_qdrant_claim_evidence(CLAIM, "inv-a", limit=2)
    # Top 2 are f-1 and the retracted f-ret; the pool statistics still cover all six.
    assert [r["evidence_id"] for r in refs] == ["f-1"]
    assert (refs[0]["pool_size"], refs[0]["pool_median"]) == (6, 0.75)
    assert status["excluded_retracted"] == 1


def test_claim_evidence_reports_a_failed_query_as_an_error(qdrant, monkeypatch):
    monkeypatch.setattr(server, "_get_qdrant", lambda: (QdrantClient(location=":memory:"), "missing_collection"))
    refs, status = server._search_qdrant_claim_evidence(CLAIM, "inv-a")
    assert refs == []
    assert status["query_attempted"] is True and status["available"] is True
    assert "missing_collection" in status["error"]


def test_claim_evidence_without_an_embedding_is_reported_not_queried(qdrant, monkeypatch):
    monkeypatch.setattr(server, "_embed", lambda text: None)
    refs, status = server._search_qdrant_claim_evidence(CLAIM, "inv-a")
    assert refs == []
    assert status == {"enabled": True, "available": True, "query_attempted": False,
                      "error": "embedding_unavailable"}


def test_claim_evidence_without_a_client_is_unavailable(monkeypatch):
    embedded = []
    monkeypatch.setattr(server, "_get_qdrant", lambda: (None, COL))
    monkeypatch.setattr(server, "_embed", lambda text: embedded.append(text))
    monkeypatch.setenv("QDRANT_URL", "")
    refs, status = server._search_qdrant_claim_evidence(CLAIM, "inv-a")
    assert refs == []
    assert status == {"enabled": False, "available": False, "query_attempted": False, "error": None}
    assert embedded == []

"""Tests for scripts/reembed_daemon — batch GPU re-embed. Qdrant + embed are stubbed.

No live Qdrant/Ollama/GPU: a fake client captures scroll/upsert and a stub embed_fn
returns deterministic vectors. Proves the safety contract (dry-run mutates nothing),
incremental targeting (only stale/missing points), batching, and per-batch fail-open.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# The module under test lives in ../../scripts.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import reembed_daemon as R  # noqa: E402

MODEL = "nomic-embed-text"
VERSION = "1"


# ── fakes ────────────────────────────────────────────────────────────────────

class FakeQdrant:
    """Captures scroll paging + upsert calls. scroll returns (points, next_offset)."""

    def __init__(self, pages):
        self.pages = pages          # list[list[point-dict]]
        self.upserts = []           # list of the `points` lists passed to upsert
        self.scroll_calls = 0

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        assert with_vectors is True and with_payload is True
        self.scroll_calls += 1
        idx = offset or 0
        if idx >= len(self.pages):
            return [], None
        nxt = idx + 1 if idx + 1 < len(self.pages) else None
        return self.pages[idx], nxt

    def upsert(self, collection_name, points):
        self.upserts.append(points)


def _pt(pid, *, text="hello", model=MODEL, version=VERSION, vector=(0.1, 0.2)):
    payload = {"text": text}
    if model is not None:
        payload["embed_model"] = model
    if version is not None:
        payload["embed_version"] = version
    return {"id": pid, "payload": payload, "vector": list(vector) if vector else vector}


def _fresh(pid):
    return _pt(pid)                                  # has vector, model+version match


def _stale_model(pid):
    return _pt(pid, model="old-model")               # wrong model


def _stale_version(pid):
    return _pt(pid, version="0")                      # wrong version


def _missing_vec(pid):
    return _pt(pid, vector=None)                      # no vector at all


def _stub_embed(texts):
    """Deterministic 2-D vectors; one per text (correct count)."""
    return [[float(len(t)), 1.0] for t in texts]


def _upserted_ids(client):
    ids = []
    for batch in client.upserts:
        for p in batch:
            ids.append(p["id"] if isinstance(p, dict) else getattr(p, "id"))
    return ids


def _payload_of(p):
    return p["payload"] if isinstance(p, dict) else getattr(p, "payload")


# ── tests ────────────────────────────────────────────────────────────────────

def test_dry_run_mutates_nothing():
    pages = [[_fresh("a"), _stale_model("b"), _missing_vec("c")]]
    client = FakeQdrant(pages)

    called = {"n": 0}

    def _boom_embed(texts):        # must NOT be called on a dry-run
        called["n"] += 1
        raise AssertionError("embed_fn called during dry-run")

    r = R.reembed("hermes_memory", qdrant_client=client, embed_fn=_boom_embed,
                  apply=False)

    assert r["dry_run"] is True and r["applied"] is False
    assert client.upserts == []            # wrote NOTHING
    assert called["n"] == 0                 # GPU never touched
    assert r["scanned"] == 3
    assert r["missing_vector"] == 1
    assert r["stale_meta"] == 1             # the wrong-model point
    assert r["targeted"] == 2               # b + c would be re-embedded
    assert r["reembedded"] == 0
    assert r["degraded"] is False


def test_apply_upserts_only_stale_and_missing():
    pages = [[
        _fresh("keep1"),
        _stale_model("m"),
        _stale_version("v"),
        _missing_vec("x"),
        _fresh("keep2"),
    ]]
    client = FakeQdrant(pages)

    r = R.reembed("hermes_memory", qdrant_client=client, embed_fn=_stub_embed,
                  apply=True, batch_size=64)

    assert r["applied"] is True and r["dry_run"] is False
    assert set(_upserted_ids(client)) == {"m", "v", "x"}   # fresh points untouched
    assert r["targeted"] == 3
    assert r["reembedded"] == 3
    assert r["errors"] == []
    # Re-embedded points get the current model/version stamped -> idempotent next run.
    for batch in client.upserts:
        for p in batch:
            pl = _payload_of(p)
            assert pl["embed_model"] == MODEL
            assert pl["embed_version"] == VERSION


def test_idempotent_second_run_targets_nothing():
    # All points already fresh -> nothing targeted, nothing written.
    pages = [[_fresh("a"), _fresh("b"), _fresh("c")]]
    client = FakeQdrant(pages)
    r = R.reembed("hermes_memory", qdrant_client=client, embed_fn=_stub_embed,
                  apply=True)
    assert r["targeted"] == 0 and r["reembedded"] == 0
    assert client.upserts == []


def test_batching_chunks_correctly():
    # 10 stale points across 2 pages, batch_size 4 -> ceil(10/4) = 3 upsert calls.
    n = 10
    bs = 4
    stale = [_stale_model(f"p{i}") for i in range(n)]
    pages = [stale[:6], stale[6:]]         # paged
    client = FakeQdrant(pages)

    r = R.reembed("hermes_memory", qdrant_client=client, embed_fn=_stub_embed,
                  apply=True, batch_size=bs, page_limit=6)

    assert r["targeted"] == n and r["reembedded"] == n
    assert len(client.upserts) == math.ceil(n / bs) == 3
    sizes = [len(b) for b in client.upserts]
    assert all(s <= bs for s in sizes)
    assert sum(sizes) == n
    assert sizes == [4, 4, 2]


def test_fail_open_on_embed_error():
    # First batch's embed raises; run must continue and upsert the rest, no exception.
    stale = [_stale_model(f"p{i}") for i in range(6)]
    pages = [stale]
    client = FakeQdrant(pages)

    calls = {"n": 0}

    def _flaky_embed(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("gpu hiccup")
        return _stub_embed(texts)

    r = R.reembed("hermes_memory", qdrant_client=client, embed_fn=_flaky_embed,
                  apply=True, batch_size=3)

    # 2 batches attempted; first failed (no upsert), second succeeded.
    assert r["embed_batches"] == 2
    assert len(client.upserts) == 1               # only the surviving batch
    assert r["reembedded"] == 3
    assert any("embed:" in e for e in r["errors"])
    assert r["degraded"] is False                 # per-batch failure is NOT a hard degrade


def test_fail_open_on_embed_count_mismatch():
    stale = [_stale_model("a"), _stale_model("b")]
    client = FakeQdrant([stale])

    def _short_embed(texts):
        return [[0.0, 1.0]]                        # wrong count (1 for 2 texts)

    r = R.reembed("hermes_memory", qdrant_client=client, embed_fn=_short_embed,
                  apply=True, batch_size=64)
    assert client.upserts == []                   # nothing written on a bad batch
    assert r["reembedded"] == 0
    assert any("embed-count" in e for e in r["errors"])


def test_scroll_error_is_fail_open_degraded():
    class Boom(FakeQdrant):
        def scroll(self, **kw):
            raise RuntimeError("qdrant down")

    r = R.reembed("hermes_memory", qdrant_client=Boom([]), embed_fn=_stub_embed,
                  apply=True)
    assert r["degraded"] is True
    assert r["reembedded"] == 0
    assert any("scroll:" in e for e in r["errors"])


def test_named_vector_missing_detection():
    # Named-vector collection: point missing the named vector is targeted.
    p_ok = {"id": "ok", "payload": {"text": "x", "embed_model": MODEL,
                                    "embed_version": VERSION},
            "vector": {"dense": [0.1, 0.2]}}
    p_missing = {"id": "miss", "payload": {"text": "y", "embed_model": MODEL,
                                           "embed_version": VERSION},
                 "vector": {"dense": None}}
    client = FakeQdrant([[p_ok, p_missing]])
    r = R.reembed("c", qdrant_client=client, embed_fn=_stub_embed, apply=False,
                  vector_name="dense")
    assert r["missing_vector"] == 1 and r["targeted"] == 1

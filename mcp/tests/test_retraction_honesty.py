"""Retraction honesty: every recall path drops retracted findings, reports the
exclusions, tolerates malformed investigation dirs, and retract/restore are
exact inverses.

Adapted from the audit repros under /tmp/loci-audit/{lifecycle,storage,health,
live,skeptic-lifecycle}. Everything runs against a temp MEMORY_DIR with Qdrant,
Mnemosyne, LLM and graph side effects stubbed out.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import grounding  # noqa: E402
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
    monkeypatch.setenv("LOCI_RAG_EXPAND", "0")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setattr(inv_store, "_STORE_LOCK_TIMEOUT_S", 0.5)
    inv_store._manifest_cache.clear()
    no = lambda *a, **k: None  # noqa: E731
    for name, value in {
        "_get_qdrant": lambda *a, **k: (None, None),
        "_qdrant_upsert": no,
        "_mnemo_remember": lambda *a, **k: False,
        "_event_log_append": no,
        "_mirror_finding_to_ladybug": no,
        "_autolink_finding_to_ladybug": no,
        "_retract_quarantine_verdict": lambda *a, **k: False,
        "_update_entities_jsonl": no,
        "_forget_finding_verdicts": lambda *a, **k: 0,
        "_semantic_neighbor_ids": lambda *a, **k: [],
        "_get_cross_encoder": lambda *a, **k: None,
        "docs_search": lambda *a, **k: json.dumps({"results": []}),
    }.items():
        monkeypatch.setattr(server, name, value, raising=False)
    no_mnemo = lambda: (None, None)  # noqa: E731
    monkeypatch.setattr(server, "_get_mnemo_funcs", no_mnemo)
    monkeypatch.setattr(mnemo_ops, "_get_mnemo_funcs", no_mnemo)
    yield mem
    inv_store._manifest_cache.clear()


def _start(inv: str) -> str:
    assert _j(server.investigation_start(investigation_id=inv, title="retraction audit")).get("status") == "created"
    return inv


def _store(inv: str, text: str, **kw) -> str:
    r = _j(server.investigation_store(inv, "observed", text, "unit-test", **kw))
    assert r.get("stored"), r
    return r["finding_id"]


def _retract(inv: str, fid: str) -> None:
    r = _j(server.memory_retract(inv, fid, reason="hallucination", dry_run=False, scope_semantic=False))
    assert r.get("applied") is True, r


def _malformed_undefined_dir(mem: Path, fid: str = "undef-f1", text: str = "undefined-dir hallucination") -> None:
    """A legacy dir named with a missing-value sentinel; never renamed or deleted."""
    d = mem / "undefined"
    d.mkdir()
    (d / "findings.jsonl").write_text(json.dumps({"id": fid, "record_type": "observed", "text": text}) + "\n")
    (d / "retractions.jsonl").write_text(json.dumps({"finding_id": fid, "active": True, "ts": "2026-01-01T00:00:00+00:00"}) + "\n")


def _fake_mnemo(monkeypatch, bank: list) -> None:
    def recall(query, top_k=10, **kw):
        return [dict(b) for b in bank]
    funcs = lambda: (None, recall)  # noqa: E731
    monkeypatch.setattr(server, "_get_mnemo_funcs", funcs)
    monkeypatch.setattr(mnemo_ops, "_get_mnemo_funcs", funcs)


def _mnemo_item(inv: str, fid: str, text: str) -> dict:
    return {"content": text, "score": 0.9,
            "metadata": {"investigation_id": inv, "record_type": "observed", "source": "unit-test", "finding_id": fid}}


def _qdrant_hits(monkeypatch, hits: list) -> None:
    monkeypatch.setattr(server, "_get_qdrant", lambda *a, **k: (object(), server.QDRANT_COLLECTION_PREFIX))
    monkeypatch.setattr(server, "_qdrant_search_collection", lambda *a, **k: [dict(h) for h in hits])


def _hit(inv: str, fid: str, text: str) -> dict:
    return {"id": fid, "investigation_id": inv, "text": text, "score": 0.9,
            "origin": server.QDRANT_COLLECTION_PREFIX, "source": "unit-test", "record_type": "observed"}


# --- lock deadlocks -------------------------------------------------------

def test_restore_after_retract_is_not_busy(store):
    """restore-self-deadlock: restore used to wait on its own retractions.jsonl flock."""
    inv = _start("rstb")
    fid = _store(inv, "The admin endpoint /api/v9/debug exists on host alpha.example.com")
    _retract(inv, fid)
    out = _j(server.memory_restore(inv, finding_id=fid))
    assert out.get("error") != "busy", out
    assert out.get("restored") is True, out
    assert fid not in inv_store._load_retracted_ids(inv)


def test_restore_after_retract_succeeds_and_is_exact_inverse(store):
    """restore-self-deadlock, restore-not-inverse-valid_until, retract-rewrites-appendonly-log."""
    inv = _start("rst")
    fid = _store(inv, "The admin endpoint /api/v9/debug exists on host alpha.example.com")
    findings_path = server._inv_dir(inv) / "findings.jsonl"
    before = findings_path.read_bytes()

    _retract(inv, fid)
    assert findings_path.read_bytes() == before, "retract must not rewrite findings.jsonl"
    later = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    assert _j(investigation_tools.investigation_as_of(inv, later))["count"] == 0

    out = _j(server.memory_restore(inv, finding_id=fid, reason="was real"))
    assert out.get("restored") is True, out
    assert fid not in inv_store._load_retracted_ids(inv)
    assert findings_path.read_bytes() == before
    assert _j(server.investigation_load(inv))["total_findings"] == 1
    assert _j(investigation_tools.investigation_as_of(inv, later))["count"] == 1


def test_as_of_still_hides_finding_during_its_retraction_interval(store):
    inv = _start("interval")
    fid = _store(inv, "Host delta.example.com runs OpenSSH 7.2")
    p = server._inv_dir(inv) / "retractions.jsonl"
    inv_store._append_jsonl(p, {"finding_id": fid, "active": True, "ts": "2099-01-01T00:00:00+00:00"})
    inv_store._append_jsonl(p, {"finding_id": fid, "active": False, "ts": "2099-02-01T00:00:00+00:00"})
    count = lambda ts: _j(investigation_tools.investigation_as_of(inv, ts))["count"]  # noqa: E731
    assert count("2098-12-01T00:00:00+00:00") == 1
    assert count("2099-01-15T00:00:00+00:00") == 0
    assert count("2099-03-01T00:00:00+00:00") == 1


def test_as_of_legacy_valid_until_stamp_defers_to_restore(store):
    """Older retracts stamped valid_until=<retraction ts>; a restore must still bring the finding back."""
    inv = _start("legacy")
    fid = _store(inv, "Legacy stamped finding")
    now = datetime.now(timezone.utc)
    ts = (now + timedelta(days=1)).isoformat()
    fp = server._inv_dir(inv) / "findings.jsonl"
    rows = server._read_jsonl(fp)
    rows[0]["valid_until"] = ts
    fp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    p = server._inv_dir(inv) / "retractions.jsonl"
    inv_store._append_jsonl(p, {"finding_id": fid, "active": True, "ts": ts})
    inv_store._append_jsonl(p, {"finding_id": fid, "active": False, "ts": (now + timedelta(days=2)).isoformat()})
    count = lambda d: _j(investigation_tools.investigation_as_of(inv, (now + timedelta(days=d)).isoformat()))["count"]  # noqa: E731
    assert count(1.5) == 0
    assert count(3) == 1


def test_wiring_obligation_resolve_succeeds(store):
    """wiring-resolve-self-deadlock."""
    inv = _start("wire")
    w = _j(server.investigation_store(inv, "gap", "Obligation: foo() must call bar()", source="t",
                                      tags="wiring_obligation"))["finding_id"]
    out = _j(server.wiring_obligation_resolve(inv, w, "bar() called at x.py:10"))
    assert out.get("resolved") is True, out


# --- malformed dirs -------------------------------------------------------

def test_global_search_filters_retracted_despite_malformed_dir(store, monkeypatch):
    """global-search-retraction-filter-disabled: an 'undefined' dir must not disable filtering."""
    inv = _start("good")
    bad = _store(inv, "worker.py POSTs telemetry to telemetry.deepthink.internal")
    live = _store(inv, "worker.py writes logs to stdout")
    _retract(inv, bad)
    _malformed_undefined_dir(store)

    rids, _texts = server._search_retraction_scope(None)
    assert bad in rids.get(inv, set())
    assert "undef-f1" in rids.get("undefined", set())

    _fake_mnemo(monkeypatch, [
        _mnemo_item(inv, bad, "worker.py POSTs telemetry to telemetry.deepthink.internal"),
        _mnemo_item(inv, live, "worker.py writes logs to stdout"),
        _mnemo_item("undefined", "undef-f1", "undefined-dir hallucination"),
    ])
    r = _j(server.investigation_search("worker telemetry", limit=2))
    texts = " ".join(x["text"] for x in r["results"])
    assert "deepthink" not in texts and "undefined-dir hallucination" not in texts
    assert r["excluded_retracted"] == 2
    assert r["retraction_filter"]["malformed_investigations"][0]["investigation_id"] == "undefined"
    assert (store / "undefined").is_dir(), "the malformed dir must be left in place"


def test_memory_self_check_global_tolerates_malformed_dir(store):
    """memory-self-check-global-crash."""
    inv = _start("sc")
    _store(inv, "some observed fact")
    _malformed_undefined_dir(store)
    out = _j(server.memory_self_check(checks="provenance", record=False))
    assert "error" not in out, out
    assert out["investigation_ids"] == [inv]
    assert [s["investigation_id"] for s in out["skipped_investigations"]] == ["undefined"]


def test_retraction_integrity_counts_and_reports_malformed_dir(store):
    """retraction-integrity-skips-but-counts."""
    inv = _start("hi")
    fid = _store(inv, "retract me")
    _retract(inv, fid)
    _malformed_undefined_dir(store)
    targets = sorted(p.name for p in store.iterdir() if p.is_dir())
    status, detail, remediation = server._health_probe_retraction_integrity(targets, None)
    assert detail["active_retractions"] == 2
    assert detail["investigations_scanned"] == 2
    assert detail["malformed_investigations"][0]["investigation_id"] == "undefined"
    assert status == "warn" and remediation


def test_consolidate_input_and_causal_edges_exclude_retracted(store):
    """causal-paths-ignore-retraction (also tolerates a malformed dir with a manifest)."""
    _malformed_undefined_dir(store)
    (store / "undefined" / "manifest.json").write_text(json.dumps({"id": "undefined", "updated_at": "9999"}))
    inv = _start("zz")
    p = _store(inv, "Gateway zeta.example.com logs show 500 errors")
    c = _store(inv, "Gateway zeta.example.com is compromised", derived_from=[p])
    _store(inv, "unrelated filler note about widgets")
    _retract(inv, c)
    server._rag_record_access([{"origin": server.QDRANT_COLLECTION_PREFIX, "id": p, "investigation_id": inv}], "q")

    got_inv, fs = server._find_most_recent_investigation()
    assert got_inv == inv
    ids = [f.get("id") for f in fs]
    assert c not in ids
    assert all(f.get("record_type") != "access" for f in fs)

    inv_store._append_jsonl(server._inv_dir(inv) / "causal_edges.jsonl", {
        "id": "e1", "source_id": p, "target_id": c, "edge_type": "caused_by", "confidence": 0.5})
    el = _j(server.causal_edges_list(inv))
    assert el["count"] == 0 and el["excluded_retracted_edges"] == 1


# --- recall paths ---------------------------------------------------------

def test_rag_context_search_drops_retracted_hits(store, monkeypatch):
    """retracted-findings-served-by-rag-and-ground / rag-surface-no-retraction-filter."""
    inv = _start("rg")
    bad = _store(inv, "Endpoint /api/v9/debug exists on alpha host")
    good = _store(inv, "Endpoint /api/v1/health exists on alpha host")
    _retract(inv, bad)
    _qdrant_hits(monkeypatch, [_hit(inv, bad, "Endpoint /api/v9/debug exists on alpha host"),
                               _hit(inv, good, "Endpoint /api/v1/health exists on alpha host")])
    r = _j(server.rag_context_search("debug endpoint", expand_query=False))
    assert "/api/v9/debug" not in r["context"]
    assert "/api/v1/health" in r["context"]
    assert r["result_count"] == 1 and r["excluded_retracted"] == 1
    # Access markers live in their own log (inv_store.ACCESS_LOG_NAME), never in findings.jsonl.
    access_ids = [x.get("id") for x in server._read_jsonl(server._inv_dir(inv) / inv_store.ACCESS_LOG_NAME)
                  if x.get("record_type") == "access"]
    assert bad not in access_ids and good in access_ids


def test_ground_rag_lane_does_not_inject_retracted_finding(store, monkeypatch):
    inv = _start("gr")
    bad = _store(inv, "Endpoint telemetry.deepthink.internal exists")
    _retract(inv, bad)
    _qdrant_hits(monkeypatch, [_hit(inv, bad, "Endpoint telemetry.deepthink.internal exists")])
    g = grounding.ground({"title": "deepthink telemetry endpoint"}, {"memoryDir": ""})
    assert "telemetry.deepthink.internal" not in g["block"]


def test_torn_retractions_line_does_not_void_the_tombstones(store, monkeypatch):
    """Review follow-up (probe P3b): one non-UTF-8 byte in retractions.jsonl made
    _read_jsonl raise, every tombstone in the investigation was dropped, and ground()
    served the retracted text with degraded=False."""
    inv = _start("torn")
    bad = _store(inv, "Host 10.4.4.44 runs the backdoor 4444")
    _retract(inv, bad)
    with open(server._inv_dir(inv) / "retractions.jsonl", "ab") as fh:
        fh.write(b"\xff\xfe torn\n")
    _qdrant_hits(monkeypatch, [_hit(inv, bad, "Host 10.4.4.44 runs the backdoor 4444")])
    r = _j(server.rag_context_search("backdoor 4444", expand_query=False))
    assert r["excluded_retracted"] == 1 and r["retraction_filter"]["status"] == "ok", r
    g = grounding.ground({"title": "backdoor 4444"}, {"memoryDir": ""})
    assert "4444" not in g["block"], g


@pytest.mark.skipif(not hasattr(__import__("os"), "geteuid") or __import__("os").geteuid() == 0,
                    reason="root ignores file permissions")
def test_ground_is_degraded_when_retractions_are_unreadable(store, monkeypatch):
    """Review follow-up (probe P3): an unreadable retractions.jsonl leaves that
    investigation unfiltered; ground() must say so instead of degraded=False."""
    inv = _start("eacces")
    bad = _store(inv, "Host 10.5.5.55 runs the backdoor 5555")
    _retract(inv, bad)
    log = server._inv_dir(inv) / "retractions.jsonl"
    log.chmod(0)
    try:
        _qdrant_hits(monkeypatch, [_hit(inv, bad, "Host 10.5.5.55 runs the backdoor 5555")])
        g = grounding.ground({"title": "backdoor 5555"}, {"memoryDir": ""})
    finally:
        log.chmod(0o600)
    assert g["degraded"] is True, g
    assert any(d["lane"] == "rag" and "retraction filter degraded" in d["reason"]
               for d in g["degraded_lanes"]), g


def test_memory_surface_drops_retracted_hits(store, monkeypatch):
    inv = _start("sf")
    bad = _store(inv, "Endpoint /api/v9/debug exists on alpha host")
    _retract(inv, bad)
    _qdrant_hits(monkeypatch, [_hit(inv, bad, "Endpoint /api/v9/debug exists on alpha host")])
    s = _j(server.memory_surface("working on the debug endpoint", investigation_id=inv))
    assert s["count"] == 0, s
    assert s["excluded_retracted"] == 1


def test_rag_marks_superseded_hits(store, monkeypatch):
    inv = _start("rs")
    old = _store(inv, "Root cause is the cache TTL of 5 seconds")
    server.finding_resolve(inv, old, "superseded")
    _qdrant_hits(monkeypatch, [_hit(inv, old, "Root cause is the cache TTL of 5 seconds")])
    r = _j(server.rag_context_search("root cause", expand_query=False))
    assert "superseded" in r["context"]


def test_recall_filter_caches_per_query_parse_and_sees_new_state(store, monkeypatch):
    """Review follow-up: every rag/surface/ground call re-parsed findings.jsonl just to
    find superseded ids. The parse is cached until a log changes, and a change is seen."""
    import recall_filter
    inv = _start("rfc")
    a = _store(inv, "first claim about the cache TTL")
    b = _store(inv, "second claim about the session writer")
    _retract(inv, b)
    reads = []
    real = recall_filter._read_jsonl
    monkeypatch.setattr(recall_filter, "_read_jsonl", lambda p: reads.append(p.name) or real(p))
    rf1 = recall_filter.build_recall_filter(store, [inv], with_superseded=True)
    assert rf1.superseded == {} and rf1.retracted_texts[inv] == {"second claim about the session writer"}
    reads.clear()
    recall_filter.build_recall_filter(store, [inv], with_superseded=True)
    assert "findings.jsonl" not in reads, reads
    server.finding_resolve(inv, a, "superseded")
    rf3 = recall_filter.build_recall_filter(store, [inv], with_superseded=True)
    assert rf3.superseded == {inv: {a}}


def test_ground_case_lane_marks_superseded_finding(store):
    """ground-superseded-as-current."""
    inv = _start("gc")
    old = _store(inv, "Root cause is the cache TTL of 5 seconds")
    _store(inv, "Root cause is actually a race in the session writer")
    server.finding_resolve(inv, old, "superseded")
    g = grounding.ground({"title": "session bug", "caseIds": [inv]}, {"budgetChars": 4000, "memoryDir": ""})
    line = next(ln for ln in g["block"].splitlines() if "cache TTL of 5 seconds" in ln or "[superseded" in ln)
    assert "superseded" in line
    assert f"case:{inv}:finding:superseded" in g["sources"]


# --- investigation_search counts and empty responses -----------------------

def test_mnemo_rows_carry_finding_id_and_real_resolution(store, monkeypatch):
    """mnemo-rows-resolution-always-open."""
    inv = _start("srch")
    a = _store(inv, "Service gamma leaks tokens in debug logs")
    b = _store(inv, "Service gamma rotates tokens hourly")
    server.finding_resolve(inv, a, "superseded", note="replaced by b")
    _fake_mnemo(monkeypatch, [_mnemo_item(inv, a, "Service gamma leaks tokens in debug logs"),
                              _mnemo_item(inv, b, "Service gamma rotates tokens hourly")])
    rows = mnemo_ops._mnemo_recall("gamma tokens", investigation_id=inv)
    assert {r.get("finding_id") for r in rows} == {a, b}
    r = _j(server.investigation_search("gamma tokens", investigation_id=inv, limit=5, resolution="superseded"))
    assert [x.get("finding_id") for x in r["results"]] == [a]
    r = _j(server.investigation_search("gamma tokens", investigation_id=inv, limit=5, resolution="open"))
    assert [x.get("finding_id") for x in r["results"]] == [b]


def test_search_excluded_retracted_counts_findings_not_rows(store, monkeypatch):
    """search-excluded-retracted-count."""
    inv = _start("cnt")
    a = _store(inv, "Service gamma leaks tokens in debug logs")
    b = _store(inv, "Service gamma rotates tokens hourly")
    _retract(inv, a)
    _fake_mnemo(monkeypatch, [_mnemo_item(inv, a, "Service gamma leaks tokens in debug logs"),
                              _mnemo_item(inv, a, "Service gamma leaks tokens in debug logs"),
                              _mnemo_item(inv, b, "Service gamma rotates tokens hourly")])
    r = _j(server.investigation_search("gamma tokens", investigation_id=inv, limit=1))
    assert r["excluded_retracted"] == 1


def test_search_all_retracted_reports_exclusions(store, monkeypatch):
    """search-empty-response-hides-exclusions-and-outage (retracted half)."""
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(server, "_qdrant_similarity_search",
                        lambda *a, **k: {"ok": True, "reason": "hybrid+reranked", "results": []})
    inv = _start("allr")
    a = _store(inv, "Service gamma leaks tokens in debug logs")
    b = _store(inv, "Service gamma rotates tokens hourly")
    _retract(inv, a)
    _retract(inv, b)
    _fake_mnemo(monkeypatch, [_mnemo_item(inv, a, "Service gamma leaks tokens in debug logs"),
                              _mnemo_item(inv, a, "Service gamma leaks tokens in debug logs"),
                              _mnemo_item(inv, b, "Service gamma rotates tokens hourly")])
    r = _j(server.investigation_search("gamma tokens", investigation_id=inv))
    assert r["results"] == []
    assert r["mode"] == "all_retracted"
    assert r["excluded_retracted"] == 2
    assert r["include_retracted"] is False


def test_search_empty_on_embedding_outage_is_not_no_matches(store, monkeypatch):
    """search-empty-response-hides-exclusions-and-outage (outage half)."""
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(server, "_qdrant_similarity_search",
                        lambda *a, **k: {"ok": False, "reason": "embedding_unavailable", "results": []})
    _start("outage")
    r = _j(server.investigation_search("anything", investigation_id="outage"))
    assert r["mode"] != "no_matches"
    assert r["mode"] == "rag_degraded" and r["reason"] == "embedding_unavailable"
    assert r["excluded_retracted"] == 0

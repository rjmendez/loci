import json

import pytest
import server
import mnemo_ops


def _json(raw: str) -> dict:
    return json.loads(raw)


@pytest.fixture
def isolated_memory(tmp_path, monkeypatch):
    original = server.MEMORY_DIR
    server.MEMORY_DIR = tmp_path
    server._session_hints.clear()
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path))
    yield tmp_path
    server.MEMORY_DIR = original
    server._session_hints.clear()


def _flybrain_claim_scope() -> dict:
    return {
        "dataset": "FlyWire",
        "dataset_version": "flywire783",
        "sex": "unspecified",
        "life_stage": "adult",
        "annotation_completeness": 0.95,
        "circuit_class": "Kenyon cell",
        "experience_window": "naive",
    }


def _flybrain_metadata() -> dict:
    return {
        "flybrain_provenance": {
            "tool_name": "query_connectivity",
            "tool_variant": "virtual-fly-brain-query_connectivity",
            "request": {
                "upstream_type_input": "FBbt_00003686",
                "weight": 20,
                "group_by_class": True,
                "exclude_dbs": ["hb", "fafb"],
            },
            "dataset_scope": {
                "included_symbols": ["fw", "mc"],
                "excluded_symbols": ["hb", "fafb"],
                "version_ids_seen": ["flywire783", "male_cns_v1_0"],
            },
            "result_contract": {
                "count": 1858,
                "count_status": "exact",
                "returned_rows": 5,
                "warnings": [],
            },
            "scope": {
                "stage": "adult-biased",
                "sex": "unspecified",
            },
            "claim_scope": _flybrain_claim_scope(),
        }
    }


def test_flybrain_envelope_persists_on_store_and_load(isolated_memory):
    inv_id = "flybrain-env-load"
    server.investigation_start(investigation_id=inv_id, title="FlyBrain envelope load test")
    stored = _json(server.investigation_store(
        investigation_id=inv_id,
        finding_type="observed",
        text="Connectivity rollup shows Kenyon cells project to targets in this dataset scope.",
        source="virtual-fly-brain-query_connectivity",
        confidence="high",
        metadata=_flybrain_metadata(),
    ))
    assert stored.get("stored"), stored

    loaded = _json(server.investigation_load(investigation_id=inv_id))
    finding = loaded["recent_findings"][0]
    env = finding["metadata"]["flybrain_provenance"]

    assert env["tool_name"] == "query_connectivity"
    assert env["tool_variant"] == "virtual-fly-brain-query_connectivity"
    assert env["dataset_scope"]["excluded_symbols"] == ["fafb", "hb"]
    assert env["result_contract"]["count_status"] == "exact"
    assert env["evidence_family"] == "connectivity"
    assert isinstance(env.get("replay_fingerprint"), str) and len(env["replay_fingerprint"]) == 64
    assert finding.get("flybrain_replay_fingerprint") == env["replay_fingerprint"]


def test_flybrain_envelope_is_carried_into_mnemo_and_recall(isolated_memory, monkeypatch):
    inv_id = "flybrain-env-mnemo"
    server.investigation_start(investigation_id=inv_id, title="FlyBrain envelope mnemo test")
    captured = {}

    def _fake_remember(content, *, importance=0.6, metadata=None):
        captured["metadata"] = metadata or {}
        return True

    monkeypatch.setattr(server, "_mnemo_remember", _fake_remember)
    stored = _json(server.investigation_store(
        investigation_id=inv_id,
        finding_type="observed",
        text="Predicted neurotransmitter profile in this dataset scope favors cholinergic signaling.",
        source="virtual-fly-brain-get_predicted_neurotransmitters",
        confidence="high",
        metadata=_flybrain_metadata(),
    ))
    assert stored.get("stored"), stored
    env = captured["metadata"]["flybrain_provenance"]
    assert env["tool_variant"] == "virtual-fly-brain-query_connectivity"
    assert isinstance(env.get("replay_fingerprint"), str)

    raw = [{
        "content": "memory row",
        "metadata": captured["metadata"],
        "score": 0.91,
    }]
    monkeypatch.setattr(mnemo_ops, "_get_mnemo_funcs", lambda: (None, lambda **kwargs: raw))
    rows = mnemo_ops._mnemo_recall("flybrain query", investigation_id=inv_id)
    assert rows and rows[0]["flybrain_provenance"]["replay_fingerprint"] == env["replay_fingerprint"]


def test_flybrain_audit_log_emits_reproducibility_envelope(isolated_memory):
    inv_id = "flybrain-env-audit"
    server.investigation_start(investigation_id=inv_id, title="FlyBrain envelope audit test")
    result = _json(server.audit_log(
        tool_name="virtual-fly-brain-query_connectivity",
        inputs_json=json.dumps({
            "upstream_type": "FBbt_00003686",
            "exclude_dbs": ["hb", "fafb"],
            "group_by_class": True,
            "limit": 5,
        }),
        output=json.dumps({
            "count": 14,
            "count_status": "exact",
            "rows": [{"db": "fw", "short_form": "flywire783"}],
            "warnings": [],
        }),
        investigation_id=inv_id,
    ))
    assert result.get("logged"), result

    entries = server._read_jsonl(server.MEMORY_DIR / inv_id / "audit.jsonl")
    assert entries, "expected scoped audit entries"
    env = entries[-1]["flybrain_provenance"]
    assert env["tool_name"] == "query_connectivity"
    assert env["dataset_scope"]["version_ids_seen"] == ["flywire783"]
    assert env["result_contract"]["count_status"] == "exact"
    assert isinstance(env.get("replay_fingerprint"), str) and len(env["replay_fingerprint"]) == 64

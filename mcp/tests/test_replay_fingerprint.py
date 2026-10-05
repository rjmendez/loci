import json
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

from replay_fingerprint import (  # noqa: E402
    apply_finding_fingerprints,
    flybrain_audit_fingerprint,
    flybrain_replay_fingerprint,
)


@pytest.mark.parametrize(
    ("request_a", "request_b", "scope_a", "scope_b"),
    [
        (
            {
                "upstream_type_input": "FBbt_00003686",
                "exclude_dbs": ["hb", "fafb"],
                "group_by_class": True,
            },
            {
                "group_by_class": True,
                "exclude_dbs": ["fafb", "hb"],
                "upstream_type_input": "FBbt_00003686",
            },
            {
                "included_symbols": ["fw", "mc"],
                "excluded_symbols": ["hb", "fafb"],
                "version_ids_seen": ["male_cns_v1_0", "flywire783"],
            },
            {
                "version_ids_seen": ["flywire783", "male_cns_v1_0"],
                "excluded_symbols": ["fafb", "hb"],
                "included_symbols": ["mc", "fw"],
            },
        ),
        (
            {
                "query": "cafe\u0301",
                "upstream_type_input": "FBbt_00003686",
                "group_by_class": True,
                "exclude_dbs": ["hb", "fafb"],
                "filters": {"dataset": "FlyWire", "label": "cafe\u0301"},
            },
            {
                "query": "café",
                "upstream_type_input": "FBbt_00003686",
                "group_by_class": True,
                "exclude_dbs": {"fafb", "hb"},
                "filters": {"label": "cafe\u0301", "dataset": "FlyWire"},
            },
            {
                "included_symbols": ["fw", "mc"],
                "excluded_symbols": ["hb", "fafb"],
                "version_ids_seen": ["male_cns_v1_0", "flywire783"],
            },
            {
                "included_symbols": ["mc", "fw"],
                "excluded_symbols": ["fafb", "hb"],
                "version_ids_seen": ["flywire783", "male_cns_v1_0"],
            },
        ),
    ],
)
def test_flybrain_replay_fingerprint_is_stable_for_semantically_identical_payloads(
    request_a,
    request_b,
    scope_a,
    scope_b,
):
    fp_a = flybrain_replay_fingerprint("virtual-fly-brain-query_connectivity", request_a, scope_a)
    fp_b = flybrain_replay_fingerprint("virtual-fly-brain-query_connectivity", request_b, scope_b)
    assert fp_a == fp_b


def test_apply_finding_fingerprints_adds_flybrain_and_provenance_fingerprints():
    finding = {
        "source": "virtual-fly-brain-query_connectivity",
        "metadata": {
            "evidence_provenance_tier": "tool_verified",
            "flybrain_provenance": {
                "tool_variant": "virtual-fly-brain-query_connectivity",
                "request": {"upstream_type_input": "FBbt_00003686"},
                "dataset_scope": {"included_symbols": ["fw"], "version_ids_seen": ["flywire783"]},
            },
        },
    }
    out = apply_finding_fingerprints(finding)
    fb = out["metadata"]["flybrain_provenance"]
    # the fingerprint of exactly this tool + request + (canonicalised) dataset scope
    assert fb["replay_fingerprint"] == flybrain_replay_fingerprint(
        "virtual-fly-brain-query_connectivity",
        {"upstream_type_input": "FBbt_00003686"},
        {"excluded_symbols": [], "included_symbols": ["fw"], "version_ids_seen": ["flywire783"]},
    )
    assert out["flybrain_replay_fingerprint"] == fb["replay_fingerprint"]
    assert fb["replay_fingerprint_version"] == "v1"
    access = out["metadata"]["provenance_access_path_fingerprint"]
    assert len(access) == 64 and int(access, 16) >= 0
    assert out["metadata"]["provenance_access_path_fingerprint_version"] == "v1"
    # the access-path fingerprint depends on the tier it certifies
    other = json.loads(json.dumps(finding))
    other["metadata"]["evidence_provenance_tier"] = "model_asserted"
    assert apply_finding_fingerprints(other)["metadata"]["provenance_access_path_fingerprint"] != access


def test_flybrain_replay_fingerprint_golden_value():
    """Pinned against a hand-written canonical form: sorted keys, set-like lists
    sorted, tool name lower-cased, compact separators, sha256."""
    import hashlib
    canonical = ('{"dataset_scope":{"included_symbols":["fw","mc"]},'
                 '"request":{"a":"x","b":1},'
                 '"tool_name":"virtual-fly-brain-query_connectivity","version":"v1"}')
    assert flybrain_replay_fingerprint(
        " Virtual-Fly-Brain-Query_Connectivity ", {"b": 1, "a": "x"}, {"included_symbols": ["mc", "fw"]},
    ) == hashlib.sha256(canonical.encode()).hexdigest()


def test_flybrain_replay_fingerprint_changes_with_each_input():
    base = ("virtual-fly-brain-query_connectivity", {"upstream_type_input": "FBbt_00003686"},
            {"included_symbols": ["fw"]})
    fp = flybrain_replay_fingerprint(*base)
    assert flybrain_replay_fingerprint("virtual-fly-brain-run_query", *base[1:]) != fp
    assert flybrain_replay_fingerprint(base[0], {"upstream_type_input": "FBbt_00000001"}, base[2]) != fp
    assert flybrain_replay_fingerprint(base[0], base[1], {"included_symbols": ["mc"]}) != fp
    # a list that is NOT set-like keeps its order
    assert flybrain_replay_fingerprint(base[0], {"path": ["a", "b"]}, base[2]) != \
        flybrain_replay_fingerprint(base[0], {"path": ["b", "a"]}, base[2])


def test_flybrain_audit_fingerprint_extracts_dataset_scope():
    payload = flybrain_audit_fingerprint(
        "virtual-fly-brain-query_connectivity",
        json.dumps(
            {
                "upstream_type": "FBbt_00003686",
                "exclude_dbs": ["hb", "fafb"],
                "group_by_class": True,
                "limit": 5,
            }
        ),
        json.dumps(
            {
                "count": 14,
                "count_status": "exact",
                "rows": [{"db": "fw", "short_form": "flywire783"}],
                "warnings": [],
            }
        ),
    )
    assert payload is not None
    assert payload["replay_fingerprint_version"] == "v1"
    assert payload["flybrain_provenance"]["dataset_scope"]["version_ids_seen"] == ["flywire783"]
    assert payload["flybrain_provenance"]["result_contract"]["count_status"] == "exact"
    assert payload["replay_fingerprint"] == payload["flybrain_provenance"]["replay_fingerprint"]


def test_replay_acceptance_golden_cases_for_claim_scope_and_contracts():
    claim_scope = {
        "dataset": "fw",
        "dataset_version": "flywire783",
        "sex": "female",
        "life_stage": "adult",
        "annotation_completeness": 0.95,
        "circuit_class": "Kenyon cell",
        "experience_window": "naive",
    }
    request_a = {
        "upstream_type_input": "FBbt_00003686",
        "exclude_dbs": ["hb", "fafb"],
        "group_by_class": True,
    }
    request_b = {
        "group_by_class": True,
        "exclude_dbs": ["fafb", "hb"],
        "upstream_type_input": "FBbt_00003686",
    }
    scope_a = {
        "included_symbols": ["fw", "mc"],
        "excluded_symbols": ["hb", "fafb"],
        "version_ids_seen": ["male_cns_v1_0", "flywire783"],
    }
    scope_b = {
        "version_ids_seen": ["flywire783", "male_cns_v1_0"],
        "excluded_symbols": ["fafb", "hb"],
        "included_symbols": ["mc", "fw"],
    }

    fp_a = flybrain_replay_fingerprint("virtual-fly-brain-query_connectivity", request_a, scope_a)
    fp_b = flybrain_replay_fingerprint("virtual-fly-brain-query_connectivity", request_b, scope_b)
    assert fp_a == fp_b

    drifted_scope = {**scope_b, "version_ids_seen": ["male_cns_v1_0", "flywire783", "new_release"]}
    assert flybrain_replay_fingerprint("virtual-fly-brain-query_connectivity", request_b, drifted_scope) != fp_a

    payload = flybrain_audit_fingerprint(
        "virtual-fly-brain-query_connectivity",
        json.dumps({"upstream_type": "FBbt_00003686", "exclude_dbs": ["hb", "fafb"], "group_by_class": True, "limit": 5}),
        json.dumps({
            "count": 14,
            "count_status": "exact",
            "rows": [{"db": "fw", "short_form": "flywire783"}],
            "warnings": ["dataset skewed toward fw"],
        }),
    )
    assert payload is not None
    assert payload["flybrain_provenance"]["result_contract"]["count"] == 14
    assert payload["flybrain_provenance"]["result_contract"]["count_status"] == "exact"
    assert payload["flybrain_provenance"]["result_contract"]["warnings"] == ["dataset skewed toward fw"]

    finding = {
        "source": "virtual-fly-brain-query_connectivity",
        "metadata": {
            "evidence_provenance_tier": "tool_verified",
            "claim_scope": claim_scope,
            "flybrain_provenance": {
                "tool_variant": "virtual-fly-brain-query_connectivity",
                "request": request_a,
                "dataset_scope": scope_a,
                "result_contract": {
                    "count": 14,
                    "count_status": "exact",
                    "warnings": ["dataset skewed toward fw"],
                },
            },
        },
    }
    out = apply_finding_fingerprints(finding)
    assert out["metadata"]["claim_scope"] == claim_scope
    assert out["metadata"]["flybrain_provenance"]["replay_fingerprint"]
    assert out["flybrain_replay_fingerprint"] == out["metadata"]["flybrain_provenance"]["replay_fingerprint"]

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
    assert fb["replay_fingerprint"]
    assert fb["replay_fingerprint_version"] == "v1"
    assert out["metadata"]["provenance_access_path_fingerprint"]
    assert out["metadata"]["provenance_access_path_fingerprint_version"] == "v1"


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

import sys
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

from replay_fingerprint import (  # noqa: E402
    apply_finding_fingerprints,
    flybrain_audit_fingerprint,
    flybrain_replay_fingerprint,
)


def test_flybrain_replay_fingerprint_is_deterministic_across_key_and_list_order():
    request_a = {
        'upstream_type_input': 'FBbt_00003686',
        'exclude_dbs': ['hb', 'fafb'],
        'group_by_class': True,
    }
    request_b = {
        'group_by_class': True,
        'exclude_dbs': ['fafb', 'hb'],
        'upstream_type_input': 'FBbt_00003686',
    }
    scope_a = {
        'included_symbols': ['fw', 'mc'],
        'excluded_symbols': ['hb', 'fafb'],
        'version_ids_seen': ['male_cns_v1_0', 'flywire783'],
    }
    scope_b = {
        'version_ids_seen': ['flywire783', 'male_cns_v1_0'],
        'excluded_symbols': ['fafb', 'hb'],
        'included_symbols': ['mc', 'fw'],
    }
    fp1 = flybrain_replay_fingerprint('virtual-fly-brain-query_connectivity', request_a, scope_a)
    fp2 = flybrain_replay_fingerprint('virtual-fly-brain-query_connectivity', request_b, scope_b)
    assert fp1 == fp2


def test_flybrain_replay_fingerprint_normalizes_unicode_and_set_variance():
    request_a = {
        'query': 'cafe\u0301',
        'upstream_type_input': 'FBbt_00003686',
        'group_by_class': True,
        'exclude_dbs': ['hb', 'fafb'],
        'filters': {'dataset': 'FlyWire', 'label': 'cafe\u0301'},
    }
    request_b = {
        'query': 'café',
        'upstream_type_input': 'FBbt_00003686',
        'group_by_class': True,
        'exclude_dbs': {'fafb', 'hb'},
        'filters': {'label': 'cafe\u0301', 'dataset': 'FlyWire'},
    }
    scope = {
        'included_symbols': ['fw', 'mc'],
        'excluded_symbols': ['hb', 'fafb'],
        'version_ids_seen': ['male_cns_v1_0', 'flywire783'],
    }
    assert flybrain_replay_fingerprint('virtual-fly-brain-query_connectivity', request_a, scope) == flybrain_replay_fingerprint(
        'virtual-fly-brain-query_connectivity', request_b, scope
    )


def test_apply_finding_fingerprints_adds_flybrain_and_provenance_fingerprints():
    finding = {
        'source': 'virtual-fly-brain-query_connectivity',
        'metadata': {
            'evidence_provenance_tier': 'tool_verified',
            'flybrain_provenance': {
                'tool_variant': 'virtual-fly-brain-query_connectivity',
                'request': {'upstream_type_input': 'FBbt_00003686'},
                'dataset_scope': {'included_symbols': ['fw'], 'version_ids_seen': ['flywire783']},
            },
        },
    }
    out = apply_finding_fingerprints(finding)
    fb = out['metadata']['flybrain_provenance']
    assert fb['replay_fingerprint']
    assert fb['replay_fingerprint_version'] == 'v1'
    assert out['metadata']['provenance_access_path_fingerprint']
    assert out['metadata']['provenance_access_path_fingerprint_version'] == 'v1'


def test_flybrain_audit_fingerprint_extracts_dataset_scope():
    payload = flybrain_audit_fingerprint(
        'virtual-fly-brain-query_connectivity',
        '{\

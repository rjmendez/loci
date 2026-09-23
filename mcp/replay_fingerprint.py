'''Deterministic replay/access-path fingerprint helpers.

Scope is intentionally narrow:
- FlyBrain-derived provenance blocks get a replay fingerprint based on
  (tool_name, request, dataset_scope).
- Findings with explicit provenance metadata can get an access-path
  fingerprint so the evidence path is traceable across exports/imports.
'''
from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from typing import Any

_SET_LIKE_KEYS = frozenset(
    {
        'exclude_dbs',
        'include_dbs',
        'excluded_symbols',
        'included_symbols',
        'version_ids_seen',
        'datasets',
        'dataset_symbols',
    }
)
_FLYBRAIN_TOOL_PREFIXES = ('virtual-fly-brain-', 'virtual_fly_brain_')
_FINGERPRINT_VERSION = 'v1'


def is_flybrain_tool_name(tool_name: str | None) -> bool:
    text = str(tool_name or '').strip().lower()
    return any(text.startswith(prefix) for prefix in _FLYBRAIN_TOOL_PREFIXES)


def _normalize_text(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode('utf-8')
        except UnicodeDecodeError:
            value = value.decode('utf-8', errors='surrogateescape')
    return unicodedata.normalize('NFC', str(value)).strip()


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonicalize(value: Any, *, key_name: str = '') -> Any:
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for raw_key, raw_value in value.items():
            key = _normalize_text(raw_key)
            items.append((key, _canonicalize(raw_value, key_name=key.lower())))
        items.sort(key=lambda item: item[0].casefold())
        return {key: val for key, val in items}
    if isinstance(value, (list, tuple)):
        canonical_items = [_canonicalize(v, key_name=key_name) for v in value]
        if key_name in _SET_LIKE_KEYS:
            unique: dict[str, Any] = {}
            for item in canonical_items:
                marker = _stable_json(item)
                unique.setdefault(marker, item)
            return [unique[key] for key in sorted(unique)]
        return canonical_items
    if isinstance(value, set):
        canonical_items = [_canonicalize(v, key_name=key_name) for v in value]
        unique: dict[str, Any] = {}
        for item in canonical_items:
            marker = _stable_json(item)
            unique.setdefault(marker, item)
        return [unique[key] for key in sorted(unique)]
    if isinstance(value, str):
        return _normalize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _normalize_text(value)


def _hash_payload(payload: dict) -> str:
    canonical = _canonicalize(payload)
    encoded = _stable_json(canonical)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _parse_json(raw: str) -> Any | None:
    text = str(raw or '').strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_dataset_scope(output_obj: Any) -> dict:
    symbols: set[str] = set()
    versions: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_l = str(key).lower()
                if key_l in {'db', 'symbol', 'dataset_symbol'} and isinstance(value, str):
                    v = _normalize_text(value)
                    if v:
                        symbols.add(v)
                elif key_l in {'short_form', 'dataset', 'dataset_id', 'version_id'} and isinstance(value, str):
                    v = _normalize_text(value)
                    if v:
                        versions.add(v)
                walk(value)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                walk(item)

    walk(output_obj)
    return {
        'included_symbols': sorted(symbols, key=str.casefold),
        'version_ids_seen': sorted(versions, key=str.casefold),
    }


def _extract_result_contract(output_obj: Any) -> dict:
    if not isinstance(output_obj, dict):
        return {}
    rows = output_obj.get('rows')
    warnings = output_obj.get('warnings')
    return {
        'count': output_obj.get('count'),
        'count_status': output_obj.get('count_status'),
        'returned_rows': len(rows) if isinstance(rows, list) else output_obj.get('returned_rows'),
        'warnings': _canonicalize(warnings if isinstance(warnings, list) else []),
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_name_from_variant(tool_variant: str) -> str:
    variant = str(tool_variant or '').strip()
    low = variant.lower()
    if low.startswith('virtual-fly-brain-'):
        return variant[len('virtual-fly-brain-'):]
    if low.startswith('virtual_fly_brain_'):
        return variant[len('virtual_fly_brain_'):]
    if '__' in variant:
        return variant.split('__')[-1]
    return variant


def _evidence_family(tool_name: str) -> str:
    name = str(tool_name or '').strip().lower()
    if name in {'query_connectivity', 'run_query'}:
        return 'connectivity'
    if 'neurotransmitter' in name:
        return 'curated_neurotransmitter' if 'known' in name else 'predicted_neurotransmitter'
    if name in {'search_terms', 'get_term_info', 'get_hierarchy'}:
        return 'ontology_resolution'
    return 'flybrain_claim'


def normalize_flybrain_provenance(
    provenance: dict,
    *,
    source_tool: str | None = None,
    executed_at: str | None = None,
) -> dict:
    '''Return a canonical, replay-ready FlyBrain provenance envelope.'''
    prov = dict(provenance or {})
    tool_variant = str(prov.get('tool_variant') or source_tool or '').strip()
    tool_name = str(prov.get('tool_name') or _tool_name_from_variant(tool_variant)).strip()
    request = prov.get('request') if isinstance(prov.get('request'), dict) else {}
    dataset_scope = prov.get('dataset_scope') if isinstance(prov.get('dataset_scope'), dict) else {}
    result_contract = prov.get('result_contract') if isinstance(prov.get('result_contract'), dict) else {}
    scope = prov.get('scope') if isinstance(prov.get('scope'), dict) else None
    resolution = prov.get('resolution') if isinstance(prov.get('resolution'), dict) else None

    normalized = {
        'tool_name': tool_name,
        'tool_variant': tool_variant,
        'executed_at': str(prov.get('executed_at') or executed_at or _now_iso()),
        'request': _canonicalize(request),
        'resolution': _canonicalize(resolution),
        'dataset_scope': _canonicalize({
            'included_symbols': dataset_scope.get('included_symbols', []),
            'excluded_symbols': dataset_scope.get('excluded_symbols', []),
            'version_ids_seen': dataset_scope.get('version_ids_seen', []),
        }),
        'result_contract': _canonicalize({
            **result_contract,
            'warnings': result_contract.get('warnings', []),
        }),
        'scope': _canonicalize(scope),
        'evidence_family': str(prov.get('evidence_family') or _evidence_family(tool_name)),
    }
    normalized['partial_provenance'] = not (
        bool(normalized['request'])
        and bool(normalized['dataset_scope'])
        and bool(normalized['result_contract'])
    )
    normalized['replay_fingerprint'] = flybrain_replay_fingerprint(
        tool_variant or tool_name,
        normalized['request'],
        normalized['dataset_scope'],
    )
    normalized['replay_fingerprint_version'] = _FINGERPRINT_VERSION
    return normalized


def flybrain_replay_fingerprint(tool_name: str | None, request: Any, dataset_scope: Any) -> str:
    payload = {
        'version': _FINGERPRINT_VERSION,
        'tool_name': str(tool_name or '').strip().lower(),
        'request': request if request is not None else {},
        'dataset_scope': dataset_scope if dataset_scope is not None else {},
    }
    return _hash_payload(payload)


def apply_finding_fingerprints(
    finding: dict,
    *,
    source: str | None = None,
    explicit_provenance_tier: str | None = None,
) -> dict:
    '''Return a copy of finding with deterministic replay/access fingerprints.'''
    if not isinstance(finding, dict):
        return finding
    out = dict(finding)
    metadata = out.get('metadata')
    if not isinstance(metadata, dict):
        return out
    metadata = dict(metadata)

    flybrain = metadata.get('flybrain_provenance')
    if isinstance(flybrain, dict):
        tool_name = (
            flybrain.get('tool_variant')
            or flybrain.get('tool_name')
            or source
            or out.get('source')
        )
        if is_flybrain_tool_name(str(tool_name or '')):
            metadata['flybrain_provenance'] = normalize_flybrain_provenance(
                flybrain,
                source_tool=str(tool_name or ''),
                executed_at=str(out.get('ts') or ''),
            )

    tier = (
        explicit_provenance_tier
        or metadata.get('evidence_provenance_tier')
        or metadata.get('provenance_tier')
        or metadata.get('evidence_kind')
    )
    if tier:
        access_payload = {
            'version': _FINGERPRINT_VERSION,
            'source': source or out.get('source') or '',
            'tier': str(tier).strip().lower(),
            'access_path': metadata.get('access_path'),
            'provenance': metadata.get('provenance'),
            'flybrain_provenance': metadata.get('flybrain_provenance'),
        }
        metadata['provenance_access_path_fingerprint'] = _hash_payload(access_payload)
        metadata['provenance_access_path_fingerprint_version'] = _FINGERPRINT_VERSION

    out['metadata'] = metadata
    if isinstance(metadata.get('flybrain_provenance'), dict):
        out['flybrain_replay_fingerprint'] = metadata['flybrain_provenance'].get('replay_fingerprint')
    return out


def flybrain_audit_fingerprint(tool_name: str, inputs_json: str, output: str) -> dict | None:
    '''Build a compact replay fingerprint block for FlyBrain audit entries.'''
    if not is_flybrain_tool_name(tool_name):
        return None
    request = _parse_json(inputs_json)
    if not isinstance(request, dict):
        request = {'raw_inputs': str(inputs_json or '').strip()}
    output_obj = _parse_json(output)
    dataset_scope = _extract_dataset_scope(output_obj)
    result_contract = _extract_result_contract(output_obj)
    provenance = normalize_flybrain_provenance(
        {
            'tool_name': _tool_name_from_variant(tool_name),
            'tool_variant': tool_name,
            'request': request,
            'dataset_scope': dataset_scope,
            'result_contract': result_contract,
        },
        source_tool=tool_name,
        executed_at=_now_iso(),
    )
    return {
        'flybrain_provenance': provenance,
        'replay_fingerprint': provenance.get('replay_fingerprint'),
        'replay_fingerprint_version': provenance.get('replay_fingerprint_version'),
        'replay_dataset_scope': provenance.get('dataset_scope'),
    }

"""Slow neuromodulation state layer for cross-session routing/confidence tuning.

The layer is deliberately low-frequency and fail-open:
- Missing/corrupt state falls back to neutral tones.
- Save failures never block caller behavior.
- Neutral state preserves existing thresholds and limits.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path


_STATE_DIR = "_slow_neuromodulation"
_STATE_FILE = "state.json"
_SCHEMA_VERSION = 1
_ALPHA = 0.08  # slow EMA update
_TONE_LIMIT = 0.4


def _finite_float(value: float, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _clamp(value: float, lo: float, hi: float, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    return max(lo, min(hi, numeric))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(memory_dir: Path) -> Path:
    return Path(memory_dir) / _STATE_DIR / _STATE_FILE


def neutral_state() -> dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "routing_tone": 0.0,
        "confidence_tone": 0.0,
        "consolidation_tone": 0.0,
        "events_seen": 0,
        "last_event": "",
        "updated_at": "",
    }


def load_state(memory_dir: Path) -> dict:
    """Load state. Never raises; returns neutral defaults when unavailable."""
    state = neutral_state()
    path = _state_path(memory_dir)
    if not path.exists():
        return state
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return state
        state["routing_tone"] = _clamp(raw.get("routing_tone", 0.0), -_TONE_LIMIT, _TONE_LIMIT, name="routing_tone")
        state["confidence_tone"] = _clamp(raw.get("confidence_tone", 0.0), -_TONE_LIMIT, _TONE_LIMIT, name="confidence_tone")
        state["consolidation_tone"] = _clamp(raw.get("consolidation_tone", 0.0), -_TONE_LIMIT, _TONE_LIMIT, name="consolidation_tone")
        state["events_seen"] = int(raw.get("events_seen") or 0)
        state["last_event"] = str(raw.get("last_event") or "")
        state["updated_at"] = str(raw.get("updated_at") or "")
        return state
    except Exception:
        return state


def save_state(memory_dir: Path, state: dict) -> bool:
    """Persist state fail-open. Returns False when persistence fails."""
    try:
        path = _state_path(memory_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def observe(
    memory_dir: Path,
    *,
    event: str,
    routing_signal: float = 0.0,
    confidence_signal: float = 0.0,
    consolidation_signal: float = 0.0,
) -> dict:
    """Fold one slow feedback event into state and persist fail-open."""
    state = load_state(memory_dir)
    alpha = _ALPHA
    state["routing_tone"] = _clamp(
        (1.0 - alpha) * float(state.get("routing_tone", 0.0)) + alpha * _clamp(routing_signal, -1.0, 1.0, name="routing_signal"),
        -_TONE_LIMIT,
        _TONE_LIMIT,
        name="routing_tone",
    )
    state["confidence_tone"] = _clamp(
        (1.0 - alpha) * float(state.get("confidence_tone", 0.0)) + alpha * _clamp(confidence_signal, -1.0, 1.0, name="confidence_signal"),
        -_TONE_LIMIT,
        _TONE_LIMIT,
        name="confidence_tone",
    )
    state["consolidation_tone"] = _clamp(
        (1.0 - alpha) * float(state.get("consolidation_tone", 0.0)) + alpha * _clamp(consolidation_signal, -1.0, 1.0, name="consolidation_signal"),
        -_TONE_LIMIT,
        _TONE_LIMIT,
        name="consolidation_tone",
    )
    state["events_seen"] = int(state.get("events_seen") or 0) + 1
    state["last_event"] = str(event or "")
    state["updated_at"] = _now()
    saved = save_state(memory_dir, state)
    return {"state": state, "degraded": not saved}


def routing_policy(state: dict, *, mnemo_top_k: int, qdrant_limit: int) -> dict:
    """Return biased but bounded retrieval limits from slow routing tone."""
    tone = _clamp(float((state or {}).get("routing_tone", 0.0)), -_TONE_LIMIT, _TONE_LIMIT, name="routing_tone")
    # Positive tone nudges semantic exploration (Qdrant), negative tone nudges memory recall.
    qdrant_factor = _clamp(1.0 + 0.5 * tone, 0.7, 1.4, name="qdrant_factor")
    mnemo_factor = _clamp(1.0 - 0.5 * tone, 0.7, 1.4, name="mnemo_factor")
    policy = {
        "routing_tone": tone,
        "mnemo_top_k": max(1, int(round(max(1, int(mnemo_top_k)) * mnemo_factor))),
        "qdrant_limit": max(1, int(round(max(1, int(qdrant_limit)) * qdrant_factor))),
    }
    assert_routing_policy_invariants(policy, minimum_top_k=1)
    return policy


def confidence_policy(state: dict, *, confidence: float) -> dict:
    """Apply a small confidence bias from the slow confidence tone."""
    tone = _clamp(float((state or {}).get("confidence_tone", 0.0)), -_TONE_LIMIT, _TONE_LIMIT, name="confidence_tone")
    delta = 0.12 * tone
    adjusted = _clamp(float(confidence) + delta, 0.0, 1.0, name="adjusted_confidence")
    policy = {
        "confidence_tone": tone,
        "delta": delta,
        "confidence": adjusted,
    }
    assert_confidence_policy_invariants(policy)
    return policy


def consolidation_policy(state: dict, *, default_min_findings: int = 3) -> dict:
    """Return a conservative threshold for consolidation-triggered inference."""
    tone = _clamp(float((state or {}).get("consolidation_tone", 0.0)), -_TONE_LIMIT, _TONE_LIMIT, name="consolidation_tone")
    base = max(1, int(default_min_findings))
    if tone >= 0.2:
        minimum = max(2, base - 1)
    elif tone <= -0.2:
        minimum = min(5, base + 1)
    else:
        minimum = base
    policy = {
        "consolidation_tone": tone,
        "min_findings_for_causal": minimum,
    }
    assert_consolidation_policy_invariants(policy)
    return policy


def assert_routing_policy_invariants(policy: dict, *, minimum_top_k: int) -> None:
    required = ("routing_tone", "mnemo_top_k", "qdrant_limit")
    for key in required:
        if key not in policy:
            raise ValueError(f"routing policy invariant failed: missing {key}")
    tone = _finite_float(policy["routing_tone"], name="routing_tone")
    if tone < -_TONE_LIMIT or tone > _TONE_LIMIT:
        raise ValueError("routing policy invariant failed: routing_tone out of bounds")
    mnemo_top_k = int(policy["mnemo_top_k"])
    qdrant_limit = int(policy["qdrant_limit"])
    min_top_k = max(1, int(minimum_top_k))
    if mnemo_top_k < min_top_k:
        raise ValueError("routing policy invariant failed: mnemo_top_k below minimum")
    if qdrant_limit < min_top_k:
        raise ValueError("routing policy invariant failed: qdrant_limit below minimum")


def assert_confidence_policy_invariants(policy: dict) -> None:
    required = ("confidence_tone", "delta", "confidence")
    for key in required:
        if key not in policy:
            raise ValueError(f"confidence policy invariant failed: missing {key}")
    tone = _finite_float(policy["confidence_tone"], name="confidence_tone")
    if tone < -_TONE_LIMIT or tone > _TONE_LIMIT:
        raise ValueError("confidence policy invariant failed: confidence_tone out of bounds")
    delta = _finite_float(policy["delta"], name="delta")
    if abs(delta) > 0.2:
        raise ValueError("confidence policy invariant failed: delta out of bounds")
    conf = _finite_float(policy["confidence"], name="confidence")
    if conf < 0.0 or conf > 1.0:
        raise ValueError("confidence policy invariant failed: confidence out of range")


def assert_consolidation_policy_invariants(policy: dict) -> None:
    required = ("consolidation_tone", "min_findings_for_causal")
    for key in required:
        if key not in policy:
            raise ValueError(f"consolidation policy invariant failed: missing {key}")
    tone = _finite_float(policy["consolidation_tone"], name="consolidation_tone")
    if tone < -_TONE_LIMIT or tone > _TONE_LIMIT:
        raise ValueError("consolidation policy invariant failed: consolidation_tone out of bounds")
    minimum = int(policy["min_findings_for_causal"])
    if minimum < 1 or minimum > 8:
        raise ValueError("consolidation policy invariant failed: min_findings_for_causal out of range")

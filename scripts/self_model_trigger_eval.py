#!/usr/bin/env python3
"""Proactive self-model and trigger evaluator for Loci.

The script builds a small durable self-model from local memory artifacts,
derives an introspection snapshot, and evaluates a bounded set of proactive
triggers. It is deliberately fail-open: unreadable inputs degrade the report
rather than crashing the cron lane.

Intended cron shape:
  - every 5m: refresh self-model + evaluate triggers
  - daily summary trigger: write a richer planning snapshot when due

The module keeps the core logic pure and testable:
  - build_self_model(...)
  - evaluate_proactive_trigger(...)
  - run_eval(...)
  - render_introspection_report(...)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MEMORY_DIR = Path(
    os.path.expanduser(os.environ.get("LOCI_MEMORY_DIR", "~/.hermes/memory-sessions"))
)
SELF_MODEL_DIR = DEFAULT_MEMORY_DIR / "_self-model"
STATE_FILE = SELF_MODEL_DIR / "state.json"
REFLECTION_STATE_FILE = DEFAULT_MEMORY_DIR / "_reflection-loop" / "state.json"
EVENT_LOG_FILE = Path(
    os.path.expanduser(os.environ.get("LOCI_EVENT_LOG", "~/.hermes/event_log.jsonl"))
)
STALE_HOURS = float(os.environ.get("LOCI_STALE_INVESTIGATION_HOURS", "72"))
QUEUE_FLOOD_THRESHOLD = int(os.environ.get("LOCI_REFLECTION_QUEUE_THRESHOLD", "200"))
BLOCKED_TODO_THRESHOLD = int(os.environ.get("LOCI_BLOCKED_TODO_THRESHOLD", "3"))
ERROR_STORM_THRESHOLD = int(os.environ.get("LOCI_ERROR_STORM_THRESHOLD", "10"))
DAILY_SUMMARY_START_HOUR = int(os.environ.get("LOCI_DAILY_SUMMARY_START_HOUR", "17"))
DAILY_SUMMARY_COOLDOWN_HOURS = float(os.environ.get("LOCI_DAILY_SUMMARY_COOLDOWN_HOURS", "18"))
T2_COOLDOWN_HOURS = float(os.environ.get("LOCI_PROACTIVE_T2_COOLDOWN_HOURS", "4"))
T1_COOLDOWN_MINUTES = int(os.environ.get("LOCI_PROACTIVE_T1_COOLDOWN_MINUTES", "15"))
T3_COOLDOWN_HOURS = float(os.environ.get("LOCI_PROACTIVE_T3_COOLDOWN_HOURS", "18"))


@dataclass
class TriggerConfig:
    queue_flood_threshold: int = QUEUE_FLOOD_THRESHOLD
    stale_hours: float = STALE_HOURS
    blocked_todo_threshold: int = BLOCKED_TODO_THRESHOLD
    error_storm_threshold: int = ERROR_STORM_THRESHOLD
    daily_summary_start_hour: int = DAILY_SUMMARY_START_HOUR
    daily_summary_cooldown_hours: float = DAILY_SUMMARY_COOLDOWN_HOURS
    t1_cooldown_minutes: int = T1_COOLDOWN_MINUTES
    t2_cooldown_hours: float = T2_COOLDOWN_HOURS
    t3_cooldown_hours: float = T3_COOLDOWN_HOURS


@dataclass
class TriggerEvaluation:
    trigger_type: str
    tier: str
    target_id: str
    condition_key: str
    fired: bool
    suppressed: bool
    reason: str
    action: str
    value: Any = None
    priority: str = "medium"
    cooldown_until: str | None = None

    def as_alert(self) -> dict[str, Any]:
        return {
            "ts": now_iso(),
            "trigger_type": self.trigger_type,
            "tier": self.tier,
            "target_id": self.target_id,
            "condition_key": self.condition_key,
            "status": "fired" if self.fired else "suppressed",
            "reason": self.reason,
            "action": self.action,
            "value": self.value,
            "priority": self.priority,
            "cooldown_until": self.cooldown_until,
        }


def now_iso(now: datetime | None = None) -> str:
    return _ensure_aware(now or datetime.now(timezone.utc)).isoformat().replace("+00:00", "Z")


def _ensure_aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _parse_dt(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None
    return _ensure_aware(dt)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _manifest_paths(memory_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if not memory_dir.exists():
        return paths
    for child in memory_dir.iterdir():
        if not child.is_dir() or child.name.startswith("_"):
            continue
        manifest = child / "manifest.json"
        if manifest.exists():
            paths.append(manifest)
    return paths


def _normalise_open_questions(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        out = []
        for item in raw:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(raw).strip()
    return [text] if text else []


def _load_previous_state(memory_dir: Path) -> dict[str, Any]:
    state = _read_json(memory_dir / "_self-model" / "state.json", {})
    return state if isinstance(state, dict) else {}


def _load_reflection_state(memory_dir: Path) -> dict[str, Any]:
    state = _read_json(memory_dir / "_reflection-loop" / "state.json", {})
    return state if isinstance(state, dict) else {}


def _load_todos(memory_dir: Path) -> list[dict[str, Any]]:
    raw = _read_json(memory_dir / "_self-model" / "todos.json", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _load_event_log_counts(path: Path, *, since: datetime) -> dict[str, Any]:
    if not path.exists():
        return {"total": 0, "errors": 0, "warnings": 0, "signatures": []}
    total = 0
    errors = 0
    warnings = 0
    sig_counts: dict[str, int] = {}
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except Exception:
                    continue
                ts = _parse_dt(ev.get("iso")) or _parse_dt(ev.get("ts"))
                if ts is not None and ts < since:
                    continue
                total += 1
                sig = str(ev.get("op") or ev.get("type") or ev.get("event") or "").strip()
                if sig:
                    sig_counts[sig] = sig_counts.get(sig, 0) + 1
                lowered = sig.lower()
                if lowered.startswith("error") or lowered.endswith("error") or lowered == "exception":
                    errors += 1
                if "warn" in lowered:
                    warnings += 1
    except Exception:
        return {"total": 0, "errors": 0, "warnings": 0, "signatures": []}
    signatures = [
        {"signature": sig, "count": count}
        for sig, count in sorted(sig_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    return {"total": total, "errors": errors, "warnings": warnings, "signatures": signatures}


def probe_loci_health() -> dict[str, Any]:
    """Best-effort read-only health probe.

    Fail-open: any import/probe error yields an empty dict, which the trigger
    evaluator treats as 'unknown' rather than 'down'.
    """
    repo_root = Path(__file__).resolve().parents[1]
    mcp_dir = repo_root / "mcp"
    inserted = False
    try:
        if str(mcp_dir) not in sys.path:
            sys.path.insert(0, str(mcp_dir))
            inserted = True
        import server as mcp_server  # type: ignore

        raw = mcp_server.loci_health()
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    finally:
        if inserted:
            try:
                sys.path.remove(str(mcp_dir))
            except ValueError:
                pass


def build_self_model(
    memory_dir: Path,
    *,
    now: datetime | None = None,
    health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = _ensure_aware(now or datetime.now(timezone.utc))
    previous = _load_previous_state(memory_dir)
    reflection = _load_reflection_state(memory_dir)
    todos = _load_todos(memory_dir)
    recent_events = _load_event_log_counts(EVENT_LOG_FILE, since=now - timedelta(hours=24))

    investigations: list[dict[str, Any]] = []
    for manifest_path in _manifest_paths(memory_dir):
        manifest = _read_json(manifest_path, {})
        if not isinstance(manifest, dict):
            continue
        updated_at = _parse_dt(manifest.get("updated_at")) or _parse_dt(manifest.get("created_at"))
        age_hours = None
        if updated_at is not None:
            age_hours = round((now - updated_at).total_seconds() / 3600.0, 2)
        open_questions = _normalise_open_questions(manifest.get("open_questions"))
        finding_counts = manifest.get("finding_counts")
        if not isinstance(finding_counts, dict):
            finding_counts = {}
        findings_total = 0
        for value in finding_counts.values():
            if isinstance(value, (int, float)):
                findings_total += int(value)
        investigations.append({
            "id": str(manifest.get("id") or manifest_path.parent.name),
            "title": str(manifest.get("title") or manifest_path.parent.name),
            "status": str(manifest.get("status") or "unknown"),
            "staleness_hours": age_hours,
            "open_questions_count": len(open_questions),
            "finding_count": findings_total,
            "next_step": str(manifest.get("next_step") or "").strip(),
            "open_questions": open_questions[:8],
        })

    investigations.sort(
        key=lambda item: (
            item["staleness_hours"] if item["staleness_hours"] is not None else 10_000.0,
            item["title"].lower(),
        )
    )
    active_investigations = [item for item in investigations if item["status"] != "closed"]

    blocked_todos = [item for item in todos if str(item.get("status") or "").lower() == "blocked"]
    open_todos = [item for item in todos if str(item.get("status") or "").lower() == "pending"]

    health_data = health if isinstance(health, dict) else {}
    backends = {
        "qdrant": bool(health_data.get("qdrant_reachable", True)),
        "ollama": bool(health_data.get("ollama_reachable", True)),
        "vllm": bool(health_data.get("vllm_reachable", True)),
        "ladybug": health_data.get("ladybug", "unknown"),
    }

    reflection_queue = 0
    if isinstance(reflection, dict):
        for key in ("queue_size", "backlog", "reflection_queue_size"):
            value = reflection.get(key)
            if isinstance(value, (int, float)):
                reflection_queue = int(value)
                break
        if not reflection_queue:
            queue = reflection.get("queue")
            if isinstance(queue, list):
                reflection_queue = len(queue)

    last_daily_summary_ts = previous.get("last_daily_summary_ts")
    current_focus = previous.get("current_focus")
    if active_investigations:
        current_focus = active_investigations[0]["title"]
        if active_investigations[0]["next_step"]:
            current_focus = active_investigations[0]["next_step"]

    state = {
        "generated_at": now_iso(now),
        "last_run_ts": now_iso(now),
        "memory_dir": str(memory_dir),
        "backends": backends,
        "active_investigations": active_investigations[:12],
        "blocked_todos": [
            {
                "id": str(item.get("id") or ""),
                "title": str(item.get("title") or ""),
                "status": str(item.get("status") or ""),
            }
            for item in blocked_todos[:12]
        ],
        "open_todos_count": len(open_todos),
        "blocked_todos_count": len(blocked_todos),
        "reflection_queue_size": reflection_queue,
        "recent_event_count_24h": recent_events["total"],
        "error_signatures_24h": recent_events["signatures"],
        "errors_24h": recent_events["errors"],
        "warnings_24h": recent_events["warnings"],
        "current_focus": current_focus or "",
        "recent_decisions": previous.get("recent_decisions", [])[:10] if isinstance(previous.get("recent_decisions"), list) else [],
        "blocked_items": previous.get("blocked_items", [])[:10] if isinstance(previous.get("blocked_items"), list) else [],
        "last_daily_summary_ts": last_daily_summary_ts or "",
        "trigger_state": previous.get("trigger_state", {}) if isinstance(previous.get("trigger_state"), dict) else {},
        "session_start": previous.get("session_start") or now_iso(now),
        "mode": previous.get("mode") or "idle",
    }
    return state


def _cooldown_hours(trigger_type: str, config: TriggerConfig) -> float:
    if trigger_type.startswith("T1-"):
        return config.t1_cooldown_minutes / 60.0
    if trigger_type.startswith("T2-"):
        return config.t2_cooldown_hours
    if trigger_type.startswith("T3-"):
        return config.t3_cooldown_hours
    return 1.0


def _cooldown_until(previous: dict[str, Any], key: str, now: datetime, hours: float) -> tuple[bool, str | None]:
    trigger_state = previous.get("trigger_state", {})
    last = None
    if isinstance(trigger_state, dict):
        entry = trigger_state.get(key)
        if isinstance(entry, dict):
            last = _parse_dt(entry.get("last_fired_at"))
    if last is None:
        return False, None
    delta = now - last
    cooldown = timedelta(hours=hours)
    if delta < cooldown:
        return True, now_iso(last + cooldown)
    return False, now_iso(last + cooldown)


def evaluate_proactive_trigger(
    trigger: dict[str, Any],
    report: dict[str, Any],
    *,
    config: TriggerConfig,
    now: datetime | None = None,
) -> TriggerEvaluation:
    now = _ensure_aware(now or datetime.now(timezone.utc))
    trigger_type = str(trigger.get("type") or "")
    tier = str(trigger.get("tier") or "T2")
    target_id = str(trigger.get("target_id") or "")
    condition_key = str(trigger.get("condition_key") or f"{trigger_type}:{target_id}")
    reason = str(trigger.get("reason") or "")
    action = str(trigger.get("action") or "")
    priority = str(trigger.get("priority") or "medium")
    value = trigger.get("value")
    suppressed, cooldown_until = _cooldown_until(
        report,
        condition_key,
        now,
        _cooldown_hours(trigger_type, config),
    )
    fired = bool(trigger.get("fired"))
    if suppressed:
        return TriggerEvaluation(
            trigger_type=trigger_type,
            tier=tier,
            target_id=target_id,
            condition_key=condition_key,
            fired=False,
            suppressed=True,
            reason=reason,
            action=action,
            value=value,
            priority=priority,
            cooldown_until=cooldown_until,
        )
    return TriggerEvaluation(
        trigger_type=trigger_type,
        tier=tier,
        target_id=target_id,
        condition_key=condition_key,
        fired=fired,
        suppressed=False,
        reason=reason,
        action=action,
        value=value,
        priority=priority,
        cooldown_until=cooldown_until,
    )


def _build_triggers(state: dict[str, Any], *, now: datetime, config: TriggerConfig) -> list[dict[str, Any]]:
    triggers: list[dict[str, Any]] = []

    backends = state.get("backends", {})
    if isinstance(backends, dict):
        for name, value in backends.items():
            if name == "ladybug":
                if value in ("available", "contended"):
                    continue
                if value == "unknown":
                    continue
                triggers.append({
                    "type": "T1-BACKEND-DOWN",
                    "tier": "T1",
                    "target_id": str(name),
                    "condition_key": f"backend:{name}",
                    "fired": False,
                    "reason": f"{name} backend is {value}",
                    "action": f"inspect {name} health and pause dependent work",
                    "priority": "critical",
                    "value": value,
                })
                continue
            if value is False:
                triggers.append({
                    "type": "T1-BACKEND-DOWN",
                    "tier": "T1",
                    "target_id": str(name),
                    "condition_key": f"backend:{name}",
                    "fired": False,
                    "reason": f"{name} backend unreachable",
                    "action": f"inspect {name} health and pause dependent work",
                    "priority": "critical",
                    "value": value,
                })

    queue_size = int(state.get("reflection_queue_size") or 0)
    if queue_size >= config.queue_flood_threshold:
        triggers.append({
            "type": "T1-QUEUE-FLOOD",
            "tier": "T1",
            "target_id": "reflection_queue",
            "condition_key": "reflection_queue_size",
            "fired": True,
            "reason": f"reflection queue size {queue_size} >= {config.queue_flood_threshold}",
            "action": "run reflection_loop_tick(max_items=20) and surface backlog",
            "priority": "critical",
            "value": queue_size,
        })

    error_count = int(state.get("errors_24h") or 0)
    if error_count >= config.error_storm_threshold:
        triggers.append({
            "type": "T1-ERROR-STORM",
            "tier": "T1",
            "target_id": "event_log",
            "condition_key": "errors_24h",
            "fired": True,
            "reason": f"{error_count} errors in 24h >= {config.error_storm_threshold}",
            "action": "surface error storm and inspect recent failures",
            "priority": "critical",
            "value": error_count,
        })

    for inv in state.get("active_investigations", []) or []:
        if not isinstance(inv, dict):
            continue
        staleness = inv.get("staleness_hours")
        open_q = int(inv.get("open_questions_count") or 0)
        if staleness is not None and float(staleness) >= config.stale_hours and open_q > 0:
            triggers.append({
                "type": "T2-STALE-INV",
                "tier": "T2",
                "target_id": str(inv.get("id") or inv.get("title") or "investigation"),
                "condition_key": f"investigation:{inv.get('id')}:stale",
                "fired": True,
                "reason": (
                    f"{inv.get('title') or inv.get('id')} is {staleness}h stale with {open_q} open questions"
                ),
                "action": "call investigation_reason(..., persist=True) for the next best step",
                "priority": "high",
                "value": {
                    "staleness_hours": staleness,
                    "open_questions": open_q,
                },
            })

    blocked_todos = int(state.get("blocked_todos_count") or 0)
    if blocked_todos > config.blocked_todo_threshold:
        triggers.append({
            "type": "T2-BLOCKED-TODOS",
            "tier": "T2",
            "target_id": "blocked_todos",
            "condition_key": "blocked_todos_count",
            "fired": True,
            "reason": f"blocked todo count {blocked_todos} > {config.blocked_todo_threshold}",
            "action": "summarize blockers and propose unblock steps",
            "priority": "high",
            "value": blocked_todos,
        })

    last_daily_summary_ts = _parse_dt(state.get("last_daily_summary_ts"))
    daily_due = False
    if now.hour >= config.daily_summary_start_hour:
        if last_daily_summary_ts is None:
            daily_due = True
        else:
            daily_due = (now - last_daily_summary_ts) >= timedelta(hours=config.daily_summary_cooldown_hours)
    if daily_due:
        triggers.append({
            "type": "T3-DAILY-SUMMARY",
            "tier": "T3",
            "target_id": "daily_summary",
            "condition_key": "last_daily_summary_ts",
            "fired": True,
            "reason": f"daily summary overdue and local hour is {now.hour}",
            "action": "render a planning summary and refresh last_daily_summary_ts",
            "priority": "medium",
            "value": state.get("last_daily_summary_ts") or "",
        })

    return triggers


def _confidence_score(state: dict[str, Any], alerts: list[TriggerEvaluation]) -> tuple[float, str]:
    penalties = 0.0
    basis = []
    stale = 0
    for inv in state.get("active_investigations", []) or []:
        if isinstance(inv, dict) and inv.get("staleness_hours") is not None:
            if float(inv["staleness_hours"]) >= STALE_HOURS:
                stale += 1
    blockers = [a for a in alerts if a.fired and a.tier == "T1"]
    corrections = [a for a in alerts if a.fired and a.tier == "T2"]
    planning = [a for a in alerts if a.fired and a.tier == "T3"]
    if blockers:
        penalties += 0.35
        basis.append("tier-1 blockers")
    if corrections:
        penalties += 0.20
        basis.append("tier-2 corrections")
    if planning:
        penalties += 0.10
        basis.append("tier-3 planning")
    if stale:
        penalties += min(0.25, stale * 0.05)
        basis.append("stale investigations")
    if int(state.get("blocked_todos_count") or 0):
        penalties += min(0.20, int(state.get("blocked_todos_count") or 0) * 0.03)
        basis.append("blocked todos")
    score = max(0.05, min(0.98, 1.0 - penalties))
    if not basis:
        basis = ["current state looks consistent"]
    return round(score, 2), ", ".join(basis)


def render_introspection_report(
    state: dict[str, Any],
    evaluations: list[TriggerEvaluation],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = _ensure_aware(now or datetime.now(timezone.utc))
    fired = [item for item in evaluations if item.fired]
    blockers = []
    uncertainty = []
    next_step = {
        "action": "continue current work",
        "priority": "low",
        "trigger_tier": "T0",
        "rationale": "no proactive trigger fired",
        "tool_hint": "",
    }

    for inv in state.get("active_investigations", []) or []:
        if not isinstance(inv, dict):
            continue
        questions = inv.get("open_questions") or []
        if not isinstance(questions, list):
            questions = []
        for q in questions:
            text = str(q).strip()
            if text:
                uncertainty.append({
                    "type": "open_question",
                    "text": text,
                    "finding_id": inv.get("id"),
                    "since": inv.get("staleness_hours"),
                })
        if inv.get("staleness_hours") is not None and float(inv["staleness_hours"]) >= STALE_HOURS and questions:
            blockers.append({
                "type": "investigation_stale",
                "resource": inv.get("id"),
                "trigger_tier": "T2",
            })

    if state.get("reflection_queue_size", 0) >= QUEUE_FLOOD_THRESHOLD:
        blockers.append({
            "type": "queue_overflow",
            "resource": "reflection_loop",
            "trigger_tier": "T1",
        })

    if int(state.get("blocked_todos_count") or 0) > BLOCKED_TODO_THRESHOLD:
        blockers.append({
            "type": "blocked_todos",
            "resource": "todo_queue",
            "trigger_tier": "T2",
        })

    if fired:
        top = sorted(
            fired,
            key=lambda item: ({"T1": 0, "T2": 1, "T3": 2}.get(item.tier, 9), item.priority),
        )[0]
        next_step = {
            "action": top.action,
            "priority": "critical" if top.tier == "T1" else top.priority,
            "trigger_tier": top.tier,
            "rationale": top.reason,
            "tool_hint": top.action,
        }

    confidence_score, confidence_basis = _confidence_score(state, evaluations)
    if any(item.tier == "T1" and item.fired for item in evaluations):
        mode = "degraded"
    elif any(item.tier == "T2" and item.fired for item in evaluations):
        mode = "blocked"
    elif any(item.tier == "T3" and item.fired for item in evaluations):
        mode = "planning"
    elif state.get("reflection_queue_size", 0) > 0:
        mode = "reflecting"
    else:
        mode = "idle"

    return {
        "generated_at": now_iso(now),
        "state": {
            "mode": mode,
            "active_investigation": state.get("current_focus") or "",
            "reflection_queue_backlog": state.get("reflection_queue_size") or 0,
            "session_tool_calls": state.get("recent_event_count_24h") or 0,
            "session_start": state.get("session_start") or "",
        },
        "confidence": {
            "overall": confidence_score,
            "basis": confidence_basis,
            "hypothesis": state.get("current_focus") or "",
            "finding_distribution": {
                "active_investigations": len(state.get("active_investigations") or []),
                "blocked_todos": int(state.get("blocked_todos_count") or 0),
                "open_todos": int(state.get("open_todos_count") or 0),
            },
            "memory_confidence_score": confidence_score,
        },
        "uncertainty": uncertainty[:20],
        "blockers": blockers[:20],
        "next_step": next_step,
        "mode": mode,
        "fired_triggers": [asdict(item) for item in fired],
    }


def run_eval(
    triggers: list[dict[str, Any]],
    state: dict[str, Any],
    config: TriggerConfig,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = _ensure_aware(now or datetime.now(timezone.utc))
    evaluations = [evaluate_proactive_trigger(trigger, state, config=config, now=now) for trigger in triggers]
    report = render_introspection_report(state, evaluations, now=now)
    return {
        "state": state,
        "report": report,
        "evaluations": evaluations,
    }


def _update_trigger_state(state: dict[str, Any], evaluations: Iterable[TriggerEvaluation], *, now: datetime) -> dict[str, Any]:
    trigger_state = dict(state.get("trigger_state") or {})
    if not isinstance(trigger_state, dict):
        trigger_state = {}
    for ev in evaluations:
        if not ev.fired:
            continue
        entry = dict(trigger_state.get(ev.condition_key) or {})
        entry["last_fired_at"] = now_iso(now)
        entry["fire_count"] = int(entry.get("fire_count") or 0) + 1
        trigger_state[ev.condition_key] = entry
    state["trigger_state"] = trigger_state
    return state


def _persist_outputs(memory_dir: Path, state: dict[str, Any], report: dict[str, Any], evaluations: list[TriggerEvaluation], *, now: datetime) -> None:
    payload_state = dict(state)
    payload_state["generated_at"] = now_iso(now)
    payload_state["last_run_ts"] = now_iso(now)
    payload_state = _update_trigger_state(payload_state, evaluations, now=now)
    _write_json_atomic(memory_dir / "_self-model" / "state.json", payload_state)
    _write_json_atomic(memory_dir / "_self-model" / "introspection_report.json", report)
    for ev in evaluations:
        if ev.fired:
            _append_jsonl(memory_dir / "_self-model" / "alerts.jsonl", ev.as_alert())
    if any(ev.fired for ev in evaluations):
        daily = next((ev for ev in evaluations if ev.trigger_type == "T3-DAILY-SUMMARY" and ev.fired), None)
        if daily is not None:
            payload_state["last_daily_summary_ts"] = now_iso(now)
            _write_json_atomic(memory_dir / "_self-model" / "state.json", payload_state)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-dir",
        default=str(DEFAULT_MEMORY_DIR),
        help="Memory root containing investigation manifests and _self-model state.",
    )
    parser.add_argument(
        "--now",
        default="",
        help="Override current time (ISO-8601). Useful for tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute state and report without writing any files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the generated introspection report JSON regardless of alerts.",
    )
    args = parser.parse_args(argv)

    memory_dir = Path(args.memory_dir).expanduser()
    now = _parse_dt(args.now) or datetime.now(timezone.utc)
    health = probe_loci_health()
    state = build_self_model(memory_dir, now=now, health=health)
    config = TriggerConfig()
    triggers = _build_triggers(state, now=now, config=config)
    result = run_eval(triggers, state, config, now=now)
    evaluations = result["evaluations"]
    report = result["report"]

    if not args.dry_run:
        _persist_outputs(memory_dir, state, report, evaluations, now=now)

    fired = [ev for ev in evaluations if ev.fired]
    if args.verbose or fired:
        summary = {
            "mode": report["mode"],
            "alerts": len(fired),
            "current_focus": state.get("current_focus") or "",
            "reflection_queue_size": state.get("reflection_queue_size") or 0,
            "blocked_todos": state.get("blocked_todos_count") or 0,
        }
        print(json.dumps(summary, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

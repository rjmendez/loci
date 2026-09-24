from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


_SCRIPTS = Path(__file__).resolve().parent.parent


def _load(name: str = "self_model_trigger_eval_uut"):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / "self_model_trigger_eval.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_manifest(root: Path, slug: str, *, title: str, updated_at: datetime, open_questions: list[str], finding_counts: dict[str, int] | None = None, status: str = "open", next_step: str = "") -> None:
    case_dir = root / slug
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": slug,
        "title": title,
        "status": status,
        "updated_at": updated_at.isoformat().replace("+00:00", "Z"),
        "open_questions": open_questions,
        "finding_counts": finding_counts or {"observed": 1},
        "next_step": next_step,
    }
    (case_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_reflection_state(root: Path, queue_size: int) -> None:
    rdir = root / "_reflection-loop"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "state.json").write_text(json.dumps({"queue_size": queue_size}), encoding="utf-8")


def _write_event_log(root: Path, rows: list[dict]) -> Path:
    path = root / "event_log.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _write_todos(root: Path, rows: list[dict]) -> None:
    tdir = root / "_self-model"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "todos.json").write_text(json.dumps(rows), encoding="utf-8")


def test_build_self_model_and_report(tmp_path, monkeypatch):
    mod = _load()
    memory_dir = tmp_path / "memory-sessions"
    now = datetime(2026, 9, 22, 15, 30, tzinfo=timezone.utc)
    _write_manifest(
        memory_dir,
        "alpha",
        title="Alpha",
        updated_at=now - timedelta(hours=80),
        open_questions=["Need validation"],
        finding_counts={"observed": 2, "inferred": 1},
        next_step="Check the queue",
    )
    _write_manifest(
        memory_dir,
        "beta",
        title="Beta",
        updated_at=now - timedelta(hours=1),
        open_questions=[],
        finding_counts={"observed": 1},
        status="open",
    )
    _write_reflection_state(memory_dir, queue_size=250)
    _write_todos(
        memory_dir,
        [
            {"id": "t1", "title": "blocked", "status": "blocked"},
            {"id": "t2", "title": "done", "status": "done"},
            {"id": "t3", "title": "pending", "status": "pending"},
            {"id": "t4", "title": "blocked", "status": "blocked"},
        ],
    )
    _write_event_log(
        tmp_path,
        [
            {"ts": now.isoformat(), "op": "store"},
            {"ts": now.isoformat(), "op": "error"},
            {"ts": now.isoformat(), "op": "warn"},
        ],
    )
    monkeypatch.setattr(mod, "EVENT_LOG_FILE", tmp_path / "event_log.jsonl")

    state = mod.build_self_model(
        memory_dir,
        now=now,
        health={"qdrant_reachable": True, "ollama_reachable": True, "vllm_reachable": True, "ladybug": "available"},
    )
    config = mod.TriggerConfig()
    triggers = mod._build_triggers(state, now=now, config=config)
    result = mod.run_eval(triggers, state, config, now=now)
    report = result["report"]

    assert state["current_focus"] == "Beta"
    assert state["reflection_queue_size"] == 250
    assert state["blocked_todos_count"] == 2
    assert state["errors_24h"] == 1
    assert state["warnings_24h"] == 1
    assert any(item["id"] == "alpha" for item in state["active_investigations"])
    assert report["mode"] == "degraded"
    assert report["next_step"]["trigger_tier"] == "T1"
    assert any(blocker["type"] == "queue_overflow" for blocker in report["blockers"])
    assert any(item["type"] == "open_question" for item in report["uncertainty"])


def test_evaluate_trigger_respects_cooldown(tmp_path):
    mod = _load("self_model_trigger_eval_uut_cooldown")
    now = datetime(2026, 9, 22, 15, 30, tzinfo=timezone.utc)
    state = {
        "trigger_state": {
            "reflection_queue_size": {"last_fired_at": (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")}
        }
    }
    trigger = {
        "type": "T1-QUEUE-FLOOD",
        "tier": "T1",
        "target_id": "reflection_queue",
        "condition_key": "reflection_queue_size",
        "fired": True,
        "reason": "queue flood",
        "action": "tick",
        "priority": "critical",
        "value": 250,
    }

    config = mod.TriggerConfig(t1_cooldown_minutes=15)
    ev = mod.evaluate_proactive_trigger(trigger, state, config=config, now=now)

    assert ev.fired is False
    assert ev.suppressed is True
    assert ev.cooldown_until == "2026-09-22T15:40:00Z"  # last fired 15:25 + 15 min

    # Expiry: the same trigger fired 16 minutes ago is live again. A cooldown
    # that never ends ("suppress forever") passed the old test.
    expired = {"trigger_state": {"reflection_queue_size": {"last_fired_at": "2026-09-22T15:14:00Z"}}}
    ev = mod.evaluate_proactive_trigger(trigger, expired, config=config, now=now)
    assert (ev.fired, ev.suppressed, ev.cooldown_until) == (True, False, "2026-09-22T15:29:00Z")

    # Boundary: exactly one cooldown later is no longer suppressed.
    boundary = {"trigger_state": {"reflection_queue_size": {"last_fired_at": "2026-09-22T15:15:00Z"}}}
    ev = mod.evaluate_proactive_trigger(trigger, boundary, config=config, now=now)
    assert (ev.fired, ev.suppressed) == (True, False)

    # Never fired: no cooldown at all.
    ev = mod.evaluate_proactive_trigger(trigger, {}, config=config, now=now)
    assert (ev.fired, ev.suppressed, ev.cooldown_until) == (True, False, None)


def test_main_writes_state_and_alerts(tmp_path, monkeypatch):
    mod = _load("self_model_trigger_eval_uut_main")
    memory_dir = tmp_path / "memory-sessions"
    now = datetime(2026, 9, 22, 18, 0, tzinfo=timezone.utc)
    _write_manifest(
        memory_dir,
        "alpha",
        title="Alpha",
        updated_at=now - timedelta(hours=80),
        open_questions=["Need validation"],
        finding_counts={"observed": 2},
    )
    _write_reflection_state(memory_dir, queue_size=250)
    monkeypatch.setattr(mod, "probe_loci_health", lambda: {})

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = mod.main([
            "--memory-dir",
            str(memory_dir),
            "--now",
            now.isoformat().replace("+00:00", "Z"),
        ])

    assert code == 0
    assert out.getvalue().strip()
    assert (memory_dir / "_self-model" / "state.json").exists()
    assert (memory_dir / "_self-model" / "introspection_report.json").exists()
    alerts = (memory_dir / "_self-model" / "alerts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert alerts
    payload = json.loads(alerts[0])
    assert payload["trigger_type"] in {"T1-QUEUE-FLOOD", "T2-STALE-INV", "T3-DAILY-SUMMARY"}

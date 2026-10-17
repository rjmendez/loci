"""Tests for reflection_triage.py — advisory semantic triage for reflection findings."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import reflection_triage  # noqa: E402


def _gen_fn(text="", *, ok=True, why=None):
    def _fn(prompt, model="", fmt=None, max_tokens=256, temperature=0.2, keep_alive="30m"):
        return {"text": text, "ok": ok, "why": why, "model": model}
    return _fn


def test_classify_reflection_observation_returns_labels_when_parseable():
    out = reflection_triage.classify_reflection_observation(
        "process_log",
        "/tmp/a.log",
        errors={"assertion failed": 1},
        gen_fn=_gen_fn('{"category":"real_regression","novelty":"novel_signal"}'),
    )
    assert out == {
        "category": "real_regression",
        "novelty": "novel_signal",
        "degraded": False,
        "ok": True,
        "error": None,
    }


def test_classify_reflection_observation_fails_open_on_generate_error():
    def _boom(*a, **k):
        raise RuntimeError("connection refused")

    out = reflection_triage.classify_reflection_observation(
        "process_log",
        "/tmp/a.log",
        errors={"assertion failed": 1},
        gen_fn=_boom,
    )
    assert out["degraded"] is True
    assert out["ok"] is False
    assert "generate() raised" in out["error"]


def test_classify_reflection_observation_invalid_labels_degrade():
    out = reflection_triage.classify_reflection_observation(
        "process_log",
        "/tmp/a.log",
        errors={"assertion failed": 1},
        gen_fn=_gen_fn('{"category":"totally_new_bucket","novelty":"novel_signal"}'),
    )
    assert out["degraded"] is True
    assert out["category"] is None
    assert out["novelty"] == "novel_signal"

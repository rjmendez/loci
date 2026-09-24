"""Shared pytest configuration for scripts/tests/."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_the_event_log(tmp_path, monkeypatch):
    """No test may append to the operator's ~/.hermes/event_log.jsonl.

    hybrid_lane_router.choose_lane() records a route event for every decision
    via mcp/route_audit.py, which reads LOCI_EVENT_LOG at call time. autouse so
    a new test file cannot forget it.
    """
    monkeypatch.setenv("LOCI_EVENT_LOG", str(tmp_path / "event_log.jsonl"))

from scripts.orchestrator import _pause_lower_priority_jobs, _resume_paused_jobs


def test_pause_lower_priority_jobs_pauses_lower_priority_running_work():
    jobs = {
        "background": {"priority": 10, "pauseable": True},
        "urgent": {"priority": 0, "pauseable": True},
        "locked": {"priority": 5, "pauseable": False},
    }
    state = {
        "background": "running",
        "urgent": "queued",
        "locked": "running",
    }

    _pause_lower_priority_jobs(jobs, state)

    assert state["background"] == "paused"
    assert state["urgent"] == "queued"
    assert state["locked"] == "running"


def test_resume_paused_jobs_when_no_higher_priority_task_remains():
    jobs = {
        "background": {"priority": 10},
        "urgent": {"priority": 0},
    }
    state = {
        "background": "paused",
        "urgent": "done",
    }

    _resume_paused_jobs(jobs, state)

    assert state["background"] == "queued"

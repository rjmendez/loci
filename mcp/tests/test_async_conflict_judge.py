"""The LLM contradiction judge runs off the investigation_store path.

It used to make up to ten serial local-LLM calls inside every store; a model that
could not load held each store for minutes. Now the store answers from the
heuristics, at most max_pairs neighbours are judged on a background worker, and a
circuit breaker stops calling a failing judge. LOCI_CONFLICT_JUDGE_SYNC=1 restores
the inline judge.
"""
import json
import logging
import threading
import time
import uuid
from unittest import mock

import pytest

import conflict_verify
import server


class _Point:
    def __init__(self, pid, score, payload):
        self.id = pid
        self.score = score
        self.payload = payload


class _Response:
    def __init__(self, points):
        self.points = points


class _Client:
    def __init__(self, points):
        self._points = points

    def query_points(self, **kwargs):
        return _Response(list(self._points))


NEW = {"id": "f-new", "record_type": "observed",
       "text": "the service exposes port 443 to the internet"}


def _obs(pid, score, text):
    return _Point(pid, score, {"id": pid, "record_type": "observed", "text": text})


def _gap(pid, score):
    return _Point(pid, score, {"id": pid, "record_type": "gap",
                               "text": "unknown whether port 443 is exposed"})


class _Judge:
    """Injected gen_fn: counts calls, says contradict when the neighbour says 'closed'."""

    def __init__(self, *, ok=True, gate=None):
        self.calls = 0
        self.finished = 0
        self.ok = ok
        self.gate = gate
        self._lock = threading.Lock()

    def __call__(self, prompt, **kwargs):
        with self._lock:
            self.calls += 1
        if self.gate is not None:
            self.gate.wait(10)
        with self._lock:
            self.finished += 1
        if not self.ok:
            return {"text": "", "ok": False, "model": "judge:7b", "why": "model failed to load"}
        verdict = "contradict" if "closed" in prompt else "agree"
        return {"text": json.dumps({"verdict": verdict, "reason": "r"}), "ok": True,
                "model": "judge:7b"}


@pytest.fixture
def inv():
    return f"async-judge-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def _fresh_judge_state(monkeypatch):
    for name in ("SYNC", "MAX_PAIRS", "BREAKER_FAILURES", "BREAKER_COOLDOWN_S", "SLOW_S"):
        monkeypatch.delenv(f"LOCI_CONFLICT_JUDGE_{name}", raising=False)
    monkeypatch.delenv("LOCI_JUDGE_VERDICT_LOG", raising=False)
    yield
    _drain()


def _drain(timeout=10.0):
    deadline = time.monotonic() + timeout
    while server._CONFLICT_JUDGE_QUEUE.unfinished_tasks:
        assert time.monotonic() < deadline, "conflict judge worker did not drain"
        time.sleep(0.01)


def _patched(points, judge):
    client = _Client(points)
    return mock.patch.multiple(
        server, _get_qdrant=lambda: (client, "loci_memory"), _embed=lambda _t: [0.1, 0.2, 0.3],
    ), mock.patch.object(conflict_verify, "_lazy_generate", judge)


def _store(inv, points, judge, *, drain=True):
    qdrant, gen = _patched(points, judge)
    with qdrant, gen:
        started = time.monotonic()
        out = server._store_conflicts(inv, dict(NEW))
        elapsed = time.monotonic() - started
        if drain:
            _drain()
    return out, elapsed


def _rows(inv):
    path = server._inv_dir(inv) / server.JUDGE_VERDICT_LOG_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _conflict_records(inv):
    path = server._inv_dir(inv) / "conflicts.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_store_returns_promptly_while_the_judge_is_slow(inv):
    gate = threading.Event()
    judge = _Judge(gate=gate)
    points = [_gap("n-gap", 0.95), _obs("n-obs", 0.9, "port 443 is closed to the internet")]
    qdrant, gen = _patched(points, judge)
    try:
        with qdrant, gen:
            started = time.monotonic()
            detected, neighbor, conflict_id = server._store_conflicts(inv, dict(NEW))
            elapsed = time.monotonic() - started
            # The heuristic answer is back while the judge is still stuck on its first pair
            # (an inline judge would have sat out the gate's 10s timeout first).
            assert judge.finished == 0
            assert elapsed < 5.0
            assert (detected, neighbor) == (True, "n-gap")
            assert [c["id"] for c in _conflict_records(inv)] == [conflict_id]
            assert _rows(inv) == []
            gate.set()
            _drain()
    finally:
        gate.set()
    assert judge.calls == 2
    assert [(r["neighbor_id"], r["verdict"]) for r in _rows(inv)] == [
        ("n-gap", "agree"), ("n-obs", "contradict")]
    # The heuristic conflict was already recorded; the judge adds no second record.
    assert len(_conflict_records(inv)) == 1


def test_investigation_store_does_not_wait_for_the_judge(inv):
    gate = threading.Event()
    judge = _Judge(gate=gate)
    server.investigation_start(investigation_id=inv, title="async judge")
    points = [_obs("n-obs", 0.9, "port 443 is closed to the internet")]
    qdrant, gen = _patched(points, judge)
    try:
        # The index write would pull a sparse-embedding model; this test is about the judge.
        with qdrant, gen, mock.patch.object(server, "_qdrant_upsert", lambda *a, **k: None):
            result = json.loads(server.investigation_store(
                investigation_id=inv, finding_type="observed", text=NEW["text"],
                source="test:async-judge"))
            assert judge.finished == 0
            assert result["stored"] is True
            assert result["conflict_detected"] is False
            gate.set()
            _drain()
    finally:
        gate.set()
    assert [r["finding_id_b"] for r in _conflict_records(inv)] == ["n-obs"]


def test_async_verdict_lands_on_the_conflict_record_and_verdict_log(inv):
    judge = _Judge()
    points = [_obs("n-agree", 0.93, "port 443 answers with a TLS banner"),
              _obs("n-obs", 0.9, "port 443 is closed to the internet")]
    (detected, neighbor, conflict_id), _ = _store(inv, points, judge)
    # No heuristic fires, so the store itself reports nothing...
    assert (detected, neighbor, conflict_id) == (False, None, None)
    # ...and the judge's contradiction is recorded afterwards, as the inline path did.
    records = _conflict_records(inv)
    assert [(r["finding_id_a"], r["finding_id_b"], r["status"]) for r in records] == [
        ("f-new", "n-obs", "open")]
    rows = _rows(inv)
    assert [(r["neighbor_id"], r["verdict"], r["judge_ok"], r["conflict_candidate"])
            for r in rows] == [("n-agree", "agree", True, False),
                               ("n-obs", "contradict", True, True)]
    assert all("judge_skipped" not in r for r in rows)


def test_cap_bounds_judge_calls_per_store(inv, monkeypatch):
    monkeypatch.setenv("LOCI_CONFLICT_JUDGE_MAX_PAIRS", "2")
    judge = _Judge()
    points = [_obs(f"n{i}", 0.99 - i / 100, f"neighbour {i}") for i in range(5)]
    _store(inv, points, judge)
    assert judge.calls == 2
    rows = _rows(inv)
    assert [r["neighbor_id"] for r in rows] == ["n0", "n1", "n2", "n3", "n4"]
    assert [r.get("judge_skipped") for r in rows] == [None, None, "cap", "cap", "cap"]
    assert [r["judge_ok"] for r in rows] == [True, True, False, False, False]


def test_default_cap_is_three(inv):
    judge = _Judge()
    points = [_obs(f"n{i}", 0.99 - i / 100, f"neighbour {i}") for i in range(10)]
    _store(inv, points, judge)
    assert judge.calls == 3


def test_cap_zero_disables_the_judge_but_not_the_heuristics(inv, monkeypatch):
    monkeypatch.setenv("LOCI_CONFLICT_JUDGE_MAX_PAIRS", "0")
    judge = _Judge()
    (detected, neighbor, _), _ = _store(inv, [_gap("n-gap", 0.95)], judge)
    assert (detected, neighbor) == (True, "n-gap")
    assert judge.calls == 0


def test_cap_from_backends_toml(inv, monkeypatch):
    import backends

    monkeypatch.setattr(backends, "_config", lambda: {"conflict_judge": {"max_pairs": 1}})
    judge = _Judge()
    points = [_obs(f"n{i}", 0.99 - i / 100, f"neighbour {i}") for i in range(4)]
    _store(inv, points, judge)
    assert judge.calls == 1


def test_breaker_opens_after_consecutive_failures_and_skips_the_judge(inv, monkeypatch):
    monkeypatch.setenv("LOCI_CONFLICT_JUDGE_BREAKER_FAILURES", "2")
    judge = _Judge(ok=False)
    points = [_obs(f"n{i}", 0.99 - i / 100, f"neighbour {i}") for i in range(3)]
    _store(inv, points, judge)
    _store(inv, points, judge)
    assert judge.calls == 2
    skipped = [r.get("judge_skipped") for r in _rows(inv)]
    assert skipped == [None, None, "circuit_open", "circuit_open", "circuit_open", "circuit_open"]


def test_slow_answers_count_as_failures(inv, monkeypatch):
    monkeypatch.setenv("LOCI_CONFLICT_JUDGE_BREAKER_FAILURES", "1")
    monkeypatch.setenv("LOCI_CONFLICT_JUDGE_SLOW_S", "-1")
    judge = _Judge()
    points = [_obs(f"n{i}", 0.99 - i / 100, f"neighbour {i}") for i in range(3)]
    _store(inv, points, judge)
    # The one slow answer is still used, but it opens the breaker for the rest.
    assert judge.calls == 1
    assert [(r["verdict"], r.get("judge_skipped")) for r in _rows(inv)] == [
        ("agree", None), (None, "circuit_open"), (None, "circuit_open")]


def test_breaker_closes_after_cooldown_and_logs_once_per_episode(monkeypatch, caplog):
    monkeypatch.setenv("LOCI_CONFLICT_JUDGE_BREAKER_FAILURES", "3")
    monkeypatch.setenv("LOCI_CONFLICT_JUDGE_BREAKER_COOLDOWN_S", "30")
    now = [1000.0]
    breaker = server._JudgeCircuitBreaker(clock=lambda: now[0])
    caplog.set_level(logging.INFO, logger=server.logger.name)

    for _ in range(2):
        breaker.record(False)
    assert breaker.allow()
    breaker.record(False)
    assert not breaker.allow()

    now[0] += 29.0
    assert not breaker.allow()
    now[0] += 2.0
    assert breaker.allow()
    # Half-open: one more failure reopens at once, without a second warning.
    breaker.record(False)
    assert not breaker.allow()
    opened = [r for r in caplog.records if "circuit breaker open" in r.getMessage()]
    assert len(opened) == 1

    now[0] += 31.0
    breaker.record(True)
    assert breaker.allow()
    assert any("circuit breaker closed" in r.getMessage() for r in caplog.records)
    # Closed means the failure count restarted.
    breaker.record(False)
    breaker.record(False)
    assert breaker.allow()


def test_sync_mode_judges_inline_like_before(inv, monkeypatch):
    monkeypatch.setenv("LOCI_CONFLICT_JUDGE_SYNC", "1")
    judge = _Judge()
    points = [_obs("n-agree", 0.93, "port 443 answers with a TLS banner"),
              _obs("n-obs", 0.9, "port 443 is closed to the internet")]
    (detected, neighbor, conflict_id), _ = _store(inv, points, judge, drain=False)
    # The judge's contradiction is in the store's own answer, nothing was queued.
    assert (detected, neighbor) == (True, "n-obs")
    assert server._CONFLICT_JUDGE_QUEUE.unfinished_tasks == 0
    assert [r["id"] for r in _conflict_records(inv)] == [conflict_id]
    assert judge.calls == 2
    assert [r["verdict"] for r in _rows(inv)] == ["agree", "contradict"]


def test_sync_mode_takes_the_first_conflict_by_rank_as_before(inv, monkeypatch):
    monkeypatch.setenv("LOCI_CONFLICT_JUDGE_SYNC", "1")
    points = [_obs("n-obs", 0.95, "port 443 is closed to the internet"), _gap("n-gap", 0.9)]
    qdrant, gen = _patched(points, _Judge())
    with qdrant, gen:
        expected = server._detect_conflicts(inv, dict(NEW))
        detected, neighbor, _ = server._store_conflicts(inv, dict(NEW))
    assert [c["neighbor_id"] for c in expected] == ["n-obs", "n-gap"]
    assert (detected, neighbor) == (True, expected[0]["neighbor_id"])


def test_full_queue_drops_the_job_and_logs_the_pairs_as_skipped(inv, monkeypatch):
    full = server.queue.Queue(maxsize=1)
    full.put_nowait(("other-inv", {"id": "x"}, [], False))
    monkeypatch.setattr(server, "_CONFLICT_JUDGE_QUEUE", full)
    judge = _Judge()
    (detected, _, _), _ = _store(inv, [_gap("n-gap", 0.95)], judge, drain=False)
    assert detected is True
    assert judge.calls == 0
    assert [(r["neighbor_id"], r.get("judge_skipped")) for r in _rows(inv)] == [
        ("n-gap", "queue_full")]
    full.get_nowait()
    full.task_done()


def test_worker_survives_a_failing_job(inv):
    judge = _Judge()
    with mock.patch.object(server, "_judge_conflict_candidates", side_effect=RuntimeError("boom")):
        _store(inv, [_obs("n-obs", 0.9, "port 443 is closed to the internet")], judge)
    _store(inv, [_obs("n-obs", 0.9, "port 443 is closed to the internet")], judge)
    assert judge.calls == 1
    assert [r["finding_id_b"] for r in _conflict_records(inv)] == ["n-obs"]


def test_store_never_fails_when_candidate_search_raises(inv):
    with mock.patch.object(server, "_conflict_candidates", side_effect=RuntimeError("qdrant")):
        assert server._store_conflicts(inv, dict(NEW)) == (False, None, None)

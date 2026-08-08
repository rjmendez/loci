"""_get_ladybug resilience: a transient lock/IO failure must NOT permanently disable the
code graph (retry later), while a genuinely-unrecoverable failure (ladybug unimportable)
still latches so we stop retrying. All fail-open — never raises."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server as S  # noqa: E402
from graph import ladybug_store as KZ  # noqa: E402


def _reset(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "MEMORY_DIR", tmp_path / "mem")
    monkeypatch.setattr(S, "_ladybug_store", None, raising=False)
    monkeypatch.setattr(S, "_ladybug_failed", False, raising=False)
    monkeypatch.setattr(S, "_ladybug_last_attempt", 0.0, raising=False)
    monkeypatch.setattr(S, "_LADYBUG_RETRY_SECONDS", 0, raising=False)  # disable backoff wait
    monkeypatch.setattr(S, "_ladybug_backfill_if_empty", lambda ks: None)  # isolate from backfill


def test_transient_lock_failure_does_not_latch_and_retries(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path)
    monkeypatch.setattr(KZ, "_HAS_LADYBUG", True, raising=False)
    state = {"n": 0}

    class Store:
        def __init__(self, path):
            state["n"] += 1
            if state["n"] == 1:  # first open loses the single-writer lock race
                raise RuntimeError("Could not set lock on file: Resource temporarily unavailable")

        def available(self):
            return True

    monkeypatch.setattr(KZ, "LadybugStore", Store)

    # First call: lock contention -> None, but the permanent latch is NOT set.
    assert S._get_ladybug() is None
    assert S._ladybug_failed is False

    # A later call retries (lock released) and succeeds -> graph self-heals.
    ks = S._get_ladybug()
    assert ks is not None
    assert state["n"] == 2


def test_transient_unavailable_open_does_not_latch(tmp_path, monkeypatch):
    """LadybugStore constructs but reports available()==False (lock held) -> transient."""
    _reset(monkeypatch, tmp_path)
    monkeypatch.setattr(KZ, "_HAS_LADYBUG", True, raising=False)
    state = {"n": 0}

    class Store:
        def __init__(self, path):
            state["n"] += 1

        def available(self):
            return state["n"] >= 2  # first open unavailable, second OK

    monkeypatch.setattr(KZ, "LadybugStore", Store)

    assert S._get_ladybug() is None
    assert S._ladybug_failed is False
    assert S._get_ladybug() is not None
    assert state["n"] == 2


def test_missing_ladybug_latches_permanently(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path)
    monkeypatch.setattr(KZ, "_HAS_LADYBUG", False, raising=False)  # ladybug not importable
    called = {"n": 0}

    class Store:
        def __init__(self, path):
            called["n"] += 1

        def available(self):
            return True

    monkeypatch.setattr(KZ, "LadybugStore", Store)

    assert S._get_ladybug() is None
    assert S._ladybug_failed is True          # permanent latch set
    # Subsequent call short-circuits on the latch; the factory is never touched.
    assert S._get_ladybug() is None
    assert called["n"] == 0


def test_import_error_latches_permanently(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path)
    monkeypatch.setattr(KZ, "_HAS_LADYBUG", True, raising=False)

    class Store:
        def __init__(self, path):
            raise ImportError("ladybug native extension unavailable")

        def available(self):
            return True

    monkeypatch.setattr(KZ, "LadybugStore", Store)

    assert S._get_ladybug() is None
    assert S._ladybug_failed is True          # ImportError is unrecoverable -> latch

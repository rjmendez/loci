"""Model leases: plan, acquire/release against a fake Ollama and fake nvidia-smi.

Nothing here touches a real Ollama, GPU or store: transport and telemetry are monkeypatched
and the ledger lives under a temp LOCI_MEMORY_DIR.
"""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import model_lease as L  # noqa: E402
import model_pool as M  # noqa: E402

GB = 1.0


def _pool():
    return [
        M.PoolEntry("main:8b", ("gen", "verify"), rank=1),
        M.PoolEntry("spare:4b", ("gen",), rank=2),
        M.PoolEntry("worst:3b", ("gen",), rank=9),
        M.PoolEntry("embedder", ("embed",), rank=1, pinned=True),
        M.PoolEntry("sticky:7b", ("code",), rank=3, evictable=False),
        M.PoolEntry("coder:7b", ("code",), rank=1),
    ]


def _res(*pairs):
    return [{"name": n, "size_gb": s} for n, s in pairs]


PRIMARIES = {"gen": "main:8b", "verify": "main:8b", "embed": "embedder", "code": "coder:7b"}


# ---- plan (pure) ---------------------------------------------------------------------

def test_plan_orders_unknown_first_then_worst_rank_and_protects_primaries_and_pins():
    resident = _res(("main:8b", 5), ("spare:4b", 4), ("worst:3b", 3), ("embedder", 0.3),
                    ("stranger:1b", 1), ("sticky:7b", 5), ("coder:7b", 5))
    p = L.plan(resident, 8, "normal", _pool(), PRIMARIES, busy=set())
    assert [v.name for v in p.victims] == ["stranger:1b", "worst:3b", "spare:4b"]
    why = dict(p.protected)
    assert why["embedder"] == "pinned" and why["sticky:7b"] == "pinned"
    assert why["main:8b"].startswith("primary for") and why["coder:7b"].startswith("primary for")


def test_critical_priority_may_evict_primaries_but_never_pinned():
    resident = _res(("main:8b", 5), ("embedder", 0.3), ("sticky:7b", 5))
    p = L.plan(resident, 8, "critical", _pool(), PRIMARIES, busy=set())
    assert [v.name for v in p.victims] == ["main:8b"]
    assert "critical" in p.victims[0].reason
    assert {n for n, _ in p.protected} == {"embedder", "sticky:7b"}


def test_a_model_in_use_is_never_a_victim():
    p = L.plan(_res(("spare:4b", 4), ("worst:3b", 3)), 6, "normal", _pool(), PRIMARIES, busy={"worst:3b"})
    assert [v.name for v in p.victims] == ["spare:4b"]
    assert ("worst:3b", "in use by a Loci call") in p.protected


def test_bigger_model_goes_first_on_equal_rank():
    pool = [M.PoolEntry("a", ("gen",), 5), M.PoolEntry("b", ("gen",), 5)]
    p = L.plan(_res(("a", 3), ("b", 6)), 4, "normal", pool, {}, set())
    assert [v.name for v in p.victims] == ["b", "a"]


def test_a_cpu_resident_model_is_not_a_victim_because_unloading_it_frees_no_vram():
    p = L.plan(_res(("worst:3b", 0.0), ("spare:4b", 4)), 4, "normal", _pool(), PRIMARIES, busy=set())
    assert [v.name for v in p.victims] == ["spare:4b"]
    assert any(n == "worst:3b" and "no VRAM" in w for n, w in p.protected)


def test_inflight_context_manager_counts_and_clears():
    with L.inflight("m1"):
        with L.inflight("m1"):
            assert "m1" in L.inflight_models()
        assert "m1" in L.inflight_models()
    assert "m1" not in L.inflight_models()


# ---- fake world ----------------------------------------------------------------------

class World:
    """Ollama + GPUs: loaded models consume VRAM on one of two cards; unloading frees it."""

    def __init__(self, loaded, gpu_total=(11.0, 12.0)):
        self.loaded = dict(loaded)               # name -> (gb, gpu index)
        self.total = list(gpu_total)
        self.unloaded: list[str] = []
        self.loaded_calls: list[str] = []
        self.refuse_unload: set[str] = set()

    def free(self):
        used = [0.0] * len(self.total)
        for gb, gpu in self.loaded.values():
            used[gpu] += gb
        return [t - u for t, u in zip(self.total, used)]

    def ps(self, base):
        return [{"name": n, "size_gb": gb} for n, (gb, _) in self.loaded.items()]

    def unload(self, base, name):
        if name in self.refuse_unload:
            return False
        self.loaded.pop(name, None)
        self.unloaded.append(name)
        return True

    def load(self, base, name, keep_alive="30m"):
        self.loaded_calls.append(name)
        self.loaded[name] = (3.0, 0)
        return True


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    L._held_cache = (0.0, -1.0, set())
    w = World({"main:8b": (5.0, 1), "spare:4b": (4.0, 0), "worst:3b": (3.0, 0), "embedder": (0.3, 1)})
    monkeypatch.setattr(L, "resident_detail", w.ps)
    monkeypatch.setattr(L, "gpu_free_gb", w.free)
    monkeypatch.setattr(L, "unload", w.unload)
    monkeypatch.setattr(L, "load", w.load)
    monkeypatch.setattr(L, "_pool_entries", _pool)
    monkeypatch.setattr(L, "_primaries", lambda: dict(PRIMARIES))
    return w


def _acquire(**kw):
    kw.setdefault("base_url", "http://fake")
    kw.setdefault("_sleep", lambda s: None)
    kw.setdefault("wait_s", 0.0)
    return L.acquire(kw.pop("job", "evo-run"), kw.pop("need_gb", 8.0), **kw)


# ---- acquire / release ---------------------------------------------------------------

def test_already_enough_room_evicts_nothing_but_still_records_a_lease(world):
    out = _acquire(need_gb=5.0)                      # GPU1 has 12-5.3 free? check: gpu0 has 11-7=4, gpu1 6.7
    assert out["granted"] and out["evicted"] == []
    assert out["lease_id"] in L.active_leases()


def test_evicts_worst_first_until_one_gpu_has_room_then_stops(world):
    out = _acquire(need_gb=9.0)                      # needs worst:3b and spare:4b off GPU0 (11 GB card)
    assert out["granted"] and out["verified"] is True
    assert [e["name"] for e in out["evicted"]] == ["worst:3b", "spare:4b"]
    assert world.unloaded == ["worst:3b", "spare:4b"]
    assert "main:8b" in world.loaded and "embedder" in world.loaded      # protected
    assert max(world.free()) >= 9.0


def test_stops_as_soon_as_a_single_gpu_fits(world):
    out = _acquire(need_gb=7.0)                      # GPU0: 11-7=4 free; unloading worst frees 3 -> 7
    assert out["granted"] and [e["name"] for e in out["evicted"]] == ["worst:3b"]


def test_insufficient_room_restores_everything_and_reports_the_shortfall(world):
    out = _acquire(need_gb=11.5)                     # only 'normal' victims: GPU0 maxes at 11 free
    assert out["granted"] is False and "insufficient" in out["reason"]
    assert out["evicted"] == []
    assert sorted(world.loaded_calls) == ["spare:4b", "worst:3b"]        # put back
    assert L.active_leases() == {}


def test_critical_priority_can_take_a_primary_to_make_room(world):
    out = _acquire(need_gb=11.5, priority="critical")
    assert out["granted"] is True
    assert "main:8b" in [e["name"] for e in out["evicted"]]
    assert "embedder" in world.loaded                                    # still pinned


def test_release_loads_evicted_models_back_and_clears_the_lease(world):
    out = _acquire(need_gb=9.0)
    rel = L.release(out["lease_id"], base_url="http://fake")
    assert rel["released"] and sorted(rel["restored"]) == ["spare:4b", "worst:3b"]
    assert L.active_leases() == {}
    assert L.release(out["lease_id"], base_url="http://fake")["released"] is False


def test_restore_false_leaves_models_out(world):
    out = _acquire(need_gb=9.0, restore=False)
    rel = L.release(out["lease_id"], base_url="http://fake")
    assert rel["restored"] == [] and world.loaded_calls == []


def test_unreachable_ollama_changes_nothing(monkeypatch, world):
    monkeypatch.setattr(L, "resident_detail", lambda base: None)
    out = _acquire(need_gb=9.0)
    assert out["granted"] is False and "unreachable" in out["reason"]
    assert world.unloaded == []


def test_a_model_that_refuses_to_unload_is_skipped(world):
    world.refuse_unload.add("worst:3b")
    out = _acquire(need_gb=7.0)                       # worst can't go; spare:4b frees 4 -> 8 free
    assert out["granted"] and [e["name"] for e in out["evicted"]] == ["spare:4b"]


def test_without_telemetry_everything_eligible_is_evicted_unverified(monkeypatch, world):
    monkeypatch.setattr(L, "gpu_free_gb", lambda: None)
    out = _acquire(need_gb=9.0)
    assert out["granted"] and out["verified"] is False
    assert sorted(e["name"] for e in out["evicted"]) == ["spare:4b", "worst:3b"]


def test_busy_models_survive_a_lease(world):
    with L.inflight("worst:3b"):
        out = _acquire(need_gb=7.0)
    assert out["granted"] is False or "worst:3b" not in [e["name"] for e in out["evicted"]]
    assert "worst:3b" in world.loaded


# ---- expiry, ledger and the pool ------------------------------------------------------

def test_expired_leases_are_reaped_and_restored(world):
    out = _acquire(need_gb=9.0, ttl_s=1)
    ledger = json.loads(L._ledger_path().read_text())
    ledger[out["lease_id"]]["expires_at"] = time.time() - 5
    L._ledger_path().write_text(json.dumps(ledger))
    assert L.reap(base_url="http://fake") == [out["lease_id"]]
    assert sorted(world.loaded_calls) == ["spare:4b", "worst:3b"] and L.active_leases() == {}


def test_held_out_lists_models_evicted_by_active_leases(world):
    assert L.held_out() == set()
    _acquire(need_gb=9.0)
    L._held_cache = (0.0, -1.0, set())
    assert L.held_out() == {"spare:4b", "worst:3b"}


def test_the_pool_will_not_pick_a_model_a_lease_evicted(world):
    inv = {"spare:4b": 4 * 10**9, "main:8b": 5 * 10**9, "worst:3b": 3 * 10**9}
    pool = [M.PoolEntry("spare:4b", ("gen",), 1), M.PoolEntry("worst:3b", ("gen",), 2)]
    kw = dict(pool=pool, inv=inv, resident=set(), cap_gb=None, bonus=0.0, fallback=False)
    assert M.rank_role("gen", held_out=set(), **kw).chosen == "spare:4b"
    d = M.rank_role("gen", held_out={"spare:4b"}, **kw)
    assert d.chosen == "worst:3b"
    assert next(c for c in d.candidates if c.name == "spare:4b").reason == "evicted by an active lease"


def test_pool_reads_active_leases_by_default(world):
    _acquire(need_gb=9.0)
    L._held_cache = (0.0, -1.0, set())
    inv = {"spare:4b": 4 * 10**9, "worst:3b": 3 * 10**9}
    pool = [M.PoolEntry("spare:4b", ("gen",), 1), M.PoolEntry("worst:3b", ("gen",), 2)]
    d = M.rank_role("gen", pool=pool, inv=inv, resident=set(), cap_gb=None, bonus=0.0, fallback=False)
    assert d.chosen == ""                            # both were evicted by the lease


def test_context_manager_releases_even_when_the_job_fails(world):
    with pytest.raises(RuntimeError):
        with L.lease("evo-run", 9.0, base_url="http://fake", wait_s=0.0, _sleep=lambda s: None) as grant:
            assert grant["granted"]
            raise RuntimeError("job crashed")
    assert L.active_leases() == {}
    assert sorted(world.loaded_calls) == ["spare:4b", "worst:3b"]


def test_status_reports_leases_resident_and_gpu_free(world):
    _acquire(need_gb=9.0)
    s = L.status(base_url="http://fake")
    assert len(s["leases"]) == 1 and isinstance(s["resident"], list)
    assert isinstance(s["gpu_free_gb"], list)


# ---- config / tool surface -------------------------------------------------------------

def test_evictable_false_parses_from_the_pool(monkeypatch):
    import backends
    monkeypatch.setattr(backends, "_config", lambda: {"models": {"pool": [
        {"name": "a", "roles": ["gen"], "evictable": False}, {"name": "b", "roles": ["gen"]}]}})
    a, b = M.entries()
    assert a.evictable is False and b.evictable is True


def test_rendered_toml_keeps_evictable_false():
    import tomllib
    text = M.render_toml([M.PoolEntry("m", ("gen",), 1, evictable=False)])
    assert tomllib.loads(text)["models"]["pool"][0]["evictable"] is False


def test_lease_tools_are_registered_and_return_json(monkeypatch):
    import llm_tools

    monkeypatch.setattr(L, "acquire", lambda *a, **k: {"granted": True, "lease_id": "x"})
    monkeypatch.setattr(L, "release", lambda lid: {"released": True})
    monkeypatch.setattr(L, "status", lambda: {"leases": {}})
    assert json.loads(llm_tools.model_lease_acquire("job", 8))["lease_id"] == "x"
    assert json.loads(llm_tools.model_lease_release("x"))["released"] is True
    assert json.loads(llm_tools.model_lease_status()) == {"leases": {}}
    from mcp.server.fastmcp import FastMCP
    srv = FastMCP("t")
    llm_tools.register(srv)
    names = {t.name for t in srv._tool_manager.list_tools()}
    assert {"model_lease_acquire", "model_lease_release", "model_lease_status"} <= names

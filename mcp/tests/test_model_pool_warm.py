"""``model_pool.py warm`` keeps ``warm = true`` models resident within a per-card VRAM budget.

Resident-only exploration (2026-10-06) only tries models Ollama already holds, because a cold load takes 25-70 s and
queues every other request behind it. Something has to keep a second model resident for it to try; this is that,
and it refuses a model that has no home card or that would not fit the card beside what is already there.
"""
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import model_lease  # noqa: E402
import model_pool as M  # noqa: E402

GB = 1024 ** 3
INV = {n: GB for n in ("a", "b", "c", "d", "gemma")}
CARDS = {0: 12.0, 1: 11.0}


@pytest.fixture(autouse=True)
def _clean():
    M.clear_cache()
    yield
    M.clear_cache()


def entry(name, gpu=1, vram=5.0, warm=True, roles=("gen",)):
    return M.PoolEntry(name, roles, vram_gb=vram, gpu=gpu, warm=warm)


def plan(pool, resident=(), inv=None, cards=None, held=()):
    return M.warm_plan(pool=pool, resident=set(resident), inv=INV if inv is None else inv,
                       gpu_gb=CARDS if cards is None else cards, held_out=set(held))


def by(rows):
    return {r["model"]: (r["action"], r["why"]) for r in rows}


class TestSchema:
    def _cfg(self, monkeypatch, models):
        monkeypatch.setattr(M, "_config", lambda: {"models": models})

    def test_only_a_literal_true_marks_a_model_warm(self, monkeypatch):
        self._cfg(monkeypatch, {"pool": [{"name": "t", "roles": ["gen"], "warm": True},
                                         {"name": "s", "roles": ["gen"], "warm": "true"},
                                         {"name": "o", "roles": ["gen"], "warm": 1},
                                         {"name": "f", "roles": ["gen"], "warm": False},
                                         {"name": "n", "roles": ["gen"]}]})
        assert {e.name: e.warm for e in M.entries()} == {"t": True, "s": False, "o": False, "f": False, "n": False}

    def test_gpu_vram_gb_keys_by_card_and_drops_bad_rows(self, monkeypatch):
        self._cfg(monkeypatch, {"gpu_vram_gb": {"0": 12, "1": 11.5, "x": 5, "-1": 4, "2": 0, "3": "big"}})
        assert M.gpu_vram_gb() == {0: 12.0, 1: 11.5}

    def test_no_budget_configured_is_an_empty_budget(self, monkeypatch):
        self._cfg(monkeypatch, {})
        assert M.gpu_vram_gb() == {}
        self._cfg(monkeypatch, {"gpu_vram_gb": [12, 11]})
        assert M.gpu_vram_gb() == {}


class TestWarmPlan:
    def test_a_warm_model_that_fits_its_card_is_loaded(self):
        assert plan([entry("a", gpu=1, vram=5.0)]) == [
            {"model": "a", "gpu": 1, "action": "load", "why": "needs 5 of 10 GB free on card 1"}]

    def test_a_model_not_marked_warm_is_never_in_the_plan(self):
        assert plan([entry("a", warm=False)]) == []

    def test_a_resident_model_is_left_alone(self):
        assert by(plan([entry("a")], resident={"a"})) == {"a": ("skip", "already resident")}
        assert by(plan([entry("a")], resident={"a:latest"})) == {"a": ("skip", "already resident")}

    def test_a_model_that_is_not_installed_is_skipped(self):
        assert by(plan([entry("zzz")])) == {"zzz": ("skip", "not installed")}

    def test_a_model_held_out_by_a_lease_is_skipped(self):
        assert by(plan([entry("a")], held={"a"})) == {"a": ("skip", "evicted by an active lease")}

    def test_a_model_with_no_home_card_is_never_warmed(self):
        action, why = by(plan([entry("a", gpu=None)]))["a"]
        assert action == "skip" and why.startswith("no home gpu")

    def test_a_model_with_no_vram_figure_or_no_card_budget_is_skipped(self):
        assert by(plan([entry("a", vram=None)])) == {"a": ("skip", "no vram_gb to budget with")}
        assert by(plan([entry("a", gpu=5)])) == {"a": ("skip", "no gpu_vram_gb for card 5")}

    def test_it_must_fit_beside_the_pool_models_already_resident_on_that_card(self):
        too_much = [entry("gemma", gpu=1, vram=6.0, warm=False), entry("a", gpu=1, vram=5.0)]
        assert by(plan(too_much, resident={"gemma"}))["a"][0] == "skip"          # 6 + 5 > 11 - 1
        fits = [entry("gemma", gpu=1, vram=4.0, warm=False), entry("a", gpu=1, vram=5.0)]
        assert by(plan(fits, resident={"gemma"}))["a"][0] == "load"              # 4 + 5 <= 10

    def test_another_cards_residents_do_not_count(self):
        pool = [entry("gemma", gpu=0, vram=11.0, warm=False), entry("a", gpu=1, vram=9.0)]
        assert by(plan(pool, resident={"gemma"}))["a"][0] == "load"

    def test_the_headroom_is_kept(self):
        assert by(plan([entry("a", gpu=1, vram=10.0)]))["a"][0] == "load"        # exactly 11 - 1
        assert by(plan([entry("a", gpu=1, vram=10.1)]))["a"][0] == "skip"

    def test_earlier_planned_loads_count_against_later_ones(self):
        rows = by(plan([entry("a", gpu=1, vram=6.0), entry("b", gpu=1, vram=6.0)]))
        assert rows["a"][0] == "load"
        assert rows["b"] == ("skip", "6 GB does not fit card 1: 6 of 10 GB used")

    def test_two_cards_are_budgeted_separately(self):
        rows = by(plan([entry("a", gpu=1, vram=8.0), entry("b", gpu=0, vram=8.0)]))
        assert (rows["a"][0], rows["b"][0]) == ("load", "load")


class TestWarmRun:
    def _run(self, monkeypatch, pool, models_cfg=None, **kw):
        loaded = []
        monkeypatch.setattr(M, "entries", lambda: pool)
        monkeypatch.setattr(M, "inventory", lambda base_url=None: INV)
        monkeypatch.setattr(M, "resident_models", lambda base_url=None: set())
        monkeypatch.setattr(M, "gpu_vram_gb", lambda: CARDS)
        monkeypatch.setattr(M, "_models_cfg", lambda: models_cfg or {})
        monkeypatch.setattr(M, "_gen_url", lambda: "http://ollama/")
        monkeypatch.setattr(model_lease, "held_out", lambda: set())
        monkeypatch.setattr(model_lease, "load", lambda base, name, keep_alive="30m":
                            (loaded.append((base, name, keep_alive)), True)[1])
        return M.warm(**kw), loaded

    def test_it_loads_each_planned_model_with_a_long_keep_alive(self, monkeypatch):
        rows, loaded = self._run(monkeypatch, [entry("a", vram=4.0), entry("b", vram=4.0)])
        assert loaded == [("http://ollama", "a", "2h"), ("http://ollama", "b", "2h")]
        assert [r["loaded"] for r in rows] == [True, True]

    def test_the_keep_alive_is_configurable(self, monkeypatch):
        _, loaded = self._run(monkeypatch, [entry("a")], models_cfg={"warm_keep_alive": "6h"})
        assert loaded[0][2] == "6h"

    def test_a_dry_run_loads_nothing(self, monkeypatch):
        rows, loaded = self._run(monkeypatch, [entry("a")], dry_run=True)
        assert loaded == [] and rows[0]["action"] == "load" and "loaded" not in rows[0]

    def test_a_skipped_model_is_not_loaded(self, monkeypatch):
        rows, loaded = self._run(monkeypatch, [entry("a", gpu=None)])
        assert loaded == [] and rows[0]["action"] == "skip" and "loaded" not in rows[0]

    def test_a_failed_load_is_reported(self, monkeypatch):
        self._run(monkeypatch, [entry("a")])
        monkeypatch.setattr(model_lease, "load", lambda base, name, keep_alive="30m": False)
        assert M.warm()[0]["loaded"] is False


class TestCli:
    def test_it_prints_the_plan_and_loads_nothing_on_a_dry_run(self, monkeypatch, capsys):
        monkeypatch.setattr(M, "warm_plan", lambda: [
            {"model": "a", "gpu": 1, "action": "load", "why": "needs 5 of 10 GB free on card 1"}])
        monkeypatch.setattr(model_lease, "load", lambda *a, **k: pytest.fail("a dry run must not load"))
        assert M._main(["warm", "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "load  a" in out and "gpu=1" in out and "needs 5 of 10 GB free on card 1" in out and "loaded" not in out

    def test_it_says_when_nothing_is_marked_warm(self, monkeypatch, capsys):
        monkeypatch.setattr(M, "warm", lambda dry_run=False: [])
        assert M._main(["warm"]) == 0
        assert "no pool entry has warm = true" in capsys.readouterr().out

"""``model_pool.py warm`` keeps ``warm = true`` models resident within a per-card VRAM budget.

Resident-only exploration (2026-10-06) only tries models Ollama already holds, because a cold load takes 25-70 s and
queues every other request behind it. Something has to keep a second model resident for it to try; this is that,
and it refuses a model that would not fit beside what is already there. A pinned model (``gpu = n``) is budgeted against
that card's static ``gpu_vram_gb``; an unpinned one (the default) against the VRAM that is free right now.
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


def plan(pool, resident=(), inv=None, cards=None, held=(), free=None):
    return M.warm_plan(pool=pool, resident=set(resident), inv=INV if inv is None else inv,
                       gpu_gb=CARDS if cards is None else cards, held_out=set(held),
                       free_gb={} if free is None else free)


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

    def test_an_unpinned_model_is_planned_against_the_vram_free_right_now(self):
        # no home card: the card with the most free VRAM must have the model plus headroom free
        assert plan([entry("a", gpu=None, vram=5.0)], free={0: 2.7, 1: 7.5}) == [
            {"model": "a", "gpu": None, "action": "load",
             "why": "needs 5 GB (+1 headroom); card 1 has 7.5 GB free"}]

    def test_an_unpinned_model_that_fits_no_card_is_skipped_with_the_best_figure(self):
        assert by(plan([entry("a", gpu=None, vram=5.0)], free={0: 2.7, 1: 5.9})) == {
            "a": ("skip", "5 GB (+1 GB headroom) does not fit: the card with the most free VRAM is 1 with 5.9 GB")}

    def test_the_headroom_boundary_is_inclusive_for_an_unpinned_model(self):
        assert by(plan([entry("a", gpu=None, vram=5.0)], free={0: 6.0}))["a"][0] == "load"
        assert by(plan([entry("a", gpu=None, vram=5.0)], free={0: 5.99}))["a"][0] == "skip"

    def test_unpinned_loads_are_charged_to_the_card_ollama_would_use(self):
        # two 5 GB models, cards with 7.5 and 7 GB free: the first goes to card 1 (most free), leaving 2.5, so the
        # second needs card 0 (7 >= 6) - not card 1 again
        rows = plan([entry("a", gpu=None, vram=5.0), entry("b", gpu=None, vram=5.0)], free={0: 7.0, 1: 7.5})
        assert [(r["model"], r["action"]) for r in rows] == [("a", "load"), ("b", "load")]
        assert rows[0]["why"].endswith("card 1 has 7.5 GB free") and rows[1]["why"].endswith("card 0 has 7.0 GB free")
        third = plan([entry(n, gpu=None, vram=5.0) for n in ("a", "b", "c")], free={0: 7.0, 1: 7.5})
        assert [r["action"] for r in third] == ["load", "load", "skip"]

    def test_unreadable_free_vram_means_not_loaded_blind(self):
        assert by(plan([entry("a", gpu=None)], free={})) == {"a": ("skip", "free VRAM could not be read, so it is not loaded blind")}

    def test_the_static_budget_is_not_used_for_an_unpinned_model(self):
        # card 1 is "budgeted" at 11 GB, but only 3 GB is really free right now: the live figure decides
        assert by(plan([entry("a", gpu=None, vram=5.0)], cards={1: 11.0}, free={1: 3.0}))["a"][0] == "skip"

    def test_a_pinned_model_still_ignores_the_live_figure(self):
        assert by(plan([entry("a", gpu=1, vram=5.0)], free={1: 0.5}))["a"][0] == "load"

    def test_free_vram_is_read_once_per_plan_and_only_when_an_unpinned_model_needs_it(self, monkeypatch):
        calls = []
        monkeypatch.setattr(M, "gpu_free_gb", lambda: calls.append(1) or {0: 8.0})
        M.warm_plan(pool=[entry("a", gpu=1), entry("b", gpu=1)], resident=set(), inv=INV, gpu_gb=CARDS, held_out=set())
        assert calls == []
        M.warm_plan(pool=[entry("a", gpu=None), entry("b", gpu=None)], resident=set(), inv=INV, gpu_gb=CARDS, held_out=set())
        assert calls == [1]

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
        rows, loaded = self._run(monkeypatch, [entry("zzz")])                 # not installed
        assert loaded == [] and rows[0]["action"] == "skip" and "loaded" not in rows[0]

    def test_an_unpinned_model_is_loaded_when_a_card_has_room_and_not_otherwise(self, monkeypatch):
        monkeypatch.setattr(M, "gpu_free_gb", lambda: {0: 8.0})
        rows, loaded = self._run(monkeypatch, [entry("a", gpu=None, vram=5.0)])
        assert loaded == [("http://ollama", "a", "2h")] and rows[0]["loaded"] is True
        monkeypatch.setattr(M, "gpu_free_gb", lambda: {0: 3.0})
        rows, loaded = self._run(monkeypatch, [entry("a", gpu=None, vram=5.0)])
        assert loaded == [] and rows[0]["action"] == "skip"

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


class TestGpuFreeGb:
    def run(self, monkeypatch, stdout=None, exc=None):
        import types

        def fake(cmd, **kw):
            assert cmd == ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
            if exc:
                raise exc
            return types.SimpleNamespace(stdout=stdout)
        monkeypatch.setattr("subprocess.run", fake)
        return M.gpu_free_gb()

    def test_parses_mib_to_gb_per_card(self, monkeypatch):
        assert self.run(monkeypatch, "0, 7677\n1, 2733\n") == {0: 7677 / 1024, 1: 2733 / 1024}

    def test_crlf_and_unreadable_rows(self, monkeypatch):
        assert self.run(monkeypatch, "0, 7677\r\n1, [N/A]\r\nx\r\n") == {0: 7677 / 1024}

    @pytest.mark.parametrize("exc", [FileNotFoundError("nvidia-smi"), RuntimeError("driver")])
    def test_a_failure_is_none_and_logged(self, monkeypatch, caplog, exc):
        import logging
        with caplog.at_level(logging.WARNING):
            assert self.run(monkeypatch, exc=exc) is None
        assert any("could not read free VRAM from nvidia-smi" in r.getMessage() for r in caplog.records)

    def test_no_rows_is_none(self, monkeypatch):
        assert self.run(monkeypatch, "\n") is None

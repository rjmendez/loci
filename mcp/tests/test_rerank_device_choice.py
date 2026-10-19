"""The reranker lands on the CUDA card with the most free VRAM at load time; no static pin.

Why: a hardcoded card (CUDA_VISIBLE_DEVICES in the launcher, a device_id in a script) is a guess that goes stale as soon as
Ollama loads something else. Ranking the cards by free memory when the model loads, trying them best first, and failing over
when a load fails lets the model use what is available, while Ollama manages its own memory (it unloads idle models).

Mutation runs (each applied to mcp/reranker.py, confirmed red, reverted):
- rank by index instead of free memory -> test_auto_ranks_the_cards_by_free_vram_most_free_first red
- ``>=`` min free -> ``>`` -> test_a_card_with_exactly_the_minimum_free_is_a_candidate red
- drop the failover ``continue`` (stop after the first failed card) -> test_a_failed_load_on_the_best_card_fails_over_to_the_next red
- fall back to CPU when every candidate failed -> test_when_every_qualifying_card_fails_it_does_not_quietly_become_cpu red
- ignore LOCI_RERANK_DEVICE=cpu -> test_explicit_cpu_is_a_choice_with_its_reason red
- cache the card list across loads -> test_free_memory_is_read_again_at_every_load red
- health drops the card index/name -> test_health_reports_the_card_the_model_landed_on red
"""
import logging
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import reranker as R  # noqa: E402

MIB = 1 << 20


def card(index, free, total=11264, name=None):
    return {"index": index, "name": name or f"GPU{index}", "free_mb": free, "total_mb": total}


# ------------------------------------------------------------------ the pure ranking rules

def test_auto_ranks_the_cards_by_free_vram_most_free_first():
    cands, reason = R.choose_devices([card(0, 2700), card(1, 7700)], "auto", 2600)
    assert [c["index"] for c in cands] == [1, 0]
    assert reason == "most free VRAM of: cuda:0 GPU0 2700 MiB free, cuda:1 GPU1 7700 MiB free"


def test_cuda_means_auto():
    assert R.choose_devices([card(0, 5000), card(1, 9000)], "cuda", 2600)[0][0]["index"] == 1


def test_a_card_with_exactly_the_minimum_free_is_a_candidate():
    cands, _ = R.choose_devices([card(0, 2600), card(1, 2599)], "auto", 2600)
    assert [c["index"] for c in cands] == [0]


def test_no_card_with_room_gives_no_candidates_and_a_reason_that_names_them():
    cands, reason = R.choose_devices([card(0, 2000), card(1, 2500)], "auto", 2600)
    assert cands == [] and reason == (
        "no card has 2600 MiB free; cards: cuda:0 GPU0 2000 MiB free, cuda:1 GPU1 2500 MiB free")


def test_no_cuda_card_at_all_says_so():
    assert R.choose_devices([], "auto", 2600) == ([], "torch sees no CUDA card")


def test_a_card_whose_memory_could_not_be_read_is_never_a_candidate_and_is_named():
    broken = {"index": 1, "name": "GPU1", "free_mb": None, "total_mb": None, "error": "RuntimeError: x"}
    cands, reason = R.choose_devices([card(0, 8000), broken], "auto", 2600)
    assert [c["index"] for c in cands] == [0] and "cuda:1 GPU1 unreadable MiB free" in reason


def test_explicit_cpu_is_a_choice_with_its_reason():
    assert R.choose_devices([card(0, 9000)], "cpu", 2600) == ([], "LOCI_RERANK_DEVICE=cpu")


def test_an_explicit_card_is_used_even_when_another_has_more_room():
    cands, reason = R.choose_devices([card(0, 2700), card(1, 9000)], "cuda:0", 2600)
    assert [c["index"] for c in cands] == [0] and reason == "LOCI_RERANK_DEVICE=cuda:0"


@pytest.mark.parametrize("setting,expect", [
    ("cuda:7", "cuda:7 not usable; cards: cuda:0 GPU0 5000 MiB free"),
    ("cuda:x", "LOCI_RERANK_DEVICE='cuda:x' is not cuda:<integer>; cards: cuda:0 GPU0 5000 MiB free"),
])
def test_an_unusable_explicit_card_is_reported_not_guessed(setting, expect):
    assert R.choose_devices([card(0, 5000)], setting, 2600) == ([], expect)


def test_the_setting_defaults_to_auto_and_is_normalised(monkeypatch):
    monkeypatch.delenv("LOCI_RERANK_DEVICE", raising=False)
    assert R._device_setting() == "auto"
    monkeypatch.setenv("LOCI_RERANK_DEVICE", "  CUDA:1 ")
    assert R._device_setting() == "cuda:1"
    monkeypatch.setenv("LOCI_RERANK_DEVICE", "   ")
    assert R._device_setting() == "auto"


# ------------------------------------------------------------------ the loader, against a fake machine

class Machine:
    """Fake torch + sentence_transformers. ``free`` is MiB per card and can change between loads; ``fail`` lists
    cuda devices on which constructing the model raises (CUDA OOM)."""

    def __init__(self, monkeypatch, free, fail=(), names=None):
        self.free, self.fail, self.created, self.cache_clears = list(free), set(fail), [], 0
        self.names = names or [f"Fake GPU {i}" for i in range(len(free))]
        machine = self
        torch = types.ModuleType("torch")
        torch.__version__ = "2.11.0+cu128"
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: len(machine.free) > 0, device_count=lambda: len(machine.free),
            get_device_name=lambda i: machine.names[i], mem_get_info=lambda i: (machine.free[i] * MIB, 11264 * MIB),
            empty_cache=self._clear)
        st = types.ModuleType("sentence_transformers")

        class CrossEncoder:
            def __init__(self, name, max_length, device):
                machine.created.append(device)
                if device in machine.fail:
                    raise RuntimeError(f"CUDA out of memory on {device}")
        st.CrossEncoder = CrossEncoder
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "sentence_transformers", st)

    def _clear(self):
        self.cache_clears += 1


@pytest.fixture
def machine(monkeypatch):
    for name, value in (("_model", None), ("_model_name", None), ("_device", None), ("_torch_build", None), ("_placement", {})):
        monkeypatch.setattr(R, name, value)
    monkeypatch.setattr(R, "_latch", R.RetryLatch("reranker", base_s=30.0, cap_s=600.0, clock=lambda: 0.0,
                                                  wall=lambda: 1_700_000_000.0))
    for var in ("RERANK_HTTP_URL", "LOCI_RERANK_DEVICE", "LOCI_RERANK_MIN_FREE_MB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RERANK_MODEL", "some/reranker")
    return lambda free, **kw: Machine(monkeypatch, free, **kw)


def test_the_model_lands_on_the_card_with_the_most_free_vram(machine):
    m = machine([2700, 7700])                     # the real situation: the 4070 Ti nearly full, the 2080 Ti roomy
    assert R._get_model() is not None
    assert m.created == ["cuda:1"]


def test_health_reports_the_card_the_model_landed_on(machine):
    machine([2700, 7700], names=["RTX A", "RTX B"])
    R._get_model()
    h = R.health()
    assert (h["state"], h["device"], h["device_index"], h["device_name"]) == ("loaded", "cuda", 1, "RTX B")
    assert h["placement"]["free_mb_at_load"] == 7700
    assert h["placement"]["reason"] == "most free VRAM of: cuda:0 RTX A 2700 MiB free, cuda:1 RTX B 7700 MiB free"
    assert [c["free_mb"] for c in h["placement"]["cards_seen"]] == [2700, 7700]


def test_a_failed_load_on_the_best_card_fails_over_to_the_next(machine, caplog):
    m = machine([2700, 7700], fail={"cuda:1"})
    with caplog.at_level(logging.WARNING, logger="reranker"):
        assert R._get_model() is not None
    assert m.created == ["cuda:1", "cuda:0"] and m.cache_clears == 1
    assert R.health()["device_index"] == 0
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert msgs == ["Reranker could not load some/reranker on cuda:1 (Fake GPU 1, 7700 MiB free at the time): "
                    "RuntimeError: CUDA out of memory on cuda:1; trying the next card"]


def test_when_every_qualifying_card_fails_it_does_not_quietly_become_cpu(machine):
    m = machine([2700, 7700], fail={"cuda:0", "cuda:1"})
    assert R._get_model() is None
    assert m.created == ["cuda:1", "cuda:0"]                   # no cpu attempt
    h = R.health()
    assert h["state"] == "unavailable" and h["latch"]["ok"] is False and h["latch"]["attempts"] == 1
    assert h["latch"]["last_error"] == "RuntimeError: CUDA out of memory on cuda:0"       # retried later by the latch


def test_when_no_card_has_room_it_runs_on_cpu_and_says_why(machine, caplog):
    m = machine([900, 1200])
    with caplog.at_level(logging.WARNING, logger="reranker"):
        assert R._get_model() is not None
    assert m.created == ["cpu"]
    h = R.health()
    assert (h["device"], h["device_index"]) == ("cpu", None)
    assert h["placement"]["reason"] == "no card has 2600 MiB free; cards: cuda:0 Fake GPU 0 900 MiB free, cuda:1 Fake GPU 1 1200 MiB free"
    assert any("no card has 2600 MiB free" in r.getMessage() for r in caplog.records)


def test_explicit_cpu_in_the_loader_is_a_choice_with_its_reason(machine, monkeypatch):
    monkeypatch.setenv("LOCI_RERANK_DEVICE", "cpu")
    m = machine([2700, 7700])
    R._get_model()
    assert m.created == ["cpu"] and R.health()["placement"]["reason"] == "LOCI_RERANK_DEVICE=cpu"


def test_an_explicit_card_overrides_the_ranking(machine, monkeypatch):
    monkeypatch.setenv("LOCI_RERANK_DEVICE", "cuda:0")
    m = machine([2700, 7700])
    R._get_model()
    assert m.created == ["cuda:0"]


def test_the_minimum_free_threshold_is_configurable(machine, monkeypatch):
    monkeypatch.setenv("LOCI_RERANK_MIN_FREE_MB", "8000")
    m = machine([2700, 7700])
    R._get_model()
    assert m.created == ["cpu"]


def test_free_memory_is_read_again_at_every_load(machine, monkeypatch):
    m = machine([8000, 2700])
    R._get_model()
    assert m.created == ["cuda:0"]
    monkeypatch.setattr(R, "_model", None)           # a fresh load (restart / reconfigure)
    m.free = [2700, 8000]                            # Ollama loaded something onto card 0 meanwhile
    R._get_model()
    assert m.created == ["cuda:0", "cuda:1"]

"""A pool entry's ``gpu`` is the model's home card, and every Ollama call for that model carries it.

2026-10-05, measured on the Windows host (RTX 4070 Ti = Ollama index 0, RTX 2080 Ti = index 1; nvidia-smi
numbers them the other way round): a request's ``options.main_gpu`` decides which card Ollama loads a model
onto, and a loaded model is reloaded (~9 s) when a request's options differ. So placement has to be sent by
every caller of a pinned model, including the lease that reloads it, or the model flaps between cards.
"""
import sys
import types
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import llm_local as L  # noqa: E402
import model_lease  # noqa: E402
import model_pool as M  # noqa: E402
from memcheck import llm as memcheck_llm  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    M.clear_cache()
    yield
    M.clear_cache()


def _config(monkeypatch, pool):
    monkeypatch.setattr(M, "_config", lambda: {"models": {"pool": pool}})


POOL = [
    {"name": "spec:Q4_K_M", "roles": ["gen"], "gpu": 1},
    {"name": "home:latest", "roles": ["gen"], "gpu": 0},
    {"name": "free:Q4_K_M", "roles": ["gen"]},
]


class TestSchema:
    @pytest.mark.parametrize("value,expected", [
        (0, 0), (1, 1), ("1", 1), (" 0 ", 0), (7, 7),
    ])
    def test_a_whole_number_is_a_home_card(self, monkeypatch, value, expected):
        _config(monkeypatch, [{"name": "a", "roles": ["gen"], "gpu": value}])
        assert [e.gpu for e in M.entries()] == [expected]

    @pytest.mark.parametrize("value", [-1, True, False, 1.5, "x", "", "-1", None, [1], {"id": 1}])
    def test_anything_else_is_no_pin_and_never_drops_the_entry(self, monkeypatch, value):
        _config(monkeypatch, [{"name": "a", "roles": ["gen"], "gpu": value}])
        assert [(e.name, e.gpu) for e in M.entries()] == [("a", None)]

    def test_an_entry_without_gpu_is_unpinned(self, monkeypatch):
        _config(monkeypatch, [{"name": "a", "roles": ["gen"]}])
        assert [e.gpu for e in M.entries()] == [None]


class TestHomeGpuLookup:
    def test_a_pinned_model_reports_its_card_and_options(self, monkeypatch):
        _config(monkeypatch, POOL)
        assert (M.home_gpu("spec:Q4_K_M"), M.options_for("spec:Q4_K_M")) == (1, {"main_gpu": 1})

    def test_card_zero_is_a_card_not_no_pin(self, monkeypatch):
        _config(monkeypatch, POOL)
        assert (M.home_gpu("home:latest"), M.options_for("home:latest")) == (0, {"main_gpu": 0})

    def test_an_unpinned_unknown_or_empty_model_gets_no_options(self, monkeypatch):
        _config(monkeypatch, POOL)
        for name in ("free:Q4_K_M", "nothing:here", "", None):
            assert (M.home_gpu(name), M.options_for(name)) == (None, {})

    def test_name_and_name_latest_are_one_model_in_both_directions(self, monkeypatch):
        _config(monkeypatch, POOL)
        assert M.home_gpu("home") == 0                 # asked without the tag, configured with it
        _config(monkeypatch, [{"name": "bare", "roles": ["gen"], "gpu": 1}])
        assert M.home_gpu("bare:latest") == 1          # asked with the tag, configured without it

    def test_a_different_tag_is_a_different_model(self, monkeypatch):
        _config(monkeypatch, POOL)
        assert M.home_gpu("spec:Q8_0") is None

    def test_a_broken_config_gives_no_options(self, monkeypatch):
        def boom():
            raise RuntimeError("config unreadable")
        monkeypatch.setattr(M, "entries", boom)
        assert M.options_for("spec:Q4_K_M") == {}


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _post_capture(monkeypatch):
    seen = []

    def fake_post(url, json=None, timeout=None):  # noqa: A002 - mirror requests' kwarg
        seen.append(json)
        return _Resp({"response": "ok"})

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=fake_post))
    monkeypatch.setattr(L, "_gen_env", lambda: "http://fake-ollama:11434")
    return seen


class TestLlmLocal:
    def test_a_pinned_model_is_sent_with_its_card_and_keeps_its_other_options(self, monkeypatch):
        _config(monkeypatch, POOL)
        seen = _post_capture(monkeypatch)
        out = L.generate("hello", model="spec:Q4_K_M", max_tokens=33, temperature=0.5)
        assert out["ok"] is True
        assert seen[0]["options"] == {"num_predict": 33, "temperature": 0.5, "main_gpu": 1}

    def test_an_unpinned_model_is_sent_exactly_as_before(self, monkeypatch):
        """Positive twin: same call, no pin, so no placement key."""
        _config(monkeypatch, POOL)
        seen = _post_capture(monkeypatch)
        L.generate("hello", model="free:Q4_K_M", max_tokens=33, temperature=0.5)
        assert seen[0]["options"] == {"num_predict": 33, "temperature": 0.5}

    def test_card_zero_is_sent(self, monkeypatch):
        _config(monkeypatch, POOL)
        seen = _post_capture(monkeypatch)
        L.generate("hello", model="home:latest")
        assert seen[0]["options"]["main_gpu"] == 0

    def test_the_supervisor_call_is_placed_too(self, monkeypatch):
        _config(monkeypatch, [{"name": "sup:Q4", "roles": ["gen"], "gpu": 1}])
        import backends
        monkeypatch.setattr(backends, "cloud_tier_enabled", lambda: True)
        monkeypatch.setattr(backends, "cloud_supervisor_model", lambda: "sup:Q4")
        seen = _post_capture(monkeypatch)
        monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(
            post=lambda url, json=None, timeout=None: (seen.append(json), _Resp({"response": "{}"}))[1]))
        L._supervisor_route("route this", fmt=None, max_tokens=100)
        assert seen[0]["model"] == "sup:Q4"
        assert seen[0]["options"] == {"num_predict": 100, "temperature": 0.0, "main_gpu": 1}


class TestModelLease:
    def _load(self, monkeypatch, model):
        _config(monkeypatch, POOL)
        sent = []
        monkeypatch.setattr(model_lease, "_http_json", lambda url, body=None, timeout=5.0: (sent.append((url, body)), {})[1])
        assert model_lease.load("http://ollama", model) is True
        return sent[0]

    def test_a_pinned_model_is_reloaded_onto_its_card(self, monkeypatch):
        url, body = self._load(monkeypatch, "spec:Q4_K_M")
        assert url == "http://ollama/api/generate"
        assert body["options"] == {"num_predict": 0, "main_gpu": 1}

    def test_an_unpinned_model_is_reloaded_as_before(self, monkeypatch):
        _, body = self._load(monkeypatch, "free:Q4_K_M")
        assert body["options"] == {"num_predict": 0}


class TestMemcheck:
    def _call(self, monkeypatch, model, options=None):
        _config(monkeypatch, POOL)
        sent = []
        monkeypatch.setattr(memcheck_llm, "_ollama_gen_base", lambda: "http://ollama")
        monkeypatch.setattr(memcheck_llm, "_post_json", lambda url, payload, headers, timeout: (sent.append(payload), {"response": "x"})[1])
        monkeypatch.setattr(memcheck_llm, "_warn_if_truncated", lambda *a, **k: None)
        memcheck_llm._call_ollama("q", False, 5.0, model=model, options=options)
        return sent[0]

    def test_a_pinned_model_gets_its_card_beside_the_callers_options(self, monkeypatch):
        assert self._call(monkeypatch, "spec:Q4_K_M", {"temperature": 0})["options"] == {"temperature": 0, "main_gpu": 1}

    def test_a_pinned_model_with_no_caller_options_still_gets_its_card(self, monkeypatch):
        assert self._call(monkeypatch, "spec:Q4_K_M")["options"] == {"main_gpu": 1}

    def test_an_unpinned_model_keeps_the_payload_it_had(self, monkeypatch):
        assert "options" not in self._call(monkeypatch, "free:Q4_K_M")
        assert self._call(monkeypatch, "free:Q4_K_M", {"temperature": 0})["options"] == {"temperature": 0}

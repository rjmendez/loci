"""loci.policy / loci.model_selector: a bad tier-map env or a failing pool lookup is reported, not silent.

Every return value stays what it was (the fallback map); only the log line and the counter are new.
"""
from __future__ import annotations

import json
import logging
import sys
import types

import pytest

from loci import model_selector, policy
from loci.policy import PolicyTier, TIER_MODEL_MAP, get_proxy_tier_model_map, get_tier_model_map


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(policy, "_DEGRADED", {})
    monkeypatch.setattr(policy, "_DEGRADED_SEEN", set())
    monkeypatch.setattr(model_selector, "_POOL_FAILURES", {})
    monkeypatch.setattr(model_selector, "_POOL_FAILURES_SEEN", set())
    monkeypatch.delenv("LOCI_TIER_MODEL_MAP", raising=False)
    monkeypatch.delenv("LOCI_LITELLM_TIER_MAP", raising=False)


def _warnings(caplog, name="loci.policy"):
    return [r.getMessage() for r in caplog.records if r.name == name and r.levelno == logging.WARNING]


def test_unparseable_tier_map_warns_once_counts_every_time_and_keeps_the_fallback(monkeypatch, caplog):
    monkeypatch.setenv("LOCI_TIER_MODEL_MAP", "{not valid json}")
    with caplog.at_level(logging.WARNING, logger="loci.policy"):
        assert get_tier_model_map() == TIER_MODEL_MAP
        assert get_tier_model_map() == TIER_MODEL_MAP
    assert _warnings(caplog) == [
        "LOCI_TIER_MODEL_MAP is not a JSON object of tier -> models "
        "(JSONDecodeError('Expecting property name enclosed in double quotes: line 1 column 2 (char 1)')); "
        "ignoring it and using the built-in fallback map. Value: '{not valid json}'"
    ]
    assert policy.policy_degraded_counts() == {"LOCI_TIER_MODEL_MAP_unparseable": 2}


def test_a_different_bad_value_is_logged_again(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="loci.policy"):
        monkeypatch.setenv("LOCI_TIER_MODEL_MAP", "{a")
        get_tier_model_map()
        monkeypatch.setenv("LOCI_TIER_MODEL_MAP", "{b")
        get_tier_model_map()
    assert len(_warnings(caplog)) == 2
    assert policy.policy_degraded_counts() == {"LOCI_TIER_MODEL_MAP_unparseable": 2}


def test_json_that_is_not_an_object_is_reported(monkeypatch, caplog):
    monkeypatch.setenv("LOCI_TIER_MODEL_MAP", "[1, 2]")
    with caplog.at_level(logging.WARNING, logger="loci.policy"):
        assert get_tier_model_map() == TIER_MODEL_MAP
    assert _warnings(caplog) == [
        "LOCI_TIER_MODEL_MAP is not a JSON object of tier -> models "
        "(AttributeError(\"'list' object has no attribute 'items'\")); "
        "ignoring it and using the built-in fallback map. Value: '[1, 2]'"
    ]


def test_map_with_only_unknown_tiers_is_reported(monkeypatch, caplog):
    monkeypatch.setenv("LOCI_TIER_MODEL_MAP", json.dumps({"ultra": ["x"], "mega": ["y"]}))
    with caplog.at_level(logging.WARNING, logger="loci.policy"):
        assert get_tier_model_map() == TIER_MODEL_MAP
    assert _warnings(caplog) == [
        "LOCI_TIER_MODEL_MAP names no valid tier (['mega', 'ultra']); "
        "ignoring it and using the built-in fallback map."
    ]
    assert policy.policy_degraded_counts() == {"LOCI_TIER_MODEL_MAP_no_valid_tier": 1}


def test_valid_map_is_used_with_no_warning_and_no_count(monkeypatch, caplog):
    monkeypatch.setenv("LOCI_TIER_MODEL_MAP", json.dumps({"light": ["tiny-a"], "ultra": ["ignored"]}))
    with caplog.at_level(logging.WARNING, logger="loci.policy"):
        assert get_tier_model_map() == {PolicyTier.LIGHT: ["tiny-a"]}
    assert _warnings(caplog) == []
    assert policy.policy_degraded_counts() == {}


def test_unset_env_is_the_normal_default_not_a_degraded_mode(caplog):
    with caplog.at_level(logging.WARNING, logger="loci.policy"):
        assert get_tier_model_map() == TIER_MODEL_MAP
        assert get_proxy_tier_model_map() == TIER_MODEL_MAP
    assert _warnings(caplog) == []
    assert policy.policy_degraded_counts() == {}


def test_unparseable_litellm_map_names_its_own_variable_and_falls_back_to_the_ordinary_map(monkeypatch, caplog):
    monkeypatch.setenv("LOCI_TIER_MODEL_MAP", json.dumps({"light": ["ordinary"]}))
    monkeypatch.setenv("LOCI_LITELLM_TIER_MAP", "nope")
    with caplog.at_level(logging.WARNING, logger="loci.policy"):
        assert get_proxy_tier_model_map() == {PolicyTier.LIGHT: ["ordinary"]}
    assert _warnings(caplog) == [
        "LOCI_LITELLM_TIER_MAP is not a JSON object of tier -> models "
        "(JSONDecodeError('Expecting value: line 1 column 1 (char 0)')); "
        "ignoring it and using the built-in fallback map. Value: 'nope'"
    ]
    assert policy.policy_degraded_counts() == {"LOCI_LITELLM_TIER_MAP_unparseable": 1}


def test_valid_litellm_map_wins_silently(monkeypatch, caplog):
    monkeypatch.setenv("LOCI_LITELLM_TIER_MAP", json.dumps({"light": ["proxy-small"]}))
    with caplog.at_level(logging.WARNING, logger="loci.policy"):
        assert get_proxy_tier_model_map() == {PolicyTier.LIGHT: ["proxy-small"]}
    assert _warnings(caplog) == []


def test_unimportable_model_selector_is_reported(monkeypatch, caplog):
    monkeypatch.setenv("LOCI_TIER_MODEL_MAP", json.dumps({"light": ["map-model"]}))
    monkeypatch.setitem(sys.modules, "loci.model_selector", None)
    with caplog.at_level(logging.WARNING, logger="loci.policy"):
        assert policy._tier_candidates(PolicyTier.LIGHT) == ["map-model"]
        assert policy._tier_candidates(PolicyTier.LIGHT) == ["map-model"]
    assert _warnings(caplog) == [
        "loci.model_selector is not importable "
        "(ModuleNotFoundError('import of loci.model_selector halted; None in sys.modules')); "
        "tier candidates are the policy map only"
    ]
    assert policy.policy_degraded_counts() == {"model_selector_unimportable": 2}


# ---- model_selector._pool_best -------------------------------------------------------------------

class FakePool(types.ModuleType):
    """Recording stand-in for mcp/model_pool.py."""

    def __init__(self, chosen: str = "", error: Exception | None = None):
        super().__init__("model_pool")
        self.asked: list = []
        self._chosen = chosen
        self._error = error

    def rank_role(self, role):
        self.asked.append(role)
        if self._error is not None:
            raise self._error
        return types.SimpleNamespace(chosen=self._chosen)


def test_pool_exception_warns_once_per_role_and_counts(monkeypatch, caplog):
    pool = FakePool(error=RuntimeError("backends.toml unreadable"))
    monkeypatch.setitem(sys.modules, "model_pool", pool)
    with caplog.at_level(logging.INFO, logger="loci.model_selector"):
        assert model_selector._pool_best("gen") == ""
        assert model_selector._pool_best("gen") == ""
        assert model_selector._pool_best("code") == ""
    assert pool.asked == ["gen", "gen", "code"]
    assert [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "loci.model_selector"] == [
        (logging.WARNING, "model pool lookup for role 'gen' failed: RuntimeError('backends.toml unreadable'); "
                          "candidates fall back to the policy map"),
        (logging.WARNING, "model pool lookup for role 'code' failed: RuntimeError('backends.toml unreadable'); "
                          "candidates fall back to the policy map"),
    ]
    assert model_selector.pool_failure_counts() == {"gen": 2, "code": 1}


def test_unimportable_pool_is_info_not_warning(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "model_pool", None)
    with caplog.at_level(logging.INFO, logger="loci.model_selector"):
        assert model_selector._pool_best("gen") == ""
    assert [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "loci.model_selector"] == [
        (logging.INFO, "model pool lookup for role 'gen' failed: "
                       "ModuleNotFoundError('import of model_pool halted; None in sys.modules'); "
                       "candidates fall back to the policy map"),
    ]
    assert model_selector.pool_failure_counts() == {"gen": 1}


def test_working_pool_returns_its_pick_with_no_log_and_no_count(monkeypatch, caplog):
    pool = FakePool(chosen="qwen-local:8b")
    monkeypatch.setitem(sys.modules, "model_pool", pool)
    with caplog.at_level(logging.INFO, logger="loci.model_selector"):
        assert model_selector._pool_best("gen") == "qwen-local:8b"
    assert [r for r in caplog.records if r.name == "loci.model_selector"] == []
    assert model_selector.pool_failure_counts() == {}


def test_pool_with_no_pick_is_empty_without_a_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "model_pool", FakePool(chosen=""))
    assert model_selector._pool_best("gen") == ""
    assert model_selector.pool_failure_counts() == {}

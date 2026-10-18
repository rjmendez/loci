"""Tier -> model candidates (loci.model_selector) and the policy functions built on them.

The model pool is the dependency: every test installs a recording fake ``model_pool`` module, so
what a tier resolves to is decided by the fake's roles, never by the machine's backends.toml.
Models named below are fixture values the fake hands back; the code under test names none.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loci import model_selector
from loci.policy import (
    PolicyTier,
    TIER_MODEL_MAP,
    downgrade_model_to_tier,
    resolve_model_for_tier,
)


class FakePool:
    """Stand-in for mcp/model_pool.py: roles -> chosen tag, every lookup recorded."""

    def __init__(self, roles=None, explode=False):
        self.roles = dict(roles or {})
        self.explode = explode
        self.lookups = []
        self.pick_calls = []     # pick() logs a routing decision; the selector must never call it

    def rank_role(self, role, **_):
        self.lookups.append(role)
        if self.explode:
            raise RuntimeError("inventory unreachable")
        return types.SimpleNamespace(role=role, chosen=self.roles.get(role, ""))

    def pick(self, role, hint=""):
        self.pick_calls.append(role)
        return self.roles.get(role, "")


@pytest.fixture
def pool(monkeypatch):
    def install(roles=None, explode=False):
        fake = FakePool(roles, explode)
        monkeypatch.setitem(sys.modules, "model_pool", fake)
        return fake
    monkeypatch.delenv("LOCI_TIER_PREFER_LOCAL", raising=False)
    monkeypatch.delenv("LOCI_TIER_MODEL_MAP", raising=False)
    return install


# ---- local_model: which pool role serves a tier ------------------------------------------------

def test_tier_role_wins_over_the_general_role(pool):
    pool({"tier_standard": "pinned:1b", "gen": "general:4b"})
    assert model_selector.local_model("standard") == "ollama/pinned:1b"


def test_general_role_serves_a_tier_with_no_tier_role(pool):
    pool({"gen": "general:4b"})
    assert model_selector.local_model("standard") == "ollama/general:4b"
    assert model_selector.local_model("light") == "ollama/general:4b"


@pytest.mark.parametrize("tier,role", [
    ("nano", "decide"), ("light", "gen"), ("standard", "gen"), ("medium", "gen"),
    ("turbo", "gen_large"), ("heavy", "gen_large"), ("research", "gen_large"),
])
def test_each_tier_falls_back_to_its_general_role(pool, tier, role):
    pool({role: "m:1"})
    assert model_selector.local_model(tier) == "ollama/m:1"


def test_task_type_uses_the_role_that_serves_it(pool):
    pool({"security": "sec:7b", "gen": "general:4b"})
    assert model_selector.local_model("standard", "security_review") == "ollama/sec:7b"
    assert model_selector.local_model("standard", "no_such_task") == "ollama/general:4b"


def test_task_type_without_a_pooled_role_falls_to_the_tier(pool):
    pool({"gen": "general:4b"})
    assert model_selector.local_model("standard", "security_review") == "ollama/general:4b"


def test_zero_tier_has_no_local_model_and_asks_the_pool_nothing(pool):
    fake = pool({"gen": "general:4b", "decide": "d:1"})
    assert model_selector.local_model("zero") == ""
    assert fake.lookups == []


def test_selector_never_calls_pick(pool):
    fake = pool({"gen": "general:4b"})
    model_selector.select_model("standard")
    assert fake.lookups == ["tier_standard", "gen"]
    assert fake.pick_calls == []


# ---- fail-open -----------------------------------------------------------------------------------

def test_no_pool_means_the_policy_map_unchanged(pool):
    pool({})
    assert model_selector.candidates("standard") == TIER_MODEL_MAP[PolicyTier.STANDARD]
    assert model_selector.select_model("standard") == "claude-sonnet-4.6"


def test_pool_error_falls_back_to_the_policy_map_and_a_working_pool_does_not(pool):
    pool({"gen": "general:4b"}, explode=True)
    assert model_selector.select_model("standard") == "claude-sonnet-4.6"
    pool({"gen": "general:4b"})
    assert model_selector.select_model("standard") == "ollama/general:4b"


def test_model_pool_not_importable_falls_back_to_the_policy_map(monkeypatch):
    monkeypatch.setitem(sys.modules, "model_pool", None)
    monkeypatch.delenv("LOCI_TIER_PREFER_LOCAL", raising=False)
    monkeypatch.delenv("LOCI_TIER_MODEL_MAP", raising=False)
    assert model_selector.select_model("heavy") == "claude-opus-4.8"


# ---- ordering --------------------------------------------------------------------------------------

def test_local_leads_by_default_then_the_policy_map_in_its_own_order(pool):
    pool({"gen": "general:4b"})
    assert model_selector.candidates("light") == [
        "ollama/general:4b", "claude-haiku-4.5", "gpt-5-mini", "mistral-7b-instruct"]


def test_prefer_local_false_puts_cloud_first_and_keeps_the_local_pick_reachable(pool):
    pool({"gen": "general:4b"})
    assert model_selector.candidates("light", prefer_local=False) == [
        "claude-haiku-4.5", "gpt-5-mini", "mistral-7b-instruct", "ollama/general:4b"]
    assert model_selector.select_model("light", prefer_local=False) == "claude-haiku-4.5"
    assert model_selector.select_model("light", prefer_local=True) == "ollama/general:4b"


def test_env_switch_turns_local_preference_off_and_on(pool, monkeypatch):
    pool({"gen": "general:4b"})
    monkeypatch.setenv("LOCI_TIER_PREFER_LOCAL", "0")
    assert model_selector.select_model("light") == "claude-haiku-4.5"
    monkeypatch.setenv("LOCI_TIER_PREFER_LOCAL", "1")
    assert model_selector.select_model("light") == "ollama/general:4b"


def test_a_pool_pick_already_in_the_map_is_listed_once(pool):
    pool({"decide": "llama3.2:3b"})
    assert model_selector.candidates("nano") == [
        "ollama/llama3.2:3b", "ollama/mistral:7b", "ollama/phi3:mini"]


def test_policy_env_override_is_the_cloud_side(pool, monkeypatch):
    pool({"gen": "general:4b"})
    monkeypatch.setenv("LOCI_TIER_MODEL_MAP", '{"standard": ["cloud-x", "cloud-y"]}')
    assert model_selector.candidates("standard") == ["ollama/general:4b", "cloud-x", "cloud-y"]
    assert model_selector.candidates("standard", prefer_local=False) == ["cloud-x", "cloud-y", "ollama/general:4b"]


# ---- errors ------------------------------------------------------------------------------------------

def test_zero_tier_is_refused(pool):
    pool({"gen": "general:4b"})
    with pytest.raises(ValueError, match="ZERO forbids LLM calls"):
        model_selector.select_model("zero")
    assert model_selector.candidates("zero") == []


def test_unknown_tier_is_refused_by_name(pool):
    pool({})
    with pytest.raises(ValueError, match="unknown policy tier 'platinum'"):
        model_selector.select_model("platinum")


# ---- the policy functions that use it ------------------------------------------------------------------

def test_downgrade_substitutes_the_pools_local_pick(pool):
    pool({"gen": "general:4b"})
    assert downgrade_model_to_tier("claude-opus-4.8", PolicyTier.LIGHT) == ("ollama/general:4b", True)


def test_downgrade_without_a_pool_substitutes_the_maps_first_model(pool):
    pool({})
    assert downgrade_model_to_tier("claude-opus-4.8", PolicyTier.LIGHT) == ("claude-haiku-4.5", True)


def test_downgrade_keeps_a_model_already_at_or_below_the_tier(pool):
    pool({"gen": "general:4b"})
    assert downgrade_model_to_tier("claude-haiku-4.5", PolicyTier.LIGHT) == ("claude-haiku-4.5", False)
    assert downgrade_model_to_tier("claude-haiku-4.5", PolicyTier.HEAVY) == ("claude-haiku-4.5", False)


def test_downgrade_treats_the_pools_own_model_as_belonging_to_its_tier(pool):
    pool({"gen": "general:4b", "gen_large": "big:27b"})
    assert downgrade_model_to_tier("ollama/general:4b", PolicyTier.STANDARD) == ("ollama/general:4b", False)
    assert downgrade_model_to_tier("ollama/big:27b", PolicyTier.STANDARD) == ("ollama/general:4b", True)


def test_downgrade_substitutes_a_model_no_tier_lists(pool):
    pool({"gen_large": "big:27b"})
    assert downgrade_model_to_tier("mystery-model", PolicyTier.HEAVY) == ("ollama/big:27b", True)
    pool({})
    assert downgrade_model_to_tier("mystery-model", PolicyTier.HEAVY) == ("claude-opus-4.8", True)


def test_downgrade_at_zero_leaves_the_request_alone(pool):
    pool({"gen": "general:4b"})
    assert downgrade_model_to_tier("claude-opus-4.8", PolicyTier.ZERO) == ("claude-opus-4.8", False)


def test_resolve_model_for_tier_leads_with_the_pool_and_honours_available(pool):
    pool({"decide": "tiny:1b"})
    assert resolve_model_for_tier(PolicyTier.NANO) == "ollama/tiny:1b"
    assert resolve_model_for_tier(PolicyTier.NANO, available=["ollama/phi3:mini"]) == "ollama/phi3:mini"
    assert resolve_model_for_tier(PolicyTier.NANO, available=["nothing-matches"]) == "ollama/tiny:1b"


def test_resolve_model_for_tier_zero_is_refused(pool):
    pool({"decide": "tiny:1b"})
    with pytest.raises(ValueError, match="PolicyTier.ZERO forbids LLM calls"):
        resolve_model_for_tier(PolicyTier.ZERO)

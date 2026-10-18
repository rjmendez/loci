"""loci/model_selector.py: a chat request never resolves to the SystemOne ``decide`` classifier.

The ``decide`` pool role holds a 0.8B decision model: it scores answer letters and cannot write a reply. Until
2026-10-18 tier ``nano`` fell back to it and task type ``triage`` mapped to it, so a nano or triage chat request was
rewritten to that classifier. These tests pin the exact roles now tried, and that the ``decide`` role is skipped even
if a table lists it. The pool is a recording fake; nothing reads backends.toml or the machine's models.

Mutation log (each applied to loci/model_selector.py; the named tests went red; then reverted):
- _TIER_ROLES["nano"] back to ("tier_nano", "decide"): test_nano_tries_its_tier_role_then_gen_and_never_the_decision_model,
  test_a_decision_model_alone_serves_no_tier
- _TASK_ROLES gets "triage": "decide" back: test_triage_is_not_special_cased_to_the_decision_model
- the _NON_GENERATING_ROLES skip removed: test_a_role_that_does_not_generate_is_skipped_wherever_a_table_lists_it
"""
from __future__ import annotations

import sys
import types

import pytest

from loci import model_selector


class FakePool:
    """Stand-in for mcp/model_pool.py: role -> chosen tag, every lookup recorded in order."""

    def __init__(self, roles):
        self.roles = dict(roles)
        self.lookups = []

    def rank_role(self, role, **_):
        self.lookups.append(role)
        return types.SimpleNamespace(role=role, chosen=self.roles.get(role, ""))


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.delenv("LOCI_TIER_PREFER_LOCAL", raising=False)
    monkeypatch.delenv("LOCI_TIER_MODEL_MAP", raising=False)

    def install(roles):
        fake = FakePool(roles)
        monkeypatch.setitem(sys.modules, "model_pool", fake)
        return fake
    return install


def test_nano_tries_its_tier_role_then_gen_and_never_the_decision_model(pool):
    fake = pool({"decide": "sys1:0.8b", "gen": "writer:3b"})
    assert model_selector.local_model("nano") == "ollama/writer:3b"
    assert fake.lookups == ["tier_nano", "gen"]


def test_an_operator_pin_on_the_tier_role_wins_over_gen(pool):
    fake = pool({"tier_nano": "pinned:1b", "decide": "sys1:0.8b", "gen": "writer:3b"})
    assert model_selector.local_model("nano") == "ollama/pinned:1b"
    assert fake.lookups == ["tier_nano"]


def test_a_decision_model_alone_serves_no_tier(pool):
    for tier in sorted(model_selector._TIER_ROLES):
        fake = pool({"decide": "sys1:0.8b"})
        assert model_selector.local_model(tier) == "", tier
        assert "decide" not in fake.lookups, tier


def test_a_decision_model_alone_leaves_nano_with_the_policy_map_only(pool):
    pool({"decide": "sys1:0.8b"})
    assert model_selector.candidates("nano") == [
        "ollama/llama3.2:3b", "ollama/mistral:7b", "ollama/phi3:mini"]
    pool({"gen": "writer:3b"})                                      # twin: with a generation model the pool leads
    assert model_selector.candidates("nano") == [
        "ollama/writer:3b", "ollama/llama3.2:3b", "ollama/mistral:7b", "ollama/phi3:mini"]


def test_triage_is_not_special_cased_to_the_decision_model(pool):
    fake = pool({"decide": "sys1:0.8b", "gen": "writer:3b"})
    assert model_selector.local_model("nano", "triage") == "ollama/writer:3b"
    assert fake.lookups == ["tier_nano", "gen"]
    fake = pool({"code": "coder:7b", "decide": "sys1:0.8b", "gen": "writer:3b"})      # twin: a real task role leads
    assert model_selector.local_model("nano", "code") == "ollama/coder:7b"
    assert fake.lookups == ["code"]


def test_a_role_that_does_not_generate_is_skipped_wherever_a_table_lists_it(pool, monkeypatch):
    monkeypatch.setitem(model_selector._TIER_ROLES, "nano", ("tier_nano", "decide", "gen"))
    fake = pool({"decide": "sys1:0.8b", "gen": "writer:3b"})
    assert model_selector.local_model("nano") == "ollama/writer:3b"
    assert fake.lookups == ["tier_nano", "gen"]
    monkeypatch.setitem(model_selector._TIER_ROLES, "nano", ("tier_nano", "gen_large", "gen"))
    fake = pool({"gen_large": "big:14b", "gen": "writer:3b"})       # twin: a generating role in the same slot is used
    assert model_selector.local_model("nano") == "ollama/big:14b"
    assert fake.lookups == ["tier_nano", "gen_large"]


def test_a_task_table_entry_for_the_decision_model_is_skipped_too(pool, monkeypatch):
    monkeypatch.setitem(model_selector._TASK_ROLES, "classify", "decide")
    fake = pool({"decide": "sys1:0.8b", "gen": "writer:3b"})
    assert model_selector.local_model("light", "classify") == "ollama/writer:3b"
    assert fake.lookups == ["tier_light", "gen"]

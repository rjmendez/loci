"""downgrade_model_to_tier fails closed when a tier names no model.

Slice S3 of the Tripwire-Governor plan.  Before, an empty candidate list returned the REQUESTED
model unchanged (policy.py ``return requested_model, False``), so a tier with nothing allowed let
any model through.  Now the result is the cheapest allowed model at or below the tier (or "") with
``tier_unresolved=True``.

Candidates come from the LOCI_TIER_MODEL_MAP override; the model pool's local pick is stubbed to
"none served" so the outcome does not depend on the machine.

Mutation runs (each applied to loci/policy.py, confirmed red, reverted):
- final ``return DowngradeResult(_cheapest_allowed(tier), True, True)`` back to
  ``return DowngradeResult(requested_model, False)`` -> every test in TestEmptyCandidates goes red.
- ``_cheapest_allowed`` always returning "" -> test_empty_light_tier_falls_back_to_the_cheapest_allowed_model goes red.
- drop ``tier_unresolved=True`` on the final return -> the tier_unresolved assertions go red.
- ``allowed[0]`` -> ``allowed[-1]`` -> test_non_empty_candidates_return_the_first_unchanged goes red.
- the ``if tier is PolicyTier.ZERO`` branch -> ``if False`` -> the two ZERO pair tests go red (and the
  pinned test_downgrade_at_zero_leaves_the_request_alone in mcp/tests/test_model_selector.py).
- DowngradeResult.usable_model returning ``self.model`` -> the ZERO tier_router and litellm_call tests go red.
- loci/hooks.py: the ``usable_model`` branch -> ``if False`` -> test_tier_router_with_a_zero_tier_returns_no_model goes red.
"""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from loci import model_selector
from loci.hooks import tier_router_surface
from loci.policy import (
    TIER_MODEL_MAP,
    DowngradeResult,
    Policy,
    PolicyBackend,
    PolicyRegion,
    PolicyTier,
    downgrade_model_to_tier,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_pool(monkeypatch):
    """The model pool serves nothing, so candidates are exactly the LOCI_TIER_MODEL_MAP entries."""
    monkeypatch.setattr(model_selector, "_pool_best", lambda role: "")


def _map(monkeypatch, mapping):
    monkeypatch.setenv("LOCI_TIER_MODEL_MAP", json.dumps(mapping))


class TestEmptyCandidates:
    def test_empty_light_tier_falls_back_to_the_cheapest_allowed_model(self, monkeypatch):
        _map(monkeypatch, {"nano": ["small-a", "small-b"], "light": []})
        result = downgrade_model_to_tier("big-x", PolicyTier.LIGHT)
        assert result == ("small-a", True)
        assert result.tier_unresolved is True

    def test_nothing_allowed_at_or_below_the_tier_gives_no_model(self, monkeypatch):
        _map(monkeypatch, {"nano": [], "light": [], "heavy": ["h-1"]})
        result = downgrade_model_to_tier("small-a", PolicyTier.LIGHT)
        assert result == ("", True)
        assert result.tier_unresolved is True
        assert result.usable_model == ""

    def test_an_unresolved_fallback_model_is_usable(self, monkeypatch):
        _map(monkeypatch, {"nano": ["small-a"], "light": []})
        assert downgrade_model_to_tier("big-x", PolicyTier.LIGHT).usable_model == "small-a"

    def test_zero_tier_keeps_the_legacy_pair_but_is_flagged_and_unusable(self, monkeypatch):
        # mcp/tests/test_model_selector.py pins ("claude-opus-4.8", False) for ZERO, so the pair stays;
        # the flag and usable_model are how callers learn the request must not be called.
        _map(monkeypatch, {"nano": ["small-a"]})
        result = downgrade_model_to_tier("small-a", PolicyTier.ZERO)
        assert result == ("small-a", False)
        assert result.tier_unresolved is True
        assert result.usable_model == ""

    @pytest.mark.xfail(strict=True, reason=(
        "ZERO still returns the requested model in the pair (pinned by an existing test in "
        "mcp/tests/test_model_selector.py, which this slice must not edit); callers fail closed "
        "through usable_model instead"))
    def test_zero_tier_returns_no_model_from_the_function_itself(self, monkeypatch):
        _map(monkeypatch, {"nano": ["small-a"]})
        assert downgrade_model_to_tier("small-a", PolicyTier.ZERO) == ("", True)

    def test_a_heavier_tier_is_never_used_to_fill_an_empty_tier(self, monkeypatch):
        _map(monkeypatch, {"light": [], "heavy": ["h-1"]})
        # nothing at or below light: the heavier h-1 must not be chosen, nor the requested model
        result = downgrade_model_to_tier("whatever", PolicyTier.LIGHT)
        assert result == ("", True)
        assert result.tier_unresolved is True

    @pytest.mark.parametrize("unknown", ["bogus", "", None, 7])
    def test_unknown_tier_is_resolved_explicitly_to_the_cheapest_model_anywhere(self, monkeypatch, unknown):
        _map(monkeypatch, {"nano": ["small-a"], "heavy": ["h-1"]})
        result = downgrade_model_to_tier("h-1", unknown)
        assert result == ("small-a", True)
        assert result.tier_unresolved is True

    def test_unknown_tier_name_that_is_a_real_tier_string_is_resolved_normally(self, monkeypatch):
        _map(monkeypatch, {"light": ["l-1"], "nano": ["small-a"]})
        result = downgrade_model_to_tier("big-x", "light")
        assert result == ("l-1", True)
        assert result.tier_unresolved is False


class TestNonEmptyCandidates:
    def test_non_empty_candidates_return_the_first_unchanged(self, monkeypatch):
        _map(monkeypatch, {"light": ["l-1", "l-2"], "nano": ["small-a"]})
        result = downgrade_model_to_tier("big-x", PolicyTier.LIGHT)
        assert result == ("l-1", True)  # rewritten to allowed[0], exactly as before
        assert result.tier_unresolved is False

    def test_a_model_already_within_the_tier_is_returned_unchanged(self, monkeypatch):
        _map(monkeypatch, {"light": ["l-1", "l-2"], "nano": ["small-a"]})
        for model in ("l-2", "small-a"):
            result = downgrade_model_to_tier(model, PolicyTier.LIGHT)
            assert result == (model, False)
            assert result.tier_unresolved is False

    def test_result_unpacks_and_compares_as_the_legacy_pair(self, monkeypatch):
        _map(monkeypatch, {"light": ["l-1"]})
        result = downgrade_model_to_tier("big-x", PolicyTier.LIGHT)
        model, rewritten = result
        assert (model, rewritten) == ("l-1", True)
        assert isinstance(result, DowngradeResult) and isinstance(result, tuple)
        assert (result.model, result.was_rewritten) == ("l-1", True)


# ---------------------------------------------------------------------------
# callers
# ---------------------------------------------------------------------------

def _store_policy(tmp_path, monkeypatch, tier):
    monkeypatch.setenv("LOCI_HOME", str(tmp_path / "loci-home"))
    PolicyBackend().store(Policy(policy_id="pol-fc", title="fail closed", tier=tier, region=PolicyRegion.ROUTING))


def test_tier_router_reports_tier_unresolved_and_passes_no_model_through(tmp_path, monkeypatch):
    _map(monkeypatch, {"nano": ["small-a"], "light": []})
    _store_policy(tmp_path, monkeypatch, PolicyTier.LIGHT)
    out = asyncio.run(tier_router_surface("big-x", [], {"policy_id": "pol-fc"}, {}))
    assert out["model"] == "small-a"
    assert out["downgraded"] is True
    assert out["original_model"] == "big-x"
    assert [(v["type"], v["severity"]) for v in out["violations"]] == [
        ("tier_unresolved", "error"), ("model_downgraded", "warning")]


def test_tier_router_with_a_zero_tier_returns_no_model(tmp_path, monkeypatch):
    _map(monkeypatch, {"nano": ["small-a"]})
    _store_policy(tmp_path, monkeypatch, PolicyTier.ZERO)
    out = asyncio.run(tier_router_surface("small-a", [], {"policy_id": "pol-fc"}, {}))
    assert out["model"] == ""
    assert out["downgraded"] is True
    assert "tier_unresolved" in [v["type"] for v in out["violations"]]


def test_tier_router_resolved_tier_has_no_tier_unresolved_violation(tmp_path, monkeypatch):
    _map(monkeypatch, {"nano": ["small-a"], "light": ["l-1"]})
    _store_policy(tmp_path, monkeypatch, PolicyTier.LIGHT)
    out = asyncio.run(tier_router_surface("big-x", [], {"policy_id": "pol-fc"}, {}))
    assert out["model"] == "l-1"
    assert [v["type"] for v in out["violations"]] == ["model_downgraded"]


def _litellm_tool(monkeypatch):
    spec = importlib.util.spec_from_file_location("litellm_tools_under_test", REPO / "mcp" / "tools" / "litellm_tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Response:
        def to_dict(self):
            return {"content": "ok"}

    calls = []

    def call_litellm_proxy(**kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(module, "call_litellm_proxy", call_litellm_proxy)
    tools = {}

    class Server:
        def tool(self):
            def deco(fn):
                tools[fn.__name__] = fn
                return fn
            return deco

    module.register(Server())
    return tools["litellm_call"], calls


def test_litellm_call_refuses_when_the_tier_allows_no_model(monkeypatch):
    _map(monkeypatch, {"nano": ["small-a"]})
    litellm_call, calls = _litellm_tool(monkeypatch)
    out = json.loads(litellm_call("small-a", [{"role": "user", "content": "hi"}], tier="zero"))
    assert out == {
        "error": "tier zero allows no model; the requested model was not called",
        "model": "",
        "original_model": "small-a",
    }
    assert calls == []


def test_litellm_call_proceeds_with_the_rewritten_model_when_the_tier_resolves(monkeypatch):
    _map(monkeypatch, {"nano": ["small-a"], "light": ["l-1"]})
    litellm_call, calls = _litellm_tool(monkeypatch)
    out = json.loads(litellm_call("big-x", [{"role": "user", "content": "hi"}], tier="light"))
    assert out == {"content": "ok", "model_rewritten": True, "original_model": "big-x"}
    assert [c["model"] for c in calls] == ["l-1"]


# ---------------------------------------------------------------------------
# known gap
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "S3 asks cloud model names to come from pool roles, not literals; TIER_MODEL_MAP still names "
    "them and existing tests pin those names, so this is deferred (see the S3 report)"))
def test_the_compiled_tier_map_names_no_literal_models():
    named = {tier.value: models for tier, models in TIER_MODEL_MAP.items() if models}
    assert named == {}

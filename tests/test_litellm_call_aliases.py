"""litellm_call names only models the LiteLLM proxy serves.

On the live machine litellm_call("simple-tier", tier=medium) came back from the proxy as
``400 Invalid model name passed in model=ollama/gemma4-e4b-hermes:16k``: the tier policy rewrote a valid proxy
group name to the model pool's local pick, which the proxy has never heard of. ``downgrade_model_to_tier(...,
pool=False)`` judges and rewrites against the tier map only (LOCI_TIER_MODEL_MAP lists the proxy's group names).

The pool is stubbed to serve a local model for every role, so the difference between pool=True and pool=False is
visible and does not depend on the machine.

Mutation runs (each applied to the named file, confirmed red, reverted):
- mcp/tools/litellm_tools.py: drop ``pool=False`` -> every end-to-end test below goes red (the proxy is asked for the ollama model)
- loci/policy.py ``_candidates``: ignore ``pool`` (always the pool's candidates) -> the pool=False downgrade tests go red
- loci/policy.py: drop the ``and pool`` guard before the local_model lookup -> test_a_pool_model_is_not_a_tier_member_when_the_pool_is_left_out goes red
- loci/policy.py: pass the default pool through ``_cheapest_allowed`` -> test_an_unresolved_tier_does_not_fall_back_to_the_pool_pick goes red
"""
import json
import sys
from pathlib import Path

import pytest

from loci import model_selector
from loci.policy import PolicyTier, downgrade_model_to_tier

REPO = Path(__file__).resolve().parents[1]
ALIASES = {
    "nano": ["simple-tier"], "light": ["simple-tier"],
    "standard": ["medium-tier"], "medium": ["medium-tier"],
    "turbo": ["complex-tier"], "heavy": ["complex-tier"], "research": ["complex-tier"],
}
POOL_PICK = "ollama/pool-model:8b"


@pytest.fixture(autouse=True)
def pool_serves_a_local_model(monkeypatch):
    monkeypatch.setattr(model_selector, "_pool_best", lambda role: "pool-model:8b")
    monkeypatch.setenv("LOCI_TIER_MODEL_MAP", json.dumps(ALIASES))


class TestDowngradeWithoutThePool:
    def test_a_costlier_alias_is_rewritten_to_the_tiers_alias_not_the_pool_pick(self):
        assert downgrade_model_to_tier("complex-tier", PolicyTier.LIGHT, pool=False) == ("simple-tier", True)

    def test_the_default_still_prefers_the_pool_pick(self):
        # the twin: without pool=False the same request lands on the model only Ollama knows
        assert downgrade_model_to_tier("complex-tier", PolicyTier.LIGHT) == (POOL_PICK, True)

    def test_an_alias_at_or_below_the_tier_is_left_alone(self):
        assert downgrade_model_to_tier("simple-tier", PolicyTier.STANDARD, pool=False) == ("simple-tier", False)
        assert downgrade_model_to_tier("medium-tier", PolicyTier.STANDARD, pool=False) == ("medium-tier", False)

    def test_a_pool_model_is_not_a_tier_member_when_the_pool_is_left_out(self):
        # POOL_PICK is the pool's own light-tier pick: with the pool it counts as light and stays, without it it is unknown
        assert downgrade_model_to_tier(POOL_PICK, PolicyTier.LIGHT) == (POOL_PICK, False)
        assert downgrade_model_to_tier(POOL_PICK, PolicyTier.LIGHT, pool=False) == ("simple-tier", True)

    def test_an_unknown_model_is_rewritten_to_the_tiers_alias(self):
        assert downgrade_model_to_tier("mystery", PolicyTier.MEDIUM, pool=False) == ("medium-tier", True)

    def test_an_unresolved_tier_does_not_fall_back_to_the_pool_pick(self, monkeypatch):
        monkeypatch.setenv("LOCI_TIER_MODEL_MAP", json.dumps({"heavy": ["complex-tier"]}))
        without = downgrade_model_to_tier("complex-tier", PolicyTier.LIGHT, pool=False)
        assert without == ("", True)
        assert without.tier_unresolved is True and without.usable_model == ""
        # twin: with the pool the same request is served by the pool's pick
        assert downgrade_model_to_tier("complex-tier", PolicyTier.LIGHT) == (POOL_PICK, True)


class _Recorder:
    def __init__(self):
        self.models = []

    def __call__(self, model, messages, max_tokens, temperature):
        self.models.append(model)

        class Resp:
            def to_dict(self_inner):
                return {"model": model, "content": "ok"}
        return Resp()


@pytest.fixture
def call(monkeypatch):
    sys.path.insert(0, str(REPO / "mcp"))
    from tools import litellm_tools

    class FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self):
            return lambda fn: self.tools.setdefault(fn.__name__, fn)

    fake = FakeServer()
    litellm_tools.register(fake)
    rec = _Recorder()
    monkeypatch.setattr(litellm_tools, "call_litellm_proxy", rec)
    msgs = [{"role": "user", "content": "hi"}]

    def go(**kw):
        return json.loads(fake.tools["litellm_call"](messages=msgs, **kw)), rec
    return go


class TestLitellmCall:
    def test_a_valid_proxy_group_is_sent_to_the_proxy_unchanged(self, call):
        out, rec = call(model="simple-tier", tier="medium")
        assert rec.models == ["simple-tier"]
        assert "model_rewritten" not in out

    def test_a_costlier_group_is_downgraded_to_a_group_the_proxy_serves(self, call):
        out, rec = call(model="complex-tier", tier="light")
        assert rec.models == ["simple-tier"]
        assert out["model_rewritten"] is True and out["original_model"] == "complex-tier"

    def test_the_pool_pick_is_never_what_the_proxy_is_asked_for(self, call):
        for requested, tier in (("complex-tier", "light"), ("simple-tier", "medium"), ("mystery", "heavy")):
            call(model=requested, tier=tier)
        _, rec = call(model="simple-tier", tier="nano")
        assert POOL_PICK not in rec.models and rec.models == ["simple-tier", "simple-tier", "complex-tier", "simple-tier"]

    def test_a_tier_that_allows_nothing_does_not_call_the_proxy(self, call):
        out, rec = call(model="simple-tier", tier="zero")
        assert rec.models == []
        assert out["error"] == "tier zero allows no model; the requested model was not called"

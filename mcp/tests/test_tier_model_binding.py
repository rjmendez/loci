"""
Tests for cost-driven model routing via PolicyTier (tier→model binding).

Covers:
- resolve_model_for_tier: no rewrite when model is in-tier
- resolve_model_for_tier: downgrade when model is in a higher tier
- resolve_model_for_tier: downgrade when model is unknown
- resolve_model_for_tier: keep cheaper model when model is in a lower tier (positive twin)
- get_tier_model_map: valid LOCI_TIER_MODEL_MAP env override
- get_tier_model_map: invalid JSON falls back to default (positive twin: valid JSON works)
- tier_router_surface: HEAVY model + LIGHT policy → downgraded=True, model rewritten
- tier_router_surface: LIGHT model + LIGHT policy → downgraded=False, model unchanged

Mutations run to confirm each test goes red:
- resolve_model_for_tier no-rewrite: deleted the `tier_order[requested_tier] <= tier_order[tier]`
  guard (always downgrade) → was_rewritten=True, test failed.
- resolve_model_for_tier heavy-downgrade: returned (requested_model, False) unconditionally
  → was_rewritten=False, resolved="claude-opus-4.8", test failed.
- resolve_model_for_tier unknown: returned (requested_model, False) unconditionally
  → was_rewritten=False, test failed.
- get_tier_model_map override: deleted `if override:` block → returned default map, test failed.
- get_tier_model_map invalid fallback: made invalid JSON succeed (removed try/except) → KeyError,
  test failed.
- tier_router_surface heavy+light: stubbed resolve_model_for_tier to return (model, False)
  → downgraded=False, test failed.
- tier_router_surface light+light: stubbed resolve_model_for_tier to return ("claude-haiku-4.5", True)
  → downgraded=True, test failed.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy import (
    PolicyTier,
    TIER_MODEL_MAP,
    get_tier_model_map,
    resolve_model_for_tier,
    Policy,
    PolicyRegion,
)
from hooks import tier_router_surface


# ---------------------------------------------------------------------------
# resolve_model_for_tier
# ---------------------------------------------------------------------------


class TestResolveModelForTier:
    def test_model_already_in_tier_no_rewrite(self):
        """LIGHT model against LIGHT policy → returned unchanged, was_rewritten=False."""
        resolved, was_rewritten = resolve_model_for_tier("claude-haiku-4.5", PolicyTier.LIGHT)
        assert resolved == "claude-haiku-4.5"
        assert was_rewritten is False

    def test_heavy_model_against_light_policy_downgrades_to_first_light(self):
        """HEAVY model against LIGHT policy → downgraded to first LIGHT model."""
        resolved, was_rewritten = resolve_model_for_tier("claude-opus-4.8", PolicyTier.LIGHT)
        assert resolved == "claude-haiku-4.5"
        assert was_rewritten is True

    def test_unknown_model_against_medium_policy_returns_first_medium(self):
        """Unknown model against MEDIUM policy → first MEDIUM model, was_rewritten=True."""
        resolved, was_rewritten = resolve_model_for_tier("gpt-4-turbo-unknown", PolicyTier.MEDIUM)
        assert resolved == "claude-sonnet-4.6"
        assert was_rewritten is True

    def test_light_model_against_medium_policy_kept_as_is(self):
        """LIGHT model against MEDIUM policy → kept unchanged (within budget)."""
        resolved, was_rewritten = resolve_model_for_tier("claude-haiku-4.5", PolicyTier.MEDIUM)
        assert resolved == "claude-haiku-4.5"
        assert was_rewritten is False

    def test_medium_model_against_heavy_policy_kept_as_is(self):
        """MEDIUM model against HEAVY policy → kept unchanged (within budget)."""
        resolved, was_rewritten = resolve_model_for_tier("claude-sonnet-4.6", PolicyTier.HEAVY)
        assert resolved == "claude-sonnet-4.6"
        assert was_rewritten is False

    def test_heavy_model_against_heavy_policy_no_rewrite(self):
        """HEAVY model against HEAVY policy → returned unchanged."""
        resolved, was_rewritten = resolve_model_for_tier("claude-opus-4.8", PolicyTier.HEAVY)
        assert resolved == "claude-opus-4.8"
        assert was_rewritten is False

    def test_unknown_model_against_light_policy_downgrades(self):
        """Unknown model against LIGHT policy → downgraded to first LIGHT model."""
        resolved, was_rewritten = resolve_model_for_tier("some-future-model-v99", PolicyTier.LIGHT)
        assert resolved == "claude-haiku-4.5"
        assert was_rewritten is True

    def test_medium_model_against_light_policy_downgrades(self):
        """MEDIUM model against LIGHT policy → downgraded to first LIGHT model."""
        resolved, was_rewritten = resolve_model_for_tier("claude-sonnet-4.6", PolicyTier.LIGHT)
        assert resolved == "claude-haiku-4.5"
        assert was_rewritten is True


# ---------------------------------------------------------------------------
# get_tier_model_map
# ---------------------------------------------------------------------------


class TestGetTierModelMap:
    def test_default_returns_tier_model_map(self, monkeypatch):
        """Without env override, returns the module-level TIER_MODEL_MAP."""
        monkeypatch.delenv("LOCI_TIER_MODEL_MAP", raising=False)
        result = get_tier_model_map()
        assert result == TIER_MODEL_MAP
        assert "claude-haiku-4.5" in result[PolicyTier.LIGHT]
        assert "claude-opus-4.8" in result[PolicyTier.HEAVY]

    def test_valid_env_override_replaces_map(self, monkeypatch):
        """Valid LOCI_TIER_MODEL_MAP JSON overrides all three tiers."""
        override = {
            "light": ["tiny-model-a", "tiny-model-b"],
            "medium": ["mid-model-x"],
            "heavy": ["big-model-z"],
        }
        monkeypatch.setenv("LOCI_TIER_MODEL_MAP", json.dumps(override))
        result = get_tier_model_map()
        assert result[PolicyTier.LIGHT] == ["tiny-model-a", "tiny-model-b"]
        assert result[PolicyTier.MEDIUM] == ["mid-model-x"]
        assert result[PolicyTier.HEAVY] == ["big-model-z"]

    def test_invalid_json_falls_back_to_default(self, monkeypatch):
        """Invalid JSON in LOCI_TIER_MODEL_MAP silently falls back to TIER_MODEL_MAP."""
        monkeypatch.setenv("LOCI_TIER_MODEL_MAP", "{not valid json}")
        result = get_tier_model_map()
        assert result == TIER_MODEL_MAP

    def test_partial_override_only_specified_tiers_present(self, monkeypatch):
        """Partial override (only 'light') returns a map with only LIGHT key."""
        override = {"light": ["nano-model"]}
        monkeypatch.setenv("LOCI_TIER_MODEL_MAP", json.dumps(override))
        result = get_tier_model_map()
        assert result == {PolicyTier.LIGHT: ["nano-model"]}
        assert PolicyTier.MEDIUM not in result
        assert PolicyTier.HEAVY not in result

    def test_override_with_unknown_tier_keys_ignored(self, monkeypatch):
        """Keys outside light/medium/heavy are silently ignored."""
        override = {
            "light": ["haiku-clone"],
            "ultra": ["should-be-ignored"],
        }
        monkeypatch.setenv("LOCI_TIER_MODEL_MAP", json.dumps(override))
        result = get_tier_model_map()
        assert list(result.keys()) == [PolicyTier.LIGHT]


# ---------------------------------------------------------------------------
# tier_router_surface (async)
# ---------------------------------------------------------------------------


def _make_policy(tier: PolicyTier) -> Policy:
    return Policy(
        policy_id="test-policy-abc",
        title="Test Policy",
        tier=tier,
        region=PolicyRegion.ROUTING,
    )


class TestTierRouterSurface:
    def test_heavy_model_light_policy_downgrades(self):
        """HEAVY model + LIGHT policy → model rewritten, downgraded=True, violation logged."""
        light_policy = _make_policy(PolicyTier.LIGHT)

        mock_backend = MagicMock()
        mock_backend.load.return_value = light_policy

        async def run():
            with patch("loci.hooks.PolicyBackend", return_value=mock_backend):
                return await tier_router_surface(
                    model="claude-opus-4.8",
                    messages=[],
                    kwargs={"policy_id": "test-policy-abc"},
                    litellm_params={},
                )

        result = asyncio.run(run())

        assert result["downgraded"] is True
        assert result["model"] == "claude-haiku-4.5"
        assert result["original_model"] == "claude-opus-4.8"
        assert result["tier_decision"] == "light"
        assert len(result["violations"]) == 1
        assert result["violations"][0]["type"] == "model_downgraded"
        assert result["violations"][0]["severity"] == "warning"
        mock_backend.load.assert_called_once_with("test-policy-abc")

    def test_light_model_light_policy_no_rewrite(self):
        """LIGHT model + LIGHT policy → model unchanged, downgraded=False, no violations."""
        light_policy = _make_policy(PolicyTier.LIGHT)

        mock_backend = MagicMock()
        mock_backend.load.return_value = light_policy

        async def run():
            with patch("loci.hooks.PolicyBackend", return_value=mock_backend):
                return await tier_router_surface(
                    model="claude-haiku-4.5",
                    messages=[],
                    kwargs={"policy_id": "test-policy-abc"},
                    litellm_params={},
                )

        result = asyncio.run(run())

        assert result["downgraded"] is False
        assert result["model"] == "claude-haiku-4.5"
        assert result["original_model"] is None
        assert result["tier_decision"] == "light"
        assert result["violations"] == []

    def test_no_policy_id_returns_default(self):
        """Missing policy_id → tier_decision='default', model unchanged."""
        not_called = MagicMock()

        async def run():
            with patch("loci.hooks.PolicyBackend", return_value=not_called):
                return await tier_router_surface(
                    model="claude-opus-4.8",
                    messages=[],
                    kwargs={},
                    litellm_params={},
                )

        result = asyncio.run(run())

        assert result["tier_decision"] == "default"
        assert result["downgraded"] is False
        assert result["model"] == "claude-opus-4.8"
        assert result["violations"] == []
        not_called.load.assert_not_called()

    def test_policy_not_found_returns_default_with_violation(self):
        """Policy load returns None → tier_decision='default', policy_not_found violation."""
        mock_backend = MagicMock()
        mock_backend.load.return_value = None

        async def run():
            with patch("loci.hooks.PolicyBackend", return_value=mock_backend):
                return await tier_router_surface(
                    model="claude-opus-4.8",
                    messages=[],
                    kwargs={"policy_id": "missing-id"},
                    litellm_params={},
                )

        result = asyncio.run(run())

        assert result["tier_decision"] == "default"
        assert result["downgraded"] is False
        assert result["model"] == "claude-opus-4.8"
        assert len(result["violations"]) == 1
        assert result["violations"][0]["type"] == "policy_not_found"

    def test_medium_model_heavy_policy_no_rewrite(self):
        """MEDIUM model + HEAVY policy → kept as-is (within budget)."""
        heavy_policy = _make_policy(PolicyTier.HEAVY)

        mock_backend = MagicMock()
        mock_backend.load.return_value = heavy_policy

        async def run():
            with patch("loci.hooks.PolicyBackend", return_value=mock_backend):
                return await tier_router_surface(
                    model="claude-sonnet-4.6",
                    messages=[],
                    kwargs={"policy_id": "test-policy-abc"},
                    litellm_params={},
                )

        result = asyncio.run(run())

        assert result["downgraded"] is False
        assert result["model"] == "claude-sonnet-4.6"
        assert result["tier_decision"] == "heavy"
        assert result["violations"] == []

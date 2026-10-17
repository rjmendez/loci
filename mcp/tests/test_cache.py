"""
Tests for mcp/cache.py — memory-augmented response caching.

Mutation proofs noted inline for each test.
"""

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache import CacheHit, check_memory_cache, format_cache_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_finding(
    score: float = 0.9,
    content: str = "cached answer text",
    age_hours: float | None = 1.0,
    finding_id: str = "f1",
    source: str = "loci_memory",
) -> dict:
    """Build a finding dict with the requested age relative to now."""
    finding: dict = {
        "id": finding_id,
        "score": score,
        "content": content,
        "source": source,
    }
    if age_hours is not None:
        ts = datetime.now(timezone.utc) - timedelta(hours=age_hours)
        finding["created_at"] = ts.isoformat()
    return finding


# ---------------------------------------------------------------------------
# check_memory_cache
# ---------------------------------------------------------------------------


class TestCheckMemoryCacheHit:
    """Positive path: a qualifying finding returns a CacheHit."""

    def test_high_score_recent_returns_cache_hit(self):
        """score=0.9 age=1h threshold=0.85 ttl=24 → CacheHit.

        Mutation: change score to 0.7 → returns None.
        """
        findings = [_fresh_finding(score=0.9, age_hours=1.0)]
        hit = check_memory_cache(findings, confidence_threshold=0.85, ttl_hours=24.0)

        assert hit is not None
        assert hit.score == 0.9
        assert hit.content == "cached answer text"
        assert hit.age_hours < 2.0  # sanity: it's recent

    def test_no_timestamp_treated_as_always_valid(self):
        """A finding with no timestamp is accepted regardless of TTL.

        Mutation: remove the ``if ts_str`` branch so missing timestamp
        triggers a skip → hit becomes None.
        """
        finding = {
            "id": "no-ts",
            "score": 0.91,
            "content": "timeless content",
            "source": "loci_memory",
            # no created_at / timestamp key
        }
        hit = check_memory_cache([finding], confidence_threshold=0.85, ttl_hours=0.001)

        assert hit is not None
        assert hit.score == 0.91
        assert hit.age_hours == 0.0

    def test_multiple_hits_returns_highest_score(self):
        """When two findings qualify, the one with the higher score wins.

        Mutation: return the first qualifying hit instead of the best →
        returns f-low (score=0.87) instead of f-high (score=0.95).
        """
        f_low = _fresh_finding(score=0.87, content="lower", finding_id="f-low", age_hours=2.0)
        f_high = _fresh_finding(score=0.95, content="higher", finding_id="f-high", age_hours=2.0)
        # Deliberately put the lower-score one first in the list.
        hit = check_memory_cache([f_low, f_high], confidence_threshold=0.85, ttl_hours=24.0)

        assert hit is not None
        assert hit.finding_id == "f-high"
        assert hit.score == 0.95


class TestCheckMemoryCacheMiss:
    """Negative paths: findings that must NOT qualify."""

    def test_expired_finding_returns_none(self):
        """score=0.9 but age=25h > TTL=24h → None.

        Positive twin: test_high_score_recent_returns_cache_hit.
        Mutation: remove the TTL check → returns a hit even for stale findings.
        """
        findings = [_fresh_finding(score=0.9, age_hours=25.0)]
        hit = check_memory_cache(findings, confidence_threshold=0.85, ttl_hours=24.0)

        assert hit is None

    def test_low_score_returns_none(self):
        """score=0.7 < threshold=0.85 → None.

        Positive twin: test_high_score_recent_returns_cache_hit (score=0.9).
        Mutation: change the threshold guard to ``<`` instead of ``>=`` → accepts 0.7.
        """
        findings = [_fresh_finding(score=0.70, age_hours=1.0)]
        hit = check_memory_cache(findings, confidence_threshold=0.85, ttl_hours=24.0)

        assert hit is None

    def test_empty_findings_returns_none(self):
        """Empty list → None immediately.

        Mutation: return a sentinel CacheHit for empty input → fails ``is None``.
        """
        hit = check_memory_cache([], confidence_threshold=0.85, ttl_hours=24.0)

        assert hit is None

    def test_finding_without_content_skipped(self):
        """A finding with score above threshold but no content is skipped.

        Positive twin: test_high_score_recent_returns_cache_hit (has content).
        Mutation: remove the ``if not content: continue`` guard → returns a hit
        with empty content.
        """
        finding = {
            "id": "empty-content",
            "score": 0.99,
            # no content/text key
            "source": "loci_memory",
        }
        hit = check_memory_cache([finding], confidence_threshold=0.85, ttl_hours=24.0)

        assert hit is None

    def test_score_exactly_at_threshold_qualifies(self):
        """Boundary: score exactly equal to threshold is accepted (>=).

        Positive of threshold boundary; negative twin is test_low_score_returns_none.
        Mutation: change guard to ``>`` strict → boundary value rejected.
        """
        findings = [_fresh_finding(score=0.85, age_hours=1.0)]
        hit = check_memory_cache(findings, confidence_threshold=0.85, ttl_hours=24.0)

        assert hit is not None
        assert hit.score == 0.85


# ---------------------------------------------------------------------------
# format_cache_response
# ---------------------------------------------------------------------------


class TestFormatCacheResponse:
    def test_produces_loci_cache_prefix(self):
        """Output starts with [LOCI-CACHE ...] and includes content.

        Mutation: remove the prefix → prefix check fails.
        """
        hit = CacheHit(
            finding_id="abc123",
            score=0.92,
            content="the cached body",
            age_hours=3.5,
            source="loci_memory",
        )
        result = format_cache_response(hit)

        assert result.startswith("[LOCI-CACHE")
        assert "hit=0.92" in result
        assert "age=3.5h" in result
        assert "src=loci_memory" in result
        assert "id=abc123" in result
        assert result.endswith("the cached body")


# ---------------------------------------------------------------------------
# grounding_hook_augmented integration
# ---------------------------------------------------------------------------


class TestGroundingHookCacheIntegration:
    """grounding_hook_augmented returns cache_hit=True when policy is enabled
    and a high-confidence finding is present."""

    def test_high_confidence_finding_returns_cache_hit_true(self, monkeypatch, tmp_path):
        """With LOCI_POLICY_ENABLE=1 and a score-0.9 finding, hook short-circuits.

        Mutation: remove the cache check block from grounding_hook_augmented →
        cache_hit key absent / False.
        """
        monkeypatch.setenv("LOCI_POLICY_ENABLE", "1")
        monkeypatch.setenv("LOCI_CACHE_CONFIDENCE", "0.85")
        monkeypatch.setenv("LOCI_CACHE_TTL_HOURS", "24.0")
        # Point PolicyBackend at tmp so it doesn't touch ~/.loci
        monkeypatch.setenv("LOCI_POLICY_DIR", str(tmp_path / "policies"))

        # Reload policy module so env vars take effect
        import importlib
        import policy as policy_mod
        import hooks as hooks_mod
        importlib.reload(policy_mod)
        importlib.reload(hooks_mod)

        from hooks import grounding_hook_augmented

        finding = _fresh_finding(score=0.9, age_hours=1.0)
        kwargs = {
            "grounding_context": {"findings": [finding]},
        }
        messages = [{"role": "user", "content": "what is X?"}]

        result = asyncio.run(
            grounding_hook_augmented(
                model="gpt-4",
                messages=messages,
                kwargs=kwargs,
                litellm_params={},
            )
        )

        assert result["cache_hit"] is True
        assert result["augmented"] is False
        assert result["r_mem"] == pytest.approx(0.9)
        assert "[LOCI-CACHE" in result["cache_response"]
        # Original messages must be returned unchanged
        assert result["messages"] == messages

    def test_disabled_policy_does_not_cache(self, monkeypatch, tmp_path):
        """With LOCI_POLICY_ENABLE=0, even a high-score finding is not cached.

        Positive twin: test_high_confidence_finding_returns_cache_hit_true.
        Mutation: remove the ``policy_config.enabled`` guard → caches even when
        policy is disabled.
        """
        monkeypatch.setenv("LOCI_POLICY_ENABLE", "0")
        monkeypatch.setenv("LOCI_CACHE_CONFIDENCE", "0.85")

        import importlib
        import policy as policy_mod
        import hooks as hooks_mod
        importlib.reload(policy_mod)
        importlib.reload(hooks_mod)

        from hooks import grounding_hook_augmented

        finding = _fresh_finding(score=0.99, age_hours=0.5)
        kwargs = {"grounding_context": {"findings": [finding]}}
        messages = [{"role": "user", "content": "anything"}]

        result = asyncio.run(
            grounding_hook_augmented(
                model="gpt-4",
                messages=messages,
                kwargs=kwargs,
                litellm_params={},
            )
        )

        assert result.get("cache_hit") is not True

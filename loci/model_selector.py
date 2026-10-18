"""Tier -> model candidates. No model is named here.

The local model for a tier is whatever the model pool (``mcp/model_pool.py``, the
``[[models.pool]]`` entries in backends.toml) ranks first for that tier's pool role; cloud
candidates are the entries of ``loci.policy.get_tier_model_map()`` (the compiled map or the
``LOCI_TIER_MODEL_MAP`` override). Nothing here picks, ranks or remembers a model: an operator
who wants a different local model for a tier edits the pool, not this file.

Pool roles, most specific first:

* a task type (``security_review``, ``rtl_verify``, ``code``) names the pool role that already
  serves it (``security``, ``rtl``, ``code``);
* each tier names its own role (``tier_nano`` .. ``tier_heavy``) so an operator can pin a model to
  a tier, and falls back to the general role a tier maps onto (``gen``, ``gen_large``).

Every role here must GENERATE text: the result is a chat model for a request. The ``decide`` role is
the SystemOne decision model, a 0.8B classifier that answers one answer-letter probability per
question and cannot write a reply, so it is never a candidate (``_NON_GENERATING_ROLES``; it is used
only through ``mcp/decide.py``). Until 2026-10-18 nano fell back to ``decide`` and ``triage`` mapped
to it, which resolved a nano or triage chat request to that classifier.

The lookup is ``model_pool.rank_role(role).chosen``, which is pure: it logs no routing decision
and never explores, unlike ``pick()``, because a tier rewrite is not itself a model call.

Fail-open: no pool, a role nobody serves, ``model_pool`` not importable (the LiteLLM proxy
process does not have ``mcp/`` on its path) or any error in the lookup leaves the local slot
empty, and the candidates are exactly the policy map's, as before the pool existed.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger("loci.model_selector")

LOCAL_PREFIX = "ollama/"

# tier -> pool roles to try, most specific first
_TIER_ROLES = {
    "nano":     ("tier_nano", "gen"),
    "light":    ("tier_light", "gen"),
    "standard": ("tier_standard", "gen"),
    "medium":   ("tier_standard", "gen"),
    "turbo":    ("tier_turbo", "gen_large"),
    "heavy":    ("tier_heavy", "gen_large"),
    "research": ("tier_research", "gen_large"),
}

# task type -> the pool role that serves it
_TASK_ROLES = {
    "security_review": "security",
    "rtl_verify": "rtl",
    "code": "code",
}

# pool roles whose models classify but do not generate; never a candidate for a chat request
_NON_GENERATING_ROLES = frozenset({"decide"})


def prefer_local_default() -> bool:
    """``LOCI_TIER_PREFER_LOCAL=0`` puts cloud candidates ahead of the pool's local pick."""
    return os.getenv("LOCI_TIER_PREFER_LOCAL", "1").strip().lower() not in ("0", "false", "no", "off")


def _pool_best(role: str) -> str:
    try:
        import model_pool
        return model_pool.rank_role(role).chosen or ""
    except Exception as exc:  # noqa: BLE001 - fail-open to the policy map
        logger.debug("model pool lookup for role %r failed: %r", role, exc)
        return ""


def local_model(tier: str, task_type: Optional[str] = None) -> str:
    """The pool's pick for *tier* (``ollama/<tag>``), or "" when the pool has none."""
    roles: List[str] = []
    if task_type and task_type in _TASK_ROLES:
        roles.append(_TASK_ROLES[task_type])
    roles.extend(_TIER_ROLES.get(tier.lower(), ()))
    for role in roles:
        if role in _NON_GENERATING_ROLES:
            continue
        tag = _pool_best(role)
        if tag:
            return LOCAL_PREFIX + tag
    return ""


def candidates(
    tier: str,
    task_type: Optional[str] = None,
    prefer_local: Optional[bool] = None,
) -> List[str]:
    """Ordered model candidates for *tier*: the pool's local pick and the policy map's entries.

    With ``prefer_local`` (default ``LOCI_TIER_PREFER_LOCAL``, on) the local pick leads and the
    policy map follows in its own order; without it the map's cloud entries lead, then the local
    pick, then the map's own ``ollama/`` entries. Duplicates are dropped. Empty for ZERO.
    Raises ValueError for a tier name PolicyTier does not know.
    """
    from loci.policy import PolicyTier, get_tier_model_map  # lazy: policy imports this module lazily too

    try:
        tier_enum = PolicyTier(tier.lower())
    except ValueError:
        raise ValueError(f"unknown policy tier {tier!r}") from None
    static = list(get_tier_model_map().get(tier_enum, []))
    local = local_model(tier_enum.value, task_type)
    if prefer_local is None:
        prefer_local = prefer_local_default()
    if prefer_local:
        ordered = [local] + static
    else:
        ordered = ([m for m in static if not m.startswith(LOCAL_PREFIX)] + [local]
                   + [m for m in static if m.startswith(LOCAL_PREFIX)])
    seen, out = set(), []
    for m in ordered:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def select_model(
    tier: str,
    task_type: Optional[str] = None,
    prefer_local: Optional[bool] = None,
) -> str:
    """First candidate for *tier* ("" when there is none).

    Raises ValueError for the ZERO tier (it forbids LLM calls) and for an unknown tier name.
    """
    if tier.lower() == "zero":
        raise ValueError("PolicyTier.ZERO forbids LLM calls: no model can be selected for it")
    found = candidates(tier, task_type, prefer_local)
    return found[0] if found else ""

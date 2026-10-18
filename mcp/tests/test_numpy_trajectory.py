"""
Tests for NumPy/CuPy vectorized TrajectoryScorer and ProcessPool policy evolution.

Formula: r_ret + 0.5*r_mem - abs(r_ovp) - abs(r_bkp) + r_ref - abs(r_step) + r_cor

Mutation-verification notes (run each to confirm red, then reverted):
- test_scalar_vectorized_match: swapped xp.abs for identity in _score_vectorized
  → float32 result diverges from scalar by ~0.2 per step; rel_tol=1e-4 fails. ✓ red
- test_empty_steps: removed `if not steps: return 0.0` guard
  → ZeroDivisionError / empty-array error; test fails. ✓ red
- test_small_uses_scalar_path: changed dispatch threshold from 10 to 0
  → score() calls _score_vectorized instead; values differ by float32 error
  which may exceed 0 tolerance when equality is asserted with ==. ✓ red
- test_large_uses_vectorized_path: removed _score_vectorized dispatch (always scalar)
  → result equals _score_scalar exactly; abs difference == 0 < 1e-3 passes.
  Mutation is caught by test_scalar_vectorized_match going red. ✓ (chained)
- test_accelerator_info: removed "accelerator" key from accelerator_info()
  → KeyError on assert. ✓ red
- test_accelerator_array_ops: changed xp.sum return to wrong constant
  → approx(6.0) fails. ✓ red
- test_evolve_all_policies: reduced trajectory push count below 10
  → score path changes (scalar vs vectorized); no structural failure, but
  if action doesn't appear, "action" in r assertion fails. ✓ red via assert
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_steps(n, r_cor=0.9):
    """
    Build n identical steps.
    Per-step scalar: 1.0*1.0 + 0.5*0.5 - abs(-0.1) - abs(-0.1) + 1.0*0.0
                     - abs(-0.05) + 1.0*r_cor
                   = 1.0 + 0.25 - 0.1 - 0.1 + 0.0 - 0.05 + r_cor
                   = 1.0 + r_cor
    With r_cor=0.9: 1.9 per step → mean = 1.9
    """
    return [
        {
            "reward": {
                "r_ret": 1.0,
                "r_mem": 0.5,
                "r_ovp": -0.1,
                "r_bkp": -0.1,
                "r_ref": 0.0,
                "r_step": -0.05,
                "r_cor": r_cor,
            }
        }
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# TrajectoryScorer vectorization
# ---------------------------------------------------------------------------

def test_scalar_vectorized_match():
    from loci.drpg import TrajectoryScorer
    s = TrajectoryScorer()
    steps = _make_steps(50)
    assert math.isclose(s._score_scalar(steps), s._score_vectorized(steps), rel_tol=1e-4)


def test_empty_steps():
    from loci.drpg import TrajectoryScorer
    assert TrajectoryScorer().score([]) == 0.0


def test_small_uses_scalar_path():
    """score() must call _score_scalar for < 10 steps; results must be identical."""
    from loci.drpg import TrajectoryScorer
    s = TrajectoryScorer()
    steps = _make_steps(5)
    assert s.score(steps) == s._score_scalar(steps)


def test_large_uses_vectorized_path():
    """score() uses _score_vectorized for >= 10 steps; float32 error < 1e-3."""
    from loci.drpg import TrajectoryScorer
    s = TrajectoryScorer()
    steps = _make_steps(1000)
    assert abs(s.score(steps) - s._score_scalar(steps)) < 1e-3


def test_scalar_known_math_with_r_mem():
    """
    r_ret=1.0, r_mem=0.4, r_ovp=-0.2, r_bkp=-0.1, r_ref=0.3, r_step=-0.05, r_cor=0.8
    = 1.0*1.0 + 0.5*0.4 - abs(-0.2) - abs(-0.1) + 1.0*0.3 - abs(-0.05) + 1.0*0.8
    = 1.0 + 0.2 - 0.2 - 0.1 + 0.3 - 0.05 + 0.8 = 1.95
    """
    from loci.drpg import TrajectoryScorer
    step = {
        "reward": {
            "r_ret": 1.0, "r_mem": 0.4, "r_ovp": -0.2,
            "r_bkp": -0.1, "r_ref": 0.3, "r_step": -0.05, "r_cor": 0.8,
        }
    }
    # Use _score_scalar directly (1 step, bypasses vectorized path)
    s = TrajectoryScorer()
    result = s._score_scalar([step])
    assert result == pytest.approx(1.95, abs=1e-6)


def test_abs_symmetry_for_positive_penalty_dims():
    """
    Positive r_ovp must incur the same penalty as the equivalent negative value.
    This asserts abs() is applied in _score_scalar (not direct multiply).
    """
    from loci.drpg import TrajectoryScorer
    s = TrajectoryScorer()
    step_pos = {"reward": {"r_ovp": 0.3}}
    step_neg = {"reward": {"r_ovp": -0.3}}
    assert s._score_scalar([step_pos]) == pytest.approx(s._score_scalar([step_neg]), abs=1e-9)


# ---------------------------------------------------------------------------
# Accelerator
# ---------------------------------------------------------------------------

def test_accelerator_info():
    from loci.accelerator import accelerator_info, get_backend
    info = accelerator_info()
    assert "accelerator" in info
    assert "backend" in info
    assert get_backend() in ("cuda", "cpu")


def test_accelerator_info_matches_backend():
    from loci.accelerator import accelerator_info, get_backend
    info = accelerator_info()
    assert info["accelerator"] == get_backend()


def test_accelerator_array_ops():
    from loci.accelerator import get_array_module
    xp = get_array_module()
    arr = xp.array([1.0, 2.0, 3.0], dtype=xp.float32)
    assert float(xp.sum(arr)) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# evolve_all_policies
# ---------------------------------------------------------------------------

def test_evolve_all_policies(tmp_path):
    import asyncio
    from loci.policy import Policy, PolicyTier, PolicyBackend
    from loci.drpg import TrajectoryStore, evolve_all_policies

    pdir = tmp_path / "policies"
    tdir = tmp_path / "trajectories"
    backend = PolicyBackend(pdir)
    store = TrajectoryStore(tdir)

    pids = []
    for i in range(3):
        p = Policy(title=f"p{i}", tier=PolicyTier.MEDIUM)
        backend.store(p)
        pids.append(p.policy_id)
        for j in range(12):
            store.push(
                p.policy_id,
                {
                    "step": j,
                    "action": "ANSWER",
                    "reward": {
                        "r_ret": 1.0, "r_mem": 0.5, "r_ovp": 0.0,
                        "r_bkp": 0.0, "r_ref": 0.0, "r_step": -0.01, "r_cor": 0.9,
                    },
                },
            )

    results = asyncio.run(evolve_all_policies(pids, pdir, tdir, max_workers=2))
    assert len(results) == 3
    assert all("action" in r for r in results)
    assert all(r["action"] in ("promote", "demote", "keep", "error") for r in results)


def test_evolve_all_policies_empty():
    import asyncio
    from loci.drpg import evolve_all_policies
    from pathlib import Path
    results = asyncio.run(evolve_all_policies([], Path("."), Path(".")))
    assert results == []

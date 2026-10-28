"""Bounded evolver (Tripwire-Governor S18): genes, caps, ES, promotion state machine, patches, revert, sandbox.

Real files under tmp_path for every store; the filesystem seam is a RECORDING OsFs subclass (it records, it
does not swallow); the clock and the vitals heartbeat are injected. No model, network or live store is touched.

Mutation log (each applied to loci/drpg.py, loci/policy.py or scripts/learn_run.py; the named tests went red;
then reverted); 49 mutations were run in all, none survives.
- draw_gene without the upper clip (a gene may exceed the human default):
  test_ten_thousand_seeded_draws_hit_exactly_the_floor_and_the_human_default
- GENES default raised to 1.2 for one gene: same test, test_clamp_gene_clips_to_the_bounds
- 24 h patch limit dropped (check_rate no-op): test_one_patch_per_cell_per_24_hours
- 24 h limit ">=" boundary flipped to ">": test_one_patch_per_cell_per_24_hours
- check_cap_change: widening allowed: test_caps_are_not_genes_widening_raises_and_a_10_percent_tightening_is_accepted
- 10% step computed in floats: test_ten_percent_step_is_decided_exactly
- promotion MIN_OUTCOMES 30 -> 29 / MIN_DAYS 3 -> 2: test_promotion_state_machine_boundaries
- stage0 requirement dropped: test_promotion_needs_a_stage0_pass_and_independent_windows
- evidence-hash check dropped from revert: test_revert_refuses_a_patch_whose_evidence_no_longer_hashes
- revert restores the proposed instead of the previous value: test_revert_restores_the_exact_previous_values
- sandbox protected-path check dropped: test_the_learner_cannot_write_protected_paths
- learned_feature_on ignores the registry: test_learned_genes_are_inert_by_default
- C11 prerequisite (rows_read == 0 refusal) deleted: test_c11_prerequisite_refuses_with_a_clear_reason_and_writes_nothing
- dry-run applies anyway (apply flag ignored): test_dry_run_is_the_default_and_writes_nothing
- ES candidate not tighten-only / 10% step bound dropped / gene floor dropped:
  test_es_candidates_are_tighten_only_within_ten_percent_and_the_bounds
- ES held-out gate dropped, gain ">=" -> ">": test_es_acceptance_gates_are_exact
- lock staleness ">" -> ">=": test_one_job_at_a_time_and_a_stale_lock_is_taken_over (SURVIVED the first draft, which
  tested one second either side; the exact-boundary assertion was added)
- ES tighten-only mutation "value = cur * factor" alone was an equivalent mutant (a later min() also clips); the
  mutation that removes both clips is killed
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from datetime import date, datetime, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from loci import drpg
from loci import policy as policy_mod
from loci.degeneracy import DEGENERATE, HEALTHY, Verdict
from loci.drpg import (
    APPROVED, COLLECTING, REVIEW_REQUIRED, ROLLBACK_TRIGGERED, Deadline, LearnerBusy, LearnWriter,
    LimitsNotApplied, LowDiskSpace, NothingToRevert, OsFs, Outcome, PatchConflict, PatchLog, PatchRateLimited,
    PatchTampered, QuotaExceeded, ReviewRequired, RunLock, SandboxViolation, TimeLimitExceeded, TrajectoryStore,
    apply_os_limits, apply_patch, evaluate_promotion, evidence_hash, evolve_genome, flip_flag_off,
    independent_windows, learner_blocked, outcomes_from_rows, plan_patch, replay_score, revert_last_patch,
    row_key, split_rows,
)
from loci.policy import (
    GENE_BY_NAME, GENES, MAX_GENES, CapViolation, GeneBoundsError, GeneSpec, Policy, PolicyBackend, PolicyTier,
    apply_genes, check_cap_change, check_genome_step, clamp_gene, default_genome, draw_gene, draw_genome,
    learned_feature_on, read_learned_genes, validate_gene_set,
)

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


learn_run = _load_script("learn_run")
check_policy_flags = _load_script("check_policy_flags")

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc).timestamp()
DAY = 86400.0
HOUR = 3600.0


@pytest.fixture(autouse=True)
def _fresh_caches():
    drpg._TRAJ_CACHE.clear()
    policy_mod._POLICY_CACHE.clear()
    yield
    drpg._TRAJ_CACHE.clear()
    policy_mod._POLICY_CACHE.clear()


class RecordingFs(OsFs):
    """Real files, with every operation recorded and the free-space / tree-size readings controllable."""

    def __init__(self, free=10 ** 13, tree=None):
        self.ops = []
        self.free = free
        self.tree = tree

    def free_bytes(self, path):
        return self.free

    def tree_size(self, path):
        return super().tree_size(path) if self.tree is None else self.tree

    def write_temp(self, final, data):
        tmp = super().write_temp(final, data)
        self.ops.append(("write_temp", Path(final).name, len(data), tmp.name))
        return tmp

    def replace(self, tmp, final):
        self.ops.append(("replace", Path(tmp).name, Path(final).name))
        super().replace(tmp, final)

    def remove(self, path):
        self.ops.append(("remove", Path(path).name))
        super().remove(path)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(Path(root).rglob("*")) if p.is_file()}


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def write_vitals(path, level=0, ts=NOW):
    path.write_text(json.dumps({"ts": ts, "level": level, "degraded": [], "readings": {}}), encoding="utf-8")
    return path


def make_policy(policy_dir, pid="cell-a", tier=PolicyTier.STANDARD, config=None):
    backend = PolicyBackend(policy_dir)
    backend.store(Policy(policy_id=pid, title="cell", tier=tier, config=dict(config or {})))
    return backend


def stored_config(policy_dir, pid="cell-a"):
    return json.loads((Path(policy_dir) / f"{pid}.json").read_text(encoding="utf-8"))["config"]


def push_rows(traj_dir, pid, n, *, usage=None, ts0=None, step=60.0, sessions=5):
    store = TrajectoryStore(traj_dir)
    usage = {g.tier_key: 1 for g in GENES} if usage is None else usage
    for i in range(n):
        ts = (NOW - 3 * DAY if ts0 is None else ts0) + i * step
        store.push(pid, {"id": f"{pid}-r{i}", "reward": {"r_ret": 0.5}, "usage": usage,
                         "session_id": f"s{i % sessions}", "_ts": iso(ts)})


class World:
    """A tmp_path laid out like a deployment: policies, trajectories, learn dir, vitals heartbeat, flag registry."""

    def __init__(self, tmp_path, level=0):
        self.root = tmp_path
        self.policy_dir = tmp_path / "policies"
        self.traj_dir = tmp_path / "trajectories"
        self.learn_dir = tmp_path / "learn"
        self.flags = tmp_path / "flags.toml"
        self.vitals = write_vitals(tmp_path / "vitals_state.json", level)
        self.fs = RecordingFs()
        self.flags.write_text('# keep me\n[[flag]]\nname = "other"\nreason = "r"\nreview_by = 2026-12-01\n',
                              encoding="utf-8")
        self.policy_dir.mkdir()
        self.traj_dir.mkdir()

    def writer(self, **kw):
        kw.setdefault("min_free_bytes", 0)
        return LearnWriter(self.learn_dir, fs=self.fs, also_allowed=[self.policy_dir, self.flags], **kw)

    def log(self, writer=None):
        return PatchLog(self.learn_dir / "patches.jsonl", writer or self.writer())

    def evolve(self, **kw):
        kw.setdefault("clock", lambda: NOW)
        kw.setdefault("fs", self.fs)
        kw.setdefault("min_free_bytes", 0)
        return learn_run.run_evolver(policy_dir=self.policy_dir, trajectory_dir=self.traj_dir,
                                     learn_dir=self.learn_dir, vitals_state=self.vitals, **kw)


# ---------------------------------------------------------------------------------------------- gene bounds

def test_the_gene_set_is_at_most_eight_and_validated():
    assert MAX_GENES == 8
    assert [g.name for g in GENES] == ["retrieval_hops_scale", "refinement_steps_scale", "perspectives_scale",
                                       "fanout_scale", "context_chars_scale"]
    nine = tuple(GeneSpec(f"g{i}", "max_fanout", 0.5) for i in range(9))
    with pytest.raises(GeneBoundsError, match="9 genes; at most 8"):
        validate_gene_set(nine)
    assert validate_gene_set(nine[:8]) == nine[:8]  # the positive twin: exactly 8 is allowed
    with pytest.raises(GeneBoundsError, match="floor < default"):
        validate_gene_set((GeneSpec("bad", "max_fanout", 1.5),))


def test_ten_thousand_seeded_draws_hit_exactly_the_floor_and_the_human_default():
    rng = random.Random(20261018)
    lows = {g.name: float("inf") for g in GENES}
    highs = {g.name: float("-inf") for g in GENES}
    for _ in range(10_000):
        genome = draw_genome(rng)
        for name, value in genome.items():
            lows[name] = min(lows[name], value)
            highs[name] = max(highs[name], value)
    assert lows == {"retrieval_hops_scale": 0.5, "refinement_steps_scale": 0.5, "perspectives_scale": 0.5,
                    "fanout_scale": 0.4, "context_chars_scale": 0.25}
    assert highs == {g.name: 1.0 for g in GENES}


def test_clamp_gene_clips_to_the_bounds_and_refuses_junk():
    assert clamp_gene("fanout_scale", 7.0) == 1.0
    assert clamp_gene("fanout_scale", 0.01) == 0.4
    assert clamp_gene("fanout_scale", 0.7) == 0.7  # inside the range: untouched
    with pytest.raises(GeneBoundsError, match="not a finite number"):
        clamp_gene("fanout_scale", float("nan"))
    with pytest.raises(GeneBoundsError, match="unknown gene 'max_spend_usd'"):
        clamp_gene("max_spend_usd", 0.5)
    assert draw_gene("fanout_scale", random.Random(1)) == 0.4230860252738027


def test_caps_are_not_genes_widening_raises_and_a_10_percent_tightening_is_accepted():
    caps = {"daily_hard_usd": 50, "hourly_hard_usd": 25}
    assert not {"daily_hard_usd", "hourly_hard_usd", "soft_cap_usd"} & set(GENE_BY_NAME)
    with pytest.raises(CapViolation, match="widen"):
        check_cap_change(caps, {"daily_hard_usd": 51})
    assert check_cap_change(caps, {"daily_hard_usd": 45}) == {"daily_hard_usd": 45, "hourly_hard_usd": 25}
    assert check_cap_change(caps, {"daily_hard_usd": 50}) == caps  # unchanged is not widening
    with pytest.raises(CapViolation, match="more than 10%"):
        check_cap_change(caps, {"daily_hard_usd": 44})
    with pytest.raises(CapViolation, match="not in the envelope"):
        check_cap_change(caps, {"brand_new_cap": 1})
    with pytest.raises(CapViolation, match="finite non-negative"):
        check_cap_change(caps, {"daily_hard_usd": float("nan")})


def test_ten_percent_step_is_decided_exactly():
    # 0.8 * 0.9 is 0.7200000000000001 in floats: a float comparison would reject the exact 10% tightening
    assert check_genome_step({"fanout_scale": 0.8}, {"fanout_scale": 0.72}) is False
    with pytest.raises(GeneBoundsError, match="more than 10%"):
        check_genome_step({"fanout_scale": 0.8}, {"fanout_scale": 0.719})
    # an absent gene starts at its default 1.0
    assert check_genome_step({}, {"retrieval_hops_scale": 0.9}) is False
    with pytest.raises(GeneBoundsError, match="more than 10%"):
        check_genome_step({}, {"retrieval_hops_scale": 0.89})


def test_a_widening_step_is_flagged_not_hidden_and_bounds_still_apply():
    assert check_genome_step({"fanout_scale": 0.7}, {"fanout_scale": 0.75}) is True
    with pytest.raises(GeneBoundsError, match=r"outside \[0.4, 1.0\]"):
        check_genome_step({"fanout_scale": 0.7}, {"fanout_scale": 1.5})
    with pytest.raises(GeneBoundsError, match="unknown gene"):
        check_genome_step({}, {"nope": 0.9})


# ------------------------------------------------------------------------- learned genes in the policy

def test_read_learned_genes_reports_what_it_refuses():
    config = {"learned_genes": {"fanout_scale": 0.5, "retrieval_hops_scale": 3.0, "nope": 0.5,
                                "perspectives_scale": float("nan"), "context_chars_scale": True}}
    genes, problems = read_learned_genes(config)
    assert genes == {"fanout_scale": 0.5}
    assert problems == ["gene retrieval_hops_scale=3.0 outside [0.5, 1.0]", "unknown gene 'nope'",
                        "gene perspectives_scale=nan outside [0.5, 1.0]", "gene context_chars_scale=True outside [0.25, 1.0]"]
    assert read_learned_genes({"learned_genes": [1]}) == ({}, ["learned_genes is not an object"])
    assert read_learned_genes({}) == ({}, [])


def test_apply_genes_only_tightens_and_never_goes_below_one():
    base = PolicyTier.STANDARD.constraints()  # hops 7, refinement 4, perspectives 3, fanout 16, context 8192
    out = apply_genes(base, {"retrieval_hops_scale": 0.5, "fanout_scale": 0.5, "context_chars_scale": 0.25,
                             "perspectives_scale": 0.5})
    assert out["max_retrieval_hops"] == 3 and out["max_fanout"] == 8 and out["max_context_chars"] == 2048
    assert out["max_perspectives"] == 1 and out["max_refinement_steps"] == 4  # floor(1.5)=1; untouched gene
    assert apply_genes(base, {"fanout_scale": 7.0}) == base  # out of bounds is ignored, never loosened
    assert apply_genes(PolicyTier.ZERO.constraints(), {"fanout_scale": 0.5}) == PolicyTier.ZERO.constraints()


def test_learned_genes_are_inert_by_default(tmp_path):
    policy = Policy(policy_id="cell-a", title="t", tier=PolicyTier.STANDARD,
                    config={"learned_genes": {"fanout_scale": 0.5}})
    base = policy.get_constraints()
    empty_registry = tmp_path / "empty.toml"
    empty_registry.write_text("", encoding="utf-8")
    on = {"LOCI_LEARNED_GENES": "1"}
    # default: no env -> unchanged, even when the registry would allow it
    assert policy.get_effective_constraints(environ={}, registry_path=empty_registry) == base
    assert policy.get_effective_constraints(environ={"LOCI_LEARNED_GENES": "true"}, registry_path=empty_registry) == base
    # the REAL registry lists learned_genes as default-off: the env alone cannot turn it on
    assert policy.get_effective_constraints(environ=on) == base
    assert learned_feature_on("learned_genes", environ=on) is False
    # an unreadable registry fails closed
    assert policy.get_effective_constraints(environ=on, registry_path=tmp_path / "missing.toml") == base
    # twin: env on AND the entry removed by the owner -> the gene applies
    assert policy.get_effective_constraints(environ=on, registry_path=empty_registry) == {**base, "max_fanout": 8}
    assert learned_feature_on("other_feature", environ=on, registry_path=empty_registry) is False


def test_the_repo_registry_lists_learned_genes_off_with_a_review_date():
    flags = check_policy_flags.load_registry(check_policy_flags.DEFAULT_REGISTRY)
    entry = next(f for f in flags if f["name"] == "learned_genes")
    assert entry["review_by"] == date(2026, 11, 17)
    assert check_policy_flags.check(flags, date(2026, 11, 17)) == []
    assert check_policy_flags.check(flags, date(2026, 11, 18)) == [
        "learned_genes: review_by 2026-11-17 has expired (today 2026-11-18); re-review the flag and move the date, or remove it"]


def test_policy_store_hands_the_bytes_to_a_writer_instead_of_opening_the_file(tmp_path):
    backend = PolicyBackend(tmp_path / "policies")
    calls = []
    policy = Policy(policy_id="cell-a", title="t")
    backend.store(policy, writer=lambda path, data: calls.append((path.name, json.loads(data)["policy_id"])))
    assert calls == [("cell-a.json", "cell-a")]
    assert not (tmp_path / "policies" / "cell-a.json").exists()
    backend.store(policy)  # the default path still writes the file
    assert (tmp_path / "policies" / "cell-a.json").exists()


# -------------------------------------------------------------------------------------- promotion

def _outcomes(n, *, days, sessions=2, start=NOW, gap=300.0):
    """n outcomes spread over `days` distinct UTC days, alternating between `sessions` sessions."""
    out = []
    for i in range(n):
        day = i % days
        out.append(Outcome(f"o{i}", start + day * DAY + (i // days) * gap, f"s{i % sessions}"))
    return out


def test_promotion_state_machine_boundaries():
    s29 = evaluate_promotion(_outcomes(29, days=3), stage0_passed=True)
    assert (s29.state, s29.reasons, s29.n, s29.days) == (COLLECTING, ("outcomes 29 < 30",), 29, 3)
    s30_two_days = evaluate_promotion(_outcomes(30, days=2), stage0_passed=True)
    assert (s30_two_days.state, s30_two_days.reasons, s30_two_days.n, s30_two_days.days) == (
        COLLECTING, ("days 2 < 3",), 30, 2)
    s30 = evaluate_promotion(_outcomes(30, days=3), stage0_passed=True)
    assert (s30.state, s30.reasons, s30.n, s30.days) == (APPROVED, (), 30, 3)


def test_promotion_needs_a_stage0_pass_and_independent_windows():
    full = _outcomes(30, days=3)
    assert evaluate_promotion(full, stage0_passed=False).reasons == ("stage0_not_passed",)
    assert evaluate_promotion(full, stage0_passed=False).state == COLLECTING
    one_session = _outcomes(30, days=3, sessions=1)
    status = evaluate_promotion(one_session, stage0_passed=True)
    assert (status.state, status.reasons) == (COLLECTING, ("no_independent_windows",))
    assert evaluate_promotion(full, stage0_passed=True).state == APPROVED  # twin: two sessions, an hour apart


def test_independent_windows_need_different_sessions_at_least_an_hour_apart():
    t = NOW
    assert independent_windows([Outcome("a", t, "s1"), Outcome("b", t + HOUR, "s2")]) is True
    assert independent_windows([Outcome("a", t, "s1"), Outcome("b", t + HOUR - 1, "s2")]) is False
    assert independent_windows([Outcome("a", t, "s1"), Outcome("b", t + 5 * HOUR, "s1")]) is False
    assert independent_windows([Outcome("a", t, None), Outcome("b", t + 5 * HOUR, "s2")]) is False
    # the earliest outcome of s2 is the one that pairs with s1's latest
    assert independent_windows([Outcome("a", t, "s1"), Outcome("b", t + 2 * HOUR, "s1"),
                                Outcome("c", t + 3 * HOUR, "s2")]) is True
    assert independent_windows([]) is False


def test_duplicate_outcomes_count_once():
    dup = [Outcome("same", NOW + i, f"s{i % 2}") for i in range(40)]
    status = evaluate_promotion(dup, stage0_passed=True)
    assert (status.state, status.n) == (COLLECTING, 1)


def test_a_widening_patch_is_always_review_required():
    status = evaluate_promotion(_outcomes(30, days=3), widening=True, stage0_passed=True)
    assert (status.state, status.reasons) == (REVIEW_REQUIRED, ("widening_patch_needs_owner",))
    assert evaluate_promotion(_outcomes(30, days=3), widening=False, stage0_passed=True).state == APPROVED


def test_a_degenerate_verdict_triggers_rollback_and_a_healthy_one_does_not():
    full = _outcomes(30, days=3)
    bad = evaluate_promotion(full, stage0_passed=True, degeneracy=Verdict(DEGENERATE, ("refusal_above_incumbent",)))
    assert (bad.state, bad.reasons) == (ROLLBACK_TRIGGERED, ("refusal_above_incumbent",))
    ok = evaluate_promotion(full, stage0_passed=True, degeneracy=Verdict(HEALTHY, ()))
    assert ok.state == APPROVED


def test_outcomes_from_rows_counts_what_it_cannot_use():
    rows = [{"id": "a", "_ts": iso(NOW), "session_id": "s1"},
            {"id": "b", "ts": NOW + 5, "session": "s2"},
            {"id": "c", "_ts": "not a time"},
            {"id": "d"},
            {"id": "old", "_ts": iso(NOW - DAY)}]
    outcomes, skipped = outcomes_from_rows(rows, since=NOW - 1)
    assert outcomes == [Outcome("a", NOW, "s1"), Outcome("b", NOW + 5, "s2")]
    assert skipped == 2


# ------------------------------------------------------------------------------------------ ES

def _cur():
    return default_genome()


def test_es_candidates_are_tighten_only_within_ten_percent_and_the_bounds():
    for current in (_cur(), {"retrieval_hops_scale": 0.8123456, "refinement_steps_scale": 0.5,
                             "perspectives_scale": 0.77, "fanout_scale": 0.4, "context_chars_scale": 0.9}):
        seen = []

        def train(genome, seen=seen):
            seen.append(dict(genome))
            return 0.5

        evolve_genome(current, random.Random(7), train, lambda g: 0.5)
        assert len(seen) == 9  # the baseline, then the 8 candidates
        for cand in seen[1:]:
            for spec in GENES:
                cur = current[spec.name]
                assert spec.floor <= cand[spec.name] <= cur
                assert Fraction(str(cand[spec.name])) >= Fraction(str(cur)) * Fraction(9, 10) or cand[spec.name] == spec.floor


def test_es_acceptance_gates_are_exact():
    cur = _cur()

    def fns(train_gain, held_gain):
        return (lambda g: 0.5 if g == cur else 0.5 + train_gain), (lambda g: 0.5 if g == cur else 0.5 + held_gain)

    ok = evolve_genome(cur, random.Random(1), *fns(0.02, 0.01))
    assert (ok.accepted, ok.reason) == (True, "accepted")
    assert ok.genome != cur
    low_train = evolve_genome(cur, random.Random(1), *fns(0.0199, 0.5))
    assert (low_train.accepted, low_train.reason, low_train.heldout_gain) == (False, "train_gain_below_min", None)
    low_held = evolve_genome(cur, random.Random(1), *fns(0.5, 0.0099))
    assert (low_held.accepted, low_held.reason) == (False, "heldout_gain_below_min")


def test_es_scores_the_heldout_rows_for_the_winner_and_the_incumbent_only():
    cur = _cur()
    held_calls = []

    def held(genome):
        held_calls.append(dict(genome))
        return 0.5

    evolve_genome(cur, random.Random(3), lambda g: 0.5 if g == cur else 0.6, held)
    assert len(held_calls) == 2 and held_calls[1] == cur
    held_calls.clear()
    evolve_genome(cur, random.Random(3), lambda g: 0.5, held)  # no train gain: held-out is never looked at
    assert held_calls == []


def test_es_discards_nonfinite_candidate_scores_and_counts_them():
    cur = _cur()
    calls = []

    def train(genome):
        calls.append(1)
        return 0.5 if genome == cur else (float("nan") if len(calls) % 2 else 0.6)

    result = evolve_genome(cur, random.Random(5), train, lambda g: 0.5 if g == cur else 0.6)
    assert result.discarded_nonfinite == 4
    assert (result.accepted, result.reason) == (True, "accepted")
    nan_base = evolve_genome(cur, random.Random(5), lambda g: float("nan"), lambda g: 0.5)
    assert (nan_base.accepted, nan_base.reason) == (False, "nonfinite_baseline")


def test_es_is_deterministic_for_a_seed():
    cur = _cur()
    a = evolve_genome(cur, random.Random(11), lambda g: -sum(g.values()), lambda g: -sum(g.values()))
    b = evolve_genome(cur, random.Random(11), lambda g: -sum(g.values()), lambda g: -sum(g.values()))
    c = evolve_genome(cur, random.Random(12), lambda g: -sum(g.values()), lambda g: -sum(g.values()))
    assert a == b and a.genome != c.genome


def test_replay_score_hinges_a_row_the_tightened_limits_would_cut_off():
    policy = Policy(policy_id="cell-a", title="t", tier=PolicyTier.STANDARD)
    cur = default_genome()
    half_hops = {**cur, "retrieval_hops_scale": 0.5}   # STANDARD hops 7 -> floor(3.5) = 3
    row = lambda used: {"id": "x", "reward": {"r_ret": 0.5}, "usage": {"max_retrieval_hops": used}}  # noqa: E731
    base = replay_score(policy, cur, cur, [row(4)])
    assert base == 0.5  # the row's own reward score; no gene moved, so unknown usage never binds
    saving = (1.0 - 0.5) / 5
    assert replay_score(policy, half_hops, cur, [row(3), {"usage": {}, "reward": {"r_ret": 0.5}}]) == (
        (0.5 + saving) + (-1.0)) / 2  # row 1 fits (used 3 is not > 3); row 2 has unknown usage: binding
    assert replay_score(policy, half_hops, cur, [row(4)]) == -1.0  # used 4 > limit 3
    assert replay_score(policy, cur, cur, []) == 0.0


def test_split_rows_is_deterministic_and_order_independent():
    rows = [{"id": f"r{i}"} for i in range(100)]
    train, held = split_rows(rows)
    train2, held2 = split_rows(list(reversed(rows)))
    assert len(held) == 20 and len(train) == 80  # pinned for this fixture (sha256 buckets)
    assert sorted(r["id"] for r in held) == sorted(r["id"] for r in held2)
    assert {r["id"] for r in train} | {r["id"] for r in held} == {r["id"] for r in rows}
    assert row_key({"id": "abc"}) == "abc"
    assert len(row_key({"a": 1})) == 64


# ------------------------------------------------------------------------------- patches, rate limit

def _applied_world(tmp_path, *, config=None):
    w = World(tmp_path)
    backend = make_policy(w.policy_dir, config=config)
    writer = w.writer()
    return w, backend, writer, w.log(writer)


def test_plan_patch_records_previous_values_and_flags_widening(tmp_path):
    patch = plan_patch("cell-a", {"retrieval_hops_scale": 0.8}, {"retrieval_hops_scale": 0.72, "fanout_scale": 0.9,
                                                                 "context_chars_scale": 1.0}, NOW)
    assert patch.previous == {"retrieval_hops_scale": 0.8, "fanout_scale": None}
    assert patch.proposed == {"retrieval_hops_scale": 0.72, "fanout_scale": 0.9}
    assert patch.widening is False
    assert patch.evidence_hash == evidence_hash(patch.evidence)
    assert plan_patch("cell-a", {}, default_genome(), NOW) is None
    assert plan_patch("cell-a", {"fanout_scale": 0.7}, {"fanout_scale": 0.75}, NOW).widening is True
    with pytest.raises(GeneBoundsError, match="more than 10%"):
        plan_patch("cell-a", {}, {"fanout_scale": 0.85}, NOW)


def test_the_evidence_hash_is_the_prereg_record_hash():
    from policy_prereg.registry import record_hash
    evidence = {"cell": "a", "ts": 1.5, "previous": {"x": None}, "proposed": {"x": 0.9}, "details": {}}
    assert evidence_hash(evidence) == record_hash(evidence)


def test_one_patch_per_cell_per_24_hours(tmp_path):
    w, backend, writer, log = _applied_world(tmp_path)
    make_policy(w.policy_dir, pid="cell-b")
    first = plan_patch("cell-a", {}, {"fanout_scale": 0.9}, NOW)
    apply_patch(first, backend=backend, writer=writer, log=log, now=NOW)
    second = plan_patch("cell-a", {"fanout_scale": 0.9}, {"fanout_scale": 0.81}, NOW + DAY - 1)
    with pytest.raises(PatchRateLimited, match="at most 1 patch per 24 h"):
        apply_patch(second, backend=backend, writer=writer, log=log, now=NOW + DAY - 1)
    assert stored_config(w.policy_dir)["learned_genes"] == {"fanout_scale": 0.9}
    # twin: exactly 24 h later it is allowed; and another cell was never limited
    apply_patch(plan_patch("cell-a", {"fanout_scale": 0.9}, {"fanout_scale": 0.81}, NOW + DAY),
                backend=backend, writer=writer, log=log, now=NOW + DAY)
    assert stored_config(w.policy_dir)["learned_genes"] == {"fanout_scale": 0.81}
    apply_patch(plan_patch("cell-b", {}, {"fanout_scale": 0.9}, NOW + 1), backend=backend, writer=writer, log=log,
                now=NOW + 1)
    assert stored_config(w.policy_dir, "cell-b")["learned_genes"] == {"fanout_scale": 0.9}


def test_a_widening_patch_is_never_applied_without_the_owner(tmp_path):
    w, backend, writer, log = _applied_world(tmp_path, config={"learned_genes": {"fanout_scale": 0.7}})
    wide = plan_patch("cell-a", {"fanout_scale": 0.7}, {"fanout_scale": 0.75}, NOW)
    before = snapshot(w.root)
    with pytest.raises(ReviewRequired, match="review_required"):
        apply_patch(wide, backend=backend, writer=writer, log=log, now=NOW)
    assert snapshot(w.root) == before
    apply_patch(wide, backend=backend, writer=writer, log=log, now=NOW, owner_approved=True)
    assert stored_config(w.policy_dir)["learned_genes"] == {"fanout_scale": 0.75}


def test_apply_refuses_a_patch_built_on_stale_genes(tmp_path):
    w, backend, writer, log = _applied_world(tmp_path, config={"learned_genes": {"fanout_scale": 0.7}})
    stale = plan_patch("cell-a", {"fanout_scale": 0.8}, {"fanout_scale": 0.72}, NOW)
    with pytest.raises(PatchConflict, match=r"gene fanout_scale of 'cell-a' is 0.7, patch expected 0.8"):
        apply_patch(stale, backend=backend, writer=writer, log=log, now=NOW)
    assert log.entries() == []
    with pytest.raises(PatchConflict, match="does not exist"):
        apply_patch(plan_patch("ghost", {}, {"fanout_scale": 0.9}, NOW), backend=backend, writer=writer, log=log, now=NOW)


def test_a_failed_policy_write_is_logged_and_does_not_count_as_applied(tmp_path):
    w, backend, writer, log = _applied_world(tmp_path)
    patch = plan_patch("cell-a", {}, {"fanout_scale": 0.9}, NOW)
    bad = LearnWriter(w.learn_dir, fs=w.fs, also_allowed=[w.policy_dir], min_free_bytes=0,
                      protected=[w.policy_dir / "cell-a.json"])
    with pytest.raises(SandboxViolation, match="protected"):
        apply_patch(patch, backend=backend, writer=bad, log=PatchLog(w.learn_dir / "patches.jsonl", bad), now=NOW)
    assert [e["type"] for e in log.entries()] == ["patch", "patch_failed"]
    assert log.applied_patches() == []
    assert "learned_genes" not in stored_config(w.policy_dir)


# ------------------------------------------------------------------------------------------ revert

def test_revert_restores_the_exact_previous_values(tmp_path):
    w, backend, writer, log = _applied_world(
        tmp_path, config={"learned_genes": {"retrieval_hops_scale": 0.8123456, "context_chars_scale": 0.5}})
    before = stored_config(w.policy_dir)
    patch = plan_patch("cell-a", {"retrieval_hops_scale": 0.8123456, "context_chars_scale": 0.5},
                       {"retrieval_hops_scale": 0.7311111, "fanout_scale": 0.9}, NOW)
    apply_patch(patch, backend=backend, writer=writer, log=log, now=NOW)
    assert stored_config(w.policy_dir)["learned_genes"] == {"retrieval_hops_scale": 0.7311111, "context_chars_scale": 0.5,
                                                            "fanout_scale": 0.9}
    summary = revert_last_patch(backend=backend, writer=writer, log=log, registry_path=w.flags, now=NOW + HOUR)
    assert stored_config(w.policy_dir) == before  # exact, including the gene that was absent
    assert summary == {"patch_id": patch.id, "cell": "cell-a", "review_by": "2026-11-17",
                       "restored": {"retrieval_hops_scale": 0.8123456, "fanout_scale": None}}
    text = w.flags.read_text(encoding="utf-8")
    flags = {f["name"]: f for f in check_policy_flags.tomllib.loads(text)["flag"]}
    assert flags["learned_genes"]["review_by"] == date(2026, 11, 17)
    assert flags["other"]["review_by"] == date(2026, 12, 1)
    assert text.startswith("# keep me\n")
    assert log.applied_patches() == []
    with pytest.raises(NothingToRevert, match="no applied patch"):
        revert_last_patch(backend=backend, writer=writer, log=log, registry_path=w.flags, now=NOW + HOUR)
    # a reverted patch still counts toward the 24 h limit
    with pytest.raises(PatchRateLimited, match="24 h"):
        apply_patch(plan_patch("cell-a", {}, {"fanout_scale": 0.9}, NOW + 2 * HOUR), backend=backend, writer=writer,
                    log=log, now=NOW + 2 * HOUR)


def test_revert_removes_the_whole_key_when_the_policy_had_no_genes_before(tmp_path):
    w, backend, writer, log = _applied_world(tmp_path)
    before = stored_config(w.policy_dir)
    apply_patch(plan_patch("cell-a", {}, {"fanout_scale": 0.9}, NOW), backend=backend, writer=writer, log=log, now=NOW)
    revert_last_patch(backend=backend, writer=writer, log=log, registry_path=w.flags, now=NOW)
    assert stored_config(w.policy_dir) == before and "learned_genes" not in before


def test_revert_refuses_a_patch_whose_evidence_no_longer_hashes(tmp_path):
    w, backend, writer, log = _applied_world(tmp_path, config={"learned_genes": {"fanout_scale": 0.8}})
    apply_patch(plan_patch("cell-a", {"fanout_scale": 0.8}, {"fanout_scale": 0.72}, NOW), backend=backend,
                writer=writer, log=log, now=NOW)
    lines = [json.loads(x) for x in (w.learn_dir / "patches.jsonl").read_text(encoding="utf-8").splitlines()]
    lines[0]["evidence"]["previous"]["fanout_scale"] = 0.4  # an attacker wants the revert to "restore" 0.4
    (w.learn_dir / "patches.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    before = snapshot(w.root)
    with pytest.raises(PatchTampered, match="no longer hashes to its evidence hash"):
        revert_last_patch(backend=backend, writer=writer, log=log, registry_path=w.flags, now=NOW)
    assert snapshot(w.root) == before


def test_revert_refuses_when_the_policy_changed_since_the_patch(tmp_path):
    w, backend, writer, log = _applied_world(tmp_path, config={"learned_genes": {"fanout_scale": 0.8}})
    apply_patch(plan_patch("cell-a", {"fanout_scale": 0.8}, {"fanout_scale": 0.72}, NOW), backend=backend,
                writer=writer, log=log, now=NOW)
    policy = backend.load("cell-a")
    policy.config["learned_genes"]["fanout_scale"] = 0.6  # someone else moved it
    backend.store(policy)
    with pytest.raises(PatchConflict, match="the policy changed since the patch"):
        revert_last_patch(backend=backend, writer=writer, log=log, registry_path=w.flags, now=NOW)
    assert stored_config(w.policy_dir)["learned_genes"] == {"fanout_scale": 0.6}


def test_flip_flag_off_adds_or_refreshes_an_entry_without_touching_the_others(tmp_path):
    w = World(tmp_path)
    writer = w.writer()
    flip_flag_off(writer, w.flags, "learned_genes", "first", date(2026, 11, 17))
    flip_flag_off(writer, w.flags, "learned_genes", "second", date(2026, 12, 25))
    text = w.flags.read_text(encoding="utf-8")
    flags = check_policy_flags.tomllib.loads(text)["flag"]
    assert [(f["name"], f["reason"], f["review_by"]) for f in flags] == [
        ("other", "r", date(2026, 12, 1)), ("learned_genes", "second", date(2026, 12, 25))]
    assert text.count("[[flag]]") == 2 and text.startswith("# keep me\n")
    assert check_policy_flags.check(flags, date(2026, 10, 18)) == []  # the checker accepts what the revert writes


# --------------------------------------------------------------------------- sandbox, quota, lock

def test_the_learner_cannot_write_protected_paths(tmp_path):
    fs = RecordingFs()
    learn = tmp_path / "learn"
    envelope = tmp_path / "gov" / "envelope.toml"
    stop = tmp_path / "gov" / "STOP"
    audit = tmp_path / "gov" / "audit"
    writer = LearnWriter(learn, fs=fs, protected=[envelope, stop, audit], min_free_bytes=0)
    for target, pattern in ((envelope, "protected path"), (stop, "protected path"), (audit / "chain.jsonl", "protected path"),
                            (learn / "STOP", "reserved name"), (learn / "pause", "reserved name"),
                            (learn / "envelope.toml", "reserved name"), (learn / "audit" / "x.jsonl", "reserved name"),
                            (learn / "audit_log.jsonl", "reserved name"),
                            (learn / "sub" / ".." / ".." / "escaped.txt", "outside the learner sandbox"),
                            (tmp_path / "elsewhere.txt", "outside the learner sandbox")):
        with pytest.raises(SandboxViolation, match=pattern):
            writer.write(target, b"x")
        with pytest.raises(SandboxViolation, match=pattern):
            writer.remove(target)
    assert fs.ops == []  # nothing was attempted, not even a temp file
    # twins: ordinary names inside the sandbox are fine, including ones that merely look similar
    writer.write(learn / "ok.json", b"1")
    writer.write(learn / "stopwatch.json", b"2")
    assert (learn / "ok.json").read_bytes() == b"1" and (learn / "stopwatch.json").read_bytes() == b"2"


def test_the_learn_dir_must_be_on_c():
    with pytest.raises(ValueError, match=r"not on C: \(drive F:\)"):
        LearnWriter(Path("F:/learn"))


def test_writes_are_temp_then_rename_and_a_failed_batch_leaves_no_temp_files(tmp_path):
    fs = RecordingFs()
    learn = tmp_path / "learn"
    writer = LearnWriter(learn, fs=fs, min_free_bytes=0)
    writer.write_many([(learn / "a.json", b"AA"), (learn / "b.json", b"BBB")])
    kinds = [op[0] for op in fs.ops]
    assert kinds == ["write_temp", "write_temp", "replace", "replace"]
    assert fs.ops[0][1:3] == ("a.json", 2) and fs.ops[1][1:3] == ("b.json", 3)
    assert [(op[2]) for op in fs.ops if op[0] == "replace"] == ["a.json", "b.json"]
    assert all(op[1].startswith(op[2] if op[0] == "replace" else "") and op[1].endswith(".tmp")
               for op in fs.ops if op[0] == "replace")
    assert sorted(p.name for p in learn.iterdir()) == ["a.json", "b.json"]

    class FailSecond(RecordingFs):
        def replace(self, tmp, final):
            if Path(final).name == "d.json":
                raise OSError("disk went away")
            super().replace(tmp, final)

    fs2 = FailSecond()
    writer2 = LearnWriter(tmp_path / "learn2", fs=fs2, min_free_bytes=0)
    with pytest.raises(OSError, match="disk went away"):
        writer2.write_many([(tmp_path / "learn2" / "c.json", b"C"), (tmp_path / "learn2" / "d.json", b"D")])
    assert sorted(p.name for p in (tmp_path / "learn2").iterdir()) == ["c.json"]  # c renamed, d's temp removed


def test_the_quota_and_free_space_boundaries_are_exact(tmp_path):
    learn = tmp_path / "learn"
    learn.mkdir()
    (learn / "old.json").write_bytes(b"12345")
    fs = RecordingFs(tree=1000)
    # a new 10-byte file: 1000 + 10 == quota is allowed, one more byte is not
    LearnWriter(learn, fs=fs, quota_bytes=1010, min_free_bytes=0).write(learn / "n1.json", b"x" * 10)
    with pytest.raises(QuotaExceeded, match="exceed its 1009 byte quota"):
        LearnWriter(learn, fs=fs, quota_bytes=1009, min_free_bytes=0).write(learn / "n2.json", b"x" * 10)
    # replacing the 5-byte file with 10 bytes adds 5
    LearnWriter(learn, fs=fs, quota_bytes=1005, min_free_bytes=0).write(learn / "old.json", b"x" * 10)
    (learn / "old.json").write_bytes(b"12345")
    with pytest.raises(QuotaExceeded, match="exceed its 1004 byte quota"):
        LearnWriter(learn, fs=fs, quota_bytes=1004, min_free_bytes=0).write(learn / "old.json", b"y" * 10)
    assert (learn / "old.json").read_bytes() == b"12345"  # the refused write changed nothing
    assert drpg.LEARN_QUOTA_BYTES == 256 * 1024 * 1024 and drpg.MIN_FREE_BYTES == 2 * 1024 ** 3
    # free space: leaving exactly min_free is allowed, one byte less is not
    LearnWriter(learn, fs=RecordingFs(free=110), min_free_bytes=100).write(learn / "f1.json", b"x" * 10)
    with pytest.raises(LowDiskSpace, match="writing 11 would leave less than 100"):
        LearnWriter(learn, fs=RecordingFs(free=110), min_free_bytes=100).write(learn / "f2.json", b"x" * 11)
    assert not (learn / "f2.json").exists()


def test_one_job_at_a_time_and_a_stale_lock_is_taken_over(tmp_path):
    now = [NOW]
    writer = LearnWriter(tmp_path / "learn", fs=RecordingFs(), min_free_bytes=0)
    lock_path = tmp_path / "learn" / "learn.lock"
    first = RunLock(writer, lock_path, lambda: now[0]).acquire()
    with pytest.raises(LearnerBusy, match="another learner run holds"):
        RunLock(writer, lock_path, lambda: now[0]).acquire()
    now[0] = NOW + drpg.LOCK_STALE_S  # exactly at the limit it is not yet stale
    with pytest.raises(LearnerBusy):
        RunLock(writer, lock_path, lambda: now[0]).acquire()
    now[0] = NOW + drpg.LOCK_STALE_S + 1
    second = RunLock(writer, lock_path, lambda: now[0]).acquire()  # stale: taken over
    assert second.held and lock_path.exists()
    second.release()
    assert not lock_path.exists()
    first.release()  # releasing a lock someone else replaced is harmless here; the file is already gone
    with RunLock(writer, lock_path, lambda: now[0]):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_the_wall_clock_deadline_is_inclusive_of_its_limit():
    now = [0.0]
    deadline = Deadline(600, lambda: now[0])
    now[0] = 600.0
    deadline.check()
    now[0] = 600.5
    with pytest.raises(TimeLimitExceeded, match="wall-clock limit of 600s exceeded"):
        deadline.check()
    assert drpg.EVOLVER_WALL_S == 1200 and drpg.CONSOLIDATE_WALL_S == 600


# ------------------------------------------------------------------------------ OS limits, vitals gate

class RecordingLimits:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def set_low_priority(self):
        self.calls.append("priority")
        if self.fail == "priority":
            raise OSError("access denied")
        return "below_normal"

    def bind_memory_job(self, memory_bytes):
        self.calls.append(("memory", memory_bytes))
        return f"job_memory={memory_bytes}"


def test_os_limits_apply_one_thread_below_normal_and_512_mib():
    env = {}
    limits = RecordingLimits()
    result = apply_os_limits(limits, environ=env)
    assert result == {"priority": "below_normal", "memory": "job_memory=536870912", "threads": 1}
    assert limits.calls == ["priority", ("memory", 512 * 1024 * 1024)]
    assert env == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "NUMEXPR_NUM_THREADS": "1"}


def test_a_cap_that_cannot_be_applied_refuses_to_run_uncapped():
    with pytest.raises(LimitsNotApplied, match=r"OSError: access denied\); refusing to run uncapped"):
        apply_os_limits(RecordingLimits(fail="priority"), environ={})


def test_the_learner_runs_at_l0_only(tmp_path):
    vitals = tmp_path / "v.json"
    assert learner_blocked(vitals, lambda: NOW) == (
        "vitals ladder is not L0 (status=missing, level=None); the learner runs at L0 only")
    write_vitals(vitals, level=0)
    assert learner_blocked(vitals, lambda: NOW) is None
    write_vitals(vitals, level=1)
    assert learner_blocked(vitals, lambda: NOW) == (
        "vitals ladder is not L0 (status=ok, level=1); the learner runs at L0 only")
    write_vitals(vitals, level=0, ts=NOW - 31)  # a stale heartbeat is not L0
    assert "status=stale" in learner_blocked(vitals, lambda: NOW)


# --------------------------------------------------------------------------- learn_run: end to end

def test_c11_prerequisite_refuses_with_a_clear_reason_and_writes_nothing(tmp_path):
    w = World(tmp_path)
    make_policy(w.policy_dir)
    before = snapshot(w.root)
    code, report = w.evolve(apply=True)
    assert code == learn_run.EXIT_PREREQ == 3
    assert report["refused"] == ("C11 prerequisite not met: rows_read == 0 (no trajectory rows exist), "
                                 "so there is nothing to learn from")
    assert report["rows_read"] == 0
    assert snapshot(w.root) == before and not w.learn_dir.exists()
    # an absent trajectory dir is the same refusal, and is not created
    absent = learn_run.run_evolver(policy_dir=w.policy_dir, trajectory_dir=tmp_path / "nope", learn_dir=w.learn_dir,
                                   vitals_state=w.vitals, apply=True, clock=lambda: NOW, fs=w.fs, min_free_bytes=0)
    assert absent[0] == 3 and not (tmp_path / "nope").exists()
    # twin: with rows the same call is not refused
    make_policy(w.policy_dir, pid="cell-b")
    push_rows(w.traj_dir, "cell-b", 100)
    assert w.evolve()[0] == 0


def test_the_learner_refuses_above_l0_and_writes_nothing(tmp_path):
    w = World(tmp_path, level=1)
    make_policy(w.policy_dir)
    push_rows(w.traj_dir, "cell-a", 100)
    before = snapshot(w.root)
    code, report = w.evolve(apply=True)
    assert code == 4 and report["refused"].startswith("vitals ladder is not L0 (status=ok, level=1)")
    assert snapshot(w.root) == before
    write_vitals(w.vitals, level=0)
    assert w.evolve()[0] == 0


def test_dry_run_is_the_default_and_writes_nothing(tmp_path):
    w = World(tmp_path)
    make_policy(w.policy_dir)
    push_rows(w.traj_dir, "cell-a", 100)
    before = snapshot(w.root)
    code, report = w.evolve()
    assert code == 0 and report["mode"] == "dry-run"
    [result] = report["policies"]
    assert result["status"] == "proposed" and result["reason"] == "accepted"
    assert set(result["patch"]["proposed"]) <= {g.name for g in GENES} and result["patch"]["widening"] is False
    assert snapshot(w.root) == before and not w.learn_dir.exists()
    assert w.fs.ops == []  # no write_temp / replace / remove was even attempted


def test_apply_writes_the_patch_the_log_and_the_heldout_ledger_once(tmp_path):
    w = World(tmp_path)
    make_policy(w.policy_dir)
    push_rows(w.traj_dir, "cell-a", 100)
    code, report = w.evolve(apply=True)
    assert code == 0 and report["applied"] == 1 and report["proposed"] == 0
    [result] = report["policies"]
    assert result["status"] == "applied"
    genes = stored_config(w.policy_dir)["learned_genes"]
    assert genes == result["patch"]["proposed"] and genes
    for name, value in genes.items():
        assert GENE_BY_NAME[name].floor <= value <= 1.0
        assert Fraction(str(value)) >= Fraction(9, 10)  # first patch from the default: at most 10%
    log = w.log().entries()
    assert [e["type"] for e in log] == ["patch"] and log[0]["evidence_hash"] == result["patch"]["evidence_hash"]
    assert evidence_hash(log[0]["evidence"]) == log[0]["evidence_hash"]
    ledger = [json.loads(x) for x in (w.learn_dir / "heldout_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(e["policy_family"], e["scored_at"]) for e in ledger] == [("cell-a", NOW)]
    assert not (w.learn_dir / "learn.lock").exists()  # the lock was released
    # the same rows again: the held-out split was consumed, nothing more is learned or written
    before = snapshot(w.root)
    code2, report2 = w.evolve(apply=True)
    assert code2 == 0 and report2["policies"][0]["reason"] == "heldout_already_scored"
    assert snapshot(w.root) == before


def test_a_dry_run_does_not_consume_the_heldout_split(tmp_path):
    w = World(tmp_path)
    make_policy(w.policy_dir)
    push_rows(w.traj_dir, "cell-a", 100)
    first = w.evolve()[1]["policies"][0]
    second = w.evolve()[1]["policies"][0]
    assert first == second and first["status"] == "proposed"
    assert w.evolve(apply=True)[1]["policies"][0]["status"] == "applied"


def test_the_rate_limit_holds_through_the_cli_path(tmp_path):
    w = World(tmp_path)
    backend = make_policy(w.policy_dir)
    push_rows(w.traj_dir, "cell-a", 100)
    apply_patch(plan_patch("cell-a", {}, {"fanout_scale": 0.9}, NOW - HOUR), backend=backend, writer=w.writer(),
                log=w.log(), now=NOW - HOUR)
    code, report = w.evolve(apply=True)
    [result] = report["policies"]
    assert (code, result["status"], result["reason"]) == (0, "skipped", "rate_limited")
    assert stored_config(w.policy_dir)["learned_genes"] == {"fanout_scale": 0.9}
    # twin: 24 h after that patch the same run is allowed to patch again
    write_vitals(w.vitals, ts=NOW - HOUR + DAY)
    code2, report2 = w.evolve(apply=True, clock=lambda: NOW - HOUR + DAY)
    assert report2["policies"][0]["status"] == "applied"


def test_skips_are_reported_not_silent(tmp_path):
    w = World(tmp_path)
    make_policy(w.policy_dir, pid="few")
    push_rows(w.traj_dir, "few", 10)
    push_rows(w.traj_dir, "ghost", 100)  # trajectory rows but no policy
    make_policy(w.policy_dir, pid="bad", config={"learned_genes": {"fanout_scale": 9.0}})
    push_rows(w.traj_dir, "bad", 100)
    code, report = w.evolve()
    by_id = {p["policy_id"]: p for p in report["policies"]}
    assert code == 0 and report["rows_read"] == 210
    assert by_id["few"]["reason"] == "insufficient_rows" and by_id["few"]["train_rows"] + by_id["few"]["heldout_rows"] == 10
    assert (by_id["ghost"]["status"], by_id["ghost"]["reason"]) == ("skipped", "policy_not_found")
    assert by_id["bad"]["gene_problems"] == ["gene fanout_scale=9.0 outside [0.4, 1.0]"]


def test_rows_that_bind_the_tightened_limits_are_not_learned_from(tmp_path):
    w = World(tmp_path)
    make_policy(w.policy_dir)
    push_rows(w.traj_dir, "cell-a", 100, usage={"max_retrieval_hops": 7, "max_refinement_steps": 4, "max_perspectives": 3,
                                                 "max_fanout": 16, "max_context_chars": 8192})
    [result] = w.evolve()[1]["policies"]
    assert (result["status"], result["reason"]) == ("rejected", "train_gain_below_min")


def test_the_learner_cannot_write_the_envelope_or_stop_files_through_a_run(tmp_path):
    w = World(tmp_path)
    make_policy(w.policy_dir)
    push_rows(w.traj_dir, "cell-a", 100)
    policy_bytes = (w.policy_dir / "cell-a.json").read_bytes()
    code, report = w.evolve(apply=True, protected=(w.policy_dir / "cell-a.json",))
    assert code == learn_run.EXIT_STORAGE == 13
    assert report["refused"].startswith("SandboxViolation: ") and "protected path" in report["refused"]
    assert (w.policy_dir / "cell-a.json").read_bytes() == policy_bytes
    assert not (w.learn_dir / "learn.lock").exists()


def test_resource_caps_are_applied_on_the_cli_path_and_a_failure_refuses(tmp_path, monkeypatch):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "8")
    w = World(tmp_path)
    make_policy(w.policy_dir)
    push_rows(w.traj_dir, "cell-a", 100)
    limits = RecordingLimits()
    code, report = w.evolve(limits=limits, apply_limits=True)
    assert code == 0 and limits.calls == ["priority", ("memory", 536870912)]
    assert report["limits"] == {"priority": "below_normal", "memory": "job_memory=536870912", "threads": 1}
    before = snapshot(w.root)
    code2, report2 = w.evolve(limits=RecordingLimits(fail="priority"), apply_limits=True, apply=True)
    assert code2 == learn_run.EXIT_LIMITS and "refusing to run uncapped" in report2["refused"]
    assert snapshot(w.root) == before


def test_the_wall_clock_limit_aborts_without_writing(tmp_path):
    w = World(tmp_path)
    make_policy(w.policy_dir)
    push_rows(w.traj_dir, "cell-a", 100)
    ticks = iter([NOW, NOW, NOW] + [NOW + 1201] * 50)
    before = snapshot(w.root)
    code, report = w.evolve(apply=True, clock=lambda: next(ticks))
    assert code == learn_run.EXIT_TIMEOUT and report["failed"] == "wall-clock limit of 1200s exceeded"
    assert snapshot(w.root) == before


# ------------------------------------------------------------------ revert + status through learn_run

def test_revert_is_dry_run_by_default_and_one_command_with_apply(tmp_path):
    w = World(tmp_path)
    make_policy(w.policy_dir)
    push_rows(w.traj_dir, "cell-a", 100)
    assert w.evolve(apply=True)[1]["applied"] == 1
    patched = snapshot(w.root)
    code, report = learn_run.run_revert(policy_dir=w.policy_dir, learn_dir=w.learn_dir, flags_path=w.flags,
                                        clock=lambda: NOW, fs=w.fs)
    assert code == 0 and report["mode"] == "dry-run" and report["patch"]["cell"] == "cell-a"
    assert snapshot(w.root) == patched
    code2, report2 = learn_run.run_revert(policy_dir=w.policy_dir, learn_dir=w.learn_dir, flags_path=w.flags,
                                          apply=True, clock=lambda: NOW, fs=w.fs)
    assert code2 == 0 and "learned_genes" not in stored_config(w.policy_dir)
    assert report2["reverted"]["review_by"] == "2026-11-17"
    again = learn_run.run_revert(policy_dir=w.policy_dir, learn_dir=w.learn_dir, flags_path=w.flags, apply=True,
                                 clock=lambda: NOW, fs=w.fs)
    assert again[0] == learn_run.EXIT_NOTHING_TO_REVERT and again[1]["refused"] == "no applied patch to revert"


def test_revert_works_above_l0_and_at_low_disk_because_it_is_the_safety_action(tmp_path):
    w = World(tmp_path, level=3)
    backend = make_policy(w.policy_dir)
    apply_patch(plan_patch("cell-a", {}, {"fanout_scale": 0.9}, NOW), backend=backend, writer=w.writer(), log=w.log(),
                now=NOW)
    w.fs.free = 1000  # far below the 2 GB floor
    code, _ = learn_run.run_revert(policy_dir=w.policy_dir, learn_dir=w.learn_dir, flags_path=w.flags, apply=True,
                                   clock=lambda: NOW, fs=w.fs)
    assert code == 0 and "learned_genes" not in stored_config(w.policy_dir)


def _register_stage0(w):
    from policy_prereg.registry import PreregStore, build_mde_table
    record = {"features": ["f"], "estimator": "e", "margin": -0.08, "cluster_unit": "session",
              "mde_table": build_mde_table({"cell-a": 0.9}, 0.08), "decay_constants": {"source": "x"},
              "split_fingerprint": "abc", "pass_rule": {"r": 1}}
    sha = PreregStore(w.learn_dir / "prereg").register("reg1", record)
    inc = {"cost_per_accepted": 1.0, "completion": 0.9, "refusal": 0.05, "p95_latency": 100, "cache_read_share": 0.5}
    arm = {"cost_per_accepted": 0.9, "completion": 0.89, "refusal": 0.06, "p95_latency": 125,
           "cache_read_share": 0.47, "cap_breaches": 0, "governor_escalations": 0}
    (w.learn_dir / "stage0").mkdir(exist_ok=True)
    (w.learn_dir / "stage0" / "cell-a.json").write_text(json.dumps(
        {"registration_id": "reg1", "registration_sha256": sha, "incumbent": inc, "arm": arm}), encoding="utf-8")
    return sha


def test_status_walks_collecting_to_approved_using_the_prereg_stage0_record(tmp_path):
    w = World(tmp_path)
    backend = make_policy(w.policy_dir)
    apply_patch(plan_patch("cell-a", {}, {"fanout_scale": 0.9}, NOW), backend=backend, writer=w.writer(), log=w.log(),
                now=NOW)

    def status():
        code, report = learn_run.run_status(trajectory_dir=w.traj_dir, learn_dir=w.learn_dir)
        assert code == 0
        [p] = report["patches"]
        return p["state"], p["reasons"], p["outcomes"], p["days"], p["stage0"]

    assert status() == (COLLECTING, ["outcomes 0 < 30", "days 0 < 3", "no_independent_windows", "stage0_not_passed"],
                        0, 0, "no stage0 record")
    store = TrajectoryStore(w.traj_dir)
    for i in range(30):  # 30 rows over 3 UTC days, 2 sessions, hours apart
        store.push("cell-a", {"id": f"o{i}", "reward": {"r_ret": 0.5}, "session_id": f"s{i % 2}",
                              "_ts": iso(NOW + HOUR + (i % 3) * DAY + (i // 3) * HOUR)})
    drpg._TRAJ_CACHE.clear()
    state, reasons, n, days, stage0 = status()
    assert (state, reasons, n, days, stage0) == (COLLECTING, ["stage0_not_passed"], 30, 3, "no stage0 record")
    _register_stage0(w)
    assert status() == (APPROVED, [], 30, 3, "all guards passed")
    # the same record with one guard failing is NOT a pass
    rec = json.loads((w.learn_dir / "stage0" / "cell-a.json").read_text(encoding="utf-8"))
    rec["arm"]["cost_per_accepted"] = 0.91
    (w.learn_dir / "stage0" / "cell-a.json").write_text(json.dumps(rec), encoding="utf-8")
    assert status()[0::4] == (COLLECTING, "guards failed: cost_down_10pct")


def test_a_tampered_registration_is_not_a_stage0_pass(tmp_path):
    w = World(tmp_path)
    _register_stage0(w)
    assert learn_run.stage0_passed(w.learn_dir, "cell-a") == (True, "all guards passed")
    reg = w.learn_dir / "prereg" / "reg1.json"
    reg.write_bytes(reg.read_bytes().replace(b'"margin":-0.08', b'"margin":-0.5'))
    passed, why = learn_run.stage0_passed(w.learn_dir, "cell-a")
    assert passed is False and why.startswith("stage0 record unusable: RegistrationTampered: ")
    (w.learn_dir / "stage0" / "cell-a.json").write_text("{not json", encoding="utf-8")
    assert learn_run.stage0_passed(w.learn_dir, "cell-a")[1].startswith("stage0 record unusable: JSONDecodeError")


def test_a_widening_patch_in_the_log_reports_review_required(tmp_path):
    w = World(tmp_path)
    backend = make_policy(w.policy_dir, config={"learned_genes": {"fanout_scale": 0.7}})
    apply_patch(plan_patch("cell-a", {"fanout_scale": 0.7}, {"fanout_scale": 0.75}, NOW), backend=backend,
                writer=w.writer(), log=w.log(), now=NOW, owner_approved=True)
    [p] = learn_run.run_status(trajectory_dir=w.traj_dir, learn_dir=w.learn_dir)[1]["patches"]
    assert (p["state"], p["reasons"]) == (REVIEW_REQUIRED, ["widening_patch_needs_owner"])


def test_main_is_dry_run_without_apply_and_needs_vitals(tmp_path, capsys):
    w = World(tmp_path)
    make_policy(w.policy_dir)
    push_rows(w.traj_dir, "cell-a", 100)
    base = ["--policy-dir", str(w.policy_dir), "--trajectory-dir", str(w.traj_dir), "--learn-dir", str(w.learn_dir),
            "--vitals-state", str(w.vitals)]
    before = snapshot(w.root)
    limits = RecordingLimits()
    assert learn_run.main(base, clock=lambda: NOW, fs=w.fs, limits=limits) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "dry-run"
    assert snapshot(w.root) == before and limits.calls == ["priority", ("memory", 536870912)]
    with pytest.raises(SystemExit):
        learn_run.main(base[:6], clock=lambda: NOW, fs=w.fs, limits=limits)
    assert "--vitals-state is required" in capsys.readouterr().err
    assert learn_run.main(base[:6] + ["--status"], clock=lambda: NOW, fs=w.fs) == 0
    assert json.loads(capsys.readouterr().out) == {"mode": "status", "patches": []}
    assert learn_run.main(["--policy-dir", "x", "--trajectory-dir", "x", "--learn-dir", "F:/learn"]) == learn_run.EXIT_FAILED
    assert "not on C:" in capsys.readouterr().err

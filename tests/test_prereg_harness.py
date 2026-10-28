"""scripts/policy_prereg: pass-rule arithmetic, commit-by-hash registrations, the scored-once ledger.

Everything runs in tmp_path; no network, no live store. Expected numbers are worked out by hand in the
comments, not by calling the function under test.

Mutation log (each applied to scripts/policy_prereg; the named tests went red; then reverted):
- ScoredOnceLedger.record_scoring: the prior-scoring check deleted: test_second_scoring_needs_an_amendment_id,
  test_an_amendment_id_is_single_use, test_scoring_log_gains_no_line_on_a_refusal
- ... the amendment-id reuse check deleted: test_an_amendment_id_is_single_use
- PreregStore.register: open(..., "xb") -> "wb" (overwrites): test_a_different_body_for_the_same_id_is_refused_and_leaves_the_file_alone
- ... the existing-hash comparison deleted (any second write is a no-op): the same test
- required_n_per_arm: math.ceil -> round: test_required_n_for_margin_08_at_p_85_is_247
- ... power 0.8 -> 0.5 in the z sum: test_required_n_for_margin_08_at_p_85_is_247, test_required_n_is_the_smallest_n_that_reaches_the_power
- holm: step-down replaced by Bonferroni (m * p for all): test_holm_matches_the_hand_worked_step_down
- holm: the running max (monotone adjusted p) removed: test_holm_matches_the_hand_worked_step_down
- evaluate_guards: <= -> < on the cost guard / >= -> > on completion: test_guards_pass_exactly_on_their_boundaries
- _q (exact Fraction of the written number) replaced by float: test_guard_boundaries_are_exact_where_float_arithmetic_is_not
- diff_lower_bound: z for alpha=0.05 replaced by 1.96: test_noninferiority_lower_bound_at_n_300_matches_the_design_arithmetic
- wilson_lower_bound: the z^2/(2n) centre term dropped: test_wilson_lower_bound_matches_hand_arithmetic
- canonical_json: sort_keys=False: test_hash_is_independent_of_key_order
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import pytest

_PKG = Path(__file__).resolve().parents[1] / "scripts" / "policy_prereg"


def _load():
    spec = importlib.util.spec_from_file_location(
        "policy_prereg", _PKG / "__init__.py", submodule_search_locations=[str(_PKG)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["policy_prereg"] = mod
    spec.loader.exec_module(mod)
    return importlib.import_module("policy_prereg.stats"), importlib.import_module("policy_prereg.registry")


stats, reg = _load()


# ---- required n per arm (the MDE arithmetic) --------------------------------------------------------
# n = ceil((z_0.95 + z_0.80)^2 * 2 p (1-p) / margin^2), z_0.95 = 1.644854, z_0.80 = 0.841621
# p=0.85, margin=0.08: (2.486475)^2 = 6.182558; 2*0.85*0.15 = 0.255; 0.255 / 0.0064 = 39.84375;
#   6.182558 * 39.84375 = 246.34 -> 247 (246 would leave power at 0.79952 < 0.8)
# p=0.85, margin=0.03: 6.182558 * 0.255 / 0.0009 = 1751.7 -> 1752  (design.md: "~1,235-2,196/arm")
# p=0.50, margin=0.10: 6.182558 * 0.5 / 0.01 = 309.13 -> 310

def test_required_n_for_margin_08_at_p_85_is_247():
    assert stats.required_n_per_arm(0.85, 0.08, alpha=0.05, power=0.8) == 247


@pytest.mark.parametrize("p, margin, expected", [(0.85, 0.03, 1752), (0.5, 0.10, 310)])
def test_required_n_other_cells(p, margin, expected):
    assert stats.required_n_per_arm(p, margin) == expected


def test_required_n_is_the_smallest_n_that_reaches_the_power():
    # Independent power formula: power(n) = Phi(sqrt(n) * margin / sqrt(2 p (1-p)) - z_0.95)
    nd = NormalDist()

    def power(n):
        return nd.cdf(math.sqrt(n) * 0.08 / math.sqrt(2 * 0.85 * 0.15) - nd.inv_cdf(0.95))

    n = stats.required_n_per_arm(0.85, 0.08)
    assert power(n) >= 0.8
    assert power(n - 1) < 0.8


def test_the_mde_table_has_one_row_per_cell_with_its_own_n():
    table = reg.build_mde_table({"code-large": 0.85, "chat-small": 0.5}, margin=0.08)
    # p=0.5: 6.182558 * 0.5 / 0.0064 = 483.01 -> 484
    assert table == {"chat-small": {"p": 0.5, "n_per_arm": 484}, "code-large": {"p": 0.85, "n_per_arm": 247}}


@pytest.mark.parametrize("args", [(0.0, 0.08), (1.0, 0.08), (0.85, 0.0), (0.85, -0.08)])
def test_required_n_refuses_arguments_outside_its_domain(args):
    with pytest.raises(stats.StatsError, match="required_n_per_arm: bad arguments"):
        stats.required_n_per_arm(*args)


# ---- Wilson and the non-inferiority bound -----------------------------------------------------------
# Wilson, k=8 n=10 z=1.96: p=0.8, z^2=3.8416; centre = 0.8 + 3.8416/20 = 0.99208;
#   spread = 1.96*sqrt(0.016 + 3.8416/400) = 1.96*sqrt(0.025604) = 0.313625; denom = 1 + 0.38416 = 1.38416;
#   lb = (0.99208 - 0.313625)/1.38416 = 0.49016
# k=n=100: centre = 1 + 3.8416/200 = 1.019208; spread = 1.96*sqrt(3.8416/40000) = 0.019208;
#   lb = 1.0 / 1.038416 = 0.96301

def test_wilson_lower_bound_matches_hand_arithmetic():
    assert stats.wilson_lower_bound(8, 10) == pytest.approx(0.49016, abs=2e-5)
    assert stats.wilson_lower_bound(100, 100) == pytest.approx(0.96301, abs=2e-5)
    assert stats.wilson_lower_bound(0, 25) == 0.0


def test_wilson_lower_bound_refuses_impossible_counts():
    with pytest.raises(stats.StatsError, match="wilson_lower_bound: need 0 <= successes <= n and n > 0, got 11/10"):
        stats.wilson_lower_bound(11, 10)
    with pytest.raises(stats.StatsError, match="got 0/0"):
        stats.wilson_lower_bound(0, 0)


# Design 5.5: n=300/arm, p=0.85: SE(diff) = sqrt(2 * 0.1275 / 300) = sqrt(0.00085) = 0.029155, so the one-sided
# 95% lower bound at a true difference of 0 is -1.644854 * 0.029155 = -0.047956: fails -0.03, passes -0.08.
def test_noninferiority_lower_bound_at_n_300_matches_the_design_arithmetic():
    lb = stats.diff_lower_bound(255, 300, 255, 300)
    assert lb == pytest.approx(-0.047956, abs=2e-5)
    assert not lb >= -0.03
    assert lb >= -0.08


def test_a_genuinely_worse_arm_fails_the_preregistered_margin():
    # 240/300 = 0.80 vs 255/300 = 0.85: diff -0.05, SE = sqrt(0.16/300 + 0.1275/300) = 0.030957,
    # lb = -0.05 - 1.644854 * 0.030957 = -0.100921 -> below -0.08
    lb = stats.diff_lower_bound(240, 300, 255, 300)
    assert lb == pytest.approx(-0.100921, abs=2e-5)
    assert not lb >= -0.08


# p-value at diff 0, margin 0.08: z = 0.08 / 0.029155 = 2.7439, p = 1 - Phi(2.7439) = 0.003035
def test_noninferiority_pvalue_matches_hand_arithmetic():
    assert stats.noninferiority_pvalue(255, 300, 255, 300, 0.08) == pytest.approx(0.003035, abs=2e-6)
    # a worse arm has a larger p-value: diff -0.05 -> z = 0.03/0.030957 = 0.9691, p = 0.1663
    assert stats.noninferiority_pvalue(240, 300, 255, 300, 0.08) == pytest.approx(0.1663, abs=2e-4)


# ---- Holm -------------------------------------------------------------------------------------------
# p = [0.01, 0.04, 0.03, 0.005], alpha 0.05, m = 4. Sorted: 0.005, 0.01, 0.03, 0.04.
#   thresholds alpha/(m-i): 0.0125, 0.016667, 0.025, 0.05  -> 0.005 ok, 0.01 ok, 0.03 > 0.025 stop.
#   adjusted: 0.005*4 = 0.02; 0.01*3 = 0.03; 0.03*2 = 0.06; 0.04*1 = 0.04 -> running max 0.06
#   input order [0.01, 0.04, 0.03, 0.005] -> adjusted [0.03, 0.06, 0.06, 0.02], reject [T, F, F, T]
# Bonferroni (m*p for all) would give [0.04, 0.16, 0.12, 0.02]: the adjusted values below differ from it.
def test_holm_matches_the_hand_worked_step_down():
    rejects, adjusted = stats.holm([0.01, 0.04, 0.03, 0.005])
    assert rejects == [True, False, False, True]
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_rejects_more_than_bonferroni_and_less_than_uncorrected():
    # p = [0.012, 0.013]: unadjusted both < 0.05; Bonferroni threshold 0.025 rejects both too, so use a pair where
    # only Holm's second step helps: [0.02, 0.04] -> Holm adj [0.04, 0.04] rejects both; Bonferroni adj [0.04, 0.08]
    # rejects only the first.
    rejects, adjusted = stats.holm([0.02, 0.04])
    assert rejects == [True, True]
    assert adjusted == pytest.approx([0.04, 0.04])
    # nothing survives when the smallest is too big: [0.03, 0.04] -> adj [0.06, 0.06]
    assert stats.holm([0.03, 0.04])[0] == [False, False]


def test_holm_refuses_an_empty_family():
    with pytest.raises(stats.StatsError, match="holm: no p-values"):
        stats.holm([])


# ---- the non-statistical guards ---------------------------------------------------------------------

_INC = {"cost_per_accepted": 1.00, "completion": 0.90, "refusal": 0.05, "p95_latency": 2.0, "cache_read_share": 0.40}
_ARM_AT_EDGE = {"cost_per_accepted": 0.90, "completion": 0.89, "refusal": 0.06, "p95_latency": 2.5,
                "cache_read_share": 0.37, "cap_breaches": 0, "governor_escalations": 0}


def test_guards_pass_exactly_on_their_boundaries():
    # cost down exactly 10%, completion exactly -1 pp, refusal exactly +1 pp, p95 exactly 1.25x, cache -3 pp
    assert stats.evaluate_guards(_INC, _ARM_AT_EDGE) == {
        "cost_down_10pct": True, "completion": True, "refusal": True, "p95_latency": True,
        "cache_read_share": True, "no_cap_breaches": True, "no_governor_escalations": True}


def test_guard_boundaries_are_exact_where_float_arithmetic_is_not():
    # Found by a float search: 0.10 - 0.01 gives 0.09000000000000001 (> 0.09), 0.06 + 0.01 gives 0.06999999999999999
    # (< 0.07) and 3.3 * 0.9 gives 2.9699999999999998 (< 2.97). Each case below is exactly on its boundary and
    # must pass; a float implementation fails all three.
    inc = {"cost_per_accepted": 3.3, "completion": 0.10, "refusal": 0.06, "p95_latency": 2.0, "cache_read_share": 0.40}
    arm = {"cost_per_accepted": 2.97, "completion": 0.09, "refusal": 0.07, "p95_latency": 2.5, "cache_read_share": 0.37}
    assert all(stats.evaluate_guards(inc, arm).values())


@pytest.mark.parametrize("guard, override", [
    ("cost_down_10pct", {"cost_per_accepted": 0.91}),
    ("completion", {"completion": 0.88}),
    ("refusal", {"refusal": 0.07}),
    ("p95_latency", {"p95_latency": 2.6}),
    ("cache_read_share", {"cache_read_share": 0.36}),
    ("no_cap_breaches", {"cap_breaches": 1}),
    ("no_governor_escalations", {"governor_escalations": 2}),
])
def test_each_guard_fails_alone_just_past_its_boundary(guard, override):
    result = stats.evaluate_guards(_INC, {**_ARM_AT_EDGE, **override})
    assert result[guard] is False
    assert [g for g, ok in result.items() if not ok] == [guard]


# ---- canonical JSON, the fingerprint, validation ----------------------------------------------------

def _record(**over):
    rec = {
        "features": ["prompt_tokens", "has_tools", "task_class"],
        "estimator": "live cluster-randomised comparison",
        "margin": -0.08,
        "cluster_unit": "session-day",
        "mde_table": reg.build_mde_table({"code-large": 0.85}, margin=0.08),
        "decay_constants": {"source": "loci/model_pool.py", "beta_decay": 0.995},
        "split_fingerprint": reg.split_fingerprint(["r1", "r2", "r3"]),
        "pass_rule": {"cost_down": 0.10, "quality_lb": -0.08, "holm": True},
    }
    rec.update(over)
    return rec


def test_canonical_json_is_sorted_and_compact():
    assert reg.canonical_json({"b": 1, "a": [2, {"d": 4, "c": 3}]}) == b'{"a":[2,{"c":3,"d":4}],"b":1}'


def test_hash_is_independent_of_key_order():
    assert reg.record_hash({"b": 1, "a": 2}) == hashlib.sha256(b'{"a":2,"b":1}').hexdigest()
    assert reg.record_hash({"a": 2, "b": 1}) == reg.record_hash({"b": 1, "a": 2})
    assert reg.record_hash({"a": 2, "b": 2}) != reg.record_hash({"a": 2, "b": 1})


def test_nonfinite_numbers_cannot_be_registered():
    with pytest.raises(reg.RegistrationError, match="not canonical-JSON serialisable"):
        reg.canonical_json({"margin": float("nan")})


def test_split_fingerprint_is_order_independent_and_deduplicated():
    expected = hashlib.sha256(b"a\nb\nc").hexdigest()
    assert reg.split_fingerprint(["c", "a", "b"]) == expected
    assert reg.split_fingerprint(["a", "b", "b", "c", "a"]) == expected
    assert reg.split_fingerprint(["a", "b", "d"]) != expected


def test_a_complete_record_validates():
    assert reg.validate_registration(_record()) is None


@pytest.mark.parametrize("field", reg.REQUIRED_FIELDS)
def test_each_missing_field_is_named(field):
    rec = _record()
    del rec[field]
    with pytest.raises(reg.RegistrationError, match=f"registration missing fields: {field}$"):
        reg.validate_registration(rec)


@pytest.mark.parametrize("override, pattern", [
    ({"margin": 0.08}, r"margin must be a negative number .*got 0\.08"),
    ({"margin": True}, "margin must be a negative number"),
    ({"features": []}, "features must not be empty"),
    ({"cluster_unit": " "}, "cluster_unit must be a non-empty string"),
    ({"mde_table": {"c": {"n_per_arm": 0}}}, r"mde_table\['c'\] needs a positive integer n_per_arm, got 0"),
    ({"decay_constants": {"beta_decay": 0.995}}, "decay_constants must name its single 'source'"),
])
def test_bad_values_are_refused_with_the_reason(override, pattern):
    with pytest.raises(reg.RegistrationError, match=pattern):
        reg.validate_registration(_record(**override))


# ---- commit by hash ---------------------------------------------------------------------------------

def test_register_writes_the_canonical_body_once_and_returns_its_sha256(tmp_path):
    store = reg.PreregStore(tmp_path / "prereg")
    digest = store.register("router-v1", _record())
    body = (tmp_path / "prereg" / "router-v1.json").read_bytes()
    assert body == reg.canonical_json(_record())
    assert digest == hashlib.sha256(body).hexdigest()
    assert store.verify("router-v1", digest)["margin"] == -0.08


def test_registering_the_identical_body_again_is_a_no_op(tmp_path):
    store = reg.PreregStore(tmp_path)
    first = store.register("router-v1", _record())
    reordered = dict(reversed(list(_record().items())))  # same content, different key order
    assert store.register("router-v1", reordered) == first


def test_a_different_body_for_the_same_id_is_refused_and_leaves_the_file_alone(tmp_path):
    store = reg.PreregStore(tmp_path)
    digest = store.register("router-v1", _record())
    before = (tmp_path / "router-v1.json").read_bytes()
    with pytest.raises(reg.RegistrationConflict, match="'router-v1' already exists with a different body"):
        store.register("router-v1", _record(margin=-0.12))
    assert (tmp_path / "router-v1.json").read_bytes() == before
    # positive twin: the changed body is accepted under a NEW id, and the original hash is unchanged
    assert store.register("router-v2", _record(margin=-0.12)) != digest
    assert reg.record_hash(_record()) == digest


def test_verify_detects_a_file_edited_after_commit(tmp_path):
    store = reg.PreregStore(tmp_path)
    digest = store.register("router-v1", _record())
    (tmp_path / "router-v1.json").write_bytes(reg.canonical_json(_record(margin=-0.5)))
    with pytest.raises(reg.RegistrationTampered, match="'router-v1' hashes to"):
        store.verify("router-v1", digest)


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "", ".hidden", "x" * 129])
def test_registration_ids_cannot_escape_the_store(tmp_path, bad_id):
    with pytest.raises(reg.RegistrationError, match="bad registration id"):
        reg.PreregStore(tmp_path).register(bad_id, _record())
    assert list(tmp_path.iterdir()) == []


def test_an_invalid_record_writes_nothing(tmp_path):
    rec = _record()
    del rec["pass_rule"]
    with pytest.raises(reg.RegistrationError, match="registration missing fields: pass_rule"):
        reg.PreregStore(tmp_path).register("router-v1", rec)
    assert list(tmp_path.iterdir()) == []


# ---- held-out scored once ---------------------------------------------------------------------------

FP = reg.split_fingerprint(["r1", "r2", "r3"])


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


def test_second_scoring_needs_an_amendment_id(tmp_path):
    ledger = reg.ScoredOnceLedger(tmp_path / "scored.jsonl", clock=_Clock())
    assert ledger.record_scoring("router", FP) == 1
    with pytest.raises(reg.ScoredTwiceError, match="scoring again needs an amendment id"):
        ledger.record_scoring("router", FP)
    # positive twin: with an amendment id the second scoring is recorded
    assert ledger.record_scoring("router", FP, amendment_id="AMD-1") == 2
    assert ledger.times_scored("router", FP) == 2


def test_a_blank_amendment_id_does_not_count(tmp_path):
    ledger = reg.ScoredOnceLedger(tmp_path / "scored.jsonl", clock=_Clock())
    ledger.record_scoring("router", FP)
    with pytest.raises(reg.ScoredTwiceError, match="needs an amendment id"):
        ledger.record_scoring("router", FP, amendment_id="  ")


def test_an_amendment_id_is_single_use(tmp_path):
    ledger = reg.ScoredOnceLedger(tmp_path / "scored.jsonl", clock=_Clock())
    ledger.record_scoring("router", FP)
    ledger.record_scoring("router", FP, amendment_id="AMD-1")
    with pytest.raises(reg.ScoredTwiceError, match="amendment id 'AMD-1' was already used"):
        ledger.record_scoring("router", FP, amendment_id="AMD-1")
    assert ledger.record_scoring("router", FP, amendment_id="AMD-2") == 3


def test_other_families_and_other_splits_are_scored_independently(tmp_path):
    ledger = reg.ScoredOnceLedger(tmp_path / "scored.jsonl", clock=_Clock())
    ledger.record_scoring("router", FP)
    assert ledger.record_scoring("downgrade", FP) == 1
    assert ledger.record_scoring("router", reg.split_fingerprint(["r9"])) == 1


def test_scoring_log_gains_no_line_on_a_refusal(tmp_path):
    path = tmp_path / "scored.jsonl"
    ledger = reg.ScoredOnceLedger(path, clock=_Clock())
    ledger.record_scoring("router", FP)
    with pytest.raises(reg.ScoredTwiceError, match="amendment"):
        ledger.record_scoring("router", FP)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"policy_family": "router", "split_fingerprint": FP,
                                    "amendment_id": None, "scored_at": 1001.0}


def test_a_corrupt_ledger_line_is_refused_not_skipped(tmp_path):
    path = tmp_path / "scored.jsonl"
    path.write_text('{"policy_family": "router", "split_fingerprint": "x"}\nnot json\n', encoding="utf-8")
    with pytest.raises(reg.LedgerCorruptError, match=r"scored\.jsonl:2: unreadable ledger line"):
        reg.ScoredOnceLedger(path).record_scoring("router", FP)

"""judge_calibration: adapters, calibration metrics and bootstrap, escalation policy, threshold, report CLI."""
import json
import random
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import judge_calibration as J  # noqa: E402
from judge_calibration import CalibrationError, Verdict  # noqa: E402


def _confusion(prefix, model, tp, fp, fn, tn):
    """(judge rows, gold rows) with exactly this confusion matrix (positive = correct)."""
    judge, gold, i = [], [], 0
    for n, (j, g) in ((tp, (True, True)), (fp, (True, False)), (fn, (False, True)), (tn, (False, False))):
        for _ in range(n):
            judge.append({"id": f"{prefix}{i}", "verdict": j, "model": model})
            gold.append({"id": f"{prefix}{i}", "gold_correct": g})
            i += 1
    return judge, gold


@pytest.fixture
def two_models():
    ja, ga = _confusion("a", "model-a", 4, 1, 1, 4)
    jb, gb = _confusion("b", "model-b", 2, 3, 1, 4)
    return ja + jb, ga + gb


class TestGoldAdapter:
    @pytest.mark.parametrize("grade, threshold, want", [(0, 1, False), (1, 1, True), (1, 2, False), (2, 2, True),
                                                        (3, 2, True), (2, 3, False), (3, 3, True)])
    def test_grade_is_correct_at_or_above_the_threshold(self, grade, threshold, want):
        assert J.normalize_gold([{"id": "x", "grade": grade}], threshold) == {"x": want}

    def test_pairs_dicts_and_bools_all_normalise(self):
        rows = [("p1", 3), ("p2", 0), {"id": "d1", "relevance": 2}, {"item": "d2", "gold_correct": False},
                ("p3", True)]
        assert J.normalize_gold(rows) == {"p1": True, "p2": False, "d1": True, "d2": False, "p3": True}

    def test_unlabelled_candidate_rows_are_skipped_not_guessed(self):
        rows = [{"id": "u", "gold_correct": None}, {"id": "k", "gold_correct": True}]
        assert J.normalize_gold(rows) == {"k": True}

    def test_a_grade_out_of_range_is_rejected(self):
        with pytest.raises(CalibrationError, match="gold item 'x': gold value must be a bool or a grade 0..3, got 4"):
            J.normalize_gold([("x", 4)])

    def test_a_repeated_gold_item_is_rejected(self):
        with pytest.raises(CalibrationError, match="gold item 'x' appears twice"):
            J.normalize_gold([("x", 1), ("x", 3)])

    def test_a_row_without_a_label_key_is_rejected(self):
        with pytest.raises(CalibrationError, match="gold row has none of"):
            J.normalize_gold([{"id": "x", "note": "?"}])

    @pytest.mark.parametrize("bad", [0, 4, True, 2.0, "2"])
    def test_the_threshold_must_be_an_int_in_range(self, bad):
        with pytest.raises(CalibrationError, match="threshold must be an int in 1..3"):
            J.normalize_gold([("x", 1)], bad)


class TestJudgeAdapter:
    def test_dict_fields_are_read(self):
        rows = [{"id": "x", "verdict": True, "model": "m", "margin": 0.25, "signal": False}]
        assert J.normalize_judge(rows) == ([Verdict("x", True, "m", 0.25, False)], 0)

    def test_confidence_stands_in_for_margin_and_judge_accept_for_verdict(self):
        rows = [{"decision_id": "y", "judge_accept": False, "judge_model": "m2", "confidence": 1}]
        assert J.normalize_judge(rows) == ([Verdict("y", False, "m2", 1.0, None)], 0)

    def test_the_last_row_of_an_item_wins(self):
        assert J.normalize_judge([("x", True), ("x", False)]) == ([Verdict("x", False)], 0)

    def test_a_null_verdict_is_counted_unusable_and_drops_the_item(self):
        rows = [{"id": "x", "verdict": None}, {"id": "y", "verdict": True}]
        assert J.normalize_judge(rows) == ([Verdict("y", True)], 1)

    def test_a_later_real_verdict_rescues_an_earlier_null(self):
        assert J.normalize_judge([{"id": "x", "verdict": None}, ("x", True)]) == ([Verdict("x", True)], 0)

    def test_a_non_bool_verdict_is_rejected(self):
        with pytest.raises(CalibrationError, match="judge item 'x': verdict must be a bool or null, got 1"):
            J.normalize_judge([("x", 1)])

    def test_a_nan_margin_is_rejected(self):
        with pytest.raises(CalibrationError, match="margin must be a finite number or null"):
            J.normalize_judge([{"id": "x", "verdict": True, "margin": float("nan")}])


class TestCalibrate:
    def test_overall_metrics_are_exact(self, two_models):
        cal = J.calibrate(*two_models, n_boot=10)
        assert {k: cal[k] for k in ("n", "tp", "fp", "fn", "tn")} == {"n": 20, "tp": 6, "fp": 4, "fn": 2, "tn": 8}
        assert cal["agreement"] == 0.7
        assert cal["precision"] == 0.6
        assert cal["recall"] == 0.75
        assert cal["leniency_bias"] == pytest.approx(4 / 12)
        assert cal["kappa"] == pytest.approx(0.4)

    def test_per_model_breakdown_tells_the_models_apart(self, two_models):
        by = J.calibrate(*two_models, n_boot=10)["by_model"]
        assert list(by) == ["model-a", "model-b"]
        a, b = by["model-a"], by["model-b"]
        assert (a["n"], a["tp"], a["fp"], a["fn"], a["tn"]) == (10, 4, 1, 1, 4)
        assert (a["agreement"], a["precision"], a["recall"], a["leniency_bias"]) == (0.8, 0.8, 0.8, 0.2)
        assert a["kappa"] == pytest.approx(0.6)
        assert (b["n"], b["tp"], b["fp"], b["fn"], b["tn"]) == (10, 2, 3, 1, 4)
        assert (b["agreement"], b["precision"]) == (0.6, 0.4)
        assert b["recall"] == pytest.approx(2 / 3)
        assert b["leniency_bias"] == pytest.approx(3 / 7)
        assert b["kappa"] == pytest.approx(0.2)

    def test_a_judge_that_accepts_everything_has_full_leniency_and_zero_kappa(self):
        judge, gold = _confusion("x", "lenient", 5, 5, 0, 0)
        cal = J.calibrate(judge, gold, n_boot=10)
        assert (cal["leniency_bias"], cal["recall"], cal["precision"], cal["agreement"], cal["kappa"]) == (
            1.0, 1.0, 0.5, 0.5, 0.0)

    def test_a_perfect_judge_has_no_leniency_and_kappa_one(self):
        judge, gold = _confusion("x", "good", 5, 0, 0, 5)
        cal = J.calibrate(judge, gold, n_boot=10)
        assert (cal["leniency_bias"], cal["recall"], cal["precision"], cal["agreement"], cal["kappa"]) == (
            0.0, 1.0, 1.0, 1.0, 1.0)

    def test_undefined_ratios_are_none_not_zero(self):
        judge, gold = _confusion("x", "m", 4, 0, 0, 0)       # no gold negatives, kappa undefined
        cal = J.calibrate(judge, gold, n_boot=10)
        assert (cal["leniency_bias"], cal["kappa"], cal["precision"], cal["recall"]) == (None, None, 1.0, 1.0)
        assert cal["ci"]["leniency_bias"] is None

    def test_join_accounting(self):
        judge = [("a", True), ("b", False), {"id": "c", "verdict": None}, ("nogold", True)]
        gold = [("a", True), ("b", True), ("c", True), ("unjudged", False)]
        cal = J.calibrate(judge, gold, n_boot=10)
        assert (cal["n"], cal["n_unusable"], cal["n_no_gold"], cal["n_not_judged"]) == (2, 1, 1, 2)
        assert (cal["tp"], cal["fn"]) == (1, 1)

    def test_pairs_work_before_a_gold_set_exists(self):
        cal = J.calibrate([("a", True), ("b", True), ("c", False)], [("a", 3), ("b", 0), ("c", 0)], n_boot=10)
        assert (cal["tp"], cal["fp"], cal["fn"], cal["tn"], cal["by_model"][""]["n"]) == (1, 1, 0, 1, 3)

    def test_the_gold_threshold_moves_the_metrics(self):
        judge = [("a", True), ("b", True)]
        gold = [("a", 3), ("b", 2)]
        assert J.calibrate(judge, gold, threshold=2, n_boot=10)["fp"] == 0
        assert J.calibrate(judge, gold, threshold=3, n_boot=10)["fp"] == 1


class TestBootstrap:
    def test_same_seed_same_interval_and_other_seed_other_interval(self, two_models):
        a = J.calibrate(*two_models, seed=1, n_boot=200)["ci"]
        assert J.calibrate(*two_models, seed=1, n_boot=200)["ci"] == a
        assert J.calibrate(*two_models, seed=2, n_boot=200)["ci"] != a

    def test_the_interval_brackets_the_point_estimate_and_stays_in_range(self, two_models):
        cal = J.calibrate(*two_models, seed=3, n_boot=500)
        for m in ("agreement", "precision", "recall", "leniency_bias"):
            lo, hi = cal["ci"][m]
            assert 0.0 <= lo <= cal[m] <= hi <= 1.0
            assert lo < hi

    def test_more_data_gives_a_tighter_interval(self):
        small = J.calibrate(*_confusion("s", "m", 6, 4, 2, 8), seed=5, n_boot=400)["ci"]["agreement"]
        big = J.calibrate(*_confusion("b", "m", 60, 40, 20, 80), seed=5, n_boot=400)["ci"]["agreement"]
        assert (big[1] - big[0]) < (small[1] - small[0])

    def test_a_judge_that_is_always_right_has_a_degenerate_interval(self):
        cal = J.calibrate(*_confusion("x", "m", 5, 0, 0, 5), seed=9, n_boot=50)
        assert cal["ci"]["agreement"] == (1.0, 1.0)

    def test_no_pairs_means_no_interval(self):
        assert J.bootstrap_ci([], seed=0, n_boot=10)["agreement"] is None

    def test_n_boot_must_be_positive(self):
        with pytest.raises(CalibrationError, match="n_boot must be a positive int, got 0"):
            J.bootstrap_ci([(True, True)], seed=0, n_boot=0)


def _v(item, verdict=True, margin=None, signal=None):
    return Verdict(item, verdict, "m", margin, signal)


class TestEscalation:
    def _pool(self):
        return [_v("agree-hi", True, 0.9, True), _v("dis-hi", True, 0.9, False), _v("dis-lo", False, 0.1, True),
                _v("low-agree", True, 0.2, True), _v("near-low", True, 0.2000001, True), _v("nosig", True, 0.05)]

    def test_disagreements_come_first_lowest_margin_first(self):
        got = J.select_escalations(self._pool(), call_cap=10)
        assert got == [J.Escalation("dis-lo", "disagreement"), J.Escalation("dis-hi", "disagreement")]

    def test_without_a_second_signal_nothing_is_a_disagreement(self):
        pool = [_v("a", True, 0.1), _v("b", False, 0.2)]
        assert J.select_escalations(pool, call_cap=10) == []

    def test_low_margin_threshold_is_inclusive_and_ranked_after_disagreements(self):
        got = J.select_escalations(self._pool(), call_cap=10, margin_threshold=0.2)
        assert got == [J.Escalation("dis-lo", "disagreement"), J.Escalation("dis-hi", "disagreement"),
                       J.Escalation("nosig", "low_margin"), J.Escalation("low-agree", "low_margin")]

    def test_the_cap_keeps_the_most_urgent(self):
        pool = self._pool()
        assert [e.item for e in J.select_escalations(pool, call_cap=1, margin_threshold=0.2)] == ["dis-lo"]
        assert J.select_escalations(pool, call_cap=0, margin_threshold=0.2, audit_fraction=1.0) == []

    def test_audit_fraction_extremes(self):
        pool = self._pool()
        none = J.select_escalations(pool, call_cap=100, audit_fraction=0.0)
        every = J.select_escalations(pool, call_cap=100, audit_fraction=1.0)
        assert [e.item for e in none] == ["dis-lo", "dis-hi"]
        assert [(e.item, e.reason) for e in every if e.reason == "audit"] == [
            ("agree-hi", "audit"), ("low-agree", "audit"), ("near-low", "audit"), ("nosig", "audit")]

    def test_a_tight_cap_drops_the_audit_before_a_disagreement(self):
        pool = self._pool()
        got = J.select_escalations(pool, call_cap=2, audit_fraction=1.0)
        assert [e.reason for e in got] == ["disagreement", "disagreement"]

    def test_audit_is_seeded_and_independent_of_input_order(self):
        pool = [_v(f"i{n:03d}", True, 0.9, True) for n in range(100)]
        a = J.select_escalations(pool, call_cap=100, audit_fraction=0.3, seed=11)
        shuffled = pool[:]
        random.Random(0).shuffle(shuffled)
        assert J.select_escalations(shuffled, call_cap=100, audit_fraction=0.3, seed=11) == a
        assert J.select_escalations(pool, call_cap=100, audit_fraction=0.3, seed=12) != a
        rng = random.Random(11)
        assert [e.item for e in a] == [p.item for p in pool if rng.random() < 0.3]
        assert len(a) == 37

    @pytest.mark.parametrize("kw, msg", [({"call_cap": -1}, "call_cap must be a non-negative int, got -1"),
                                         ({"call_cap": 1.5}, "call_cap must be a non-negative int, got 1.5"),
                                         ({"call_cap": 1, "audit_fraction": 1.5}, "audit_fraction must be in"),
                                         ({"call_cap": 1, "margin_threshold": float("inf")}, "margin_threshold")])
    def test_bad_parameters_are_rejected(self, kw, msg):
        with pytest.raises(CalibrationError, match=msg):
            J.select_escalations([], **kw)


class TestRecommendThreshold:
    def _items(self):
        # errors (verdict != gold) at margins .1 .2 .3 .9; correct verdicts at .25 .5 .6 .95
        errs = [(0.1, True), (0.2, True), (0.3, True), (0.9, True)]
        oks = [(0.25, False), (0.5, False), (0.6, False), (0.95, False)]
        verdicts, gold = [], {}
        for i, (m, is_err) in enumerate(errs + oks):
            verdicts.append(_v(f"i{i}", True, m))
            gold[f"i{i}"] = not is_err
        verdicts.append(_v("nomargin", True, None))
        gold["nomargin"] = False
        return verdicts, gold

    def test_smallest_cut_that_reaches_the_target(self):
        verdicts, gold = self._items()
        rec = J.recommend_threshold(verdicts, gold, target_capture=0.75)
        assert rec == {"threshold": 0.3, "errors": 4, "caught": 3, "error_capture": 0.75, "escalated": 4,
                       "escalate_fraction": 0.5}

    def test_a_full_target_needs_the_highest_error_margin(self):
        verdicts, gold = self._items()
        rec = J.recommend_threshold(verdicts, gold, target_capture=1.0)
        assert (rec["threshold"], rec["caught"], rec["escalated"]) == (0.9, 4, 7)

    def test_no_errors_means_no_recommendation(self):
        verdicts, gold = self._items()
        assert J.recommend_threshold(verdicts[4:8], gold) is None

    def test_target_must_be_in_range(self):
        with pytest.raises(CalibrationError, match="target_capture must be in"):
            J.recommend_threshold([], {}, 0)


class TestReportCli:
    def _files(self, tmp_path):
        judge = [{"id": "a", "verdict": True, "model": "m1", "margin": 0.9},
                 {"id": "b", "verdict": True, "model": "m1", "margin": 0.3},
                 {"id": "c", "verdict": False, "model": "m1", "margin": 0.8},
                 {"id": "d", "verdict": False, "model": "m1", "margin": 0.7}]
        gold = [{"id": "a", "grade": 3}, {"id": "b", "grade": 1}, {"id": "c", "grade": 0}, {"id": "d", "grade": 2}]
        jp, gp = tmp_path / "judge.jsonl", tmp_path / "gold.jsonl"
        jp.write_text("\n".join(json.dumps(r) for r in judge) + "\n\n", encoding="utf-8")
        gp.write_text("\n".join(json.dumps(r) for r in gold) + "\n", encoding="utf-8")
        return str(jp), str(gp)

    def test_report_prints_the_calibration_and_the_threshold(self, tmp_path, capsys):
        jp, gp = self._files(tmp_path)
        assert J.main(["report", "--judge", jp, "--gold", gp, "--boot", "50", "--target-capture", "1"]) == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert lines[0] == "judge calibration (gold threshold 2, seed 0, boot 50)"
        assert lines[1] == "unusable verdicts 0, judged without gold 0, gold not judged 0"
        assert lines[2] == "overall n=4 tp=1 fp=1 fn=1 tn=1"
        assert lines[3].startswith("    agreement 0.500 [")
        assert "    leniency_bias 0.500 [" in out
        assert "model m1 n=4 tp=1 fp=1 fn=1 tn=1" in out
        assert lines[-1] == ("recommended escalation threshold: margin <= 0.7 catches 2/2 judge errors "
                             "(target 1) by escalating 2 verdicts (50.0%)")

    def test_json_output_carries_the_same_numbers(self, tmp_path, capsys):
        jp, gp = self._files(tmp_path)
        assert J.main(["report", "--judge", jp, "--gold", gp, "--boot", "20", "--json", "--threshold", "3"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert (data["calibration"]["threshold"], data["calibration"]["n"], data["calibration"]["tp"]) == (3, 4, 1)
        assert data["recommended_threshold"]["threshold"] == 0.3

    def test_a_missing_file_is_exit_2_with_a_message(self, tmp_path, capsys):
        jp, _ = self._files(tmp_path)
        assert J.main(["report", "--judge", jp, "--gold", str(tmp_path / "nope.jsonl")]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("error: ")

    def test_a_non_json_line_is_exit_2_naming_the_line(self, tmp_path, capsys):
        jp, gp = self._files(tmp_path)
        Path(gp).write_text('{"id": "a", "grade": 1}\nnot json\n', encoding="utf-8")
        assert J.main(["report", "--judge", jp, "--gold", gp]) == 2
        assert capsys.readouterr().err.startswith(f"error: {gp}:2: not JSON")

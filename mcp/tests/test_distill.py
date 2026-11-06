"""distill: splits keyed by query id, D10 trainer input, promote/hold gate, label budget."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import distill as D

REPO = Path(__file__).resolve().parents[2]


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _gold(n_queries, per_query=2, prefix="q"):
    rows = []
    for i in range(n_queries):
        for k in range(per_query):
            rows.append({"query_id": f"{prefix}{i:02d}", "claim": f"claim {i}", "evidence": f"ev {i} {k}",
                         "label": (i + k) % 2})
    return rows


def _norm(rows, source="gold"):
    return D.normalize_rows(rows, source)


# ------------------------------------------------------------------ splits

def test_split_sizes_and_query_disjoint():
    rows = _norm(_gold(20))
    sp = D.split_rows(rows, seed=7)
    assert {s: len(sp[s]) for s in D.SPLITS} == {"train": 28, "validation": 6, "test": 6}
    ids = {s: {r["query_id"] for r in sp[s]} for s in D.SPLITS}
    assert len(ids["train"]) == 14 and len(ids["validation"]) == 3 and len(ids["test"]) == 3
    assert ids["train"] & ids["validation"] == set()
    assert ids["train"] & ids["test"] == set()
    assert ids["validation"] & ids["test"] == set()
    assert ids["train"] | ids["validation"] | ids["test"] == {f"q{i:02d}" for i in range(20)}


def test_split_deterministic_across_input_order_and_seed_sensitive():
    rows = _norm(_gold(20))
    a = D.split_rows(rows, seed=7)
    b = D.split_rows(list(reversed(rows)), seed=7)
    assert {r["row_id"] for r in a["test"]} == {r["row_id"] for r in b["test"]}
    c = D.split_rows(rows, seed=8)
    assert {r["query_id"] for r in a["test"]} != {r["query_id"] for r in c["test"]}


def test_split_needs_three_queries_and_valid_fracs():
    with pytest.raises(D.DistillError, match="at least 3 distinct query ids"):
        D.split_rows(_norm(_gold(2)), seed=1)
    with pytest.raises(D.DistillError, match="must be in"):
        D.split_rows(_norm(_gold(10)), seed=1, validation_frac=0.6, test_frac=0.5)
    assert len(D.split_rows(_norm(_gold(3)), seed=1)["train"]) == 2  # 1 query, 2 rows


# ------------------------------------------------------------------ rows

def test_normalize_accepts_id_and_gold_correct_bool():
    r = D.normalize_rows([{"id": "a", "claim": "c", "evidence": "e", "gold_correct": True}], "gold")
    assert r == [{"query_id": "a", "row_id": "a#0", "claim": "c", "evidence": "e", "label": 1, "source": "gold"}]


@pytest.mark.parametrize("bad,pattern", [
    ({"claim": "c", "evidence": "e", "label": 1}, "gold row 1: query_id"),
    ({"id": "a", "claim": "", "evidence": "e", "label": 1}, "gold row 1: claim and evidence"),
    ({"id": "a", "claim": "c", "evidence": "e"}, "gold row 1: label"),
    ({"id": "a", "claim": "c", "evidence": "e", "label": 2}, "label must be 0, 1 or bool, got 2"),
])
def test_normalize_rejects_bad_rows(bad, pattern):
    with pytest.raises(D.DistillError, match=pattern):
        D.normalize_rows([bad], "gold")


def test_gold_wins_over_teacher_for_same_query():
    gold = _norm([{"id": "q1", "claim": "c", "evidence": "e", "label": 1}])
    teacher = _norm([{"id": "q1", "claim": "c", "evidence": "e", "label": 0},
                     {"id": "q2", "claim": "c2", "evidence": "e2", "label": 0}], "teacher")
    rows, dropped = D.merge_sources(gold, teacher)
    assert [(r["query_id"], r["label"], r["source"]) for r in rows] == [("q1", 1, "gold"), ("q2", 0, "teacher")]
    assert dropped == 1


# ------------------------------------------------------------------ export + D10 format

def test_export_manifest_files_and_hashes(tmp_path):
    gold = _write(tmp_path / "gold.jsonl", _gold(10))
    teacher = _write(tmp_path / "t.jsonl", _gold(4, prefix="t"))
    out = tmp_path / "out"
    m = D.export(gold, out, seed=3, teacher_path=teacher)
    assert m["seed"] == 3
    assert m["source"]["gold"]["sha256"] == hashlib.sha256(gold.read_bytes()).hexdigest()
    assert m["source"]["teacher"]["rows"] == 8
    assert [m["splits"][s]["rows"] for s in D.SPLITS] == [20, 4, 4]
    assert [m["splits"][s]["queries"] for s in D.SPLITS] == [10, 2, 2]
    for s in D.SPLITS:
        data = (out / f"{s}.jsonl").read_bytes()
        assert m["splits"][s]["sha256"] == hashlib.sha256(data).hexdigest()
        rows = [json.loads(x) for x in data.decode().splitlines()]
        assert m["splits"][s]["labels"] == {"0": sum(r["label"] == 0 for r in rows),
                                            "1": sum(r["label"] == 1 for r in rows)}
        assert m["splits"][s]["queries"] == len({r["query_id"] for r in rows})
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == m
    # same inputs, same seed: byte-identical hashes
    m2 = D.export(gold, tmp_path / "out2", seed=3, teacher_path=teacher)
    assert m2["splits"] == m["splits"]
    m3 = D.export(gold, tmp_path / "out3", seed=4, teacher_path=teacher)
    assert m3["splits"]["test"]["sha256"] != m["splits"]["test"]["sha256"]


def test_d10_train_file_is_exactly_the_trainer_format_and_trainer_reads_it(tmp_path):
    gold = _write(tmp_path / "gold.jsonl", _gold(10))
    out = tmp_path / "out"
    D.export(gold, out, seed=3)
    lines = [json.loads(x) for x in (out / "d10_train.jsonl").read_text(encoding="utf-8").splitlines()]
    train = [json.loads(x) for x in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(train) > 0
    assert all(set(r) == {"claim", "evidence", "label", "signal"} for r in lines)
    assert [(r["claim"], r["evidence"], r["label"], r["signal"]) for r in lines] == \
        [(t["claim"], t["evidence"], t["label"], "gold") for t in train]
    # the real trainer's own reader accepts the file
    spec = importlib.util.spec_from_file_location("train_d10", REPO / "deep_think_loci/grounding/train_d10_pair_gate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_d10"] = mod
    try:
        spec.loader.exec_module(mod)
        rows, digest = mod.read_pairs(out / "d10_train.jsonl")
    finally:
        sys.modules.pop("train_d10", None)
    assert [(r["claim"], r["evidence"], r["label"], r["signal"]) for r in rows] == \
        [(r["claim"], r["evidence"], r["label"], r["signal"]) for r in lines]
    assert digest == hashlib.sha256((out / "d10_train.jsonl").read_bytes()).hexdigest()


def test_trainer_command_exact():
    cmd = D.trainer_command(Path("o/d10_train.jsonl"), "abc123", Path("o/model"), "http://h:11434")
    assert cmd == ("python deep_think_loci/grounding/train_d10_pair_gate.py --loci-commit abc123 "
                   "--dataset o/d10_train.jsonl --out o/model --embed-url http://h:11434")


def test_read_jsonl_reports_line_of_bad_json(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{bad\n', encoding="utf-8")
    with pytest.raises(D.DistillError, match=r"x.jsonl line 2: invalid JSON"):
        D.read_jsonl(p)


# ------------------------------------------------------------------ evaluate

def _split4(labels, teacher=None, prefix="r"):
    rows = []
    for i, y in enumerate(labels):
        r = {"query_id": f"{prefix}{i}", "row_id": f"{prefix}{i}#0", "claim": "c", "evidence": "e",
             "label": y, "source": "gold"}
        if teacher is not None:
            r["teacher_label"] = teacher[i]
        rows.append(r)
    return rows


def _scores(split, vals):
    return {r["row_id"]: v for r, v in zip(split, vals)}


def test_evaluate_exact_metrics():
    sp = _split4([1, 1, 0, 0])
    m = D.evaluate(_scores(sp, [0.9, 0.4, 0.6, 0.1]), sp, 0.5)
    assert m == {"n": 4, "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "auc": 0.75}
    perfect = D.evaluate(_scores(sp, [0.9, 0.8, 0.2, 0.1]), sp, 0.5)
    assert (perfect["accuracy"], perfect["f1"], perfect["auc"]) == (1.0, 1.0, 1.0)


def test_evaluate_threshold_is_inclusive_and_missing_scores_refuse():
    sp = _split4([1, 0])
    assert D.evaluate(_scores(sp, [0.5, 0.49]), sp, 0.5)["accuracy"] == 1.0
    with pytest.raises(D.DistillError, match="1 rows have no score, first: r1#0"):
        D.evaluate({"r0#0": 0.5}, sp, 0.5)
    with pytest.raises(D.DistillError, match="gold split is empty"):
        D.evaluate({}, [], 0.5)


# ------------------------------------------------------------------ decide

CRIT = {"metric": "accuracy", "threshold": 0.5, "min_margin_over_incumbent": 0.05,
        "max_teacher_gap_on_validation": 0.1, "min_test_rows": 4, "min_validation_rows": 4}


def _fixture():
    test = _split4([1, 1, 0, 0], prefix="t")
    val = _split4([1, 0, 1, 0], teacher=[1, 0, 1, 1], prefix="v")  # teacher accuracy vs gold = 0.75
    good_test = _scores(test, [0.9, 0.8, 0.2, 0.1])      # accuracy 1.0
    inc_test = _scores(test, [0.9, 0.4, 0.6, 0.1])       # accuracy 0.5
    good_val = _scores(val, [0.9, 0.1, 0.8, 0.2])        # accuracy 1.0
    return test, val, good_test, inc_test, good_val


def test_decide_promotes_when_all_criteria_hold():
    test, val, good_test, inc_test, good_val = _fixture()
    res = D.decide(CRIT, test, val, good_test, inc_test, good_val)
    assert res["decision"] == "promote"
    assert res["reasons"] == ["all criteria met"]
    assert res["metrics"]["student_test"]["accuracy"] == 1.0
    assert res["metrics"]["incumbent_test"]["accuracy"] == 0.5
    assert res["metrics"]["teacher_validation"]["accuracy"] == 0.75


def test_decide_holds_when_student_does_not_beat_incumbent_by_margin():
    test, val, good_test, _, good_val = _fixture()
    res = D.decide(CRIT, test, val, good_test, dict(good_test), good_val)
    assert res["decision"] == "hold"
    assert res["reasons"] == ["test accuracy 1.0000 does not beat incumbent 1.0000 by 0.05"]


def test_decide_holds_when_student_far_below_teacher_on_validation():
    test, val, good_test, inc_test, _ = _fixture()
    bad_val = _scores(val, [0.9, 0.8, 0.8, 0.8])         # accuracy 0.5 < 0.75 - 0.1
    res = D.decide(CRIT, test, val, good_test, inc_test, bad_val)
    assert res["decision"] == "hold"
    assert res["reasons"] == ["validation accuracy 0.5000 is below teacher 0.7500 by more than 0.1"]
    ok_val = _scores(val, [0.9, 0.8, 0.8, 0.1])          # accuracy 0.75: at teacher level, inside tolerance
    assert D.decide(CRIT, test, val, good_test, inc_test, ok_val)["decision"] == "promote"


def test_decide_margin_boundary_is_inclusive():
    test, val, _, _, good_val = _fixture()
    student = _scores(test, [0.9, 0.8, 0.2, 0.6])        # accuracy 0.75
    incumbent = _scores(test, [0.9, 0.4, 0.2, 0.6])      # accuracy 0.5
    crit = dict(CRIT, min_margin_over_incumbent=0.25)
    assert D.decide(crit, test, val, student, incumbent, good_val)["decision"] == "promote"
    crit = dict(CRIT, min_margin_over_incumbent=0.26)
    assert D.decide(crit, test, val, student, incumbent, good_val)["decision"] == "hold"


def test_decide_holds_on_small_splits_and_missing_teacher_labels():
    test, val, good_test, inc_test, good_val = _fixture()
    res = D.decide(dict(CRIT, min_test_rows=5), test, val, good_test, inc_test, good_val)
    assert res["decision"] == "hold"
    assert res["reasons"] == ["test split has 4 rows, need 5"]
    val_nt = _split4([1, 0, 1, 0], prefix="v")
    res = D.decide(CRIT, test, val_nt, good_test, inc_test, _scores(val_nt, [0.9, 0.1, 0.8, 0.2]))
    assert res["decision"] == "hold"
    assert res["reasons"] == ["4 validation rows lack teacher_label"]


def test_load_criteria_requires_all_keys_and_known_metric(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(CRIT), encoding="utf-8")
    assert D.load_criteria(p) == CRIT
    p.write_text(json.dumps({k: v for k, v in CRIT.items() if k != "threshold"}), encoding="utf-8")
    with pytest.raises(D.DistillError, match=r"missing keys \['threshold'\]"):
        D.load_criteria(p)
    p.write_text(json.dumps(dict(CRIT, metric="vibes")), encoding="utf-8")
    with pytest.raises(D.DistillError, match="unknown metric 'vibes'"):
        D.load_criteria(p)


# ------------------------------------------------------------------ budget

def test_plan_extrapolates_stop_point_and_budget():
    curve = [(100, 0.60), (200, 0.70), (300, 0.75)]  # slopes 0.10, 0.05 per 100 labels: ratio 0.5
    res = D.plan_budget(12, 0.5, curve, min_gain_per_step=0.01, step_labels=100)
    assert res["affordable_labels"] == 24
    assert res["stop_labels"] == 600                  # ceil(log(0.01/0.05)/log(0.5)) = 3 more steps
    assert res["additional_labels_needed"] == 300
    assert res["stop_within_budget"] is False
    assert res["saturated"] is False
    res = D.plan_budget(200, 0.5, curve, 0.01, 100)
    assert res["affordable_labels"] == 400 and res["stop_within_budget"] is True


def test_plan_saturated_curve_stops_now():
    res = D.plan_budget(10, 0.5, [(100, 0.800), (200, 0.805)], 0.01, 100)
    assert res["saturated"] is True
    assert res["stop_labels"] == 200
    assert res["additional_labels_needed"] == 0


def test_plan_cannot_estimate_without_decay_or_points():
    res = D.plan_budget(10, 0.5, [(100, 0.5), (200, 0.55), (300, 0.65)], 0.01, 100)  # slope growing
    assert res["stop_labels"] is None and res["saturated"] is False
    assert res["note"] == "slope is not decaying; no stop point estimable"
    assert "stop_within_budget" not in res
    res = D.plan_budget(10, 0.5, [(100, 0.5)], 0.01, 100)
    assert res["stop_labels"] is None
    assert res["note"] == "need at least 2 curve points to judge saturation"


def test_plan_rejects_bad_inputs():
    with pytest.raises(D.DistillError, match="cost_per_label_usd"):
        D.plan_budget(10, 0, [], 0.01, 100)
    with pytest.raises(D.DistillError, match="duplicate label counts"):
        D.plan_budget(10, 0.5, [(100, 0.5), (100, 0.6)], 0.01, 100)


# ------------------------------------------------------------------ CLI

def test_cli_export_prints_manifest_and_command(tmp_path, capsys):
    gold = _write(tmp_path / "gold.jsonl", _gold(10))
    rc = D.main(["export", "--gold", str(gold), "--out", str(tmp_path / "o"), "--seed", "3",
                 "--loci-commit", "abc", "--embed-url", "http://h:1", "--model-out", str(tmp_path / "m")])
    out = capsys.readouterr().out.rstrip("\n").splitlines()
    assert rc == 0
    assert out[-1] == D.trainer_command(tmp_path / "o" / "d10_train.jsonl", "abc", tmp_path / "m", "http://h:1")
    assert (tmp_path / "o" / "manifest.json").is_file()


def test_cli_gate_exit_codes_and_error(tmp_path, capsys):
    test, val, good_test, inc_test, good_val = _fixture()
    f = {"criteria": tmp_path / "c.json"}
    f["criteria"].write_text(json.dumps(CRIT), encoding="utf-8")
    t, v = _write(tmp_path / "t.jsonl", test), _write(tmp_path / "v.jsonl", val)

    def run(student_test):
        (tmp_path / "st.json").write_text(json.dumps(student_test), encoding="utf-8")
        (tmp_path / "it.json").write_text(json.dumps(inc_test), encoding="utf-8")
        (tmp_path / "sv.json").write_text(json.dumps(good_val), encoding="utf-8")
        return D.main(["gate", "--criteria", str(f["criteria"]), "--test", str(t), "--validation", str(v),
                       "--student-test", str(tmp_path / "st.json"), "--incumbent-test", str(tmp_path / "it.json"),
                       "--student-validation", str(tmp_path / "sv.json")])

    assert run(good_test) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "promote"
    assert run(inc_test) == 1
    assert json.loads(capsys.readouterr().out)["decision"] == "hold"
    assert run({}) == 2
    assert "4 rows have no score" in capsys.readouterr().err

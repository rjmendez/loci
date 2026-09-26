"""d10_shadow_report joins the shadow log with judge verdicts, aggregates only, and writes nothing."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d10_shadow_report as R  # noqa: E402


def _row(call, inv, fid, cos_keep, mlp_keep, **kw):
    return {"schema": 1, "event": "d10_shadow", "call_id": call, "investigation_id": inv, "finding_id": fid,
            "n_pairs": 4, "n_logged": 4, "cos_keep": cos_keep, "mlp_keep": mlp_keep,
            "cos_in_context": cos_keep, "mlp_in_context": mlp_keep, "latency_us_per_pair": kw.get("lat", 10.0),
            "gate_version": "v1+abc", "shadow_errors_total": kw.get("err", 0)}


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _tree(tmp_path):
    mem = tmp_path / "memory-sessions"
    log = tmp_path / "instrumentation" / "d10_shadow.jsonl"
    _write(log.with_name(log.name + ".1"), [_row("c1", "inv-a", "f1", True, True, lat=5.0)])
    _write(log, [
        _row("c1", "inv-a", "f2", True, False, lat=7.0),
        _row("c1", "inv-a", "f3", False, False),
        _row("c1", "inv-a", "f4", False, True, err=2),
        _row("c2", "../escape", "f9", True, True),
        {"event": "something_else"},
    ])
    _write(mem / "inv-a" / "judge_verdicts.jsonl", [
        {"new_finding_id": "f1", "neighbor_id": "f2", "verdict": "agree", "judge_ok": True},
        {"new_finding_id": "f1", "neighbor_id": "f3", "verdict": "contradict", "judge_ok": True},
        {"new_finding_id": "f3", "neighbor_id": "f4", "verdict": "same_topic_no_conflict", "judge_ok": True},
        {"new_finding_id": "f1", "neighbor_id": "zz", "verdict": "agree", "judge_ok": True},
        {"new_finding_id": "f1", "neighbor_id": "f4", "verdict": None, "judge_ok": False},
    ])
    return mem, log


def _snapshot(root):
    return {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}


def test_report_aggregates_gates_latency_and_the_judge_join(tmp_path):
    mem, log = _tree(tmp_path)
    out = R.report(log, mem)
    assert out["n_rows"] == 5 and out["n_calls"] == 2 and out["n_investigations"] == 2
    assert out["keep_vs_keep"] == {"both_keep": 2, "both_drop": 1, "cosine_only_keep": 1,
                                   "mlp_only_keep": 1, "agreement": 0.6}
    assert out["shadow_errors_total_max"] == 2
    assert out["latency_us_per_pair"]["max"] == 10.0
    j = out["judge"]
    assert j["investigation_ids_skipped_as_unsafe"] == 1
    assert j["verdict_rows_read"] == 5
    # f1 keep/keep, f2 keep/drop: agree pair -> cosine both_keep, mlp split
    assert j["by_verdict"]["agree"] == {"joined": 1, "cosine_both_keep": 1, "mlp_split": 1,
                                        "not_in_one_shadow_call": 1}
    # f1 keep/keep, f3 drop/drop: both gates split
    assert j["by_verdict"]["contradict"] == {"joined": 1, "cosine_split": 1, "mlp_split": 1}
    # f3 drop/drop, f4 drop/keep
    assert j["by_verdict"]["same_topic_no_conflict"] == {"joined": 1, "cosine_both_drop": 1, "mlp_split": 1}
    assert j["by_verdict"]["no_verdict"] == {"joined": 1, "cosine_split": 1, "mlp_both_keep": 1}
    assert j["cosine_coherence_on_judged_pairs"] == round(2 / 3, 4)
    assert j["mlp_coherence_on_judged_pairs"] == 0.0


def test_report_is_read_only(tmp_path, capsys):
    mem, log = _tree(tmp_path)
    before = _snapshot(tmp_path)
    assert R.main(["--memory-dir", str(mem), "--shadow-log", str(log)]) == 0
    assert _snapshot(tmp_path) == before
    assert json.loads(capsys.readouterr().out)["n_rows"] == 5


def test_default_log_sits_beside_the_memory_dir(tmp_path, monkeypatch, capsys):
    mem, log = _tree(tmp_path)
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(mem))
    assert R.main([]) == 0
    assert json.loads(capsys.readouterr().out)["shadow_log"] == str(log)


def test_an_empty_log_reports_zeros(tmp_path):
    out = R.report(tmp_path / "missing.jsonl", tmp_path)
    assert out["n_rows"] == 0 and out["keep_vs_keep"]["agreement"] is None
    assert out["judge"]["cosine_coherence_on_judged_pairs"] is None

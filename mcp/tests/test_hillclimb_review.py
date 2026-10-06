"""Reviewing a model's proposed labels: a proposal is never a label until a person has answered it.

2026-10-06: the owner's own 15 hand labels turned out to be careless, and a model's rule-based labels agreed with
them 1 time in 15, so neither side was trustworthy. A review mode turns the model's proposals into checked gold cheaply
and measures, per rule, how often the person accepted them, which is the model's accuracy on that rule.
"""
import json
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import hillclimb as H  # noqa: E402

SUITE = "reflection_triage"


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "mem" / "sessions"))
    H._overlay_cache.clear()
    return tmp_path


def _obs(i, key):
    return {"status": "processed", "kind": "claude_code_event", "path": f"workflows/wf_x/agent-{i}.jsonl",
            "events": {"user": i + 1}, "tools": {}, "errors": {key: 1}, "warnings": {}}


def _setup(rules):
    """``rules``: {rule name: [category, ...]}. One observation per proposal; returns {id: proposal}."""
    keys = [f"claude tool_result error: failure {r} {j}" for r, cats in rules.items() for j, _ in enumerate(cats)]
    items = [_obs(i, k) for i, k in enumerate(keys)]
    H.capture_observations(items)
    obs = H._read_jsonl(H.suite_dir(SUITE) / "observations.jsonl")
    by_key = {next(iter(o["errors"])): o for o in obs}
    props, k = {}, 0
    for r, cats in rules.items():
        for j, cat in enumerate(cats):
            o = by_key[f"claude tool_result error: failure {r} {j}"]
            props[o["id"]] = {"id": o["id"], "gold": cat, "rule": r, "note": f"{r}: evidence {j}", "labeler": "claude-rules-test"}
            k += 1
    path = H.suite_dir(SUITE) / H.PROPOSALS_FILE
    path.write_text("".join(json.dumps(p) + "\n" for p in props.values()), encoding="utf-8")
    return props


def _answers(*a):
    it = iter(a)
    return lambda prompt: next(it)


def _labels():
    return H._read_jsonl(H.suite_dir(SUITE) / "labels.jsonl")


class TestAnswers:
    def test_enter_accepts_the_proposal_and_writes_a_reviewed_label(self):
        props = _setup({"path-missing": ["config_or_environment"]})
        out = []
        t = H.review(n=5, input_fn=_answers(""), print_fn=out.append)
        assert (t["accepted"], t["changed"], t["rejected"], t["skipped"]) == (1, 0, 0, 0)
        (lab,) = _labels()
        p = next(iter(props.values()))
        assert (lab["id"], lab["gold"], lab["proposed"], lab["rule"]) == (p["id"], "config_or_environment", "config_or_environment", "path-missing")
        assert lab["labeler"] == "claude-rules-test+human-review"
        assert set(lab) >= {"id", "gold", "kind", "path", "events", "tools", "errors", "warnings", "novelty", "note"}

    def test_a_letter_changes_the_proposal_and_keeps_what_was_proposed(self):
        _setup({"path-missing": ["config_or_environment"]})
        t = H.review(n=5, input_fn=_answers("r"), print_fn=lambda s: None)
        assert (t["accepted"], t["changed"]) == (0, 1)
        (lab,) = _labels()
        assert (lab["gold"], lab["proposed"]) == ("real_regression", "config_or_environment")

    def test_every_letter_maps_to_its_category(self):
        for letter, cat in H.LABELS.items():
            for f in (H.suite_dir(SUITE) / "labels.jsonl", H.suite_dir(SUITE) / H.PROPOSALS_FILE):
                if f.exists():
                    f.unlink()
            if (H.suite_dir(SUITE) / "observations.jsonl").exists():
                (H.suite_dir(SUITE) / "observations.jsonl").unlink()
            _setup({"r": ["unknown" if cat != "unknown" else "noise_or_benign"]})
            H.review(n=1, input_fn=_answers(letter.upper()), print_fn=lambda s: None)
            assert _labels()[-1]["gold"] == cat

    def test_a_longer_word_counts_by_its_first_letter(self):
        _setup({"r": ["unknown"]})
        H.review(n=1, input_fn=_answers("noise"), print_fn=lambda s: None)
        assert _labels()[0]["gold"] == "noise_or_benign"

    def test_x_rejects_the_observation_and_writes_no_label(self):
        props = _setup({"r": ["noise_or_benign"]})
        t = H.review(n=5, input_fn=_answers("x", "harness text"), print_fn=lambda s: None)
        assert t["rejected"] == 1 and _labels() == []
        (row,) = H._read_jsonl(H.suite_dir(SUITE) / "rejected.jsonl")
        assert row["id"] in props and row["note"] == "harness text" and row["keys"]

    def test_skip_is_remembered_for_review_only_and_not_labelled(self):
        props = _setup({"r": ["noise_or_benign"]})
        t = H.review(n=5, input_fn=_answers("s"), print_fn=lambda s: None)
        assert t["skipped"] == 1 and _labels() == []
        assert H._read_json(H.suite_dir(SUITE) / "review_skipped.json")["ids"] == sorted(props)
        assert not (H.suite_dir(SUITE) / "skipped.json").exists()                   # label()'s own skips are separate
        out = []
        again = H.review(n=5, input_fn=_answers(), print_fn=out.append)
        assert again["accepted"] == 0 and out[0].startswith("0 to review") and "1 skipped in review" in out[0]

    def test_quit_stops_and_keeps_what_was_answered(self):
        _setup({"r": ["noise_or_benign", "unknown", "unknown"]})
        t = H.review(n=5, input_fn=_answers("", "q"), print_fn=lambda s: None)
        assert t["accepted"] == 1 and len(_labels()) == 1

    def test_an_invalid_key_asks_again(self):
        _setup({"r": ["noise_or_benign"]})
        out = []
        H.review(n=1, input_fn=_answers("z", "7", ""), print_fn=out.append)
        assert len(_labels()) == 1 and sum(1 for line in out if "Enter r f c n u x s q" in line) == 2


class TestSelection:
    def test_what_a_person_already_answered_is_never_offered_again(self):
        props = _setup({"r": ["noise_or_benign", "unknown", "unknown"]})
        ids = sorted(props)
        (H.suite_dir(SUITE) / "labels.jsonl").write_text(json.dumps({"id": ids[0], "gold": "unknown"}) + "\n", encoding="utf-8")
        (H.suite_dir(SUITE) / "rejected.jsonl").write_text(json.dumps({"id": ids[1], "keys": ["k"], "note": ""}) + "\n", encoding="utf-8")
        out = []
        H.review(n=9, input_fn=_answers("", "q"), print_fn=out.append)
        assert out[0].startswith("1 to review (3 proposed, 2 already labelled or rejected")
        assert _labels()[-1]["id"] == ids[2]

    def test_a_proposal_for_an_observation_that_no_longer_exists_is_ignored(self):
        _setup({"r": ["noise_or_benign"]})
        path = H.suite_dir(SUITE) / H.PROPOSALS_FILE
        path.write_text(path.read_text(encoding="utf-8") + json.dumps({"id": "r_gone", "gold": "unknown", "rule": "r"}) + "\n", encoding="utf-8")
        out = []
        H.review(n=9, input_fn=_answers("q"), print_fn=out.append)
        assert out[0].startswith("1 to review (1 proposed")

    def test_no_proposals_file_reviews_nothing(self):
        H.capture_observations([_obs(1, "claude tool_result error: x")])
        out = []
        t = H.review(n=5, input_fn=_answers(), print_fn=out.append)
        assert out[0].startswith("0 to review (0 proposed") and t["accepted"] == 0

    def test_the_first_items_cover_the_rules_largest_first_not_just_the_biggest_rule(self):
        _setup({"big": ["unknown"] * 6, "mid": ["unknown"] * 3, "small": ["unknown"]})
        out = []
        H.review(n=3, input_fn=_answers("s", "s", "s"), print_fn=out.append)
        shown = [line.split("rule: ")[1].splitlines()[0] for line in out if "rule: " in line]
        assert shown == ["big", "mid", "small"]

    def test_the_order_is_deterministic_for_a_seed_and_changes_with_it(self):
        _setup({"a": ["unknown"] * 8})
        proposals = H._read_jsonl(H.suite_dir(SUITE) / H.PROPOSALS_FILE)
        key = lambda p: p["rule"]  # noqa: E731
        one = [p["id"] for p in H._stratified(proposals, key, 7)]
        assert one == [p["id"] for p in H._stratified(proposals, key, 7)]
        assert one != [p["id"] for p in H._stratified(proposals, key, 8)]
        assert sorted(one) == sorted(p["id"] for p in proposals)

    def test_n_limits_how_many_are_shown(self):
        _setup({"r": ["unknown"] * 5})
        out = []
        H.review(n=2, input_fn=_answers("", ""), print_fn=out.append)
        assert len(_labels()) == 2 and out[0].startswith("2 to review (5 proposed")


class TestAccuracyReport:
    def test_it_reports_per_rule_how_often_the_proposal_was_accepted(self):
        _setup({"a": ["unknown", "unknown"], "b": ["noise_or_benign"]})
        out = []
        t = H.review(n=3, input_fn=_answers("", "r", "x", ""), print_fn=out.append)    # a: accept; b: reject; a: change ... order below
        assert t["by_rule"]["a"]["shown"] == 2 and t["by_rule"]["b"]["shown"] == 1
        assert sum(s["accepted"] for s in t["by_rule"].values()) == t["accepted"]
        text = "\n".join(out)
        assert f"accepted {t['accepted']}/3 reviewed proposals" in text
        assert f"{t['by_rule']['a']['accepted']}/2  a" in text and "0/1  b" in text

    def test_a_skipped_item_is_not_part_of_the_accuracy(self):
        _setup({"a": ["unknown", "unknown"]})
        out = []
        t = H.review(n=2, input_fn=_answers("s", ""), print_fn=out.append)
        assert t["by_rule"]["a"] == {"shown": 1, "accepted": 1}
        assert "accepted 1/1 reviewed proposals (100%)" in "\n".join(out)

    def test_a_changed_label_to_the_same_category_counts_as_accepted(self):
        _setup({"a": ["real_regression"]})
        t = H.review(n=1, input_fn=_answers("r"), print_fn=lambda s: None)
        assert (t["accepted"], t["changed"]) == (1, 0)
        assert _labels()[0]["proposed"] == "real_regression"


class TestInteractionWithLabel:
    def test_reviewed_labels_count_as_labelled_for_label_and_stats(self):
        _setup({"a": ["unknown", "noise_or_benign"]})
        H.review(n=1, input_fn=_answers(""), print_fn=lambda s: None)
        assert H.label_stats(SUITE)["labelled"] == 1
        out = []
        H.label(n=5, input_fn=_answers("q"), print_fn=out.append)
        assert out[0].startswith("1 to label (2 captured, 1 labelled or rejected")      # the reviewed one is not asked again

    def test_a_labelled_observation_is_not_proposed_again_in_review(self):
        _setup({"a": ["unknown", "noise_or_benign"]})
        H.review(n=1, input_fn=_answers(""), print_fn=lambda s: None)
        out = []
        H.review(n=5, input_fn=_answers("q"), print_fn=out.append)
        assert out[0].startswith("1 to review (2 proposed, 1 already labelled or rejected")

    def test_the_cli_flag_runs_the_review_with_its_arguments(self, monkeypatch, capsys):
        seen = []
        monkeypatch.setattr(H, "review", lambda suite, n, proposals_file: (seen.append((suite, n, proposals_file)), {"accepted": 1})[1])
        assert H._main(["label", "--review", "--n", "3", "--proposals", "mine.jsonl"]) == 0
        assert seen == [("reflection_triage", 3, "mine.jsonl")] and '"accepted": 1' in capsys.readouterr().out
        assert H._main(["label", "--review"]) == 0
        assert seen[-1] == ("reflection_triage", 30, H.PROPOSALS_FILE)

    def test_without_the_flag_the_cli_still_labels_from_scratch(self, monkeypatch, capsys):
        monkeypatch.setattr(H, "review", lambda *a, **k: pytest.fail("--review not given"))
        monkeypatch.setattr(H, "label", lambda *a, **k: {"labelled": 0})
        assert H._main(["label"]) == 0

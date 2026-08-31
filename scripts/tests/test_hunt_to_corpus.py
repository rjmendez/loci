"""Hunt output has to reach the corpus, and must not poison it on the way.

Two days of hunting produced 1,139 adjudicated findings with real text. The
grounding builder globs dt-loci-*/findings.jsonl and the hunts wrote plain JSON
into ~/.loci/scratch/, so the corpus stayed at 303 findings across both days and
the retrained model came out bit-identical.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "hunt_to_corpus.py"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("h2c", SCRIPT)
    assert spec and spec.loader, f"cannot load {SCRIPT}"
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _hunt(tmp_path, rows, name="FINAL-verdicts.json"):
    d = tmp_path / "20260101-0000-demo-hunt"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(rows))
    return d


def _run(mod, monkeypatch, hunt, corpus):
    monkeypatch.setattr(sys, "argv",
                        ["hunt_to_corpus.py", str(hunt), "--corpus", str(corpus)])
    assert mod.main() == 0
    return [json.loads(l) for f in corpus.rglob("findings.jsonl")
            for l in f.read_text().splitlines() if l.strip()]


def test_a_record_without_text_is_never_written(mod, monkeypatch, tmp_path):
    """The builder was fixed in #241 to drop text-less rows. Writing them from
    this end would undo that fix: two empty strings embed identically, score
    cos=1.0, and become a positive training pair."""
    hunt = _hunt(tmp_path, [
        {"claim": "a real finding about the thing", "file": "a/b.py", "verdict": "CONFIRMED"},
        {"claim": "", "file": "a/b.py", "verdict": "CONFIRMED"},
        {"claim": "   ", "file": "a/c.py", "verdict": "REFUTED"},
    ])
    out = _run(mod, monkeypatch, hunt, tmp_path / "corpus")
    assert len(out) == 1
    assert all(r["text"].strip() for r in out)


def test_rerunning_does_not_duplicate(mod, monkeypatch, tmp_path):
    """Ids are content hashes. A duplicate would silently reweight the corpus,
    and the builder keeps one row per id, so the drift would be invisible."""
    hunt = _hunt(tmp_path, [{"claim": "finding one", "file": "x.py", "verdict": "CONFIRMED"},
                            {"claim": "finding two", "file": "y.py", "verdict": "DARK"}])
    first = _run(mod, monkeypatch, hunt, tmp_path / "corpus")
    second = _run(mod, monkeypatch, hunt, tmp_path / "corpus")
    assert len(first) == len(second) == 2
    assert {r["id"] for r in first} == {r["id"] for r in second}


def test_the_output_is_globbed_by_the_builder(mod, monkeypatch, tmp_path):
    """dt-loci-* is the builder's glob. A different prefix means the export runs,
    reports success, and the model never sees any of it — which is the failure
    this script exists to end."""
    hunt = _hunt(tmp_path, [{"claim": "a finding", "file": "x.py", "verdict": "CONFIRMED"}])
    corpus = tmp_path / "corpus"
    _run(mod, monkeypatch, hunt, corpus)
    inv = [p.name for p in corpus.iterdir() if p.is_dir()]
    assert inv and all(n.startswith("dt-loci-") for n in inv), inv


def test_every_record_carries_the_tags_the_builder_groups_on(mod, monkeypatch, tmp_path):
    hunt = _hunt(tmp_path, [{"claim": "a finding", "file": "pkg/mod.py",
                             "verdict": "CONFIRMED", "severity": "high"}])
    out = _run(mod, monkeypatch, hunt, tmp_path / "corpus")
    tags = out[0]["tags"]
    assert any(t.startswith("dt_target:") for t in tags), tags
    assert "dt_target:mod" in tags, tags
    assert any(t == "hunt_verdict:confirmed" for t in tags), tags
    assert out[0]["record_type"] == "inferred"


def test_an_unrecognised_directory_is_skipped_loudly(mod, monkeypatch, tmp_path, capsys):
    d = tmp_path / "20260101-0000-not-a-hunt"; d.mkdir()
    (d / "something-else.json").write_text("[]")
    monkeypatch.setattr(sys, "argv",
                        ["hunt_to_corpus.py", str(d), "--corpus", str(tmp_path / "c")])
    assert mod.main() == 0
    assert "no recognised hunt output" in capsys.readouterr().err

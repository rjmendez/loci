"""The builder lives under deep_think_loci/, which CI does not test. It feeds
mlops/loop.py, so its regression test lives here where `pytest mlops` runs it.

On 2026-08-29 the first full loop run rebuilt the dataset from 5,418 rows to
312,291, of which 303,089 were duplicates and 215,496 were one row: claim="",
evidence="", label=1, cos=1.0. findings.jsonl is a mixed log — 492 of 795 rows
carried no text — and the builder kept them, so two empty strings embedded
identically, scored 1.0, and were labelled a positive match.
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
BUILDER = REPO / "deep_think_loci" / "grounding" / "build_grounding_dataset.py"


@pytest.fixture
def builder():
    spec = importlib.util.spec_from_file_location("bgd", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _findings(tmp_path, rows):
    d = tmp_path / "dt-loci-test"
    d.mkdir()
    f = d / "findings.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(f)


def _run(builder, monkeypatch, tmp_path, rows, out):
    """Drive main() with a deterministic stand-in for the embedding service."""
    def fake_embed(texts, url):
        assert all(t.strip() for t in texts), "empty text reached the embedder"
        v = np.array([[float(len(t)), float(sum(map(ord, t[:8])))] for t in texts],
                     dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    monkeypatch.setattr(builder, "embed", fake_embed)
    monkeypatch.setattr(sys, "argv", [
        "build_grounding_dataset.py",
        "--findings", _findings(tmp_path, rows),
        "--out", str(out),
    ])
    builder.main()
    return [json.loads(x) for x in (out / "grounding_dataset.jsonl").read_text().splitlines() if x]


def test_text_less_access_rows_never_become_training_pairs(builder, monkeypatch, tmp_path):
    """The 492-of-795 case. Access rows share an id with a real finding and carry
    no text; pairing them yielded cos=1.0 positives between two empty strings."""
    # Blank rows come first for shared ids (the old dict was last-write-wins, so
    # either order lost), and z0..z39 are access rows with no real finding at all.
    rows = [{"id": f"a{i}", "text": "", "tags": []} for i in range(6)] * 10
    rows += [{"id": f"z{i}", "text": "  ", "tags": []} for i in range(40)]
    rows += [{"id": f"a{i}", "text": f"finding number {i} about sensors",
              "tags": ["dt_target:sensor-fusion"]} for i in range(6)]
    rows += [{"id": f"b{i}", "text": f"note number {i} on the governance gate",
              "tags": ["dt_target:governance-gate"]} for i in range(4)]
    out = tmp_path / "out"; out.mkdir()
    ds = _run(builder, monkeypatch, tmp_path, rows, out)

    assert ds, "builder produced nothing"
    assert not [r for r in ds if not r["claim"].strip() or not r["evidence"].strip()]
    # 46 of the 100 input rows carried no text; only the 10 real findings pair up.
    positives = [r for r in ds if r["label"] == 1 and r["signal"] == "topical"]
    assert len(positives) == 6 * 5 // 2 + 4 * 3 // 2


def test_one_row_per_id_so_pairs_are_not_multiplied(builder, monkeypatch, tmp_path):
    """ids were not deduped before combinations(), so k copies of an id produced
    C(k,2) identical pairs. Six findings must yield C(6,2)=15 positives, not more."""
    rows = []
    for i in range(6):
        for _ in range(30):  # same id repeated, as the real log does
            rows.append({"id": f"a{i}", "text": f"finding number {i} about sensors",
                         "tags": ["dt_target:sensor-fusion"]})
    for i in range(4):
        rows.append({"id": f"b{i}", "text": f"note number {i} on the governance gate",
                     "tags": ["dt_target:governance-gate"]})
    out = tmp_path / "out"; out.mkdir()
    ds = _run(builder, monkeypatch, tmp_path, rows, out)

    positives = [r for r in ds if r["label"] == 1 and r["signal"] == "topical"]
    expected = 6 * 5 // 2 + 4 * 3 // 2   # C(6,2) + C(4,2)
    assert len(positives) == expected, f"expected {expected} positives, got {len(positives)}"
    seen = {(r["claim"], r["evidence"]) for r in positives}
    assert len(seen) == len(positives), "duplicate pairs in the dataset"


def test_a_log_with_no_usable_rows_fails_loudly(builder, monkeypatch, tmp_path):
    """Silently writing an empty dataset is how this went unnoticed for months."""
    rows = [{"id": f"f{i}", "text": "", "tags": []} for i in range(20)]
    out = tmp_path / "out"; out.mkdir()
    with pytest.raises(SystemExit, match="no usable findings"):
        _run(builder, monkeypatch, tmp_path, rows, out)

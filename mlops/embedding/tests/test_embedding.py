"""Characterization tests for mlops/embedding/ — drift.py and contrastive.py.

These pin CURRENT behaviour, warts included. Several assertions below deliberately
lock in behaviour that is arguably wrong (see the `BUG:` comments); they exist so a
refactor cannot change it silently.

No external services are touched: Ollama HTTP is stubbed at ``urllib.request.urlopen``
or by replacing ``drift._embed``, and no sentence-transformers model is ever downloaded.
"""

import importlib
import json
import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import drift as D  # noqa: E402


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _write_jsonl(path: Path, records) -> Path:
    """`records` may be dicts (json-encoded) or raw strings (written verbatim)."""
    lines = []
    for rec in records:
        lines.append(rec if isinstance(rec, str) else json.dumps(rec))
    path.write_text("\n".join(lines) + "\n")
    return path


def _long(ch: str, n: int = 40) -> str:
    return ch * n


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ======================================================================================
# module constants
# ======================================================================================

def test_module_constants_are_pinned():
    assert D.DRIFT_THRESHOLD == 0.02
    assert D.DEFAULT_N == 100
    # anchor defaults to a file sitting next to drift.py
    assert Path(D.DEFAULT_ANCHOR).name == "anchor.npz"
    assert Path(D.DEFAULT_ANCHOR).parent == Path(D.__file__).parent


def test_defaults_are_read_from_env_at_import_time(monkeypatch):
    """DEFAULT_OLLAMA / DEFAULT_MODEL are module-level constants, not per-call lookups."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://elsewhere:9999/")
    monkeypatch.setenv("EMBED_MODEL", "some-other-model")
    reloaded = importlib.reload(D)
    try:
        assert reloaded.DEFAULT_OLLAMA == "http://elsewhere:9999"  # trailing slash stripped
        assert reloaded.DEFAULT_MODEL == "some-other-model"
    finally:
        monkeypatch.undo()
        importlib.reload(D)


def test_empty_env_falls_back_to_localhost():
    """`os.environ.get(...) or default` — an empty string falls back, unlike a plain get()."""
    saved = os.environ.get("OLLAMA_BASE_URL")
    os.environ["OLLAMA_BASE_URL"] = ""
    try:
        reloaded = importlib.reload(D)
        assert reloaded.DEFAULT_OLLAMA == "http://localhost:11434"
    finally:
        if saved is None:
            os.environ.pop("OLLAMA_BASE_URL", None)
        else:
            os.environ["OLLAMA_BASE_URL"] = saved
        importlib.reload(D)


# ======================================================================================
# _cosine
# ======================================================================================

def test_cosine_identical_orthogonal_opposite():
    assert D._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert D._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert D._cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_is_scale_invariant():
    assert D._cosine([3.0, 4.0], [30.0, 40.0]) == pytest.approx(1.0)


def test_cosine_zero_vector_returns_zero_not_nan():
    assert D._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert D._cosine([1.0, 1.0], [0.0, 0.0]) == 0.0
    assert D._cosine([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_empty_vectors_return_zero():
    assert D._cosine([], []) == 0.0
    assert D._cosine([], [1.0, 2.0]) == 0.0


def test_cosine_rejects_ragged_vectors():
    """Mismatched dimensionality (e.g. anchor built under a different EMBED_MODEL)
    must not silently truncate via zip() into a plausible-looking score. It now
    matches the guard already used by mcp/embed_ops.py, mcp/memcheck/backend.py,
    and mcp/memcheck/llm.py.
    """
    assert D._cosine([1.0, 0.0], [1.0]) == 0.0
    assert D._cosine([1.0, 1.0], [1.0]) == 0.0
    assert D._cosine([1.0], [1.0, 1.0]) == 0.0


def test_cosine_result_is_a_plain_float():
    assert isinstance(D._cosine([1.0, 2.0], [3.0, 4.0]), float)


# ======================================================================================
# _sample_texts
# ======================================================================================

def test_sample_texts_key_precedence_text_content_query(tmp_path):
    ds = _write_jsonl(tmp_path / "d.jsonl", [
        {"text": _long("a"), "content": _long("b"), "query": _long("c")},
    ])
    assert D._sample_texts(str(ds), 10) == [_long("a")]

    ds2 = _write_jsonl(tmp_path / "d2.jsonl", [{"content": _long("b"), "query": _long("c")}])
    assert D._sample_texts(str(ds2), 10) == [_long("b")]

    ds3 = _write_jsonl(tmp_path / "d3.jsonl", [{"query": _long("c")}])
    assert D._sample_texts(str(ds3), 10) == [_long("c")]


def test_sample_texts_falsy_values_fall_through_to_next_key(tmp_path):
    """`or` chaining means an empty/None `text` is treated as absent."""
    ds = _write_jsonl(tmp_path / "d.jsonl", [
        {"text": None, "content": _long("z")},
        {"text": "", "query": _long("y")},
    ])
    assert sorted(D._sample_texts(str(ds), 10)) == sorted([_long("z"), _long("y")])


def test_sample_texts_length_filter_is_strictly_greater_than_30(tmp_path):
    ds = _write_jsonl(tmp_path / "d.jsonl", [
        {"text": "x" * 30},  # excluded
        {"text": "y" * 31},  # included
    ])
    assert D._sample_texts(str(ds), 10) == ["y" * 31]


def test_sample_texts_skips_blank_and_malformed_lines(tmp_path):
    ds = _write_jsonl(tmp_path / "d.jsonl", [
        {"text": _long("a")},
        "",
        "   ",
        "not json at all",
        "{broken",
        {"text": _long("b")},
    ])
    assert sorted(D._sample_texts(str(ds), 10)) == sorted([_long("a"), _long("b")])


def test_sample_texts_missing_file_returns_empty_list(tmp_path):
    assert D._sample_texts(str(tmp_path / "nope.jsonl"), 10) == []


def test_sample_texts_no_qualifying_rows_returns_empty_list(tmp_path):
    ds = _write_jsonl(tmp_path / "d.jsonl", [{"text": "tiny"}, {"other": _long("a")}])
    assert D._sample_texts(str(ds), 10) == []


def test_sample_texts_non_dict_json_line_raises_attributeerror(tmp_path):
    """BUG: only JSONDecodeError is caught. A valid-but-non-object line crashes."""
    ds = _write_jsonl(tmp_path / "d.jsonl", ["[1, 2, 3]"])
    with pytest.raises(AttributeError):
        D._sample_texts(str(ds), 10)


def test_sample_texts_non_string_text_value_raises_typeerror(tmp_path):
    """BUG: len() is applied without a type check."""
    ds = _write_jsonl(tmp_path / "d.jsonl", [{"text": 1234567890}])
    with pytest.raises(TypeError):
        D._sample_texts(str(ds), 10)


def test_sample_texts_is_deterministic_and_shuffles(tmp_path):
    """seed(42) is set on every call, so the sample (and its ORDER) is reproducible.

    Note the sample is a permutation, not a prefix, even when n >= len(texts).
    """
    ds = _write_jsonl(tmp_path / "d.jsonl", [{"text": _long(c)} for c in "ABC"])
    first = D._sample_texts(str(ds), 100)
    second = D._sample_texts(str(ds), 100)
    assert first == second
    assert sorted(first) == sorted([_long("A"), _long("B"), _long("C")])
    # pinned order under CPython's seeded random.sample
    assert [t[0] for t in first] == ["C", "A", "B"]


def test_sample_texts_n_caps_the_sample_size(tmp_path):
    ds = _write_jsonl(tmp_path / "d.jsonl", [{"text": _long(c)} for c in "ABCDE"])
    out = D._sample_texts(str(ds), 2)
    assert len(out) == 2
    assert len(set(out)) == 2  # no duplicates
    assert all(t in {_long(c) for c in "ABCDE"} for t in out)


def test_sample_texts_reseeds_the_global_random_module(tmp_path):
    """Side effect: random.seed(42) clobbers process-wide RNG state."""
    ds = _write_jsonl(tmp_path / "d.jsonl", [{"text": _long(c)} for c in "ABC"])

    random.seed(1)
    D._sample_texts(str(ds), 100)
    after_a = random.random()

    random.seed(999)
    D._sample_texts(str(ds), 100)
    after_b = random.random()

    assert after_a == after_b  # caller's seed was thrown away


# ======================================================================================
# _embed
# ======================================================================================

def test_embed_builds_the_expected_ollama_request(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["body"] = json.loads(req.data.decode())
        seen["ctype"] = req.get_header("Content-type")
        seen["timeout"] = timeout
        return _FakeResponse(json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out = D._embed("hello world", "http://host:1234", "my-model")

    assert out == [0.1, 0.2, 0.3]
    assert seen["url"] == "http://host:1234/api/embeddings"
    assert seen["method"] == "POST"
    assert seen["ctype"] == "application/json"
    assert seen["timeout"] == 30
    # note the key is "prompt", not "input"
    assert seen["body"] == {"model": "my-model", "prompt": "hello world"}


def test_embed_does_not_normalise_a_trailing_slash(monkeypatch):
    """Only the DEFAULT_OLLAMA constant is rstrip()'d; _embed concatenates blindly."""
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return _FakeResponse(json.dumps({"embedding": []}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    D._embed("t", "http://host:1234/", "m")
    assert seen["url"] == "http://host:1234//api/embeddings"


def test_embed_raises_keyerror_when_response_lacks_embedding(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: _FakeResponse(json.dumps({"error": "no model"}).encode()),
    )
    with pytest.raises(KeyError):
        D._embed("t", "http://h", "m")


def test_embed_propagates_transport_errors(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(OSError):
        D._embed("t", "http://h", "m")


# ======================================================================================
# build_anchor
# ======================================================================================

def test_build_anchor_without_numpy_returns_error_dict(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "numpy", None)
    assert D.build_anchor(str(tmp_path / "x.jsonl")) == {"error": "numpy not installed"}


def test_build_anchor_missing_dataset_reports_no_texts(tmp_path):
    assert D.build_anchor(str(tmp_path / "nope.jsonl")) == {"error": "no texts found in dataset"}


def test_build_anchor_all_embeddings_failing(tmp_path, monkeypatch, capsys):
    ds = _write_jsonl(tmp_path / "d.jsonl", [{"text": _long(c)} for c in "AB"])

    def boom(text, url, model):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(D, "_embed", boom)
    result = D.build_anchor(str(ds), anchor_path=str(tmp_path / "a.npz"))

    assert result == {"error": "all embeddings failed"}
    err = capsys.readouterr().err
    # one stderr line per failed sample, indexed by position in the sample
    assert err.count("[drift] embed failed for sample") == 2
    assert "ollama down" in err
    assert not (tmp_path / "a.npz").exists()


def test_build_anchor_happy_path_writes_npz_and_returns_counts(tmp_path, monkeypatch, capsys):
    import numpy as np

    ds = _write_jsonl(tmp_path / "d.jsonl", [{"text": _long(c)} for c in "ABC"])
    vecs = {"A": [1.0, 0.0], "B": [0.0, 1.0], "C": [0.5, 0.5]}
    monkeypatch.setattr(D, "_embed", lambda t, u, m: vecs[t[0]])

    anchor = tmp_path / "nested" / "deeper" / "anchor.npz"
    result = D.build_anchor(str(ds), ollama_url="http://x", model="m",
                            n=100, anchor_path=str(anchor))

    assert result == {"n_anchored": 3, "anchor_path": str(anchor)}
    assert anchor.exists()  # parent dirs are created
    assert "[drift] anchor built: 3 embeddings" in capsys.readouterr().out

    z = np.load(anchor, allow_pickle=True)
    assert set(z.files) == {"embeddings", "texts"}
    assert z["embeddings"].dtype == np.float32
    assert z["embeddings"].shape == (3, 2)
    stored = z["texts"].tolist()
    assert sorted(t[0] for t in stored) == ["A", "B", "C"]
    # rows are stored in the sampled (shuffled) order, not the file order
    assert [t[0] for t in stored] == ["C", "A", "B"]


def test_build_anchor_respects_n(tmp_path, monkeypatch):
    ds = _write_jsonl(tmp_path / "d.jsonl", [{"text": _long(c)} for c in "ABCDE"])
    monkeypatch.setattr(D, "_embed", lambda t, u, m: [1.0, 0.0])
    result = D.build_anchor(str(ds), n=2, anchor_path=str(tmp_path / "a.npz"))
    assert result["n_anchored"] == 2


def test_build_anchor_partial_failure_misaligns_texts_and_embeddings(tmp_path, monkeypatch):
    """BUG: on a mid-list embedding failure, texts are truncated with `texts[:len(embs)]`
    instead of tracking which texts actually succeeded.

    Sampled order for this dataset is C, A, B. Failing on A drops row 1, so the saved
    embeddings are [vec(C), vec(B)] while the saved texts are ["C...", "A..."]. Row 1 now
    pairs text "A" with B's vector.
    """
    import numpy as np

    ds = _write_jsonl(tmp_path / "d.jsonl", [{"text": _long(c)} for c in "ABC"])
    vecs = {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0], "C": [0.0, 0.0, 1.0]}

    def flaky(text, url, model):
        if text.startswith("A"):
            raise RuntimeError("boom")
        return vecs[text[0]]

    monkeypatch.setattr(D, "_embed", flaky)
    anchor = tmp_path / "a.npz"
    assert D.build_anchor(str(ds), anchor_path=str(anchor))["n_anchored"] == 2

    z = np.load(anchor, allow_pickle=True)
    assert [t[0] for t in z["texts"].tolist()] == ["C", "A"]
    assert z["embeddings"].tolist() == [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]  # C's vec, then B's
    # -> row 1 claims to be text "A" but holds the embedding of text "B"


# ======================================================================================
# measure_drift
# ======================================================================================

def _make_anchor(path: Path, texts, embeddings):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        embeddings=np.array(embeddings, dtype=np.float32),
        texts=texts,
    )
    return path


def test_measure_drift_without_numpy(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "numpy", None)
    assert D.measure_drift(str(tmp_path / "a.npz")) == {"error": "numpy not installed"}


def test_measure_drift_missing_anchor_includes_path_in_error(tmp_path):
    missing = tmp_path / "a.npz"
    assert D.measure_drift(str(missing)) == {"error": f"anchor not found: {missing}"}


def test_measure_drift_no_change_returns_full_result_shape(tmp_path, monkeypatch):
    anchor = _make_anchor(tmp_path / "a.npz", ["t one", "t two"], [[1.0, 0.0], [0.0, 1.0]])
    live = {"t one": [1.0, 0.0], "t two": [0.0, 1.0]}
    monkeypatch.setattr(D, "_embed", lambda t, u, m: live[t])

    r = D.measure_drift(str(anchor))

    assert set(r) == {
        "mean_cosine", "drift_score", "n_drifted_095", "n_texts", "threshold", "exceeded",
    }
    assert r["mean_cosine"] == pytest.approx(1.0)
    assert r["drift_score"] == pytest.approx(0.0)
    assert r["n_drifted_095"] == 0
    assert r["n_texts"] == 2
    assert r["threshold"] == D.DRIFT_THRESHOLD
    assert r["exceeded"] is False


def test_measure_drift_passes_plain_strings_to_embed(tmp_path, monkeypatch):
    """texts come back from npz as numpy str_; measure_drift wraps them in str()."""
    anchor = _make_anchor(tmp_path / "a.npz", ["hello"], [[1.0, 0.0]])
    seen = []

    def rec(text, url, model):
        seen.append((type(text), text, url, model))
        return [1.0, 0.0]

    monkeypatch.setattr(D, "_embed", rec)
    D.measure_drift(str(anchor), ollama_url="http://u", model="mm")

    assert seen == [(str, "hello", "http://u", "mm")]


def test_measure_drift_detects_real_drift(tmp_path, monkeypatch):
    anchor = _make_anchor(tmp_path / "a.npz", ["a", "b"], [[1.0, 0.0], [1.0, 0.0]])
    monkeypatch.setattr(D, "_embed", lambda t, u, m: [0.0, 1.0])  # totally different

    r = D.measure_drift(str(anchor))
    assert r["mean_cosine"] == pytest.approx(0.0)
    assert r["drift_score"] == pytest.approx(1.0)
    assert r["n_drifted_095"] == 2
    assert r["exceeded"] is True


def test_n_drifted_095_actually_uses_the_threshold_not_0_95(tmp_path, monkeypatch):
    """Key is named `n_drifted_095` but the comparison is `cos < 1.0 - threshold`.

    cos == 0.96 is ABOVE 0.95 yet still counted as drifted at the default threshold.
    """
    anchor = _make_anchor(tmp_path / "a.npz", ["a"], [[1.0, 0.0]])
    monkeypatch.setattr(D, "_embed", lambda t, u, m: [0.96, 0.28])  # unit vector, cos = 0.96

    r = D.measure_drift(str(anchor), threshold=0.02)
    assert r["mean_cosine"] == pytest.approx(0.96)
    assert r["n_drifted_095"] == 1

    r2 = D.measure_drift(str(anchor), threshold=0.10)  # cos 0.96 >= 0.90 -> not drifted
    assert r2["n_drifted_095"] == 0
    assert r2["exceeded"] is False


def test_measure_drift_exceeded_is_a_strict_greater_than(tmp_path, monkeypatch):
    anchor = _make_anchor(tmp_path / "a.npz", ["a"], [[1.0, 0.0]])
    live = [1.0, 1.0]
    monkeypatch.setattr(D, "_embed", lambda t, u, m: live)

    exact = 1.0 - D._cosine([1.0, 0.0], live)
    assert D.measure_drift(str(anchor), threshold=exact)["exceeded"] is False
    assert D.measure_drift(str(anchor), threshold=exact * 0.5)["exceeded"] is True


def test_measure_drift_skips_failed_embeds_silently_but_keeps_alignment(
    tmp_path, monkeypatch, capsys
):
    """A failed live embed is `continue`d with no log line at all (unlike build_anchor),
    but the anchor row is still indexed by enumerate(), so surviving pairs stay aligned.
    """
    anchor = _make_anchor(
        tmp_path / "a.npz",
        ["a", "b", "c"],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    )

    def flaky(text, url, model):
        if text == "b":
            raise RuntimeError("boom")
        return [1.0, 0.0]

    monkeypatch.setattr(D, "_embed", flaky)
    r = D.measure_drift(str(anchor))

    assert r["n_texts"] == 2  # only successes counted
    assert r["mean_cosine"] == pytest.approx(1.0)  # a and c both matched their own rows
    assert r["exceeded"] is False
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""  # failures are invisible


def test_measure_drift_all_embeds_failing(tmp_path, monkeypatch):
    anchor = _make_anchor(tmp_path / "a.npz", ["a"], [[1.0, 0.0]])

    def boom(t, u, m):
        raise RuntimeError("down")

    monkeypatch.setattr(D, "_embed", boom)
    assert D.measure_drift(str(anchor)) == {"error": "no embeddings produced"}


def test_measure_drift_zero_anchor_vector_scores_as_total_drift(tmp_path, monkeypatch):
    """_cosine returns 0.0 for a degenerate vector, which reads as 100% drift."""
    anchor = _make_anchor(tmp_path / "a.npz", ["a"], [[0.0, 0.0]])
    monkeypatch.setattr(D, "_embed", lambda t, u, m: [1.0, 0.0])

    r = D.measure_drift(str(anchor))
    assert r["mean_cosine"] == 0.0
    assert r["drift_score"] == 1.0
    assert r["exceeded"] is True


# ======================================================================================
# main() / CLI
# ======================================================================================

def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["drift.py"] + argv)
    with pytest.raises(SystemExit) as ei:
        D.main()
    return ei.value.code


def test_main_requires_dataset_even_when_only_measuring(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["drift.py"])
    with pytest.raises(SystemExit) as ei:
        D.main()
    assert ei.value.code == 2  # argparse usage error
    assert "--dataset" in capsys.readouterr().err


def test_main_measure_path_ignores_the_dataset_argument(monkeypatch, tmp_path):
    """--dataset is required but never used on the measure path."""
    seen = {}

    def fake_measure(anchor, ollama, model, threshold):
        seen["args"] = (anchor, ollama, model, threshold)
        return {"mean_cosine": 1.0, "drift_score": 0.0, "n_drifted_095": 0,
                "n_texts": 1, "threshold": threshold, "exceeded": False}

    monkeypatch.setattr(D, "measure_drift", fake_measure)
    code = _run_main(monkeypatch, [
        "--dataset", str(tmp_path / "does-not-exist.jsonl"),
        "--anchor", "/tmp/an.npz", "--ollama", "http://o", "--model", "mm",
        "--threshold", "0.5",
    ])
    assert code == 0
    assert seen["args"] == ("/tmp/an.npz", "http://o", "mm", 0.5)


def test_main_exits_1_when_drift_exceeded_and_writes_out_file(monkeypatch, tmp_path, capsys):
    result = {"mean_cosine": 0.9, "drift_score": 0.1, "n_drifted_095": 3,
              "n_texts": 5, "threshold": 0.02, "exceeded": True}
    monkeypatch.setattr(D, "measure_drift", lambda *a, **k: result)

    out_file = tmp_path / "nested" / "drift.json"
    code = _run_main(monkeypatch, ["--dataset", "x.jsonl", "--out", str(out_file)])

    assert code == 1
    assert json.loads(out_file.read_text()) == result  # parent dirs created
    stdout = capsys.readouterr().out
    assert "mean_cosine=0.9000" in stdout
    assert "drift_score=0.1000" in stdout
    assert "n_drifted=3/5" in stdout
    assert "exceeded=True" in stdout


def test_main_exits_1_on_measure_error_same_code_as_drift(monkeypatch, capsys):
    """BUG-ish: an operational failure and a genuine drift alarm both exit 1,
    so a caller cannot tell 'ollama is down' from 'the model drifted'."""
    monkeypatch.setattr(D, "measure_drift", lambda *a, **k: {"error": "anchor not found: /x"})
    code = _run_main(monkeypatch, ["--dataset", "x.jsonl"])
    assert code == 1
    assert "[drift] ERROR: anchor not found: /x" in capsys.readouterr().err


def test_main_does_not_write_out_file_on_error(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "measure_drift", lambda *a, **k: {"error": "boom"})
    out_file = tmp_path / "drift.json"
    assert _run_main(monkeypatch, ["--dataset", "x.jsonl", "--out", str(out_file)]) == 1
    assert not out_file.exists()


def test_main_build_anchor_success(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_build(dataset, ollama, model, n, anchor):
        seen["args"] = (dataset, ollama, model, n, anchor)
        return {"n_anchored": 7, "anchor_path": anchor}

    monkeypatch.setattr(D, "build_anchor", fake_build)
    out_file = tmp_path / "ignored.json"
    code = _run_main(monkeypatch, [
        "--dataset", "ds.jsonl", "--build-anchor", "--n", "7",
        "--anchor", "/tmp/a.npz", "--ollama", "http://o", "--model", "mm",
        "--out", str(out_file),
    ])

    assert code == 0
    assert seen["args"] == ("ds.jsonl", "http://o", "mm", 7, "/tmp/a.npz")
    assert "[drift] anchor built: n=7" in capsys.readouterr().out
    # --out is silently ignored on the build path
    assert not out_file.exists()


def test_main_build_anchor_error_exits_1(monkeypatch, capsys):
    monkeypatch.setattr(D, "build_anchor", lambda *a, **k: {"error": "no texts found in dataset"})
    code = _run_main(monkeypatch, ["--dataset", "ds.jsonl", "--build-anchor"])
    assert code == 1
    assert "[drift] ERROR: no texts found in dataset" in capsys.readouterr().err


def test_main_uses_module_defaults_when_flags_omitted(monkeypatch):
    seen = {}
    monkeypatch.setattr(D, "measure_drift",
                        lambda *a: (seen.update(args=a), {"error": "stop"})[1])
    _run_main(monkeypatch, ["--dataset", "ds.jsonl"])
    assert seen["args"] == (D.DEFAULT_ANCHOR, D.DEFAULT_OLLAMA, D.DEFAULT_MODEL,
                            D.DRIFT_THRESHOLD)


# ======================================================================================
# contrastive.py
# ======================================================================================

@pytest.fixture(scope="module")
def C():
    try:
        import contrastive as mod
    except (ImportError, SystemExit) as exc:  # pragma: no cover - env dependent
        pytest.skip(f"contrastive training deps unavailable: {exc}")
    return mod


def test_contrastive_hard_exits_when_sentence_transformers_missing(monkeypatch):
    """Import-time fail-closed: the module calls sys.exit() (SystemExit), it does not
    raise ImportError and does not degrade gracefully."""
    monkeypatch.delitem(sys.modules, "contrastive", raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(SystemExit) as ei:
        importlib.import_module("contrastive")
    assert ei.value.code == "pip install sentence-transformers torch"


def test_model_configs_are_pinned(C):
    assert C.MODEL_CONFIGS == {
        "small": {
            "base": "sentence-transformers/all-MiniLM-L6-v2",
            "epochs": 3,
            "trust_remote_code": False,
        },
        "medium": {
            "base": "nomic-ai/nomic-embed-text-v1",
            "epochs": 5,
            "trust_remote_code": True,
        },
    }


def test_load_dataset_skips_blank_lines_and_preserves_order(C, tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        '{"claim":"a","evidence":"b","label":1}\n'
        "\n"
        "   \n"
        '{"claim":"c","evidence":"d","label":0}\n'
    )
    rows = C.load_dataset(p)
    assert rows == [
        {"claim": "a", "evidence": "b", "label": 1},
        {"claim": "c", "evidence": "d", "label": 0},
    ]


def test_load_dataset_propagates_malformed_json(C, tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("{not json}\n")
    with pytest.raises(json.JSONDecodeError):
        C.load_dataset(p)


def test_load_dataset_propagates_missing_file(C, tmp_path):
    with pytest.raises(FileNotFoundError):
        C.load_dataset(tmp_path / "nope.jsonl")


def test_make_examples_label_mapping_is_equality_to_int_1(C):
    rows = [
        {"claim": "c0", "evidence": "e0", "label": 1},
        {"claim": "c1", "evidence": "e1", "label": 0},
        {"claim": "c2", "evidence": "e2", "label": True},   # True == 1 -> positive
        {"claim": "c3", "evidence": "e3", "label": 1.0},    # 1.0 == 1 -> positive
        {"claim": "c4", "evidence": "e4", "label": "1"},    # string -> negative
        {"claim": "c5", "evidence": "e5", "label": 2},      # anything else -> negative
    ]
    examples, scores = C.make_examples(rows)
    assert scores == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert [e.label for e in examples] == scores
    assert [e.texts for e in examples] == [[f"c{i}", f"e{i}"] for i in range(6)]
    assert len(examples) == len(scores) == 6


def test_make_examples_requires_label_claim_evidence(C):
    with pytest.raises(KeyError):
        C.make_examples([{"claim": "c", "evidence": "e"}])
    with pytest.raises(KeyError):
        C.make_examples([{"evidence": "e", "label": 1}])


def test_make_examples_empty_input(C):
    assert C.make_examples([]) == ([], [])


def test_print_stats_output_format(C, capsys):
    rows = [
        {"label": 1, "signal": "entail", "cos": 0.8},
        {"label": 0, "signal": "entail", "cos": 0.2},
        {"label": 0},  # no signal key -> counted as "unknown"
    ]
    C.print_stats(rows)
    out = capsys.readouterr().out
    assert "Dataset: 3 rows — pos=1, neg=2" in out
    assert "Signal breakdown: {'entail': 2, 'unknown': 1}" in out
    assert "Cosine similarity — mean=0.500, std=0.300" in out


def test_print_stats_omits_cosine_line_when_no_cos_key(C, capsys):
    C.print_stats([{"label": 1}, {"label": 0}])
    out = capsys.readouterr().out
    assert "Cosine similarity" not in out
    assert "pos=1, neg=1" in out


def test_print_stats_counts_by_summing_labels_not_by_counting_positives(C, capsys):
    """BUG: `pos = sum(labels)` assumes labels are 0/1. A stray label of 2 produces a
    negative count instead of an error."""
    C.print_stats([{"label": 2}])
    assert "pos=2, neg=-1" in capsys.readouterr().out


def test_print_stats_on_empty_rows(C, capsys):
    C.print_stats([])
    out = capsys.readouterr().out
    assert "Dataset: 0 rows — pos=0, neg=0" in out
    assert "Signal breakdown: {}" in out


def test_print_ollama_instructions_embeds_the_model_dir(C, capsys):
    C.print_ollama_instructions(Path("/models/loci-embed-small"))
    out = capsys.readouterr().out
    assert "convert_hf_to_gguf.py /models/loci-embed-small" in out
    assert "--outfile /models/loci-embed-small/model.gguf --outtype q8_0" in out
    assert "ollama create loci-embed -f Modelfile" in out
    assert out.count("=" * 60) == 2


class _StubCard:
    def set_evaluation_metrics(self, *args, **kwargs):
        return None


class _StubModel:
    """Minimal duck-typed stand-in for SentenceTransformer — no download, no network."""

    similarity_fn_name = "cosine"
    truncate_dim = None

    def __init__(self):
        self.model_card_data = _StubCard()

    def encode(self, sentences, **kwargs):
        import torch

        rows = [[float(len(s)), 1.0] for s in sentences]
        return torch.tensor(rows)


def test_baseline_spearman_returns_a_dict_despite_its_float_annotation(C, tmp_path):
    """BUG: annotated `-> float`, but sentence-transformers' evaluator returns a metrics
    dict. main() formats the equivalent value with `:.4f`, so the training entrypoint
    raises TypeError against the installed sentence-transformers.
    """
    examples = [
        C.InputExample(texts=["short", "a longer piece of evidence"], label=1.0),
        C.InputExample(texts=["tiny", "x"], label=0.0),
        C.InputExample(texts=["medium text", "another evidence"], label=1.0),
    ]
    result = C.baseline_spearman(_StubModel(), examples, [1.0, 0.0, 1.0], tmp_path)

    assert isinstance(result, dict)
    assert "val-baseline_spearman_cosine" in result
    with pytest.raises(TypeError):
        format(result, ".4f")


def test_baseline_spearman_ignores_its_val_scores_argument(C, tmp_path):
    """The scores are taken from the InputExample labels; val_scores is dead weight."""
    examples = [
        C.InputExample(texts=["short", "a longer piece of evidence"], label=1.0),
        C.InputExample(texts=["tiny", "x"], label=0.0),
        C.InputExample(texts=["medium text", "another evidence"], label=1.0),
    ]
    a = C.baseline_spearman(_StubModel(), examples, [1.0, 0.0, 1.0], tmp_path)
    b = C.baseline_spearman(_StubModel(), examples, [0.0, 9.0, -3.0], tmp_path)
    assert a == b


def test_baseline_spearman_does_not_write_a_csv(C, tmp_path):
    examples = [
        C.InputExample(texts=["short", "a longer piece of evidence"], label=1.0),
        C.InputExample(texts=["tiny", "x"], label=0.0),
    ]
    C.baseline_spearman(_StubModel(), examples, [1.0, 0.0], tmp_path)
    assert list(tmp_path.glob("*.csv")) == []


def test_contrastive_main_rejects_a_missing_dataset(C, monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys, "argv", ["contrastive.py", "--dataset", str(tmp_path / "nope.jsonl")]
    )
    with pytest.raises(SystemExit) as ei:
        C.main()
    assert str(ei.value.code).startswith("Dataset not found:")


def test_contrastive_main_dry_run_stops_before_training(C, monkeypatch, tmp_path, capsys):
    ds = tmp_path / "d.jsonl"
    ds.write_text('{"claim":"a","evidence":"b","label":1,"signal":"s"}\n')

    def explode(*args, **kwargs):  # would be hit only if training started
        raise AssertionError("SentenceTransformer must not be constructed on --dry-run")

    monkeypatch.setattr(C, "SentenceTransformer", explode)
    monkeypatch.setattr(sys, "argv", ["contrastive.py", "--dataset", str(ds), "--dry-run"])

    assert C.main() is None  # returns, does not sys.exit
    out = capsys.readouterr().out
    assert "Dataset: 1 rows — pos=1, neg=0" in out
    assert "--dry-run: exiting before training." in out
    assert not (tmp_path / "loci-embed-small").exists()


def test_contrastive_argparse_defaults(C, monkeypatch, tmp_path):
    """Defaults are repo-relative paths, resolved against the process CWD."""
    captured = {}

    real_exit = sys.exit

    def capture_exit(msg):
        captured["msg"] = msg
        real_exit(msg)

    monkeypatch.setattr(C.sys, "exit", capture_exit)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["contrastive.py"])
    with pytest.raises(SystemExit):
        C.main()
    assert captured["msg"] == (
        "Dataset not found: deep_think_loci/grounding/grounding_dataset.jsonl"
    )

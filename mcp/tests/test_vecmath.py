"""A cosine that cannot answer must not return a number that looks like an answer.

Six copies of the same eight lines existed. Four checked the lengths and returned
0.0; two did not check at all, and `zip` truncates silently:

    cosine([0.1]*768, [0.1]*384) == 0.707107

This repo mixes 768-float nomic embeddings with the 384-float hash_embed in
mcp/memcheck/vectors.py, and scripts/glymphatic_sweep.py used cosine to decide
which memories to DELETE. 0.707 clears its duplicate threshold.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vecmath import cosine, cosine_or  # noqa: E402


def test_a_length_mismatch_is_unanswerable_not_a_low_score():
    """The regression, in one line. 0.707 came from the first 384 dimensions."""
    assert cosine([0.1] * 768, [0.1] * 384) is None


def test_identical_vectors_still_score_one():
    assert round(cosine([0.1] * 768, [0.1] * 768), 9) == 1.0


def test_orthogonal_vectors_score_zero_and_that_zero_is_real():
    """0.0 has to stay available as a genuine answer, which is exactly why it
    cannot double as 'unanswerable'."""
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_empty_and_zero_magnitude_are_unanswerable():
    assert cosine([], [1.0]) is None
    assert cosine([1.0], []) is None
    assert cosine([0.0] * 8, [1.0] * 8) is None


def test_cosine_or_makes_the_default_visible_at_the_call_site():
    assert cosine_or([0.1] * 768, [0.1] * 384, default=0.0) == 0.0
    assert cosine_or([0.1] * 768, [0.1] * 384, default=-1.0) == -1.0
    assert round(cosine_or([1.0, 0.0], [1.0, 0.0]), 6) == 1.0


def test_no_module_keeps_its_own_copy_of_the_arithmetic():
    """Six copies is how two of them missed the length check. Any file that both
    zips two vectors together and divides by their norms is a seventh."""
    repo = pathlib.Path(__file__).resolve().parents[2]
    offenders = []
    for p in list(repo.glob("mcp/**/*.py")) + list(repo.glob("scripts/**/*.py")):
        if ".venv" in p.parts or p.name in ("vecmath.py", "test_vecmath.py"):
            continue
        src = p.read_text(errors="ignore")
        if "for x, y in zip(a, b)" in src or "x * y for x, y in zip(" in src:
            offenders.append(str(p.relative_to(repo)))
    assert not offenders, (
        "these still compute a cosine by hand instead of using vecmath.cosine:\n  "
        + "\n  ".join(offenders)
    )

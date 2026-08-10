"""Session-scoped fixtures so the ~1.3s full-corpus build against `--rev
HEAD` runs once per test session, not once per test function.

Also defines the two ENVIRONMENT guards below. Both exist because a set of
tests silently depended on the developer's environment and passed locally for
the wrong reason -- CI caught them the first time it ran the suite.
"""
import importlib.util
import shutil
import subprocess

import pytest

from ..config import REPO_ROOT

# Guard 1: tests that build at a HISTORICAL revision need real git history.
# A shallow clone (actions/checkout defaults to fetch-depth 1) makes
# `git rev-parse <sha>^` exit 128. The CI job sets fetch-depth: 0; this guard
# keeps the suite honest anywhere else that history is truncated.
def _has_git_history() -> bool:
    if shutil.which("git") is None:
        return False
    try:
        subprocess.run(["git", "rev-parse", "HEAD~5"], cwd=REPO_ROOT,
                       capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False


needs_git_history = pytest.mark.skipif(
    not _has_git_history(),
    reason="shallow clone: no history to build historical revisions from",
)

# Guard 2: the resolver classifies an import as third-party by IMPORTABILITY,
# so tests asserting that `numpy` resolves as installed third-party are really
# asserting something about the environment, not about the tool. Skip them
# where the corpus's own dependencies are absent (e.g. a stdlib-only CI job)
# rather than pretending the tool regressed.
_CORPUS_DEPS = ("numpy", "fastapi", "qdrant_client")


def _corpus_deps_installed() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in _CORPUS_DEPS)


needs_corpus_deps = pytest.mark.skipif(
    not _corpus_deps_installed(),
    reason=(
        "corpus third-party deps not installed; import resolution would "
        "classify them as unresolved, which measures the environment not the tool"
    ),
)

from ..ingest import load_corpus
from ..pipeline import build_graph
from ..resolve import ResolutionTable


@pytest.fixture(scope="session")
def head_build():
    return build_graph(rev="HEAD")


@pytest.fixture(scope="session")
def head_sources():
    sources, _ = load_corpus(rev="HEAD")
    return sources


@pytest.fixture(scope="session")
def head_table(head_sources):
    return ResolutionTable(head_sources)

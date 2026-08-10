"""Session-scoped fixtures so the ~1.3s full-corpus build against `--rev
HEAD` runs once per test session, not once per test function."""
import pytest

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

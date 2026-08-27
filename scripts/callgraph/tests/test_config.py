from .conftest import needs_corpus_deps, needs_git_history  # noqa: F401
"""config.py against the REAL repo: acceptance criterion is an exact file
count (119), not a vibe. If this drifts, either the repo changed shape or
the exclusion rules regressed — both worth failing loudly on."""
from .. import config


def test_corpus_is_exactly_118_files():
    files = config.iter_corpus_files_worktree()
    assert len(files) == 119, sorted(files)


def test_corpus_excludes_test_directories():
    files = config.iter_corpus_files_worktree()
    assert not any("/tests/" in f for f in files)


def test_corpus_excludes_venv_and_pycache():
    files = config.iter_corpus_files_worktree()
    assert not any(".venv" in f or "__pycache__" in f for f in files)


def test_corpus_excludes_its_own_package():
    files = config.iter_corpus_files_worktree()
    assert not any(f.startswith("scripts/callgraph/") for f in files)


def test_corpus_covers_all_five_roots():
    files = config.iter_corpus_files_worktree()
    roots_seen = {f.split("/", 1)[0] for f in files}
    assert roots_seen == set(config.CORPUS_ROOTS)


def test_in_corpus_matches_iter_corpus_files():
    files = set(config.iter_corpus_files_worktree())
    for f in files:
        assert config.in_corpus(f)
    assert not config.in_corpus("mcp/tests/test_foo.py")
    assert not config.in_corpus("scripts/callgraph/census.py")
    assert not config.in_corpus("README.md")


def test_stdlib_module_names_contains_known_stdlib():
    names = config.stdlib_module_names()
    assert "os" in names and "json" in names and "ast" in names
    assert "graph_tools" not in names


@needs_corpus_deps
def test_third_party_names_finds_installed_packages():
    # The venv is fully provisioned per the task brief; fastmcp's dependency
    # stack (dotenv, starlette, ...) should be discoverable.
    names = config.third_party_top_level_names()
    assert "dotenv" in names or "starlette" in names

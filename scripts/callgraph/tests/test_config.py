from .conftest import needs_corpus_deps, needs_git_history  # noqa: F401
"""config.py against the REAL repo: acceptance criterion is an exact file
count (126), not a vibe. If this drifts, either the repo changed shape or
the exclusion rules regressed — both worth failing loudly on."""
from .. import config


def test_corpus_is_exactly_126_files():
    files = config.iter_corpus_files_worktree()
    assert len(files) == 126, sorted(files)


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
    names = config.third_party_top_level_names()
    assert "dotenv" in names or "starlette" in names


# ── tool caches are not corpus ────────────────────────────────────────────────
# Found by running the suite on a developer machine: pytest's tmp_path fixtures
# leave .py files under mcp/.pytest_cache/tmp, and the corpus walk counted 339
# files instead of 126. CI never saw it because it checks out clean and runs the
# callgraph job in its own job, before anything populates the cache.

def test_dot_directories_are_excluded_as_a_class():
    for d in (".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".venv", ".git"):
        assert config.is_excluded_rel(f"mcp/{d}/tmp/x.py"), d
        assert config.is_excluded_rel(f"mcp/{d}"), d


def test_a_leading_dot_does_not_exclude_a_corpus_root_relative_path():
    assert not config.is_excluded_rel("mcp/server.py")
    assert not config.is_excluded_rel("scripts/hunt_to_corpus.py")


def _tree(root, rel_paths):
    for rel in rel_paths:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x = 1\n")


def test_corpus_walk_skips_tool_cache_directories(tmp_path):
    _tree(tmp_path, [
        "mcp/server.py",
        "mcp/.pytest_cache/tmp/test_thing0/fixture.py",
        "mcp/.mypy_cache/3.11/mod.py",
        "mcp/__pycache__/server.cpython-311.py",
    ])
    assert config.iter_corpus_files_worktree(tmp_path) == ["mcp/server.py"]


def test_the_two_walkers_share_one_exclusion_rule(tmp_path):
    """iter_test_files_worktree is the complement of the corpus walk, so it must
    differ on `tests` and nothing else."""
    _tree(tmp_path, [
        "mcp/server.py",
        "mcp/tests/test_server.py",
        "mcp/.pytest_cache/tmp/test_thing0/fixture.py",
        "mcp/tests/.pytest_cache/junk.py",
    ])
    assert config.iter_corpus_files_worktree(tmp_path) == ["mcp/server.py"]
    assert config.iter_test_files_worktree(tmp_path) == ["mcp/tests/test_server.py"]

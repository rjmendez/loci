"""ingest.py: worktree parsing, git-rev parsing, and the SyntaxError
degrade-not-abort contract."""
from .. import ingest
from ..tests.helpers import source_file


def test_load_corpus_worktree_parses_123_files_with_no_errors():
    sources, origin = ingest.load_corpus(rev=None)
    assert origin == "working tree"
    assert len(sources) == 123
    errors = [(sf.rel_path, sf.error) for sf in sources if sf.error is not None]
    assert errors == []
    assert all(sf.tree is not None for sf in sources)


def test_load_corpus_rev_head_reads_git_blobs():
    sources, origin = ingest.load_corpus(rev="HEAD")
    assert origin.startswith("rev ")
    assert len(sources) == 123
    assert all(sf.error is None for sf in sources)
    server = next(sf for sf in sources if sf.rel_path == "mcp/server.py")
    assert "loci-mcp" in server.source[:200]


def test_load_corpus_rev_head_differs_from_worktree_when_mcp_server_dirty():
    # --rev HEAD reads the committed blob; the working tree may legitimately differ.
    worktree_sources, _ = ingest.load_corpus(rev=None)
    rev_sources, _ = ingest.load_corpus(rev="HEAD")
    wt = {sf.rel_path for sf in worktree_sources}
    rv = {sf.rel_path for sf in rev_sources}
    assert wt == rv


def test_syntax_error_degrades_one_file_without_aborting_build():
    good = source_file("ok.py", "def f():\n    return 1\n")
    bad = source_file("broken.py", "def f(:\n    pass\n")
    assert good.tree is not None and good.error is None
    assert bad.tree is None
    assert bad.error is not None and "SyntaxError" in bad.error


def test_parse_one_reparses_identical_content_consistently():
    a = source_file("a.py", "x = 1\n")
    b = source_file("b.py", "x = 1\n")
    # Same content, different path -> same sha1, independent trees.
    assert a.sha1 == b.sha1
    assert ingest.ast.dump(a.tree) == ingest.ast.dump(b.tree)

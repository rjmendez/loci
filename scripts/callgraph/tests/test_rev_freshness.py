from .conftest import needs_corpus_deps, needs_git_history  # noqa: F401
"""Cache-invalidation / staleness tests for ingest.py's `--rev` path.

ingest.py's own docstring is explicit: there is NO parse cache (measured
cold-parse cost for the whole corpus is small enough that a cache would add
staleness risk for no real speedup at this corpus size — `--no-cache` is
accepted and is a documented no-op). "Cache invalidation tested" therefore
means proving the ABSENCE of a staleness bug, not exercising an eviction
path that doesn't exist: back-to-back builds at different revisions must
never leak content from one into the other, a build must be exactly
reproducible against the same rev, and `--rev` must read the git-committed
blob rather than a possibly-mid-edit working tree file — the concurrent-
edit scenario build_steps step 13 was written under (another workflow is
mid-editing mcp/server.py while these tests run)."""
from ..ingest import load_corpus


def _source_for(sources, rel_path):
    return next(sf for sf in sources if sf.rel_path == rel_path)


def test_rebuilding_the_same_rev_is_byte_identical():
    sources_a, origin_a = load_corpus(rev="HEAD")
    sources_b, origin_b = load_corpus(rev="HEAD")
    assert origin_a == origin_b
    by_path_a = {sf.rel_path: sf.source for sf in sources_a}
    by_path_b = {sf.rel_path: sf.source for sf in sources_b}
    assert by_path_a == by_path_b


@needs_git_history
def test_switching_revs_does_not_leak_content_between_builds():
    # mcp/grounding.py genuinely differs between 69adfa4^ (BUG B present)
    # and 69adfa4 (the fix commit) — see docs/LIMITS.md's BUG B section and
    # tests/test_cli_step10_12.py's flags regression, which depend on this
    # same pair of revisions actually differing.
    before, _ = load_corpus(rev="69adfa4^")
    after, _ = load_corpus(rev="69adfa4")
    src_before = _source_for(before, "mcp/grounding.py").source
    src_after = _source_for(after, "mcp/grounding.py").source
    assert src_before != src_after

    # Read the OLDER rev again, interleaved after the newer one, and prove
    # the result is exactly what it was the first time — nothing from the
    # `69adfa4` build (nor any other in-process state) contaminated it.
    before_again, _ = load_corpus(rev="69adfa4^")
    assert _source_for(before_again, "mcp/grounding.py").source == src_before


def test_no_cache_flag_is_accepted_and_does_not_change_the_result():
    default_sources, _ = load_corpus(rev="HEAD")
    no_cache_sources, _ = load_corpus(rev="HEAD", no_cache=True)
    by_path_default = {sf.rel_path: sf.source for sf in default_sources}
    by_path_no_cache = {sf.rel_path: sf.source for sf in no_cache_sources}
    assert by_path_default == by_path_no_cache


def test_rev_head_is_immune_to_a_concurrently_mid_edited_working_tree(tmp_path, monkeypatch):
    # Simulate "another workflow is mid-editing mcp/server.py" by writing a
    # syntactically broken decoy directly into a scratch copy of the repo
    # root's mcp/server.py path resolution — without touching the real
    # working tree file (this package must never write under mcp/). Instead
    # we assert the CONTRACT directly: load_corpus(rev="HEAD") never reads
    # config.REPO_ROOT / rel_path from disk at all when rev is not None —
    # it goes through git plumbing exclusively. Verified by monkeypatching
    # Path.read_text to explode if anything tries a direct filesystem read
    # while a rev is requested.
    from pathlib import Path

    original_read_text = Path.read_text

    def _boom(self, *a, **kw):
        raise AssertionError(f"load_corpus(rev=...) must not read {self} off disk directly")

    monkeypatch.setattr(Path, "read_text", _boom)
    try:
        sources, origin = load_corpus(rev="HEAD")
    finally:
        monkeypatch.setattr(Path, "read_text", original_read_text)
    assert origin.startswith("rev ")
    assert len(sources) == 117
    server = _source_for(sources, "mcp/server.py")
    assert server.error is None and "loci-mcp" in server.source[:200]

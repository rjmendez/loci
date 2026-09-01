"""generate_memory_index.py — the curated MEMORY.md index must never exceed its
context-load budget, and must never silently drop a memory file.

MEMORY.md overflowed its 24.4KiB load cap twice: once naturally, once again
within four days of a hand trim. Hand-maintenance doesn't hold; this generates
the index instead and enforces the budget structurally.
"""
import importlib.util
import os
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "generate_memory_index.py"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("generate_memory_index", SCRIPT)
    assert spec and spec.loader, f"cannot load {SCRIPT}"
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_mem(tmp_path, filename, name, description, mtype="feedback", body="body text"):
    (tmp_path / filename).write_text(
        f'---\nname: {name}\ndescription: "{description}"\nmetadata:\n  type: {mtype}\n---\n\n{body}\n',
        encoding="utf-8",
    )


def test_over_budget_corpus_is_shortened_to_fit(tmp_path, mod):
    """The core guarantee: a corpus that overflows the budget unfixed must be
    shortened until it fits — never left over budget, never dropped."""
    for i in range(12):
        _write_mem(tmp_path, f"mem-{i:02d}.md", f"mem-{i:02d}", "x" * 300)
    entries, malformed = mod.load_entries(tmp_path)
    assert not malformed
    sections = mod.build_sections(entries, [], {})
    budget = 900

    naive = mod.render(sections)  # no budget step applied yet — this is the pre-fix shape
    assert len(naive) > budget, "fixture must actually overflow the budget to test anything"

    assert mod.fit_to_budget(sections, budget) is True
    fitted = mod.render(sections)
    assert len(fitted) <= budget
    for i in range(12):
        assert f"(mem-{i:02d}.md)" in fitted  # shortened, not dropped


def test_every_input_file_appears_in_output(tmp_path, mod):
    names = [f"topic-{i}" for i in range(9)]
    for n in names:
        _write_mem(tmp_path, f"{n}.md", n, "a reasonably normal description", mtype="infra")
    entries, _ = mod.load_entries(tmp_path)
    assert len(entries) == len(names)
    sections = mod.build_sections(entries, [], {})
    assert mod.fit_to_budget(sections, 100_000)
    doc = mod.render(sections)
    for n in names:
        assert f"({n}.md)" in doc, f"{n}.md missing from generated index — orphaned"


def test_malformed_frontmatter_is_reported_not_dropped(tmp_path, mod, capsys):
    (tmp_path / "broken.md").write_text("just some prose, no frontmatter at all\n", encoding="utf-8")
    _write_mem(tmp_path, "good.md", "good", "fine description")

    entries, malformed = mod.load_entries(tmp_path)
    err = capsys.readouterr().err

    assert any(fname == "broken.md" for fname, _reason in malformed)
    assert "broken.md" in err  # reported to stderr, not silently swallowed

    by_name = {e.filename: e for e in entries}
    assert "broken.md" in by_name          # still present in the entry list
    assert by_name["broken.md"].title       # got a usable (filename-derived) title

    doc = mod.render(mod.build_sections(entries, [], {}))
    assert "(broken.md)" in doc            # and still linked in the rendered index


def test_preserves_existing_section_grouping(tmp_path, mod):
    _write_mem(tmp_path, "a.md", "a-mem", "desc a", mtype="feedback")
    _write_mem(tmp_path, "b.md", "b-mem", "desc b", mtype="infra")
    source = tmp_path / "old_index.md"
    source.write_text("# Memory index\n\n## Custom Group\n- [x](a.md) — old hook text\n",
                       encoding="utf-8")

    entries, _ = mod.load_entries(tmp_path)
    order, secs = mod.parse_source_sections(source.read_text(encoding="utf-8"))
    sections = mod.build_sections(entries, order, secs)

    assert "Custom Group" in sections
    assert [e.filename for e in sections["Custom Group"]] == ["a.md"]
    # b.md wasn't in the source index -> falls into a type-derived section instead
    other_files = {e.filename for sec, ents in sections.items() if sec != "Custom Group" for e in ents}
    assert "b.md" in other_files


def test_cli_end_to_end_writes_and_backs_up(tmp_path, mod):
    for i in range(3):
        _write_mem(tmp_path, f"m{i}.md", f"m{i}", "hook text " * 5)

    rc = mod.main([str(tmp_path), "--budget-chars", "2000"])
    assert rc == 0
    out = tmp_path / "MEMORY.md"
    assert out.exists()
    assert len(out.read_text(encoding="utf-8")) <= 2000

    rc2 = mod.main([str(tmp_path), "--budget-chars", "2000"])
    assert rc2 == 0
    assert list(tmp_path.glob("MEMORY.md.bak-*")), "regenerating must back up, not clobber blind"


def test_dry_run_writes_nothing(tmp_path, mod):
    _write_mem(tmp_path, "m.md", "m", "hook")
    rc = mod.main([str(tmp_path), "--dry-run"])
    assert rc == 0
    assert not (tmp_path / "MEMORY.md").exists()


def test_infeasible_budget_exits_nonzero(tmp_path, mod):
    # 50 entries with long slugs can't fit a 10-char budget even with every hook emptied.
    for i in range(50):
        slug = f"mem-{i:03d}-with-a-fairly-long-descriptive-slug-name"
        _write_mem(tmp_path, f"{slug}.md", slug, "x")
    rc = mod.main([str(tmp_path), "--budget-chars", "10", "--dry-run"])
    assert rc == 3


def test_no_source_run_preserves_existing_index_headings(tmp_path, mod):
    """[E1] --source has no CLI default, but the primary documented invocation
    is `generate_memory_index.py <memory_dir>` with no --source — and that run
    is destructive by default: it overwrites <memory_dir>/MEMORY.md, so if it
    doesn't also default to reading grouping FROM that same file, it collapses
    curated section headings into metadata.type buckets on every unqualified
    run. Default --source to the file being replaced."""
    _write_mem(tmp_path, "a.md", "a-mem", "desc a", mtype="feedback")
    _write_mem(tmp_path, "b.md", "b-mem", "desc b", mtype="infra")
    (tmp_path / "MEMORY.md").write_text(
        "# Memory index\n\n"
        "## Custom Group\n- [x](a.md) — old hook\n\n"
        "## Another Group\n- [y](b.md) — old hook2\n",
        encoding="utf-8",
    )

    rc = mod.main([str(tmp_path), "--budget-chars", "5000"])
    assert rc == 0
    doc = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert "## Custom Group" in doc
    assert "## Another Group" in doc
    assert "## Feedback" not in doc and "## Infra" not in doc


def test_unreadable_file_does_not_abort_the_run(tmp_path, mod, capsys):
    """[E2] A directory named *.md (IsADirectoryError) or an unreadable file
    must not drop the other files in the run — report and continue, matching
    the contract already honoured for malformed frontmatter."""
    (tmp_path / "weird.md").mkdir()
    _write_mem(tmp_path, "good.md", "good", "fine description")

    entries, malformed = mod.load_entries(tmp_path)
    err = capsys.readouterr().err

    assert any(fname == "weird.md" for fname, _reason in malformed)
    assert "weird.md" in err
    by_name = {e.filename: e for e in entries}
    assert "good.md" in by_name and by_name["good.md"].hook == "fine description"

    doc = mod.render(mod.build_sections(entries, [], {}))
    assert "(good.md)" in doc


def test_unreadable_permission_denied_file_is_reported_not_fatal(tmp_path, mod):
    if os.geteuid() == 0:
        pytest.skip("root bypasses file permission bits")
    blocked = tmp_path / "blocked.md"
    blocked.write_text("---\nname: blocked\ndescription: x\n---\n", encoding="utf-8")
    blocked.chmod(0)
    try:
        _write_mem(tmp_path, "good.md", "good", "fine description")
        entries, malformed = mod.load_entries(tmp_path)
        assert any(fname == "blocked.md" for fname, _reason in malformed)
        assert any(e.filename == "good.md" for e in entries)
    finally:
        blocked.chmod(0o644)


def test_marked_entry_keeps_more_hook_at_forced_truncation(tmp_path, mod):
    """[E3] Salience-weighted allocation: an entry carrying the warning marker
    must keep a larger share of its hook than a routine entry when the budget
    forces truncation of both."""
    _write_mem(tmp_path, "warn.md", "warn-mem", "⚠ " + "x" * 200, mtype="feedback")
    _write_mem(tmp_path, "plain.md", "plain-mem", "y" * 200, mtype="feedback")
    entries, _ = mod.load_entries(tmp_path)
    sections = mod.build_sections(entries, [], {})

    assert mod.fit_to_budget(sections, 260) is True
    by_file = {e.filename: e for ents in sections.values() for e in ents}
    warn_len = len(by_file["warn.md"].hook)
    plain_len = len(by_file["plain.md"].hook)
    assert 0 < plain_len < 200, "fixture must actually force truncation of the unmarked entry"
    assert warn_len > plain_len


def test_truncation_prefers_word_boundary_over_mid_word(tmp_path, mod):
    """[E3] Prefer a word boundary over a mid-word cut when one is available
    near the truncation point."""
    hook = "read the whole runbook before merging anything at all here today"
    # a hard cut at this allowance lands mid-word (hook[:24] == 'read the whole runbook b') —
    # chosen specifically so a boundary-unaware truncator fails this assertion.
    truncated = mod._truncate_hook(hook, 25)
    assert truncated.endswith("…")
    body = truncated[:-1].rstrip()
    assert hook.startswith(body)
    # the char right after the kept text in the original hook is a space —
    # i.e. we stopped at a word boundary, not mid-word.
    assert hook[len(body):len(body) + 1] in (" ", "")


def test_truncation_does_not_split_a_backtick_span(tmp_path, mod):
    """[E3] Never drop a backticked identifier/path/PR-number by splitting it
    in half — retract before the opening backtick instead."""
    hook = "see the fix in `scripts/generate_memory_index.py` for details"
    truncated = mod._truncate_hook(hook, 25)
    assert truncated.count("`") % 2 == 0


def test_headroom_pct_pure_function(mod):
    """[E4] --headroom-pct targets below the hard cap."""
    assert mod._apply_headroom(1000, 5.0) == 950
    assert mod._apply_headroom(1000, 0.0) == 1000
    assert mod._apply_headroom(1000, 100.0) == 0


def test_default_headroom_leaves_margin_under_hard_cap(tmp_path, mod, capsys):
    """[E4] End-to-end: a run with plenty of truncatable content lands strictly
    under --budget-chars by default, leaving room for a manual edit."""
    for i in range(12):
        _write_mem(tmp_path, f"mem-{i:02d}.md", f"mem-{i:02d}", "x" * 300)
    rc = mod.main([str(tmp_path), "--budget-chars", "900", "--dry-run"])
    assert rc == 0
    printed = capsys.readouterr().out
    doc_len = len(printed) - 1  # main() does print(doc), doc already ends in "\n"
    assert doc_len <= mod._apply_headroom(900, mod.DEFAULT_HEADROOM_PCT)


def test_check_flag_detects_drift_without_writing(tmp_path, mod):
    """[E5] --check is the wiring point (cron/pre-commit) that makes the
    budget guarantee real instead of only aspirational: it must fail on a
    stale index and never write."""
    _write_mem(tmp_path, "a.md", "a-mem", "desc a")
    rc = mod.main([str(tmp_path), "--check"])
    assert rc == 4
    assert not (tmp_path / "MEMORY.md").exists()

    rc = mod.main([str(tmp_path), "--budget-chars", "5000"])
    assert rc == 0
    after_first = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")

    rc = mod.main([str(tmp_path), "--budget-chars", "5000", "--check"])
    assert rc == 0
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == after_first

    _write_mem(tmp_path, "b.md", "b-mem", "a brand new entry")
    rc = mod.main([str(tmp_path), "--budget-chars", "5000", "--check"])
    assert rc == 4
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == after_first  # unchanged


def test_inline_error_shown_even_when_description_present(tmp_path, mod):
    """A missing 'name' with a present 'description' must still surface the
    frontmatter error inline, not just on stderr — otherwise the rendered
    entry looks perfectly normal and the problem is invisible in the index."""
    (tmp_path / "noname.md").write_text(
        '---\ndescription: "has a description but no name"\n---\n\nbody\n', encoding="utf-8")
    entries, malformed = mod.load_entries(tmp_path)
    assert any(fname == "noname.md" for fname, _ in malformed)
    by_name = {e.filename: e for e in entries}
    assert "frontmatter" in by_name["noname.md"].hook

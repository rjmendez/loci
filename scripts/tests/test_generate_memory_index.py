"""generate_memory_index.py — the curated MEMORY.md index must never exceed its
context-load budget, and must never silently drop a memory file.

MEMORY.md overflowed its 24.4KiB load cap twice: once naturally, once again
within four days of a hand trim. Hand-maintenance doesn't hold; this generates
the index instead and enforces the budget structurally.
"""
import importlib.util
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

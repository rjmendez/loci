"""delete_code_under on Windows-style paths (backslash separators), against the real LadybugDB engine.

Found 2026-10-18: it built the "strictly beneath" prefix with "/" only, so a tree ingested from Windows (paths kept with
backslashes) was never pruned and ``code_graph_ingest(replace=True)`` silently left every stale file in place. The over-broad
guard also only knew "/" so a bare drive root ("C:\\") was not refused.

No skip on a broken engine: a store that cannot open a database must fail these tests, not hide behind importorskip.

Mutation runs (each applied to graph/ladybug_store.py, confirmed red, reverted):
- drop the backslash alternative (``OR s.file STARTS WITH $sub2`` in the four statements) -> test_a_backslash_tree_is_pruned... red
- ``stem = prefix.rstrip("/\\\\")`` back to ``rstrip("/")`` -> the trailing-backslash twin red
- drop the drive-root guard -> test_over_broad_windows_prefixes_are_refused red
- ``prefix.strip("/\\\\")`` back to ``strip("/")`` -> test_over_broad_windows_prefixes_are_refused red
"""
import pytest

pytest.importorskip("ladybug")

from graph.ladybug_store import LadybugStore  # noqa: E402


def _sym(sid, name, file, line=1, lang="python"):
    return {"id": sid, "name": name, "kind": "function", "line": line, "lang": lang, "file": file}


def _tree(store):
    a = r"C:\repo\a\x.py"
    ab = r"C:\repo\ab\y.py"          # shares the string prefix "C:\repo\a" and must survive
    store.ingest_code([
        {"file": a, "lang": "python", "symbols": [_sym(a + "::f", "f", a)], "edges": [], "imports": []},
        {"file": ab, "lang": "python", "symbols": [_sym(ab + "::g", "g", ab)], "edges": [], "imports": []},
    ])
    return a, ab


def _ids(store):
    return sorted(r[0] for r in store.code_query("MATCH (s:CodeSymbol) RETURN s.id"))


def _paths(store):
    return sorted(r[0] for r in store.code_query("MATCH (c:CodeFile) RETURN c.path"))


def test_a_backslash_tree_is_pruned_and_its_string_prefix_sibling_is_not(tmp_path):
    store = LadybugStore(str(tmp_path / "winpaths"))
    assert store.available()
    a, ab = _tree(store)
    assert store.delete_code_under(r"C:\repo\a") == {"files_deleted": 1, "symbols_deleted": 1}
    assert _ids(store) == [ab + "::g"]
    assert _paths(store) == [ab]


def test_a_prefix_with_a_trailing_backslash_prunes_the_same_tree(tmp_path):
    store = LadybugStore(str(tmp_path / "winpaths2"))
    assert store.available()
    a, ab = _tree(store)
    assert store.delete_code_under("C:\\repo\\a\\") == {"files_deleted": 1, "symbols_deleted": 1}
    assert _ids(store) == [ab + "::g"]


def test_a_forward_slash_tree_is_still_pruned(tmp_path):
    store = LadybugStore(str(tmp_path / "posixpaths"))
    assert store.available()
    store.ingest_code([
        {"file": "/repo/a/x.py", "lang": "python", "symbols": [_sym("/repo/a/x.py::f", "f", "/repo/a/x.py")],
         "edges": [], "imports": []},
        {"file": "/repo/ab/y.py", "lang": "python", "symbols": [_sym("/repo/ab/y.py::g", "g", "/repo/ab/y.py")],
         "edges": [], "imports": []},
    ])
    assert store.delete_code_under("/repo/a") == {"files_deleted": 1, "symbols_deleted": 1}
    assert _ids(store) == ["/repo/ab/y.py::g"]


@pytest.mark.parametrize("bad", ["\\", "\\\\", "C:\\", "C:", "c:/", "D:\\\\"])
def test_over_broad_windows_prefixes_are_refused(tmp_path, bad):
    store = LadybugStore(str(tmp_path / "guard"))
    assert store.available()
    a, ab = _tree(store)
    assert store.delete_code_under(bad) == {"files_deleted": 0, "symbols_deleted": 0}
    assert _ids(store) == sorted([a + "::f", ab + "::g"])             # nothing was wiped


@pytest.mark.parametrize("bad", ["\\", "\\\\"])
def test_a_backslash_only_prefix_does_not_wipe_a_unc_tree(tmp_path, bad):
    # without a path that starts with a backslash, refusing "\\" or "\\\\" is indistinguishable from matching nothing
    store = LadybugStore(str(tmp_path / "unc"))
    assert store.available()
    unc = "\\\\srv\\share\\z.py"
    store.ingest_code([{"file": unc, "lang": "python", "symbols": [_sym(unc + "::h", "h", unc)],
                        "edges": [], "imports": []}])
    assert store.delete_code_under(bad) == {"files_deleted": 0, "symbols_deleted": 0}
    assert _ids(store) == [unc + "::h"]
    assert _paths(store) == [unc]


def test_a_real_folder_on_a_unc_share_is_pruned(tmp_path):
    store = LadybugStore(str(tmp_path / "unc2"))
    assert store.available()
    unc = "\\\\srv\\share\\z.py"
    store.ingest_code([{"file": unc, "lang": "python", "symbols": [_sym(unc + "::h", "h", unc)],
                        "edges": [], "imports": []}])
    assert store.delete_code_under("\\\\srv\\share") == {"files_deleted": 1, "symbols_deleted": 1}
    assert _ids(store) == []


def test_a_real_folder_on_a_drive_is_not_refused(tmp_path):
    store = LadybugStore(str(tmp_path / "notroot"))
    assert store.available()
    _tree(store)
    assert store.delete_code_under("C:\\repo") == {"files_deleted": 2, "symbols_deleted": 2}
    assert _ids(store) == []

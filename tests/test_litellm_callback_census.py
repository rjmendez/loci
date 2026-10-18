"""scripts/litellm_callback_census.py: read-only census of the callbacks in a LiteLLM config.yaml.

Every tree is a tiny fake package written to tmp_path; the census is run as a real subprocess with
PYTHONPATH set to exactly those trees, so the find_spec-style results are the interpreter's own. The live
config, the live proxy interpreter and the real trees are never touched.

Mutation log (each applied to scripts/litellm_callback_census.py; the named tests went red; then reverted):
- a callback that is not found is reported resolved=True (the "swallow the ImportError" mutation, ok = True):
  test_missing_module_is_unresolved_with_the_exact_reason, test_main_exits_unresolved_then_zero_when_everything_resolves,
  test_release_tree_lacks_tier_router_dev_tree_has_it, test_census_rows_follow_config_order_with_path_and_sha256
- compare() matches by file name instead of sha256: test_comparison_flags_a_changed_file_even_with_the_same_name,
  test_comparison_matches_identical_content_in_different_trees
- sha256 of the wrong thing (path text instead of file bytes): test_census_rows_follow_config_order_with_path_and_sha256
- the child inherits the caller's PYTHONPATH: test_callers_pythonpath_is_not_inherited_by_the_probe
- the probe imports the target module (__import__(name)): test_census_does_not_execute_parent_or_target_modules
- the same plus PYTHONDONTWRITEBYTECODE dropped: test_census_writes_nothing_into_the_inspected_tree (dropping the env var alone
  survives, correctly, because the probe never imports; the var is defence in depth and only the combination is observable)
- attribute check dropped (module found => resolved): test_module_found_but_attribute_missing_is_unresolved
- parent-package walk replaced by a whole-name lookup: 32 of 40 tests, among them test_parent_package_missing_reports_the_first_missing_level
- the namespace stand-in module removed (KeyError in the child): test_namespace_directory_resolves_without_a_file
- main() exits 0 despite unresolved callbacks: test_main_exits_unresolved_then_zero_when_everything_resolves
- built-in names counted as unresolved: test_builtin_name_is_not_checked_and_not_a_failure
- an invalid entry echoed verbatim: test_invalid_entry_is_never_echoed_and_no_config_value_leaks
- resolved_from picks the first sys.path entry instead of the longest containing one: test_resolved_from_names_the_path_entry_that_provides_the_file
- reordering the rows: test_census_rows_follow_config_order_with_path_and_sha256
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "litellm_callback_census.py"


def _load():
    spec = importlib.util.spec_from_file_location("litellm_callback_census_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cen = _load()

PKG = "censusfix.integrations.loci"
CALLBACKS = [
    f"{PKG}.grounding_hook.hook_a",
    f"{PKG}.memory_writer.hook_b",
    f"{PKG}.tier_router.hook_c",
    f"{PKG}.cost_ledger.hook_d",
    f"{PKG}.capability_gate.hook_e",
]


def _tree(root: Path, files: dict) -> Path:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return root


def _plugin_files(names, attr_by_module=None, body_extra=""):
    """censusfix/integrations/loci/<name>.py for each name, with the package __init__ files."""
    files = {"censusfix/__init__.py": "", "censusfix/integrations/__init__.py": "", "censusfix/integrations/loci/__init__.py": ""}
    letters = {"grounding_hook": "hook_a", "memory_writer": "hook_b", "tier_router": "hook_c", "cost_ledger": "hook_d",
               "capability_gate": "hook_e"}
    for n in names:
        attr = (attr_by_module or {}).get(n, letters.get(n, "hook"))
        files[f"censusfix/integrations/loci/{n}.py"] = f"# {n}\n{attr} = object()\n{body_extra}"
    return files


def _config(callbacks, extra=""):
    items = "\n".join(f"    - {c}" for c in callbacks)
    return f"model_list: []\nlitellm_settings:\n  drop_params: true\n  callbacks:\n{items}\n{extra}"


def _census(entries, paths, python=None):
    return cen.census(entries, cen.subprocess_resolver(python or sys.executable, [str(p) for p in paths]))


def _row(report, entry):
    return next(r for r in report["rows"] if r["entry"] == entry)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- config parsing


def test_parse_block_list_ignores_comments_quotes_and_other_sections():
    text = (
        "general_settings:\n  callbacks:\n    - not.this.one\n"
        "litellm_settings:\n  # a comment\n  drop_params: true\n  callbacks:\n"
        "    - a.b.c   # trailing comment\n"
        "    - \"d.e.f\"\n"
        "    - 'g.h.i'\n"
        "    # interleaved comment\n"
        "    - j.k.l\n"
        "  success_callback:\n    - x.y.z\n"
        "other:\n  callbacks:\n    - q.r.s\n"
    )
    assert cen.parse_callbacks(text) == ["a.b.c", "d.e.f", "g.h.i", "j.k.l"]


def test_parse_inline_list_scalar_and_dash_at_key_indent():
    assert cen.parse_callbacks("litellm_settings:\n  callbacks: [a.b.c, \"d.e.f\"]\n") == ["a.b.c", "d.e.f"]
    assert cen.parse_callbacks("litellm_settings:\n  callbacks: a.b.c\n") == ["a.b.c"]
    assert cen.parse_callbacks("litellm_settings:\n  callbacks:\n  - a.b.c\n  - d.e.f\n  drop_params: true\n") == ["a.b.c", "d.e.f"]


def test_parse_returns_nothing_without_the_key():
    assert cen.parse_callbacks("litellm_settings:\n  drop_params: true\n") == []
    assert cen.parse_callbacks("general_settings:\n  callbacks:\n    - a.b.c\n") == []
    assert cen.parse_callbacks("litellm_settings:\n  callbacks:\n    - a.b.c\n") == ["a.b.c"]


def test_parse_handles_crlf_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_bytes(b"litellm_settings:\r\n  callbacks:\r\n    - a.b.c\r\n    - d.e.f\r\n")
    assert cen.parse_callbacks(cfg.read_text(encoding="utf-8-sig")) == ["a.b.c", "d.e.f"]


# ---------------------------------------------------------------- rows with a fake resolver


def test_census_rows_follow_config_order_with_path_and_sha256(tmp_path):
    files = {}
    for n in ("one", "two", "three", "four", "five"):
        p = tmp_path / f"{n}.py"
        p.write_text(f"# distinct body of {n}\n", encoding="utf-8")
        files[n] = p
    entries = [f"pkg.{n}.cb" for n in ("one", "two", "three", "four", "five")]
    seen = []

    def resolver(pairs):
        seen.append(pairs)
        return {"interpreter": {"executable": "fake", "version": "0", "sys_path": [str(tmp_path)]},
                "results": [{"found": True, "origin": str(files[m.split(".")[1]]), "kind": "module", "reason": None,
                             "resolved_prefix": m, "attr_defined": True} for m, _a in pairs]}

    report = cen.census(entries, resolver)
    assert seen == [[["pkg.one", "cb"], ["pkg.two", "cb"], ["pkg.three", "cb"], ["pkg.four", "cb"], ["pkg.five", "cb"]]]
    assert [r["entry"] for r in report["rows"]] == entries
    assert [r["index"] for r in report["rows"]] == [0, 1, 2, 3, 4]
    assert [r["origin"] for r in report["rows"]] == [str(files[n]) for n in ("one", "two", "three", "four", "five")]
    assert [r["sha256"] for r in report["rows"]] == [_sha(files[n]) for n in ("one", "two", "three", "four", "five")]
    assert len({r["sha256"] for r in report["rows"]}) == 5
    assert [r["resolved"] for r in report["rows"]] == [True] * 5
    assert report["unresolved"] == []
    assert all(r["resolved_from"] == str(tmp_path) for r in report["rows"])


def test_resolver_returning_the_wrong_number_of_results_is_an_error():
    with pytest.raises(SystemExit, match="resolver returned 0 results for 1 modules"):
        cen.census(["a.b.c"], lambda pairs: {"interpreter": {}, "results": []})


# ---------------------------------------------------------------- real find_spec in a subprocess


def test_resolves_present_module_with_file_sha_and_source_entry(tmp_path):
    tree = _tree(tmp_path / "tree", _plugin_files(["cost_ledger"]))
    report = _census([f"{PKG}.cost_ledger.hook_d"], [tree])
    row = report["rows"][0]
    target = tree / "censusfix" / "integrations" / "loci" / "cost_ledger.py"
    assert row["resolved"] is True
    assert row["reason"] is None
    assert row["module"] == f"{PKG}.cost_ledger"
    assert row["attr"] == "hook_d"
    assert row["origin"] == str(target)
    assert row["sha256"] == _sha(target)
    assert row["kind"] == "module"
    assert row["resolved_from"] == str(tree)
    assert row["attr_defined"] is True
    assert row["resolved_prefix"] == f"{PKG}.cost_ledger"
    assert report["unresolved"] == []


def test_missing_module_is_unresolved_with_the_exact_reason(tmp_path):
    tree = _tree(tmp_path / "tree", _plugin_files(["circuit_monitor"]))
    report = _census([f"{PKG}.tier_router.hook_c"], [tree])
    row = report["rows"][0]
    init = tree / "censusfix" / "integrations" / "loci" / "__init__.py"
    assert row["resolved"] is False
    assert row["reason"] == f"module '{PKG}.tier_router' not found in package '{PKG}' (resolved from {init})"
    assert row["resolved_prefix"] == PKG
    assert row["origin"] is None
    assert row["sha256"] is None
    assert report["unresolved"] == [f"{PKG}.tier_router.hook_c"]
    # positive twin: the same entry resolves once the file exists
    twin = _tree(tmp_path / "twin", _plugin_files(["tier_router"]))
    twin_row = _census([f"{PKG}.tier_router.hook_c"], [twin])["rows"][0]
    assert twin_row["resolved"] is True
    assert twin_row["reason"] is None


def test_parent_package_missing_reports_the_first_missing_level(tmp_path):
    # the module file exists, but only under a layout that lacks the 'censusfix.integrations' package
    tree = _tree(tmp_path / "tree", {"censusfix/__init__.py": "", "censusfix/loci/tier_router.py": "hook_c = 1\n"})
    row = _census([f"{PKG}.tier_router.hook_c"], [tree])["rows"][0]
    init = tree / "censusfix" / "__init__.py"
    assert row["resolved"] is False
    assert row["reason"] == f"module 'censusfix.integrations' not found in package 'censusfix' (resolved from {init})"
    assert row["resolved_prefix"] == "censusfix"
    # twin: the same file under the full parent chain resolves
    ok = _tree(tmp_path / "ok", _plugin_files(["tier_router"]))
    assert _census([f"{PKG}.tier_router.hook_c"], [ok])["rows"][0]["resolved"] is True


def test_top_level_package_missing(tmp_path):
    tree = _tree(tmp_path / "tree", {"otherpkg/__init__.py": ""})
    row = _census([f"{PKG}.tier_router.hook_c"], [tree])["rows"][0]
    assert row["resolved"] is False
    assert row["reason"] == "top-level module 'censusfix' not found on sys.path"
    assert row["resolved_prefix"] is None
    twin = _tree(tmp_path / "twin", _plugin_files(["tier_router"]))
    assert _census([f"{PKG}.tier_router.hook_c"], [tree, twin])["rows"][0]["resolved"] is True


def test_parent_that_is_a_plain_module_is_not_a_package(tmp_path):
    tree = _tree(tmp_path / "tree", {"censusfix/__init__.py": "", "censusfix/integrations.py": "x = 1\n"})
    row = _census([f"{PKG}.tier_router.hook_c"], [tree])["rows"][0]
    assert row["resolved"] is False
    assert row["reason"] == f"'censusfix.integrations' is not a package (it resolves to {tree / 'censusfix' / 'integrations.py'})"
    assert row["resolved_prefix"] == "censusfix.integrations"


def test_module_found_but_attribute_missing_is_unresolved(tmp_path):
    tree = _tree(tmp_path / "tree", _plugin_files(["tier_router"], {"tier_router": "something_else"}))
    row = _census([f"{PKG}.tier_router.hook_c"], [tree])["rows"][0]
    target = tree / "censusfix" / "integrations" / "loci" / "tier_router.py"
    assert row["resolved"] is False
    assert row["attr_defined"] is False
    assert row["reason"] == f"module '{PKG}.tier_router' resolves to {target} but defines no top-level name 'hook_c'"
    assert row["origin"] == str(target)
    assert row["sha256"] == _sha(target)
    # twin: with the attribute defined it resolves
    ok = _tree(tmp_path / "ok", _plugin_files(["tier_router"], {"tier_router": "hook_c"}))
    assert _census([f"{PKG}.tier_router.hook_c"], [ok])["rows"][0]["resolved"] is True


@pytest.mark.parametrize("body,expected", [
    ("def hook_c(): pass\n", True),
    ("class hook_c: pass\n", True),
    ("from somewhere import thing as hook_c\n", True),
    ("import os as hook_c\n", True),
    ("try:\n    hook_c = 1\nexcept Exception:\n    hook_c = 2\n", True),
    ("if True:\n    hook_c: int = 1\n", True),
    ("def hook_other(): pass\n", False),
    ("def f():\n    hook_c = 1\n", False),  # a local, not a module-level binding
    ("def __getattr__(name): return 1\n", None),  # dynamic: unknown, not held against it
    ("def broken(:\n", None),  # unparseable: unknown
])
def test_attribute_detection_is_static_and_exact(tmp_path, body, expected):
    tree = _tree(tmp_path / "tree", {**_plugin_files([]), "censusfix/integrations/loci/tier_router.py": body})
    row = _census([f"{PKG}.tier_router.hook_c"], [tree])["rows"][0]
    assert row["attr_defined"] is expected
    assert row["resolved"] is (expected is not False)


def test_namespace_directory_resolves_without_a_file(tmp_path):
    tree = _tree(tmp_path / "tree", {"censusfix/__init__.py": "", "censusfix/ns/inner.py": "hook = 1\n"})
    row = _census(["censusfix.ns.hook"], [tree])["rows"][0]
    assert row["resolved"] is True
    assert row["kind"] == "namespace"
    assert row["origin"] is None
    assert row["sha256"] is None
    assert row["attr_defined"] is None


def test_census_does_not_execute_parent_or_target_modules(tmp_path):
    sentinel = tmp_path / "ran.txt"
    boom = f"open({str(sentinel)!r}, 'a').write('x')\nraise RuntimeError('imported')\n"
    files = _plugin_files(["tier_router"], {"tier_router": "hook_c"})
    files["censusfix/__init__.py"] = boom
    files["censusfix/integrations/loci/tier_router.py"] = boom + "hook_c = 1\n"
    tree = _tree(tmp_path / "tree", files)
    row = _census([f"{PKG}.tier_router.hook_c"], [tree])["rows"][0]
    assert row["resolved"] is True
    assert not sentinel.exists()
    # positive twin: a real import of the same tree does run that code, so the sentinel check can fail
    imp = subprocess.run([sys.executable, "-c", f"import {PKG}.tier_router"], env={"PYTHONPATH": str(tree), "SYSTEMROOT": "C:/Windows"},
                         capture_output=True, text=True)
    assert imp.returncode != 0
    assert sentinel.read_text() == "x"


def test_census_writes_nothing_into_the_inspected_tree(tmp_path):
    tree = _tree(tmp_path / "tree", _plugin_files(["tier_router"]))
    before = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*"))
    assert _census([f"{PKG}.tier_router.hook_c"], [tree])["rows"][0]["resolved"] is True
    after = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*"))
    assert after == before
    assert [p for p in after if "__pycache__" in p] == []


def test_callers_pythonpath_is_not_inherited_by_the_probe(tmp_path, monkeypatch):
    tree = _tree(tmp_path / "tree", _plugin_files(["tier_router"]))
    monkeypatch.setenv("PYTHONPATH", str(tree))
    assert _census([f"{PKG}.tier_router.hook_c"], [])["rows"][0]["resolved"] is False
    assert _census([f"{PKG}.tier_router.hook_c"], [tree])["rows"][0]["resolved"] is True


def test_resolved_from_names_the_path_entry_that_provides_the_file(tmp_path):
    a = _tree(tmp_path / "a", _plugin_files(["tier_router"]))
    b = _tree(tmp_path / "b", {**_plugin_files(["tier_router"]), "censusfix/integrations/loci/tier_router.py": "hook_c = 2\n"})
    entry = f"{PKG}.tier_router.hook_c"
    first = _census([entry], [a, b])["rows"][0]
    second = _census([entry], [b, a])["rows"][0]
    assert first["resolved_from"] == str(a)
    assert first["origin"] == str(a / "censusfix" / "integrations" / "loci" / "tier_router.py")
    assert second["resolved_from"] == str(b)
    assert second["sha256"] == _sha(b / "censusfix" / "integrations" / "loci" / "tier_router.py")
    assert first["sha256"] != second["sha256"]


def test_resolved_from_prefers_the_longest_containing_entry(tmp_path):
    tree = _tree(tmp_path / "tree", _plugin_files(["tier_router"]))
    inner = tree / "censusfix" / "integrations"
    sys_path = [str(tree), str(inner), "", "C:/nowhere"]
    origin = str(inner / "loci" / "tier_router.py")
    assert cen._resolved_from(origin, sys_path) == str(inner)
    assert cen._resolved_from(origin, [str(tree)]) == str(tree)
    assert cen._resolved_from(origin, ["C:/nowhere"]) is None


# ---------------------------------------------------------------- the open question, as fixtures


def test_release_tree_lacks_tier_router_dev_tree_has_it(tmp_path):
    release = _tree(tmp_path / "release", _plugin_files(["grounding_hook", "memory_writer", "circuit_monitor", "cost_ledger", "capability_gate"]))
    dev = _tree(tmp_path / "dev", _plugin_files(["grounding_hook", "memory_writer", "tier_router", "cost_ledger", "capability_gate"]))
    rel = _census(CALLBACKS, [release])
    assert [r["resolved"] for r in rel["rows"]] == [True, True, False, True, True]
    assert rel["unresolved"] == [f"{PKG}.tier_router.hook_c"]
    d = _census(CALLBACKS, [dev])
    assert [r["resolved"] for r in d["rows"]] == [True] * 5
    assert d["unresolved"] == []


# ---------------------------------------------------------------- comparison


def _two_trees(tmp_path, left_body, right_body):
    left = _tree(tmp_path / "left", _plugin_files(["cost_ledger"], {"cost_ledger": "hook_d"}))
    right = _tree(tmp_path / "right", _plugin_files(["cost_ledger"], {"cost_ledger": "hook_d"}))
    (left / "censusfix/integrations/loci/cost_ledger.py").write_bytes(left_body.encode())
    (right / "censusfix/integrations/loci/cost_ledger.py").write_bytes(right_body.encode())
    return left, right


def _compare(entries, left, right):
    return cen.compare(_census(entries, [left]), _census(entries, [right]))


def test_comparison_flags_a_changed_file_even_with_the_same_name(tmp_path):
    left, right = _two_trees(tmp_path, "hook_d = 1\n", "hook_d = 2\n")
    (result,) = _compare([f"{PKG}.cost_ledger.hook_d"], left, right)
    assert result["status"] == "mismatch"
    assert result["left_sha256"] == _sha(left / "censusfix/integrations/loci/cost_ledger.py")
    assert result["right_sha256"] == _sha(right / "censusfix/integrations/loci/cost_ledger.py")
    assert result["left_sha256"] != result["right_sha256"]


def test_comparison_matches_identical_content_in_different_trees(tmp_path):
    left, right = _two_trees(tmp_path, "hook_d = 1\n", "hook_d = 1\n")
    (result,) = _compare([f"{PKG}.cost_ledger.hook_d"], left, right)
    assert result["status"] == "match"
    assert result["left_sha256"] == result["right_sha256"] == hashlib.sha256(b"hook_d = 1\n").hexdigest()
    assert result["left_origin"] != result["right_origin"]


def test_comparison_reports_one_sided_and_neither(tmp_path):
    left = _tree(tmp_path / "left", _plugin_files(["cost_ledger"]))
    right = _tree(tmp_path / "right", _plugin_files(["capability_gate"]))
    entries = [f"{PKG}.cost_ledger.hook_d", f"{PKG}.capability_gate.hook_e", f"{PKG}.tier_router.hook_c"]
    results = _compare(entries, left, right)
    assert [(r["entry"], r["status"]) for r in results] == [
        (entries[0], "only_left"), (entries[1], "only_right"), (entries[2], "neither")]


def test_comparison_skips_builtin_names_and_marks_unhashable_namespaces(tmp_path):
    left = _tree(tmp_path / "left", {"censusfix/__init__.py": "", "censusfix/ns/x.py": ""})
    right = _tree(tmp_path / "right", {"censusfix/__init__.py": "", "censusfix/ns/x.py": ""})
    results = _compare(["prometheus", "censusfix.ns.hook"], left, right)
    assert [(r["entry"], r["status"]) for r in results] == [("censusfix.ns.hook", "unhashable")]


# ---------------------------------------------------------------- built-ins, invalid entries, secrets


def test_builtin_name_is_not_checked_and_not_a_failure(tmp_path):
    tree = _tree(tmp_path / "tree", _plugin_files(["cost_ledger"]))
    report = _census(["prometheus", f"{PKG}.cost_ledger.hook_d"], [tree])
    assert report["rows"][0]["resolved"] is None
    assert report["rows"][0]["reason"] == "no dot: a LiteLLM built-in callback name, not checkable as a module path"
    assert report["rows"][1]["resolved"] is True
    assert report["unresolved"] == []


def test_invalid_entry_is_never_echoed_and_no_config_value_leaks(tmp_path, capsys):
    tree = _tree(tmp_path / "tree", _plugin_files(["cost_ledger"]))
    secret = "sk-SECRET-0123456789"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        _config([f"{PKG}.cost_ledger.hook_d", secret],
                extra=f"general_settings:\n  master_key: {secret}-master\nmodel_list:\n  - litellm_params:\n      api_key: {secret}-api\n"),
        encoding="utf-8")
    for as_json in (False, True):
        argv = ["--config", str(cfg), "--path", str(tree)] + (["--json"] if as_json else [])
        with pytest.raises(SystemExit, match=r"1 unresolved: <invalid entry #2>$"):
            cen.main(argv)
        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert "sk-SECRET" not in text
        assert "<invalid entry #2>" in text
    report = _census([f"{PKG}.cost_ledger.hook_d", secret], [tree])
    assert report["rows"][1]["entry"] == "<invalid entry #2>"
    assert report["rows"][1]["reason"] == "not a dotted python path"
    assert report["rows"][1]["resolved"] is False
    # twin: a valid entry is printed as itself
    assert report["rows"][0]["entry"] == f"{PKG}.cost_ledger.hook_d"


# ---------------------------------------------------------------- the CLI


def _write_cfg(tmp_path, callbacks):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_config(callbacks), encoding="utf-8")
    return cfg


def test_main_exits_unresolved_then_zero_when_everything_resolves(tmp_path, capsys):
    release = _tree(tmp_path / "release", _plugin_files(["grounding_hook", "memory_writer", "circuit_monitor", "cost_ledger", "capability_gate"]))
    dev = _tree(tmp_path / "dev", _plugin_files(["grounding_hook", "memory_writer", "tier_router", "cost_ledger", "capability_gate"]))
    cfg = _write_cfg(tmp_path, CALLBACKS)
    with pytest.raises(SystemExit, match=rf"^litellm_callback_census: 1 unresolved: {PKG}\.tier_router\.hook_c$"):
        cen.main(["--config", str(cfg), "--path", str(release)])
    out = capsys.readouterr().out
    assert f"3. {PKG}.tier_router.hook_c  UNRESOLVED" in out
    assert "5 callbacks, 1 unresolved" in out
    assert cen.main(["--config", str(cfg), "--path", str(dev)]) == 0
    out = capsys.readouterr().out
    assert f"3. {PKG}.tier_router.hook_c  RESOLVED" in out
    assert "5 callbacks, 0 unresolved" in out


def test_main_json_report_has_the_exact_shape(tmp_path, capsys):
    tree = _tree(tmp_path / "tree", _plugin_files(["cost_ledger"]))
    cfg = _write_cfg(tmp_path, [f"{PKG}.cost_ledger.hook_d"])
    assert cen.main(["--config", str(cfg), "--path", str(tree), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    target = tree / "censusfix" / "integrations" / "loci" / "cost_ledger.py"
    assert sorted(report) == ["config", "interpreter", "pythonpath", "rows", "unresolved"]
    assert report["config"] == str(cfg)
    assert report["pythonpath"] == [str(tree)]
    assert report["unresolved"] == []
    assert report["interpreter"]["executable"] == sys.executable
    assert report["rows"] == [{
        "index": 0, "entry": f"{PKG}.cost_ledger.hook_d", "module": f"{PKG}.cost_ledger", "attr": "hook_d", "resolved": True,
        "reason": None, "origin": str(target), "kind": "module", "sha256": _sha(target), "resolved_from": str(tree),
        "resolved_prefix": f"{PKG}.cost_ledger", "attr_defined": True}]


def test_main_compare_path_prints_the_comparison(tmp_path, capsys):
    left, right = _two_trees(tmp_path, "hook_d = 1\n", "hook_d = 2\n")
    cfg = _write_cfg(tmp_path, [f"{PKG}.cost_ledger.hook_d"])
    assert cen.main(["--config", str(cfg), "--path", str(left), "--compare-path", str(right), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["compare_pythonpath"] == [str(right)]
    assert [c["status"] for c in report["comparison"]] == ["mismatch"]
    assert cen.main(["--config", str(cfg), "--path", str(left), "--compare-path", str(left)]) == 0
    assert f"     match  {PKG}.cost_ledger.hook_d" in capsys.readouterr().out


def test_main_errors_are_specific(tmp_path):
    tree = _tree(tmp_path / "tree", _plugin_files(["cost_ledger"]))
    empty = tmp_path / "empty.yaml"
    empty.write_text("litellm_settings:\n  drop_params: true\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no callbacks found under litellm_settings in .*empty.yaml"):
        cen.main(["--config", str(empty), "--path", str(tree)])
    with pytest.raises(SystemExit, match="cannot read config"):
        cen.main(["--config", str(tmp_path / "missing.yaml")])
    cfg = _write_cfg(tmp_path, [f"{PKG}.cost_ledger.hook_d"])
    with pytest.raises(SystemExit, match="interpreter not found: .*no_such_python"):
        cen.main(["--config", str(cfg), "--python", str(tmp_path / "no_such_python.exe")])
    # twin: the same config with a real interpreter and the tree succeeds
    assert cen.main(["--config", str(cfg), "--path", str(tree)]) == 0


def test_a_failing_interpreter_is_reported_not_swallowed(tmp_path):
    fake = tmp_path / "fakepy.cmd"
    fake.write_text("@echo boom 1>&2\r\n@exit /b 3\r\n", encoding="utf-8")
    if sys.platform != "win32":
        fake = tmp_path / "fakepy.sh"
        fake.write_text("#!/bin/sh\necho boom >&2\nexit 3\n", encoding="utf-8")
        fake.chmod(0o755)
    with pytest.raises(SystemExit, match=r"failed \(exit 3\): boom"):
        cen.subprocess_resolver(str(fake), [])([["a.b", "c"]])

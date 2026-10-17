"""Every module-scope import under mlops/ must be installed in the job that runs it.

A module-scope import of a package the runner does not have does not fail that
one test -- it fails COLLECTION, and pytest reports the whole file as an error.
This shape cost five separate files across two repos on 2026-08-31 (scipy,
paho-mqtt, Pillow, pandas, python-dotenv), each time as "N errors" rather than a
missing dependency.

The CI job installs a hand-written list:

    pip install pytest pytest-timeout
    pip install numpy transformers datasets sentence-transformers

Nothing tied that list to the imports. sklearn and joblib are imported at module
scope and appear in it nowhere -- they arrive transitively through
sentence-transformers, which is luck, not a contract. This is the contract.
"""
import ast
import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
MLOPS = REPO / "mlops"
SKIP_DIRS = (".venv", "__pycache__", ".pytest_cache", "node_modules")


def _first_party() -> set:
    """Sibling modules imported after a sys.path insert -- `import features`,
    `import train`. They are files in this repo, not packages to install."""
    names = set()
    for d in (MLOPS, REPO / "deep_think_loci", REPO / "scripts"):
        for p in d.rglob("*.py"):
            if not any(k in str(p) for k in SKIP_DIRS):
                names.add(p.stem)
        names.add(d.name)
    return names


def _module_scope_imports() -> dict:
    """{top-level package: [files that import it at module scope]}."""
    out: dict = {}
    for p in sorted(MLOPS.rglob("*.py")):
        if any(k in str(p) for k in SKIP_DIRS):
            continue
        try:
            tree = ast.parse(p.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for node in tree.body:                      # module scope only, not nested
            if isinstance(node, ast.Import):
                for a in node.names:
                    out.setdefault(a.name.split(".")[0], []).append(p)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                out.setdefault(node.module.split(".")[0], []).append(p)
    return out


def test_every_module_scope_import_is_installed():
    first_party = _first_party()
    missing = {}
    for mod, files in sorted(_module_scope_imports().items()):
        if mod in sys.stdlib_module_names or mod in first_party:
            continue
        if importlib.util.find_spec(mod) is None:
            missing[mod] = sorted(str(f.relative_to(REPO)) for f in files)[:3]
    assert not missing, (
        "these are imported at module scope under mlops/ and are not installed here, "
        "so pytest will report them as collection ERRORS rather than as a missing "
        "dependency — add them to the 'Install dependencies' step of the MLOps job "
        f"in .github/workflows/ci.yml:\n"
        + "\n".join(f"  {m}: {', '.join(f)}" for m, f in missing.items())
    )


def test_the_scan_finds_the_imports_it_is_supposed_to_find():
    """A scan that silently found nothing would pass the test above forever."""
    found = _module_scope_imports()
    assert "numpy" in found, "the scan is not seeing module-scope imports at all"
    assert len(found) > 10, f"only {len(found)} modules seen — the walk is too narrow"
    assert any("sklearn" == m for m in found), (
        "sklearn is imported at module scope and is not named in the CI install list; "
        "if that stops being true this test should be updated deliberately"
    )

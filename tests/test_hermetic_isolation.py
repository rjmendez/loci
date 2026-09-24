"""tests/ runs under the same live-store isolation as mcp/tests and scripts/tests.

Without tests/conftest.py, the glymphatic tests resolved MUTEX_FLAG and DB_PATH
under the real ~/.hermes, so a run from the repo root touched the operator's
~/.hermes/glymphatic.lock.
"""
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_the_hermetic_isolation_is_installed_for_root_tests():
    hermetic = sys.modules.get("loci_hermetic")
    assert hermetic is not None and hermetic.installed(), "tests/conftest.py did not install loci_hermetic"
    root = Path(hermetic.root())
    assert Path(os.environ["HOME"]) == root / "home"
    assert hermetic.forbidden_roots(), "no live store is guarded"


def test_glymphatic_paths_resolve_inside_the_isolated_home():
    hermetic = sys.modules["loci_hermetic"]
    spec = importlib.util.spec_from_file_location(
        "glymphatic_sweep_isolation_probe", REPO / "scripts" / "glymphatic_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    home = str(Path(hermetic.root()) / "home")
    for path in (mod.MUTEX_FLAG, mod.DB_PATH, mod.SHIFT_SNAPSHOT_FILE):
        assert path.startswith(home + os.sep), path
        assert not hermetic.is_forbidden(path), path

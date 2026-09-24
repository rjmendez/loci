"""Isolate scripts/hooks/tests from the live Loci stores.

The scripts under test default to ~/.hermes and ~/.loci paths (event log,
memory-sessions, Mnemosyne, backends.toml) and several import mcp/backends,
which loads the repo .env. testsupport/loci_hermetic.py explains the mechanism;
it must be installed here, at conftest import, before any test module imports
the code it isolates. Opt out only for a deliberate live smoke test with
LOCI_TESTS_LIVE=1.
"""
import importlib.util
import sys
from pathlib import Path


def _load_hermetic():
    if "loci_hermetic" in sys.modules:
        return sys.modules["loci_hermetic"]
    path = Path(__file__).resolve().parents[3] / "testsupport" / "loci_hermetic.py"
    spec = importlib.util.spec_from_file_location("loci_hermetic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["loci_hermetic"] = module
    spec.loader.exec_module(module)
    return module


loci_hermetic = _load_hermetic()
loci_hermetic.install()
globals().update(loci_hermetic.pytest_hooks())

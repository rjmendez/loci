"""Isolate the root tests/ suite from the live Loci stores.

The glymphatic tests import scripts/glymphatic_sweep.py, which takes its lock
at ~/.hermes/glymphatic.lock and reads the Mnemosyne DB under ~/.hermes by
default. With no conftest here, a run from the repo root touched the real lock
file on the machine in daily use. testsupport/loci_hermetic.py explains the
mechanism; it must be installed at conftest import, before any test module
imports the code it isolates. Opt out only for a deliberate live smoke test
with LOCI_TESTS_LIVE=1.
"""
import importlib.util
import sys
from pathlib import Path


def _load_hermetic():
    if "loci_hermetic" in sys.modules:
        return sys.modules["loci_hermetic"]
    path = Path(__file__).resolve().parents[1] / "testsupport" / "loci_hermetic.py"
    spec = importlib.util.spec_from_file_location("loci_hermetic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["loci_hermetic"] = module
    spec.loader.exec_module(module)
    return module


loci_hermetic = _load_hermetic()
loci_hermetic.install()
globals().update(loci_hermetic.pytest_hooks())

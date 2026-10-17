"""Each consolidation script owns its batch-size knob.

They shared MAX_PER_RUN with three different defaults (100 / 50 / 20), so
exporting it to tune one silently reconfigured the other two.
"""
import importlib.util
import os
import pathlib
import unittest
from unittest import mock

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent

CASES = [
    ("amem_consolidation.py", "AMEM_MAX_PER_RUN", 100),
    ("ebbinghaus_consolidation.py", "EBBINGHAUS_MAX_PER_RUN", 50),
    ("agentHER_relabeler.py", "AGENTHER_MAX_PER_RUN", 20),
]

_ALL_NAMES = ["MAX_PER_RUN"] + [n for _, n, _ in CASES]


def _load(fname):
    spec = importlib.util.spec_from_file_location(f"_bs_{fname}", _SCRIPTS / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _max_per_run(fname, env):
    clean = {k: v for k, v in os.environ.items() if k not in _ALL_NAMES}
    with mock.patch.dict(os.environ, {**clean, **env}, clear=True):
        return _load(fname).MAX_PER_RUN


class TestBatchSizeEnv(unittest.TestCase):
    def test_defaults_are_unchanged(self):
        for fname, _, default in CASES:
            self.assertEqual(_max_per_run(fname, {}), default, fname)

    def test_own_name_wins(self):
        for fname, own, _ in CASES:
            self.assertEqual(_max_per_run(fname, {own: "7"}), 7, fname)

    def test_shared_name_is_still_honoured(self):
        for fname, _, _ in CASES:
            self.assertEqual(_max_per_run(fname, {"MAX_PER_RUN": "11"}), 11, fname)

    def test_own_name_overrides_the_shared_one(self):
        for fname, own, _ in CASES:
            self.assertEqual(_max_per_run(fname, {"MAX_PER_RUN": "11", own: "3"}), 3, fname)

    def test_tuning_one_script_leaves_the_others_alone(self):
        env = {"AMEM_MAX_PER_RUN": "500"}
        self.assertEqual(_max_per_run("amem_consolidation.py", env), 500)
        self.assertEqual(_max_per_run("ebbinghaus_consolidation.py", env), 50)
        self.assertEqual(_max_per_run("agentHER_relabeler.py", env), 20)


if __name__ == "__main__":
    unittest.main()

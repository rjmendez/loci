"""LOCI_LOG_FILE: optional rotating file log."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import log_setup  # noqa: E402


def _root():
    r = logging.Logger("test-root")
    r.setLevel(logging.INFO)
    return r


def test_disabled_without_env():
    r = _root()
    assert log_setup.install_file_logging({}, r) is None
    assert r.handlers == []


def test_writes_and_rotates(tmp_path):
    r = _root()
    path = tmp_path / "logs" / "loci.log"
    h = log_setup.install_file_logging(
        {"LOCI_LOG_FILE": str(path), "LOCI_LOG_MAX_BYTES": "1024", "LOCI_LOG_BACKUPS": "2"}, r)
    assert isinstance(h, logging.handlers.RotatingFileHandler)
    for i in range(200):
        r.info("line %d %s", i, "x" * 40)
    h.close()
    assert path.exists()
    assert (tmp_path / "logs" / "loci.log.1").exists()
    assert not (tmp_path / "logs" / "loci.log.3").exists()


def test_idempotent_for_same_path(tmp_path):
    r = _root()
    env = {"LOCI_LOG_FILE": str(tmp_path / "a.log")}
    assert log_setup.install_file_logging(env, r) is log_setup.install_file_logging(env, r)
    assert len(r.handlers) == 1
    r.handlers[0].close()


def test_unusable_path_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    r = _root()
    assert log_setup.install_file_logging({"LOCI_LOG_FILE": str(blocker / "sub" / "x.log")}, r) is None
    assert r.handlers == []

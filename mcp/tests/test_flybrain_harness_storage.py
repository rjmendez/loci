import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_harness_storage as fhs  # noqa: E402


def test_build_layout_from_env(monkeypatch, tmp_path):
    root = tmp_path / "flybrain-root"
    monkeypatch.setenv(fhs.ENV_FLYBRAIN_STORAGE_ROOT, str(root))
    layout = fhs.build_flybrain_harness_layout()
    assert layout.root == root.resolve(strict=False)
    assert layout.graph == layout.root / "graph"
    assert layout.snapshots == layout.root / "snapshots"
    assert layout.cache == layout.root / "cache"
    assert layout.backups == layout.root / "backups"
    assert layout.logs == layout.root / "logs"


def test_create_layout_directories(monkeypatch, tmp_path):
    root = tmp_path / "state-root"
    monkeypatch.setenv(fhs.ENV_FLYBRAIN_STORAGE_ROOT, str(root))
    layout = fhs.build_flybrain_harness_layout(create=True)
    for path in (
        layout.root,
        layout.graph,
        layout.snapshots,
        layout.cache,
        layout.backups,
        layout.logs,
    ):
        assert path.exists()
        assert path.is_dir()


def test_missing_storage_root_fails(monkeypatch):
    monkeypatch.delenv(fhs.ENV_FLYBRAIN_STORAGE_ROOT, raising=False)
    with pytest.raises(ValueError, match=fhs.ENV_FLYBRAIN_STORAGE_ROOT):
        fhs.resolve_flybrain_storage_root()


def test_rejects_relative_storage_root():
    with pytest.raises(ValueError, match="absolute"):
        fhs.resolve_flybrain_storage_root(root_override="relative\\flybrain")


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific guard")
def test_rejects_drive_root_on_windows():
    with pytest.raises(ValueError, match="drive root"):
        fhs.resolve_flybrain_storage_root(root_override="C:\\")

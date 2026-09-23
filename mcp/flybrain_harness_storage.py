"""FlyBrain harness storage-root layout helpers.

The harness writes only under ``LOCI_FLYBRAIN_STORAGE_ROOT``. This module
materializes a stable subtree contract for durable state, snapshots, cache,
backups, and logs without hardcoding machine-specific drive paths.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ENV_FLYBRAIN_STORAGE_ROOT = "LOCI_FLYBRAIN_STORAGE_ROOT"

_ROOT_DRIVEN_SUBDIRS = (
    "graph",
    "snapshots",
    "cache",
    "backups",
    "logs",
)


@dataclass(frozen=True)
class FlyBrainHarnessLayout:
    root: Path
    graph: Path
    snapshots: Path
    cache: Path
    backups: Path
    logs: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "graph": str(self.graph),
            "snapshots": str(self.snapshots),
            "cache": str(self.cache),
            "backups": str(self.backups),
            "logs": str(self.logs),
        }


def _expand_path(raw_path: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw_path.strip()))
    return Path(expanded).resolve(strict=False)


def _is_windows_reparse_point(path: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        st = os.lstat(path)
        attrs = getattr(st, "st_file_attributes", 0)
        return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except OSError:
        return False


def _iter_existing_components(path: Path):
    probe = Path(path.anchor) if path.anchor else Path(path.root or os.sep)
    if probe.exists():
        yield probe
    for part in path.parts[len(probe.parts):]:
        probe = probe / part
        if probe.exists():
            yield probe


def _validate_root(root: Path) -> None:
    if not root.is_absolute():
        raise ValueError("LOCI_FLYBRAIN_STORAGE_ROOT must resolve to an absolute path")

    if os.name == "nt":
        if str(root).startswith("\\\\"):
            raise ValueError("LOCI_FLYBRAIN_STORAGE_ROOT must be a local-drive path, not UNC")
        if root == Path(root.anchor):
            raise ValueError("LOCI_FLYBRAIN_STORAGE_ROOT cannot be a drive root")

    for component in _iter_existing_components(root):
        if component.is_symlink() or _is_windows_reparse_point(component):
            raise ValueError(
                f"LOCI_FLYBRAIN_STORAGE_ROOT may not traverse symlink/reparse path: {component}"
            )


def resolve_flybrain_storage_root(
    *,
    root_override: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    env_map = env if env is not None else os.environ
    raw = str(root_override).strip() if root_override is not None else (env_map.get(ENV_FLYBRAIN_STORAGE_ROOT, "").strip())
    if not raw:
        raise ValueError(f"{ENV_FLYBRAIN_STORAGE_ROOT} is required")
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if not Path(expanded).is_absolute():
        raise ValueError("LOCI_FLYBRAIN_STORAGE_ROOT must be absolute before normalization")
    root = _expand_path(raw)
    _validate_root(root)
    return root


def build_flybrain_harness_layout(
    *,
    root_override: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    create: bool = False,
) -> FlyBrainHarnessLayout:
    root = resolve_flybrain_storage_root(root_override=root_override, env=env)
    layout = FlyBrainHarnessLayout(
        root=root,
        graph=root / "graph",
        snapshots=root / "snapshots",
        cache=root / "cache",
        backups=root / "backups",
        logs=root / "logs",
    )
    if create:
        layout.root.mkdir(parents=True, exist_ok=True)
        for dirname in _ROOT_DRIVEN_SUBDIRS:
            (layout.root / dirname).mkdir(parents=True, exist_ok=True)
    return layout

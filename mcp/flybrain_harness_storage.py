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
ENV_FLYBRAIN_ALLOWLIST_ROOT = "LOCI_FLYBRAIN_ALLOWED_ROOT"
ENV_FLYBRAIN_STORAGE_ROOT_ALIASES = (
    ENV_FLYBRAIN_STORAGE_ROOT,
    "HARNESS_DATA_ROOT",
    "HARNESS_STORAGE_ROOT",
    ENV_FLYBRAIN_ALLOWLIST_ROOT,
)

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


def _validate_root(root: Path, *, env_name: str = ENV_FLYBRAIN_STORAGE_ROOT) -> None:
    if not root.is_absolute():
        raise ValueError(f"{env_name} must resolve to an absolute path")

    if os.name == "nt":
        if str(root).startswith("\\\\"):
            raise ValueError(f"{env_name} must be a local-drive path, not UNC")
        if root == Path(root.anchor):
            raise ValueError(f"{env_name} cannot be a drive root")

    for component in _iter_existing_components(root):
        if component.is_symlink() or _is_windows_reparse_point(component):
            raise ValueError(
                f"{env_name} may not traverse symlink/reparse path: {component}"
            )


def _reject_link_components(path: Path, *, env_name: str) -> None:
    """Fail closed if any existing prefix of the unresolved path is a link.

    Walks the components exactly as given (no lexical ``..`` collapsing), so
    ``<root>/link/..`` is caught at ``<root>/link`` rather than normalised away.
    """
    probe = Path(path.anchor) if path.anchor else Path()
    for part in path.parts[len(probe.parts):]:
        probe = probe / part
        if part in (".", ".."):
            continue
        if probe.is_symlink() or _is_windows_reparse_point(probe):
            raise ValueError(
                f"{env_name} may not traverse symlink/reparse path: {probe}"
            )


def _resolve_valid_root_value(raw: str, *, env_name: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw.strip()))
    if not expanded:
        raise ValueError(f"{env_name} is empty")
    if not Path(expanded).is_absolute():
        raise ValueError(f"{env_name} must resolve to an absolute path")
    # Check the path as written, BEFORE resolving it: Path.resolve() follows
    # symlinks, so validating only the resolved path can never see a symlinked
    # component and the symlink/reparse guard would be a no-op.
    _reject_link_components(Path(expanded), env_name=env_name)
    root = _expand_path(expanded)
    _validate_root(root, env_name=env_name)
    return root


def resolve_flybrain_storage_root(
    *,
    root_override: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    env_map = env if env is not None else os.environ

    if root_override is not None:
        return _resolve_valid_root_value(str(root_override), env_name="root_override")

    last_error: ValueError | None = None
    for candidate in ENV_FLYBRAIN_STORAGE_ROOT_ALIASES:
        value = str(env_map.get(candidate, "")).strip()
        if not value:
            continue
        try:
            return _resolve_valid_root_value(value, env_name=candidate)
        except ValueError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    allowed = ", ".join(ENV_FLYBRAIN_STORAGE_ROOT_ALIASES)
    raise ValueError(f"No FlyBrain storage root configured. Set one of: {allowed}")


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

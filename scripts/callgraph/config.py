"""Corpus definition — the single source of truth for "what is in the corpus".

Every count `cg` prints is only meaningful relative to this answer, so it
lives in one place and nothing else in the package re-derives it.

stdlib only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# scripts/callgraph/config.py -> scripts/callgraph -> scripts -> <repo root>
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

# The five top-level directories that make up the analyzed corpus (non-test
# Python only). Order matters only for display.
CORPUS_ROOTS: tuple[str, ...] = ("mcp", "scripts", "a2a_server", "mlops", "eval")

# Directory (path-component) names excluded anywhere they occur under a
# corpus root. Dot-directories are excluded as a class by is_excluded_rel --
# naming them one at a time is how .pytest_cache got in (213 fixture files
# under mcp/.pytest_cache/tmp, 2.7x the real corpus, on a machine that had run
# the test suite). The explicit names below are the non-dot ones plus the two
# that predate the class rule.
EXCLUDE_DIR_NAMES: frozenset[str] = frozenset({
    "tests",
    "__pycache__",
    ".venv",
    "venv",
    ".git",
    ".cache",
    "node_modules",
})
EXCLUDE_DIR_SUFFIXES: tuple[str, ...] = (".egg-info",)

# This tool's own package must never be walked as part of the corpus it
# analyzes — two AST walkers over the same code is how rule drift starts,
# and self-analysis produces noise no engineer asked for.
SELF_PACKAGE_REL = "scripts/callgraph"

CACHE_DIR = PACKAGE_ROOT / ".cache"


def is_excluded_rel(rel_posix: str) -> bool:
    """True if a repo-relative POSIX path (file or dir) must be skipped."""
    if rel_posix == SELF_PACKAGE_REL or rel_posix.startswith(SELF_PACKAGE_REL + "/"):
        return True
    parts = rel_posix.split("/")
    for part in parts:
        if part in EXCLUDE_DIR_NAMES:
            return True
        # Any dot-directory: .pytest_cache, .mypy_cache, .ruff_cache, .tox,
        # .venv, .git. Tool state, never analyzed source.
        if len(part) > 1 and part.startswith("."):
            return True
        if part.endswith(EXCLUDE_DIR_SUFFIXES):
            return True
    return False


def iter_corpus_files_worktree(repo_root: Path = REPO_ROOT) -> list[str]:
    """Repo-relative POSIX paths of every non-test .py file under the corpus
    roots, read from the working tree."""
    out: list[str] = []
    for root_name in CORPUS_ROOTS:
        base = repo_root / root_name
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = Path(dirpath).relative_to(repo_root).as_posix()
            if rel_dir != "." and is_excluded_rel(rel_dir):
                dirnames[:] = []
                continue
            dirnames[:] = [
                d for d in dirnames
                if not is_excluded_rel(f"{rel_dir}/{d}" if rel_dir != "." else d)
            ]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                rel_file = f"{rel_dir}/{fn}" if rel_dir != "." else fn
                if is_excluded_rel(rel_file):
                    continue
                out.append(rel_file)
    return sorted(out)


def iter_test_files_worktree(repo_root: Path = REPO_ROOT) -> list[str]:
    """Repo-relative POSIX paths of .py files under any `tests/` directory
    nested inside a corpus root — the deliberate COMPLEMENT of
    iter_corpus_files_worktree (which prunes `tests/` on purpose, see
    EXCLUDE_DIR_NAMES). Used only by `cg writes-dead --include-tests` as a
    textual mitigation pass over the weak write-with-no-read check; test
    files are never added to the modelled GraphStore itself."""
    out: list[str] = []
    for root_name in CORPUS_ROOTS:
        base = repo_root / root_name
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = Path(dirpath).relative_to(repo_root).as_posix()
            parts = [] if rel_dir == "." else rel_dir.split("/")
            # Same exclusions as the corpus walk minus `tests`, which this
            # function exists to collect. Derived from is_excluded_rel rather
            # than restated -- the restated copy had already drifted (it was
            # missing .cache, and every dot-directory).
            dirnames[:] = [
                d for d in dirnames
                if d == "tests" or not is_excluded_rel(f"{rel_dir}/{d}" if rel_dir != "." else d)
            ]
            if "tests" not in parts:
                continue
            for fn in filenames:
                if fn.endswith(".py"):
                    out.append(f"{rel_dir}/{fn}")
    return sorted(out)


def in_corpus(rel_posix: str) -> bool:
    """True if a repo-relative POSIX path falls under a corpus root and is
    not excluded. Does not check existence."""
    if not rel_posix.endswith(".py"):
        return False
    top = rel_posix.split("/", 1)[0]
    if top not in CORPUS_ROOTS:
        return False
    return not is_excluded_rel(rel_posix)


_STDLIB_NAMES: frozenset[str] | None = None


def stdlib_module_names() -> frozenset[str]:
    global _STDLIB_NAMES
    if _STDLIB_NAMES is None:
        names = set(getattr(sys, "stdlib_module_names", ()))
        # A handful of names that show up as top-level import targets but are
        # not in every interpreter build's stdlib_module_names set.
        names |= {"_thread", "__future__"}
        _STDLIB_NAMES = frozenset(names)
    return _STDLIB_NAMES


_THIRD_PARTY_NAMES: frozenset[str] | None = None


def _site_packages_dirs() -> list[Path]:
    """Every site-packages directory that could satisfy an import in the corpus.

    The project venvs (mcp/.venv, a2a_server/.venv) come first, but a git
    worktree or a CI checkout has no venv of its own; the interpreter running
    `cg` is then the only environment with anything installed. When only the
    venvs were consulted, a worktree run reported every installed dependency
    (fastapi, numpy, aiohttp, ...) as "unresolved".
    """
    dirs: list[Path] = []
    for lib_dir in (REPO_ROOT / "mcp" / ".venv" / "lib", REPO_ROOT / "a2a_server" / ".venv" / "lib"):
        if lib_dir.is_dir():
            dirs.extend(py_dir / "site-packages" for py_dir in sorted(lib_dir.glob("python3.*")))
    for entry in sys.path:
        if entry and Path(entry).name in ("site-packages", "dist-packages"):
            dirs.append(Path(entry))
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        try:
            key = d.resolve()
        except OSError:
            continue
        if key in seen or not d.is_dir():
            continue
        seen.add(key)
        out.append(d)
    return out


def third_party_top_level_names() -> frozenset[str]:
    """Top-level importable names installed in the project's venvs or in the
    running interpreter, used only to classify an external import as
    'third-party' (installed) vs 'unresolved' (nothing on disk answers to that
    name) for reporting purposes."""
    global _THIRD_PARTY_NAMES
    if _THIRD_PARTY_NAMES is not None:
        return _THIRD_PARTY_NAMES
    names: set[str] = set()
    for site in _site_packages_dirs():
        try:
            entries = list(site.iterdir())
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if name.endswith(".pth"):
                continue
            if name.endswith(".dist-info") or name.endswith(".egg-info"):
                name = name.split("-")[0]
            elif name.endswith(".py"):
                name = name[:-3]
            elif name.endswith((".so", ".pyd")):
                name = name.split(".")[0]
            if name:
                names.add(name)
    _THIRD_PARTY_NAMES = frozenset(names)
    return _THIRD_PARTY_NAMES

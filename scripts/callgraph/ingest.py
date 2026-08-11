"""Corpus walk + ast.parse, and a `--rev` path that streams file content
from git instead of the working tree.

While another workflow is mid-editing a file in the corpus, the working
tree copy of that file may be inconsistent at any single moment. Reading a
specific git revision (default: the working tree; pass rev="HEAD" or any
commit-ish) is the correct way to get stable line numbers, and is required
whenever a concurrent edit is possible. Every report built on top of this
module names which source it read.

A SyntaxError on one file degrades that file to a stub (tree=None,
error=<message>) and never aborts the rest of the build.

No parse cache: measured cold-parsing all 114 corpus files takes ~0.2s
(ast.parse itself; pickling a tree back out costs about as much as
re-parsing it), so a cache would add staleness risk for no real speedup at
this corpus size. `--no-cache` is accepted and is a no-op today; if the
corpus grows enough for this to matter, revisit here first.

stdlib only.
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from . import config


@dataclass
class SourceFile:
    rel_path: str            # repo-relative POSIX path, e.g. "mcp/server.py"
    source: str
    tree: Optional[ast.AST]
    error: Optional[str]     # SyntaxError message, else None
    origin: str               # "working tree" or "rev <sha>"
    sha1: str = field(default="", repr=False)


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=config.REPO_ROOT,
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def read_git_blob(rel_path: str, rev: str) -> str:
    """Single-file read via `git show`. Kept for callers that only need one
    blob; load_corpus below uses the batched path (one subprocess for the
    whole corpus, not one per file) because it runs on every `--rev` build,
    including while another workflow is mid-editing files this tool must
    still read a stable copy of."""
    result = subprocess.run(
        ["git", "show", f"{rev}:{rel_path}"], cwd=config.REPO_ROOT,
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def list_git_files_with_sha(rev: str) -> list[tuple[str, str]]:
    """(rel_path, blob_sha) for every corpus .py file at `rev`, sorted by
    path -- one `git ls-tree` call, no per-file subprocess. `git ls-tree -r`
    (without --name-only) lines look like `<mode> blob <sha>\\t<path>`."""
    out = _run_git(["ls-tree", "-r", rev])
    result: list[tuple[str, str]] = []
    for line in out.splitlines():
        if "\t" not in line:
            continue
        meta, path = line.split("\t", 1)
        parts = meta.split()
        if len(parts) != 3 or parts[1] != "blob":
            continue
        if config.in_corpus(path):
            result.append((path, parts[2]))
    result.sort(key=lambda pair: pair[0])
    return result


def _read_git_blobs_batch(shas: list[str]) -> dict[str, bytes]:
    """Every blob in `shas`, read via ONE `git cat-file --batch` subprocess
    instead of one `git show`/`git cat-file` per file -- the difference
    between ~114 process spawns and 1 for a full-corpus `--rev` build.
    `--batch` writes, per requested object in request order: a header line
    `<sha> <type> <size>\\n`, exactly `<size>` bytes of content, then a
    trailing `\\n`."""
    if not shas:
        return {}
    proc = subprocess.run(
        ["git", "cat-file", "--batch"], cwd=config.REPO_ROOT,
        input=("\n".join(shas) + "\n").encode("ascii"),
        capture_output=True, check=True,
    )
    out = proc.stdout
    result: dict[str, bytes] = {}
    pos = 0
    for sha in shas:
        nl = out.index(b"\n", pos)
        header = out[pos:nl].decode("ascii")
        parts = header.split()
        if len(parts) != 3:
            raise RuntimeError(f"git cat-file --batch: unexpected header {header!r} for {sha}")
        obj_sha, _obj_type, size_s = parts
        size = int(size_s)
        start = nl + 1
        result[obj_sha] = out[start:start + size]
        pos = start + size + 1  # skip the trailing newline after the object body
    return result


def resolve_rev_sha(rev: str) -> str:
    out = _run_git(["rev-parse", rev])
    return out.strip()


def _parse_one(rel_path: str, source: str, origin: str) -> SourceFile:
    sha1 = hashlib.sha1(source.encode("utf-8", errors="surrogateescape")).hexdigest()
    try:
        tree = ast.parse(source, filename=rel_path)
        return SourceFile(rel_path, source, tree, None, origin, sha1)
    except SyntaxError as exc:
        msg = f"{exc.__class__.__name__}: {exc.msg} (line {exc.lineno})"
        return SourceFile(rel_path, source, None, msg, origin, sha1)
    except (ValueError, RecursionError) as exc:  # e.g. null bytes, too-deep nesting
        return SourceFile(rel_path, source, None, f"{exc.__class__.__name__}: {exc}", origin, sha1)


def load_corpus(rev: Optional[str] = None, no_cache: bool = False) -> tuple[list[SourceFile], str]:
    """Returns (source files, origin label). origin is "working tree" or
    "rev <sha>" — every report prints this so a result read mid-edit is
    never mistaken for a stable one."""
    if rev is None:
        rel_paths = config.iter_corpus_files_worktree()
        origin = "working tree"
        out: list[SourceFile] = []
        for rel_path in rel_paths:
            abs_path = config.REPO_ROOT / rel_path
            try:
                source = abs_path.read_text(encoding="utf-8", errors="surrogateescape")
            except OSError as exc:
                out.append(SourceFile(rel_path, "", None, f"OSError: {exc}", origin, ""))
                continue
            out.append(_parse_one(rel_path, source, origin))
        return out, origin

    sha = resolve_rev_sha(rev)
    origin = f"rev {sha[:12]}"
    files_with_sha = list_git_files_with_sha(rev)
    blobs = _read_git_blobs_batch([s for _p, s in files_with_sha])
    out = []
    for rel_path, blob_sha in files_with_sha:
        raw = blobs.get(blob_sha)
        if raw is None:
            out.append(SourceFile(rel_path, "", None, f"git cat-file: blob {blob_sha} missing", origin, ""))
            continue
        source = raw.decode("utf-8", errors="surrogateescape")
        out.append(_parse_one(rel_path, source, origin))
    return out, origin


def list_git_test_files(rev: str) -> list[str]:
    """Repo-relative POSIX paths of .py files under a `tests/` directory
    nested inside a corpus root, as they existed at `rev` — the test-side
    counterpart of list_git_files, used only by `cg writes-dead
    --include-tests`."""
    out = _run_git(["ls-tree", "-r", "--name-only", rev])
    result: list[str] = []
    for p in out.splitlines():
        if not p.endswith(".py"):
            continue
        top = p.split("/", 1)[0]
        if top not in config.CORPUS_ROOTS:
            continue
        if "tests" not in p.split("/")[:-1]:
            continue
        result.append(p)
    return sorted(result)


def load_test_sources(rev: Optional[str] = None) -> list[SourceFile]:
    """Test-tree counterpart of load_corpus — NOT part of the modelled
    corpus (config.in_corpus excludes tests/ on purpose); this is read-only
    source text for `cg writes-dead --include-tests`'s textual mitigation
    scan, never parsed into the GraphStore."""
    if rev is None:
        rel_paths = config.iter_test_files_worktree()
        origin = "working tree"
    else:
        sha = resolve_rev_sha(rev)
        rel_paths = list_git_test_files(rev)
        origin = f"rev {sha[:12]}"

    out: list[SourceFile] = []
    for rel_path in rel_paths:
        if rev is None:
            abs_path = config.REPO_ROOT / rel_path
            try:
                source = abs_path.read_text(encoding="utf-8", errors="surrogateescape")
            except OSError as exc:
                out.append(SourceFile(rel_path, "", None, f"OSError: {exc}", origin, ""))
                continue
        else:
            source = read_git_blob(rel_path, rev)
        out.append(_parse_one(rel_path, source, origin))
    return out


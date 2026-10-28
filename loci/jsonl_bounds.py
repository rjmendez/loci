"""Bounded JSONL files: rotate on append, read only the tail.

The per-policy trajectory and adherence files used to grow without limit and were read whole.
Both now go through this module:

* ``append_bounded`` appends one line, first rotating the file when the line would take it past
  ``max_rows`` rows or ``max_bytes`` bytes.  A rotated file is plain JSONL named ``<name>.<k>``
  (k = 1, 2, ...), so it stays readable and does not match the ``*.jsonl`` listing glob.
* ``tail_lines`` returns the last ``n`` complete lines by reading blocks backwards from the end,
  so the bytes read depend on ``n``, not on the file size.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

MAX_FILE_BYTES = 1024 * 1024
MAX_FILE_ROWS = 5000
TAIL_CHUNK_BYTES = 4096

# path -> (size, rows) as of our last append/count; valid only while the file size still matches.
_ROW_CACHE: Dict[str, Tuple[int, int]] = {}


def _count_rows(path: Path, size: int) -> int:
    rows = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(65536)
            if not block:
                break
            rows += block.count(b"\n")
    _ROW_CACHE[str(path)] = (size, rows)
    return rows


def _next_rotation_index(path: Path) -> int:
    highest = 0
    prefix = path.name + "."
    for other in path.parent.glob(path.name + ".*"):
        suffix = other.name[len(prefix):]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def append_bounded(
    path: Path,
    line: str,
    *,
    max_rows: int = MAX_FILE_ROWS,
    max_bytes: int = MAX_FILE_BYTES,
) -> bool:
    """Append *line* (one JSON document, no newline) to *path*; return True when the file was rotated first.

    The current file never exceeds ``max_bytes`` bytes or ``max_rows`` rows: when appending would,
    it is renamed to ``<name>.<k>`` and the line starts a fresh file.  A single line longer than
    ``max_bytes`` is refused (ValueError) rather than written.
    """
    path = Path(path)
    data = line.encode("utf-8") + b"\n"
    if len(data) > max_bytes:
        raise ValueError(f"row of {len(data)} bytes exceeds the {max_bytes}-byte file limit")
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        size = 0
    rotate = False
    if size:
        if size + len(data) > max_bytes:
            rotate = True  # also covers a legacy file already over the cap, without counting its rows
        else:
            cached = _ROW_CACHE.get(str(path))
            rows = cached[1] if cached is not None and cached[0] == size else _count_rows(path, size)
            rotate = rows >= max_rows
    if rotate:
        try:
            os.replace(path, path.with_name(f"{path.name}.{_next_rotation_index(path)}"))
        except FileNotFoundError:
            pass  # another writer rotated it first
        size = 0
        _ROW_CACHE.pop(str(path), None)
    with open(path, "ab") as fh:
        fh.write(data)
    cached = _ROW_CACHE.get(str(path))
    rows_before = cached[1] if cached is not None and cached[0] == size else (0 if size == 0 else None)
    if rows_before is None:
        _ROW_CACHE.pop(str(path), None)
    else:
        _ROW_CACHE[str(path)] = (size + len(data), rows_before + 1)
    return rotate


def tail_lines(
    path: Path,
    n: int,
    *,
    opener: Callable[..., Any] = open,
    chunk: int = TAIL_CHUNK_BYTES,
) -> List[bytes]:
    """The last *n* complete lines of *path* (oldest first), reading blocks backwards from the end.

    Only as many ``chunk``-sized blocks as are needed to hold n+1 newlines are read.  A missing file
    or n <= 0 gives [].  *opener* is injectable so a test can count the bytes read.
    """
    if n <= 0:
        return []
    try:
        fh = opener(path, "rb")
    except FileNotFoundError:
        return []
    with fh:
        fh.seek(0, os.SEEK_END)
        pos = fh.tell()
        buf = b""
        while pos > 0 and buf.count(b"\n") <= n:
            start = max(0, pos - chunk)
            fh.seek(start)
            buf = fh.read(pos - start) + buf
            pos = start
    pieces = buf.split(b"\n")
    if pos > 0:
        pieces = pieces[1:]  # the first piece starts mid-line
    if pieces and pieces[-1] == b"":
        pieces = pieces[:-1]
    return [p for p in pieces[-n:] if p.strip()]

"""memcheck checks — pure, decoupled rule functions over plain dicts.

Each check takes plain ``dict`` records (findings / audit entries), NOT server
domain objects, so it is unit-testable without importing the MCP server or a
live qdrant. Each returns a ``list[Verdict]`` with ``subject_kind="memory"``.

**Advisory-only.** A verdict annotates and surfaces; it never hides, deletes, or
mutates a finding. Findings stay append-only.

Every check is fail-safe: a malformed record dict is skipped, never raised.
"""

from .code_hallucination import run_code_checks
from .contagion import find_contamination
from .contradiction import run_contradiction
from .provenance import run_provenance

__all__ = [
    "run_provenance",
    "run_contradiction",
    "find_contamination",
    "run_code_checks",
]

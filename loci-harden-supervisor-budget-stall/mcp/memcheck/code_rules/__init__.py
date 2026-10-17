"""Vendored LLM code-hallucination checker.

A pinned copy of the upstream ``llm_hallucination_checks`` ruff plugin, used by
``memcheck.checks.code_hallucination`` to map static code-smell findings into
``Verdict`` objects on the PostToolUse code path.

VENDORED — do not edit the checker here. Edit upstream and run
``scripts/sync_hallucination_rules.sh`` to refresh the pin. See ``VENDORED.md``.

Re-exports the stable surface the rest of memcheck depends on: ``check_file``
and the ``Issue`` dataclass.
"""

from .extended_checks import (
    ALL_PATTERN_IDS,
    PATTERN_META,
    run_extended_checks,
)
from .llm_hallucination_checks import Issue, check_file

__all__ = [
    "check_file",
    "Issue",
    "PATTERN_META",
    "ALL_PATTERN_IDS",
    "run_extended_checks",
]

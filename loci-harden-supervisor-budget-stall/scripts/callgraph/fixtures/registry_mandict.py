"""Fixture: MAN-DICT — the `_SKILL_MAP` shape (a module-level, annotated
dict-of-bare-callables), and a config dict with literal values that must
NOT be classified as a dispatch table."""
from typing import Any


def skill_one(task):
    return task


def skill_two(task):
    return task


_SKILL_MAP: dict[str, Any] = {
    "one": skill_one,
    "two": skill_two,
}

_CONFIG_MAP = {
    "high": 0.9,
    "low": 0.1,
}


def dispatch(skill_id, task):
    """Mirrors a2a_server/server.py's `_dispatch`: a bare-name callsite
    (`handler(task)`) whose callee was assigned two lines earlier from
    `_SKILL_MAP.get(skill_id)` — resolvable only via extract/dispatch.py's
    rung 5 (dict-dispatch), which fans out via DISPATCHES to every
    registered member instead of guessing one."""
    handler = _SKILL_MAP.get(skill_id)
    if not handler:
        return None
    return handler(task)


def dispatch_subscript(skill_id, task):
    """Same shape, subscript form instead of `.get(...)`."""
    handler = _SKILL_MAP[skill_id]
    return handler(task)

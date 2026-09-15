"""Helpers for surfacing stored memory as inert data in LLM-facing contexts."""
from __future__ import annotations

import html

_OPEN = "<untrusted_memory_content"
_CLOSE = "</untrusted_memory_content>"


def wrap_untrusted_memory_text(text: str, **attrs) -> str:
    """Frame stored memory as data, not executable instructions.

    Idempotent: an already-wrapped block is returned unchanged so callers can
    safely apply the wrapper at multiple surfacing layers.
    """
    body = str(text or "").strip()
    if not body:
        return ""
    if body.startswith(_OPEN) and body.endswith(_CLOSE):
        return body
    rendered = []
    for key, val in attrs.items():
        if val is None or val == "":
            continue
        rendered.append(f'{key}="{html.escape(str(val), quote=True)}"')
    attr_blob = f" {' '.join(rendered)}" if rendered else ""
    return f"{_OPEN}{attr_blob}>\n{body}\n{_CLOSE}"

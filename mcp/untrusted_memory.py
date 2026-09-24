"""Helpers for surfacing stored memory as inert data in LLM-facing contexts."""
from __future__ import annotations

import html
import re

_OPEN = "<untrusted_memory_content"
_CLOSE = "</untrusted_memory_content>"

# Any open or close frame tag, including case and whitespace variants
# ("< /Untrusted_Memory_Content"), that stored text could use to close the
# frame early or open a forged one.
_FRAME_TAG_RE = re.compile(r"<(\s*/?\s*untrusted_memory_content)", re.IGNORECASE)
# One frame this module emitted: bodies are always neutralised, so a genuine
# frame never contains a raw frame tag inside it.
_SINGLE_FRAME_RE = re.compile(
    r"(?s)^<untrusted_memory_content(?:\s[^<>]*)?>\n?(?P<body>.*?)\n?</untrusted_memory_content>$"
)


def neutralize_frame_tags(text: str) -> str:
    """Escape every frame tag in ``text`` so it cannot close or open a frame."""
    return _FRAME_TAG_RE.sub(r"&lt;\1", str(text or ""))


def wrap_untrusted_memory_text(text: str, **attrs) -> str:
    """Frame stored memory as data, not executable instructions.

    The frame attributes always come from ``attrs``, never from the text. A body
    that is already a single clean frame (a surfacing layer re-wrapping output
    from another layer) is unwrapped and re-framed, so the call stays idempotent
    in shape while a stored, forged frame loses its attacker-chosen attributes.
    Every frame tag left in the body is escaped, so stored text can neither close
    the frame early nor forge a second one.
    """
    body = str(text or "").strip()
    if not body:
        return ""
    m = _SINGLE_FRAME_RE.match(body)
    if m and not _FRAME_TAG_RE.search(m.group("body")):
        body = m.group("body").strip()
        if not body:
            return ""
    body = neutralize_frame_tags(body)
    rendered = []
    for key, val in attrs.items():
        if val is None or val == "":
            continue
        rendered.append(f'{key}="{html.escape(str(val), quote=True)}"')
    attr_blob = f" {' '.join(rendered)}" if rendered else ""
    return f"{_OPEN}{attr_blob}>\n{body}\n{_CLOSE}"

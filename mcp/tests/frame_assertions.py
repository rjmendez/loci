"""Balanced untrusted-memory frame assertions shared by the wrap-site tests.

A wrap site is only safe when every frame it emits is closed and the stored
payload sits INSIDE a frame. Checking ``startswith("<untrusted_memory_content")``
or ``"payload" in text`` passes for a truncated frame with no closing tag and for
a raw payload printed next to a decoy frame, so these helpers parse the frames.
"""
from __future__ import annotations

import re

# Any raw (un-neutralised) frame tag, including case and whitespace variants a
# forged tag could use. Neutralised tags start with "&lt;" and do not match.
_TAG_RE = re.compile(r"<\s*(?P<close>/)?\s*untrusted_memory_content\b(?P<attrs>[^<>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r'([A-Za-z_][\w-]*)="([^"]*)"')


def frame_spans(text: str) -> list[dict]:
    """Parse every frame in ``text``; fail on a nested, stray or unclosed tag."""
    spans: list[dict] = []
    open_m = None
    for m in _TAG_RE.finditer(text):
        if m.group("close"):
            assert open_m is not None, f"stray closing frame tag at {m.start()}: {text!r}"
            spans.append({
                "start": open_m.start(),
                "end": m.end(),
                "open_tag": open_m.group(0),
                "attrs": dict(_ATTR_RE.findall(open_m.group("attrs"))),
                "body": text[open_m.end():m.start()],
            })
            open_m = None
        else:
            assert open_m is None, f"frame opened inside an open frame at {m.start()}: {text!r}"
            open_m = m
    assert open_m is None, f"frame opened at {open_m.start()} is never closed: {text!r}"
    return spans


def outside_frames(text: str) -> str:
    """``text`` with every frame span removed."""
    out, pos = [], 0
    for span in frame_spans(text):
        out.append(text[pos:span["start"]])
        pos = span["end"]
    out.append(text[pos:])
    return "".join(out)


def assert_payload_framed(text: str, payload: str, **attrs) -> dict:
    """Assert ``payload`` sits inside one balanced frame and nowhere outside one.

    ``attrs`` must all appear, exactly, on the frame holding the payload.
    Returns that frame's span.
    """
    spans = frame_spans(text)
    holding = [s for s in spans if payload in s["body"]]
    assert holding, f"payload {payload!r} is not inside any balanced frame: {text!r}"
    assert payload not in outside_frames(text), f"payload {payload!r} also appears outside a frame: {text!r}"
    span = holding[0]
    for key, val in attrs.items():
        assert span["attrs"].get(key) == str(val), (key, span["attrs"], text)
    return span


def assert_single_frame(text: str, payload: str, **attrs) -> dict:
    """``text`` is exactly one balanced frame whose body is exactly ``payload``."""
    spans = frame_spans(text)
    assert len(spans) == 1, spans
    span = spans[0]
    assert (span["start"], span["end"]) == (0, len(text)), f"text outside the frame: {text!r}"
    assert span["body"].strip() == payload, span["body"]
    return assert_payload_framed(text, payload, **attrs)

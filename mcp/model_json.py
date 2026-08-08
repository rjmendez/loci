"""Defensive JSON-object extraction from noisy model output.

A leaf module (stdlib only, imports no other loci module) shared by the generation-tier
callers that ask a local model for "ONLY a JSON object" and have to cope with what
actually comes back: clean JSON, JSON in a ```json fence, or JSON buried in prose.

Kept separate from mcp/text_ops.py on purpose — that module is the classify/compress
wrapper over Ollama; this is a pure parser with no backend of its own.
"""
from __future__ import annotations

import json
import re
from typing import Optional


def extract_json_object(text: str) -> Optional[dict]:
    """Defensively pull a JSON object out of possibly-noisy model text.

    Handles: clean JSON, JSON wrapped in ```json fences, and JSON embedded in stray prose.
    Returns the parsed dict, or None if nothing parseable is found.
    """
    if not text or not isinstance(text, str):
        return None
    # 1) straight parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) strip code fences and retry
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 3) brace-match scan: first '{' whose balanced span parses as an object
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        start = -1  # keep scanning for a later valid object
    return None

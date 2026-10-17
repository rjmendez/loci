"""Loads rules.toml — the pattern set for decorator/registrar classification,
kept as data so a new registrar or wrapper is a one-line edit for any
engineer, not a code change.

stdlib only (tomllib, stdlib since Python 3.11).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import lru_cache

from . import config

RULES_PATH = config.PACKAGE_ROOT / "rules.toml"


@dataclass
class Rules:
    decorator_registering: tuple[str, ...] = ()
    decorator_wrapping: tuple[str, ...] = ()
    register_fn_names: tuple[str, ...] = ()
    roots: tuple[dict, ...] = ()
    raw: dict = field(default_factory=dict)

    def classify_decorator(self, head: str) -> str:
        """head: the dotted tail of the decorator with any call parens
        stripped, e.g. "mcp.tool" from "@mcp.tool()", "app.get" from
        "@app.get(\"/a2a\")". Matched against the LAST dotted component so
        both "tool" and "mcp.tool" style entries in rules.toml work."""
        last = head.rsplit(".", 1)[-1]
        if head in self.decorator_registering or last in self.decorator_registering:
            return "registering"
        if head in self.decorator_wrapping or last in self.decorator_wrapping:
            return "wrapping"
        return "unknown"

    def is_register_fn_name(self, name: str) -> bool:
        for pattern in self.register_fn_names:
            if pattern.endswith("*"):
                if name.startswith(pattern[:-1]):
                    return True
            elif name == pattern:
                return True
        return False


@lru_cache(maxsize=1)
def load_rules() -> Rules:
    with RULES_PATH.open("rb") as f:
        data = tomllib.load(f)
    dec = data.get("decorators", {})
    registrars = data.get("registrars", {})
    return Rules(
        decorator_registering=tuple(dec.get("registering", [])),
        decorator_wrapping=tuple(dec.get("wrapping", [])),
        register_fn_names=tuple(registrars.get("register_fn_names", [])),
        roots=tuple(data.get("roots", [])),
        raw=data,
    )

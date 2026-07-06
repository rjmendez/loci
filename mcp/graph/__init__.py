"""Loci MCP graph package — code symbol/reference extraction + (optional) Kuzu store."""

from .code_parse import (
    LANG_BY_EXT,
    detect_lang,
    parse_source,
    parse_path,
)

__all__ = ["LANG_BY_EXT", "detect_lang", "parse_source", "parse_path"]

# KuzuStore is an optional companion module; keep the package importable even
# when the (in-progress) kuzu_store module is not yet present.
try:  # pragma: no cover - depends on optional sibling module
    from .kuzu_store import KuzuStore  # noqa: F401

    __all__.append("KuzuStore")
except Exception:  # pragma: no cover
    pass

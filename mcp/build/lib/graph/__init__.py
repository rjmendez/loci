"""Loci MCP graph package — code symbol/reference extraction + (optional) LadybugDB store."""

from .code_parse import (
    LANG_BY_EXT,
    detect_lang,
    parse_source,
    parse_path,
)

__all__ = ["LANG_BY_EXT", "detect_lang", "parse_source", "parse_path"]

# LadybugStore is an optional companion module; keep the package importable even
# when the (in-progress) ladybug_store module is not yet present.
try:  # pragma: no cover - depends on optional sibling module
    from .ladybug_store import LadybugStore  # noqa: F401

    __all__.append("LadybugStore")
except Exception:  # pragma: no cover
    pass

"""Fixture: the module reached only via pkg/scripts/entry.py's sys.path
insert — a flat sibling import with no __init__.py anywhere in sight."""


def helper() -> str:
    return "ok"

"""Fixture: mirrors BUG D's producer half — mcp/server.py:261's
`LadybugStore(str(MEMORY_DIR / "graph.ladybug"))`. `SomeStore(...)` is
matched by the Store-ctor suffix rule; the PATHEXPR's tail literal is
`graph.ladybug`. `write_text` on a composed path is the other produce
shape this fixture exercises."""
from pathlib import Path

MEMORY_DIR = Path.home() / ".hermes"


class SomeStore:
    def __init__(self, path: str) -> None:
        self.path = path


def open_store():
    return SomeStore(str(MEMORY_DIR / "graph.ladybug"))


def write_note(text: str) -> None:
    (MEMORY_DIR / "notes.jsonl").write_text(text)

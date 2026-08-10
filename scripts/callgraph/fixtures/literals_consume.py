"""Fixture: mirrors BUG D's consumer half — scripts/graph_facts.py:33's
`glob.glob(os.path.expanduser("~/.hermes/**/graph.kuzu"), recursive=True)`.
`graph.kuzu` (tail literal after `**` -> `*` normalization: `graph.kuzu`)
shares NOTHING textually with `graph.ladybug` in literals_produce.py except
the `graph` stem — a --near-miss lead, not a match. `open(path)` (default
mode) is the other consume shape this fixture exercises, deliberately using
a DIFFERENT literal (`notes.jsonl`, matching literals_produce.py's
`write_text` producer exactly) so the base literal_table view has one
genuinely matched pair to assert on."""
import glob
import os

from literals_produce import MEMORY_DIR


def find_databases():
    return glob.glob(os.path.expanduser("~/.hermes/**/graph.kuzu"), recursive=True)


def read_note():
    with open(MEMORY_DIR / "notes.jsonl") as f:
        return f.read()

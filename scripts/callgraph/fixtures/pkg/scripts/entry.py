"""Fixture: the "10 scripts prepend ../mcp" pattern, so resolve.py's
constant-folding of sys.path.insert(0, os.path.join(dirname(abspath(
__file__)), "..", "mcp")) can be tested against something other than the
live, editable repo."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))

import sibling  # noqa: E402  (import after the sys.path.insert, on purpose)

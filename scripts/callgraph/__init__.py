"""callgraph — a dependency-light, stdlib-only static call-graph tool for the
loci corpus (mcp/, scripts/, a2a_server/, mlops/, eval/).

No LadybugDB, no network, no third-party imports. Designed to run during
debugging when nothing else in the stack is up, and to read a specific git
revision (``--rev``) so it stays correct while another workflow is mid-edit
in a file it needs to look at.

See scripts/callgraph/README.md for the tour and scripts/callgraph/docs/ for
the design notes and known limitations.
"""

__version__ = "0.1.0"

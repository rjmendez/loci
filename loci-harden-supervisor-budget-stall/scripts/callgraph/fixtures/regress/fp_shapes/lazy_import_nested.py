"""FP shape: a FUNCTION-LOCAL import used from inside a NESTED scope.

Real source: mcp/server.py::loci_health does a function-local
`import backends`, then calls it from inside a lambda:
    ("qdrant_reachable", lambda: backends.qdrant()[0])
Python closures make the binding visible to the lambda; keyed on the
lambda's own qualname it is invisible, and mcp/backends.py::qdrant was
reported dead despite a live call site.
"""


def loci_health():
    import lazy_import_target
    probes = [lambda: lazy_import_target.helper(0.5)]
    return [p() for p in probes]


# Stands in for the @mcp.tool() registration that makes this entry point live
# in the real code: without a root, EVERY function in the fixture is trivially
# unreachable and the test would pass for the wrong reason.
if __name__ == "__main__":
    loci_health()

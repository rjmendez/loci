"""FP shape: a function called ONLY from inside a lambda body.

Real source: mcp/server.py --
    ("qdrant_reachable", lambda: backends.qdrant()[0]),
A lambda is consumed at the point it is written, so its body is reachable
exactly when the scope constructing it is.
"""


def probe_via_lambda(timeout):
    return timeout > 0


def loci_health():
    out = {}
    for key, resolver in (
        ("qdrant_reachable", lambda: probe_via_lambda(0.5)),
    ):
        out[key] = resolver()
    return out


# Stands in for the @mcp.tool() registration that makes this entry point live
# in the real code: without a root, EVERY function in the fixture is trivially
# unreachable and the test would pass for the wrong reason.
if __name__ == "__main__":
    loci_health()

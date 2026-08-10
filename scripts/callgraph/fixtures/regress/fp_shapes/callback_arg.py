"""FP shape: a function passed by BARE NAME as a call argument.

Real source: mcp/server.py's health block --
    checks.append(_health_check("embeddings_sparse", _health_probe_embeddings_sparse))
Nine `_health_probe_*` functions were reported dead this way.
"""


def probe_sparse():
    return {"ok": True}


def run_check(label, fn):
    return label, fn()


def loci_health():
    checks = []
    checks.append(run_check("embeddings_sparse", probe_sparse))
    return checks


# Stands in for the @mcp.tool() registration that makes this entry point live
# in the real code: without a root, EVERY function in the fixture is trivially
# unreachable and the test would pass for the wrong reason.
if __name__ == "__main__":
    loci_health()

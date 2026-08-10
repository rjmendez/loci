"""FP shape: relative from-import of a SIBLING SUBMODULE.

Real source: mcp/graph/analytics.py --
    from . import queries as Q
    ...
    for s in Q.finding_symbols(ks, finding_id):
For a relative import the display module is just ".", so the dotted-name
fallback probed "..queries" and "queries" and never the package-qualified
name; Q aliased an external sink instead of the MODULE and four live
queries.py functions were reported dead.
"""
from . import queries as Q


def finding_report(ks, finding_id):
    return [s for s in Q.finding_symbols(ks, finding_id)]


if __name__ == "__main__":
    finding_report(None, None)

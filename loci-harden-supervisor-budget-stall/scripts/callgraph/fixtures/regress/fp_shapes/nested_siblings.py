"""FP shape: a nested def called by bare name from a SIBLING nested def.

Real source: mcp/graph/code_parse.py::parse_source defines
`enclosing_def_node` and `_in_method` side by side, and `_in_method` calls
`enclosing_def_node(node)`. Python resolves a bare name
local -> ENCLOSING -> global, so resolution must walk outward.
"""


def outer(nodes):

    def sibling_a(node):
        return getattr(node, "parent", None)

    def sibling_b(node):
        return sibling_a(node) is not None

    return [n for n in nodes if sibling_b(n)]


if __name__ == "__main__":
    outer([])

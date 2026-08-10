"""Fixture: a function-local ("lazy") import, plus a module-level one, so
IMPORTS edges can be asserted to carry the right `scope` and
`enclosing_fn`."""
import json  # module-level


def resolve_vllm():
    import numpy  # function-local: only live inside this function's scope
    return numpy.array([1, 2, 3])


class Widget:
    def render(self):
        import textwrap  # function-local, inside a method
        return textwrap.dedent(json.dumps({"ok": True}))

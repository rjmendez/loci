"""Fixture: nesting shapes for qualname construction — a plain function, a
method, a function nested inside a function, and a lambda nested inside a
function (the `register.<locals>.<lambda>@N`-shaped case)."""


def outer():
    def inner():
        return 1
    return inner


def register(mcp, callback):
    mcp.tool()(callback)


register(None, lambda: 42)


class Widget:
    def method(self):
        return self

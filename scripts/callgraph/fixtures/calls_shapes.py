"""Fixture: one of every CALLS resolution rung this slice implements
(name-def-local, nested-def-local, name-constructor, self-attribute,
module-attribute in-corpus, module-attribute-external via a bare stdlib
import, builtin, function-local import), plus the two "not yet resolvable"
shapes (param-call, attribute-unknown-receiver) that must land on `?`."""
import json

import calls_target


def helper(x):
    return x + 1


def caller_name_def_local():
    return helper(1)


def outer_with_nested():
    def helper_nested():
        return 1
    return helper_nested()


class Widget:
    def __init__(self, n):
        self.n = n

    def bump(self):
        return self.other()

    def other(self):
        return self.n


def make_widget():
    return Widget(3)


def caller_module_attribute_external():
    return json.dumps({"a": 1})


def caller_module_attribute_corpus():
    return calls_target.target_fn(2)


def caller_builtin():
    return len([1, 2, 3])


def caller_local_import():
    import textwrap
    return textwrap.dedent("  x")


def caller_param_call(callback):
    return callback()


def caller_unknown_attribute(obj):
    return obj.mystery_method()

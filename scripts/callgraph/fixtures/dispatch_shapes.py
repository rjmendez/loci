"""Fixture: rung 6 (unique-method-name) and the getattr-literal rung for
extract/dispatch.py — a call whose receiver's type this tool cannot know
(so rungs 1-3 park it on `?` with reason "attribute-unknown-receiver"), one
case where exactly one corpus-wide method answers to the call's attribute
name and one where two do, plus getattr(<module>, "<literal>") vs
getattr(<module>, <variable>)."""
import dispatch_target


class OnlyOwner:
    def unique_op(self):
        return 1


class OwnerA:
    def shared_op(self):
        return "a"


class OwnerB:
    def shared_op(self):
        return "b"


def call_unique_method(obj):
    return obj.unique_op()


def call_ambiguous_method(obj):
    return obj.shared_op()


def call_getattr_literal():
    return getattr(dispatch_target, "target_fn")(1)


def call_getattr_variable(name):
    return getattr(dispatch_target, name)(1)

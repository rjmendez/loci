"""Fixture: mirrors BUG B's shape (mcp/grounding.py's `ground()`) in
miniature — a constant-initialized local, reassigned only inside guarded
branches, escaping as a dict value, with several guarded early exits that
skip it entirely. Also exercises the other escape forms (direct-return,
tuple-element, closure) and the non-escaping-local negative case."""


def assemble(items, use_extra):
    ok = False              # init: top-level-of-body, constant
    parts = []
    for item in items:
        if not item:
            break            # guard exit: skips `ok`
        try:
            parts.append(str(item))
        except Exception:
            continue          # guard exit: skips `ok`

    if use_extra and parts:
        ok = True             # guarded assignment #1
    elif use_extra:
        ok = True             # guarded assignment #2 (same branch family, elif)
    else:
        for p in parts:
            if not p:
                continue       # guard exit: skips `ok`

    return {"parts": parts, "ok": ok}


def direct_return_case(x):
    result = x * 2
    return result


def tuple_return_case(a, b):
    total = a + b
    return (a, total, b)


def make_adder(n):
    base = n
    step = 1

    def add(x):
        return x + base + step

    return add


def no_escape_case():
    unused_local = compute_something()
    return "constant, never returned"


def compute_something():
    return 42

"""COUNTER-TEST: the false-positive fixes must not make everything reachable.

`never_mentioned_anywhere` appears nowhere but its own def, and must still be
reported. If this stops failing, `cg dead` has become a query that never says
anything.
"""


def used():
    return 1


def passed_around():
    return 3


def never_mentioned_anywhere():
    return 2


def take(fn):
    return fn


def main():
    take(passed_around)
    return used()


if __name__ == "__main__":
    main()

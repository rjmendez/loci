"""Fixture: REG-FN's RE-ENTRANT registration side — a second call to
register() with a different function reference for `get_thing`, widening
the injected value into a set (`register()` called twice — tests do this in
the real corpus too). Kept in its own file, separate from
registry_injection_caller.py, so the single-value INJECTS tests
(test_extract_registry.py) stay exactly single-value; only the multi-value
DISPATCHES test loads this file."""
import registry_injection


def other_thing():
    return 99


registry_injection.register(other_thing, {"helper_a": other_thing})

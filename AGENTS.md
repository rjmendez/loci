# AGENTS.md

Instructions for coding agents working in this repository. Behavioural rules for
agents *using* Loci are in `rules/` and `docs/onboarding/for-agents.md`; this file is
about changing the code.

## Running tests

Tests are hermetic: every `conftest.py` loads `testsupport/loci_hermetic.py`, which
points Qdrant, Ollama and vLLM at `127.0.0.1:1` and refuses access to `~/.loci`,
`~/.hermes` and the live collections. Run the suite for the area you changed, e.g.
`python -m pytest mcp/tests/test_x.py -q`. Never point a test at a live store.

## Test-writing rules for agents

An audit of this suite (2026-09, commit 3a1ad78) mutated the code under ~330 suspect
tests: 180 stayed green with the behaviour they name removed, and ~70 asserted a
known bug as the expected result, so fixing the bug turned them red. These rules
exist because of that. They apply to every test you add or edit.

1. **Assert the exact value the claim names.** Not "key present", not a range, not
   `>= 0`, not `in (a, b)`. If the right answer is known, pin it.
2. **Never stub, re-implement or source-grep the unit under test.** Stub only its
   dependencies. A test that reads the source for a string, or copies the logic into
   the test, proves nothing about behaviour.
3. **Every negative test has a positive twin on the same fixture.** Otherwise "always
   deny" / "always empty" / "always skip" passes both.
4. **A stub that must not be called records its calls; assert the list is empty.**
   Never raise from a stub into code with `except Exception` -- the fail-open path
   swallows your `AssertionError` and the test passes.
5. **Error, degraded or fail-open output is not a pass.** If the hermetic env makes
   the success path unreachable, add a fake for the dependency (`_get_qdrant`, the
   verdict backend, the RAG layer, `_lazy_generate`, ...) so the success branch runs
   and its values are asserted. Test the degraded branch separately and assert it is
   *reported* as degraded.
6. **`pytest.raises(SpecificError, match=...)`.** Bare `raises(Exception)`, or a
   broad built-in (`ValueError`, `TypeError`, `KeyError`, `RuntimeError`,
   `OSError`, `AttributeError`, `IndexError`, ...) without a real `match=`, is
   rejected by CI: an unrelated error on the way in satisfies it. A pattern that
   matches anything (`match=""`, `".*"`) does not count. Same for `assertRaises`
   -- use `assertRaisesRegex` with a real pattern.
7. **The fixture must be able to tell right from wrong:** distinct embeddings,
   learnable labels, sizes past the caps, boundary values, duplicates where dedup
   matters, and winners that are not also first in input or alphabetical order.
8. **Never pin a known bug as expected.** Write the correct assertion; if the fix is
   genuinely out of scope, mark it `xfail(strict=True, reason="<follow-up>")`.
   Non-strict xfail is rejected by CI.
9. **Do not weaken a test in the commit that changes the code it tests** -- no
   editing an expected value, `==` to `>=`, dropping an assert, widening a skip --
   unless the commit message says why in a `Test-weakened-because:` trailer. CI
   flags such commits.
10. **No state leaks.** `monkeypatch` for module globals and env, `tmp_path` for every
    file, and restore any `importlib.reload` outside the env patch.
11. **Races need a deterministic interleave** (`threading.Event`/`Barrier`), not a
    sleep. Timing budgets must not scale with the thing being measured.
12. **An always-skipped test is no test.** A skip may depend on the platform or an
    installed tool, or on one of the live-smoke opt-ins below -- and then a small
    checked-in fixture must cover the same path in every run.
13. **Prove it before claiming coverage.** Apply the obvious mutation to the code
    (delete the guard, return the constant, invert the check), confirm the test goes
    red, revert. Say which mutation you ran in the commit or PR.
14. **No test without an assertion; no `assert True`; no `assert x or True`; no
    `try/except` that swallows an `AssertionError`.**

### Enforcement

`scripts/check_test_honesty.py` (CI `lint` job) rejects rules 6, 8 and 14 and
env-gated skips mechanically:

    python3 scripts/check_test_honesty.py --fail-stale   # whole repo, as CI runs it
    python3 scripts/check_test_honesty.py path.py        # one file

Pre-existing violations are listed, with reasons, in
`scripts/test_honesty_allowlist.toml`. That list only shrinks: when you fix a listed
test, delete its entry in the same commit (CI fails on a stale entry). Do not add an
entry to get a new test through -- fix the test.

`scripts/check_test_weakening.py` (CI, pull requests, warning only) flags a commit
that changes source and, in the same commit, removes assertions or adds violations
in a test file, unless the message carries `Test-weakened-because:`.

### Live-smoke opt-ins

The only environment variables a skip condition may read:

| Variable | Meaning |
|---|---|
| `LOCI_TESTS_LIVE=1` | Disables hermetic isolation for a deliberate live smoke run against the real stores. Never set in CI. |

Adding one means adding it here *and* to `[live_smoke_opt_ins]` in the allowlist.

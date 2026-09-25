# CLAUDE.md

Read `AGENTS.md` first; it applies to Claude Code too. In particular, follow its
**Test-writing rules for agents** for every test you add or edit: exact values,
never stub or grep the unit under test, a positive twin for every negative test,
recording stubs instead of raising ones, no error/degraded output accepted as a pass,
`pytest.raises(SpecificError, match=...)`, `xfail(strict=True)` instead of pinning a
bug, and a mutation you ran to prove the test goes red.

CI enforces the mechanical part with `python3 scripts/check_test_honesty.py`;
its allowlist (`scripts/test_honesty_allowlist.toml`) only shrinks.

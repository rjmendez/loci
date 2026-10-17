# Vendored: LLM code-hallucination checker

This directory contains a pinned, vendored copy of the upstream LLM
code-hallucination checker. It is **not** a submodule or a pip dependency —
the single checker file is copied in verbatim and pinned to a commit.

**Do not edit `llm_hallucination_checks.py` here.** Edit upstream, push, then
refresh the pin with:

```bash
bash scripts/sync_hallucination_rules.sh
```

| Field | Value |
|-------|-------|
| Source repo | https://github.com/example/llm-code-hallucination-patterns |
| Source file | `rules/ruff_plugin/llm_hallucination_checks.py` |
| Commit SHA | `b9c6b44b092fc3669dd7197d46c051465f59e193` |
| License | MIT (Copyright (c) 2026 contributors) |
| Vendored on | 2026-06-03 |

## Rules

| Code | Pattern | Description |
|------|---------|-------------|
| LH001 | H1 — Private attribute fabrication | Private attribute (`obj._attr`) access on an object that is not `self`; may break across library versions. |
| LH003 | H3 — Missing import | Name used in a call has no import (documented upstream; supplements `F821`). |
| LH007 | H7 — Vacuous test | Bare comparison with no `assert` inside a `test_*` function; the comparison tests nothing. |
| LH009 | H9 — asyncio.run in async | `asyncio.run()` called inside an `async def`; raises `RuntimeError: This event loop is already running`. |

`LH000` is emitted for a `SyntaxError` (the file could not be parsed).

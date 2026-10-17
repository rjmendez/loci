# Quality and Verification Rules
# Distilled from: quality-verification cluster (18 skills)
# Generated: 2026-06-13

## Investigate Before Acting

Before editing any file, creating any artifact, or executing any destructive
command, gather and present concrete facts: which files will be affected, what
the current state is, and what the user actually asked. Demanding facts is not
the same as self-evaluation — self-evaluation ("am I sure?") is experimentally
ineffective; forced investigation changes the output. Minimum required facts
depend on context but must be grounded in tool-retrieved evidence, never
assumption.
See skill: gateguard, agent-introspection-debugging, benchmark-optimization-loop.

## Baseline Before Change

Establish a recorded baseline — metrics, screenshots, passing test results, or
response-time measurements — before making any change intended to improve
performance, quality, or reliability. Comparisons must be against the same input
shape and conditions; a "delta" with no prior baseline is unverifiable and must
be treated as unknown. Baseline artifacts should be committed or stored so the
full team shares them.
See skill: benchmark, benchmark-optimization-loop, canary-watch, browser-qa.

## Independent Verification

Never treat an AI agent's self-assessment that code is correct as equivalent to
an independent verification. The same model that writes a fix carries the same
assumptions when reviewing it, producing systematic blind spots. All quality
gates — tests, type checks, linters, evals — must be executed by an external,
deterministic tool and the raw pass/fail output must be reported verbatim, not
paraphrased.
See skill: ai-regression-testing, gateguard, skill-comply.

## No Silent Errors

Every catch or error-handling block must take an explicit action: handle the
error, rethrow it, or log it with full context. Silently discarding errors is a
bug. In agent workflows this extends to tool failures — a failed tool call must
be captured and reported rather than quietly retried in a loop. User-facing
messages may be friendly, but the full technical context must be logged.
See skill: error-handling, agent-introspection-debugging.

## No Hardcoded Secrets

Secrets — API keys, tokens, passwords, private URLs — must never appear as
literals in source files, configuration, or prompt content. Store secrets in
environment variables, secrets managers, or hosting-platform vaults. After
writing any code or config that references credentials, verify with a grep scan
that no literal secret was introduced. Production secrets must never be committed
to git history.
See skill: security-review, security-scan, verification-loop.

## Criteria First

Before implementing a feature, optimization, or agent task, define explicit
success and failure criteria: what passing looks like, what metrics must be met,
what must not regress. Criteria must be stated in terms that an automated grader
or deterministic test can evaluate. Defining criteria after the fact is a quality
anti-pattern that allows goal-post shifting.
See skill: eval-harness, benchmark-optimization-loop, agent-eval.

## Minimum Write Scope

Restrict write and destructive operations to the smallest directory or resource
scope that the task requires. For autonomous or long-running agents, enforce a
write boundary explicitly before execution begins. Destructive commands (rm -rf,
force-push, DROP TABLE, docker prune) must trigger confirmation or be blocked by
default. Isolation should be codified — not remembered — so it survives context
loss.
See skill: safety-guard, benchmark-optimization-loop, agent-eval.

## Structured Reports

Quality, audit, and verification outputs must use a defined structured format
with explicit PASS/FAIL states per check, counts, and any raw evidence (error
messages, metric values). Free-form prose is not a substitute for a structured
report. Reports must include a clear overall verdict that a downstream agent or
human can act on without re-reading the full output.
See skill: verification-loop, benchmark-optimization-loop, browser-qa, canary-watch.

## Validate External Input

All data that crosses a trust boundary — HTTP request bodies, query params, file
uploads, environment variables, inter-agent messages — must be validated against
an explicit schema before use. Validation must reject invalid input explicitly;
do not rely on downstream code to handle malformed data gracefully. Use typed
schemas (Zod, Pydantic, JSON Schema) rather than ad-hoc conditional checks.
See skill: security-review, security-bounty-hunter, error-handling.

## Stop On Failure

When a build, type-check, test suite, or correctness gate fails, stop and fix
before proceeding. Do not run later verification phases on a broken foundation —
their results will be misleading. Report the failure clearly with the raw output,
not a paraphrase. A variant, optimization, or deploy candidate that fails a
correctness gate must be rejected, not promoted with a note.
See skill: verification-loop, benchmark-optimization-loop, ai-regression-testing.

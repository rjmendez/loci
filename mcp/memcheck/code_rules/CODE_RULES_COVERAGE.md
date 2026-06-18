# Code-Rules Coverage Matrix

Honest coverage report for the LLM code-hallucination taxonomy
(https://github.com/example/llm-code-hallucination-patterns) as implemented by
loci's static code checks.

Two layers contribute:

- **Vendored** `llm_hallucination_checks.py` (pinned upstream ruff plugin) —
  covers H1/H3/H7/H9 via `LH001`/`LH003`/`LH007`/`LH009`. Not edited here.
- **Loci-owned** `extended_checks.py` — covers the rest of the taxonomy at the
  strongest fidelity static analysis of a single `.py` file allows.

`run_code_checks` (in `../checks/code_hallucination.py`) aggregates both,
dedupes on subject signature, and stamps each verdict's confidence + a
`tier:N`/`advisory` marker from `PATTERN_META`. Every verdict is advisory
(`decision="warn"`) — these never block.

## Taxonomy size note

The build brief referenced "52 patterns." The cloned source-of-truth repo
actually defines **58** pattern ids across its 16 `patterns/*.md` files
(the extra count is `MC1b` plus the full `AT1–AT6` agent-tooling series).
`PATTERN_META` lists **all 58** so nothing is silently dropped; the unit test
asserts the set matches the source repo exactly.

## Detection-method tally (58 patterns)

| Method | Count | Meaning |
|--------|------:|---------|
| `ast` | 17 | Structural AST check — highest-fidelity static signal (conf ~0.45–0.70) |
| `regex` | 5 | Source/regex heuristic mirroring the markdown's grep detections (conf ~0.40–0.45) |
| `ruff_core` | 3 | Already covered by ruff core (F821/B015) or the vendored LH plugin |
| `advisory` | 33 | Best-effort low-confidence heuristic **or** no static signal at all (conf 0.0) |

Of the 33 `advisory` rows, **8** carry a runnable low-confidence heuristic
(H5, SD2, SS1*, plus shared signals) and **the rest (conf 0.0)** are recorded
with a rationale for why a single-`.py`-file static check cannot see them
(runtime dynamics, deployment/infra config, cross-file/cross-service contracts,
non-`.py` formats, or live-agent behavior). `*` SS1 is recorded advisory/0.25
but no standalone check is wired — see its row.

## Per-pattern matrix

Status legend: `ast` / `regex` / `ruff_core` = a runnable check fires;
`advisory` with confidence > 0 = runnable low-confidence heuristic;
`advisory` with confidence 0.0 = **no static signal** (recorded, not checked).

| Pattern | Category | Tier | Detection | Confidence | Rationale |
|---------|----------|:----:|-----------|:----------:|-----------|
| H1 | Hallucination | 1 | ruff_core | 0.7 | Private/internal attr access on a non-self object. Covered by vendored LH001. |
| H2 | Hallucination | 1 | ast | 0.6 | Duplicate prometheus metric name registered with the same literal name in the same file. |
| H3 | Hallucination | 1 | ruff_core | 0.7 | Name used at call site but never imported/defined. Covered by ruff F821 + vendored LH003; no loci backstop (high false-positive risk without full scope analysis). |
| H4 | Hallucination | 2 | regex | 0.45 | Method chain 3+ deep `a().b().c()` where an early link may return None — flag for None-safety review. |
| H5 | Hallucination | 2 | advisory | 0.25 | Hardcoded float `assert x == <float>` in a test — value may be inferred, not observed. Heuristic. |
| H6 | Hallucination | 1 | ast | 0.7 | Frozen/deprecated API: `datetime.utcnow()`, `asyncio.get_event_loop()`, `pkg_resources`, `collections.Callable`. |
| H7 | Hallucination | 1 | ruff_core | 0.7 | Bare comparison statement in a test (vacuous test). Covered by vendored LH007 / ruff B015. |
| H8 | Hallucination | 3 | advisory | 0.0 | Dict-vs-list-vs-tuple shape mismatch at a call site. Needs cross-file signature resolution / type inference; no reliable single-file static signal. |
| H9 | Hallucination | 1 | ast | 0.7 | `async def __init__` (illegal async constructor). Complements vendored LH009. |
| SD1 | Schema Drift | 2 | regex | 0.45 | Positional integer index on a query-result row (`row[0]`, `values[2]`) in a test — column insertion shifts it silently. |
| SD2 | Schema Drift | 3 | advisory | 0.25 | Hardcoded column/table name in a raw SQL string — may drift from live schema. Heuristic on SELECT/INSERT literals. |
| SD3 | Schema Drift | 2 | ast | 0.55 | prometheus metric constructed WITH a labels list but called via bare `.inc()`/`.set()` (must go through `.labels(...)`). |
| SD4 | Schema Drift | 3 | advisory | 0.0 | YAML multi-doc missing `---` separator. Lives in `.yaml` files, not `.py` source; out of scope. |
| SS1 | Silent Swallowing | 2 | advisory | 0.25 | Wrong-type arg accepted, conservative default fires. Needs call-site vs signature type comparison; no standalone check wired. |
| SS2 | Silent Swallowing | 3 | advisory | 0.0 | Dedup/rate-limit gate absorbs identical test stimuli. Test-design interaction; no reliable single-file signal. |
| SS3 | Silent Swallowing | 1 | ast | 0.6 | Validation `raise` placed AFTER an early-return guard — unreachable on the guard path. |
| SS4 | Silent Swallowing | 3 | advisory | 0.0 | Metric `.set(0)` on an irrelevant else-branch (means "no data" but zeroes signal). Needs semantic intent. |
| SS5 | Silent Swallowing | 1 | ast | 0.6 | Per-record try/except in a loop wrapping a DB execute, swallowed (pass/continue) — breaks batch atomicity. |
| SS6 | Silent Swallowing | 3 | advisory | 0.0 | Two test files assert conflicting contracts for one function. Cross-file reconciliation; out of scope. |
| CP1 | Cascade Propagation | 3 | advisory | 0.0 | Compiler-surfaces-one-callsite cascade. A build/process workflow concern (non-Python); no source signal. |
| CP2 | Cascade Propagation | 3 | advisory | 0.0 | Wiring-patch secondary impact scan. A diff/process checklist spanning many files; not a single-file check. |
| SL1 | Scope/Lifetime | 2 | ast | 0.5 | Variable assigned only inside an `if` then used after it on a path where the `if` may not run — possible UnboundLocalError. |
| SL2 | Scope/Lifetime | 1 | ast | 0.65 | pytest fixture uses `return` inside a `with mock.patch(...)` block — patch tears down before the test body runs. |
| SB1 | Serialization Boundary | 2 | regex | 0.4 | `json.dumps(...)` in a module that also uses numpy — ndarray is not JSON serializable. Heuristic. |
| SB2 | Serialization Boundary | 1 | ast | 0.65 | `datetime.utcnow()` produces a naive datetime; mixing with tz-aware records breaks comparisons. (Shares signal with H6.) |
| AC1 | Async/Concurrency | 1 | ast | 0.7 | `asyncio.get_event_loop()` (deprecated 3.10+; wrong loop in executor threads). Use `get_running_loop()`. |
| AC2 | Async/Concurrency | 2 | ast | 0.45 | `while True` retry loop whose handler only sleeps and retries — unbounded retry, no circuit breaker. |
| AC3 | Async/Concurrency | 3 | advisory | 0.0 | No graceful shutdown drains in-flight queue. Absence-of-code pattern; not reliably detectable statically. |
| AC4 | Async/Concurrency | 2 | ast | 0.5 | `asyncio.create_task(...)` as a bare statement with no reference/done-callback — exception silently lost. |
| TEC1 | Test Contamination | 2 | regex | 0.4 | Test builds a prometheus metric / reads `._value` but has no autouse reset fixture — registry leaks across tests. |
| TEC2 | Test Contamination | 3 | advisory | 0.0 | `git stash` baseline-count is misleading. Workflow/process concern; nothing in `.py` source. |
| AT1 | Agent Tooling | 3 | advisory | 0.0 | Agent wrote a file without reading it first. Agent-runtime behavior; not in code. |
| AT2 | Agent Tooling | 3 | advisory | 0.0 | Parallel subagent write conflict. Orchestration-runtime behavior; not in code. |
| AT3 | Agent Tooling | 3 | advisory | 0.0 | `execute_code` side effect persists across turns. Session-state behavior; not in code. |
| AT4 | Agent Tooling | 3 | advisory | 0.0 | Stub-completion fallacy (reported done, never ran). Agent-reporting behavior; not in code. |
| AT5 | Agent Tooling | 3 | advisory | 0.0 | Tool-error confabulation. Agent-narration behavior; not in code. |
| AT6 | Agent Tooling | 3 | advisory | 0.0 | Compaction-horizon amnesia. Agent-memory behavior; not in code. |
| PB1 | Parse Boundary | 1 | ast | 0.55 | State mutation (`self.x =` / `+=` / db insert) BEFORE a validation `raise` — partial mutation persists on failure. |
| PB2 | Parse Boundary | 3 | advisory | 0.0 | Pydantic validation bypassed via dict round-trip with extra keys. Needs model + runtime-data knowledge. |
| PB3 | Parse Boundary | 3 | advisory | 0.0 | Optional field defaults to None across a network boundary. Cross-service contract; not single-file detectable. |
| PB4 | Parse Boundary | 3 | advisory | 0.0 | YAML float-vs-string ambiguity. Lives in `.yaml` config, not `.py` source; out of scope. |
| MF1 | Metastable Failure | 3 | advisory | 0.0 | Backpressure queue saturates under load. Runtime dynamics; no static signal. |
| MF2 | Metastable Failure | 2 | ast | 0.4 | Reconnect loop sleeps a constant with no jitter / exponential backoff — reconnect-storm risk. |
| MF3 | Metastable Failure | 2 | ast | 0.4 | `logger.warning/error` called unconditionally inside a `while True` loop — log-volume feedback loop risk. |
| DC1 | Distributed Consistency | 3 | advisory | 0.0 | Write-behind cache serves stale read. Cross-component runtime behavior; no single-file signal. |
| DC2 | Distributed Consistency | 3 | advisory | 0.0 | Split singleton: process-global registry diverges across workers. Deployment-topology behavior; not statically certain. |
| DC3 | Distributed Consistency | 3 | advisory | 0.0 | K8s ConfigMap reload lag. Infra behavior; not in `.py` source. |
| DC4 | Distributed Consistency | 3 | advisory | 0.0 | Lost write under MQTT retain burst. Broker runtime behavior; not in source. |
| CD1 | Configuration Drift | 3 | advisory | 0.0 | Env var staging/prod mismatch. Code is identical across envs; nothing to detect in source. |
| CD2 | Configuration Drift | 3 | advisory | 0.0 | `LOG_LEVEL=DEBUG` in prod I/O bottleneck. Deployment-config; no static signal. |
| CD3 | Configuration Drift | 3 | advisory | 0.0 | Container memory limit too low → OOMKill. Infra config; not in source. |
| OG1 | Observability Gap | 1 | ast | 0.6 | except handler whose body is only `pass` / `return None` with no logging — exception swallowed without a trace. |
| OG2 | Observability Gap | 3 | advisory | 0.0 | Metric defined but never incremented in the hot path. Needs whole-program reachability. |
| OG3 | Observability Gap | 3 | advisory | 0.0 | No correlation/trace ID across service boundaries. Absence-of-code across services; not statically certain. |
| OG4 | Observability Gap | 3 | advisory | 0.0 | Prometheus scrape interval longer than event duration. Lives in `prometheus.yml`, not `.py` source. |
| WG1 | Wiring Gap | 2 | ast | 0.45 | Class named `*Publisher/*Notifier/*Exporter/*Sender` whose send/publish method makes no outbound call — no-op backend. |
| MC1 | Metric Misconfiguration | 3 | advisory | 0.0 | Gauge sampled at wrong aggregation level (post-flush). Needs temporal/semantic intent; no static signal. |
| MC1b | Metric Misconfiguration | 2 | regex | 0.4 | Counter used for a decreasing quantity (name contains depth/size/active/count/usage…) — should be a Gauge. |

## Patterns deliberately left with no check (confidence 0.0) — and why

These have genuinely **no static signal in a single `.py` file** and are
recorded as `advisory`/0.0 rather than faked:

- **H8** — needs cross-file signature resolution / type inference.
- **SD4, PB4, OG4** — live in `.yaml` / `prometheus.yml`, not Python source.
- **SS2, SS4, SS6, OG2** — test-design / semantic-intent / whole-program
  reachability concerns; no reliable single-file signal.
- **CP1, CP2** — multi-file diff/process checklists (CP1 is also non-Python).
- **MF1, DC1–DC4, MC1** — runtime/topology/broker dynamics.
- **CD1–CD3, TEC2** — deployment/infra config and git-workflow concerns; the
  code is identical regardless of the misconfiguration.
- **PB2, PB3** — Pydantic/runtime-data and cross-service contract knowledge.
- **AT1–AT6** — live-agent behavior (read/write ordering, parallel writes,
  session state, confabulation, compaction); nothing in the generated code.
- **SS1** — recorded advisory/0.25; a reliable check needs call-site-vs-signature
  type comparison, so no standalone check is wired yet.

`PATTERN_META` is the machine-readable source of this table.

"""Extended LLM code-hallucination checks — loci-owned.

The vendored ``llm_hallucination_checks`` ruff plugin covers only H1, H3, H7,
H9 of the upstream taxonomy
(https://github.com/example/llm-code-hallucination-patterns). This module
extends coverage to the FULL taxonomy (H1–H9, SD, SS, CP, SL, SB, AC, TEC,
AT, PB, MF, DC, CD, OG, WG, MC) at the strongest fidelity that *static*
analysis of a single ``.py`` file allows.

Design rules (do not violate):

* **Loci-owned, not vendored.** All new logic lives here; the vendored
  checker is never edited. We reuse its :class:`Issue` dataclass for a uniform
  surface into :func:`memcheck.checks.run_code_checks`.
* **Advisory-only.** Every issue is a warning (the caller sets
  ``decision="warn"``); nothing here blocks.
* **Fail-safe per check.** Any check that errors on a file is skipped, never
  raised. The top-level :func:`run_extended_checks` swallows per-check errors.
* **stdlib only.** ``ast`` + ``re``. No new deps.
* **Honest confidence.** Each pattern records a confidence reflecting how
  reliably a *static* signal maps to the real failure mode:
    - clean AST structural check  ~0.70
    - regex / source heuristic    ~0.45
    - advisory grep-wrap of a Tier-3 / runtime-only pattern ~0.25
    - ruff_core (already covered by F821/B015/etc. upstream) — noted, optional
      light regex backstop
    - genuinely no static signal   0.00 (recorded in PATTERN_META, no check)

``PATTERN_META`` is the source of truth for the per-pattern status; the
coverage matrix in ``CODE_RULES_COVERAGE.md`` is generated from the same facts.
Every pattern id in the taxonomy appears in ``PATTERN_META`` even when there is
no runnable check (``detection="advisory"``, ``confidence=0.0``) so nothing is
silently dropped.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from .llm_hallucination_checks import Issue

__all__ = [
    "PATTERN_META",
    "run_extended_checks",
    "ALL_PATTERN_IDS",
]


# ---------------------------------------------------------------------------
# Pattern metadata. Every taxonomy id appears here exactly once.
#
# detection: "ast" | "regex" | "advisory" | "ruff_core"
#   ast        — structural AST check, highest fidelity static signal
#   regex      — source/regex heuristic (the markdown's grep detections)
#   advisory   — best-effort low-confidence heuristic, OR no static signal at
#                all (confidence 0.0) for runtime/log-only / process-level
#                patterns that cannot be seen in one .py file
#   ruff_core  — already covered by ruff core (F821, B015) or the vendored
#                LH plugin; we note it and may add a light backstop
#
# confidence: how reliably the static signal implies the real failure mode.
# ---------------------------------------------------------------------------
PATTERN_META: dict[str, dict] = {
    # --- H — Hallucination -------------------------------------------------
    "H1": {
        "category": "Hallucination",
        "tier": 1,
        "detection": "ruff_core",
        "confidence": 0.7,
        "summary": "Private/internal attr access on a non-self object. Covered "
                   "by vendored LH001; backstopped here.",
    },
    "H2": {
        "category": "Hallucination",
        "tier": 1,
        "detection": "ast",
        "confidence": 0.6,
        "summary": "Duplicate prometheus metric name registered with the same "
                   "literal name in the same file.",
    },
    "H3": {
        "category": "Hallucination",
        "tier": 1,
        "detection": "ruff_core",
        "confidence": 0.7,
        "summary": "Name used at call site but never imported/defined. Covered "
                   "by ruff F821 and vendored LH003; no loci backstop (high"
                   "false-positive risk without full scope analysis).",
    },
    "H4": {
        "category": "Hallucination",
        "tier": 2,
        "detection": "regex",
        "confidence": 0.45,
        "summary": "Method chain 3+ deep (a().b().c()) where an early link may "
                   "return None — flag for None-safety review.",
    },
    "H5": {
        "category": "Hallucination",
        "tier": 2,
        "detection": "advisory",
        "confidence": 0.25,
        "summary": "Hardcoded float assertion (assert x == <float literal>) in a "
                   "test — value may be inferred not observed. Heuristic.",
    },
    "H6": {
        "category": "Hallucination",
        "tier": 1,
        "detection": "ast",
        "confidence": 0.7,
        "summary": "Frozen/deprecated API: datetime.utcnow(), "
                   "asyncio.get_event_loop(), pkg_resources, collections.Callable.",
    },
    "H7": {
        "category": "Hallucination",
        "tier": 1,
        "detection": "ruff_core",
        "confidence": 0.7,
        "summary": "Bare comparison statement in a test (vacuous test). Covered "
                   "by vendored LH007 / ruff B015; backstopped here.",
    },
    "H8": {
        "category": "Hallucination",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Dict-vs-list-vs-tuple shape mismatch at a call site. Needs "
                   "cross-file signature resolution / type inference; no reliable "
                   "single-file static signal.",
    },
    "H9": {
        "category": "Hallucination",
        "tier": 1,
        "detection": "ast",
        "confidence": 0.7,
        "summary": "async def __init__ (illegal async constructor). Complements "
                   "vendored LH009 (asyncio.run in async).",
    },
    # --- SD — Schema Drift --------------------------------------------------
    "SD1": {
        "category": "Schema Drift",
        "tier": 2,
        "detection": "regex",
        "confidence": 0.45,
        "summary": "Positional integer index on a query result row (row[0], "
                   "values[2]) in a test — column insertion shifts it silently.",
    },
    "SD2": {
        "category": "Schema Drift",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.25,
        "summary": "Hardcoded column/table name in a raw SQL string — may drift "
                   "from live schema. Heuristic flag on SELECT/INSERT literals.",
    },
    "SD3": {
        "category": "Schema Drift",
        "tier": 2,
        "detection": "ast",
        "confidence": 0.55,
        "summary": "prometheus metric constructed WITH a labels list but called "
                   "with bare .inc()/.set() (must go through .labels(...)).",
    },
    "SD4": {
        "category": "Schema Drift",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "YAML multi-doc missing --- separator. Lives in .yaml files, "
                   "not .py source; out of scope for a Python AST/regex checker.",
    },
    # --- SS — Silent Swallowing --------------------------------------------
    "SS1": {
        "category": "Silent Swallowing",
        "tier": 2,
        "detection": "advisory",
        "confidence": 0.25,
        "summary": "Wrong-type arg accepted, conservative default fires. Needs "
                   "call-site vs signature type comparison; heuristic only.",
    },
    "SS2": {
        "category": "Silent Swallowing",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Dedup/rate-limit gate absorbs identical test stimuli. A "
                   "test-design interaction; no reliable static signal in one file.",
    },
    "SS3": {
        "category": "Silent Swallowing",
        "tier": 1,
        "detection": "ast",
        "confidence": 0.6,
        "summary": "raise (validation) placed AFTER an early-return guard in the "
                   "same function — validation is unreachable on the guard path.",
    },
    "SS4": {
        "category": "Silent Swallowing",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Metric .set(0) on an irrelevant else-branch (means 'no data' "
                   "but zeroes signal). Needs semantic intent; no static signal.",
    },
    "SS5": {
        "category": "Silent Swallowing",
        "tier": 1,
        "detection": "ast",
        "confidence": 0.6,
        "summary": "Per-record try/except inside a loop that wraps a DB execute "
                   "and swallows (pass/continue) — breaks batch atomicity.",
    },
    "SS6": {
        "category": "Silent Swallowing",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Two test files assert conflicting contracts for one function. "
                   "Cross-file reconciliation; out of scope for single-file scan.",
    },
    # --- CP — Cascade Propagation ------------------------------------------
    "CP1": {
        "category": "Cascade Propagation",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Compiler-surfaces-one-callsite cascade. A build/process "
                   "workflow concern (and not Python); no source-level signal.",
    },
    "CP2": {
        "category": "Cascade Propagation",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Wiring-patch secondary impact scan. A diff/process checklist "
                   "spanning many files; not a single-file static check.",
    },
    # --- SL — Scope / Lifetime ---------------------------------------------
    "SL1": {
        "category": "Scope/Lifetime",
        "tier": 2,
        "detection": "ast",
        "confidence": 0.5,
        "summary": "Variable assigned only inside an if-block then used after it "
                   "on a path where the if may not run — possible UnboundLocalError.",
    },
    "SL2": {
        "category": "Scope/Lifetime",
        "tier": 1,
        "detection": "ast",
        "confidence": 0.65,
        "summary": "pytest fixture uses `return` inside a `with mock.patch(...)` "
                   "block — the patch tears down before the test body runs.",
    },
    # --- SB — Serialization Boundary ---------------------------------------
    "SB1": {
        "category": "Serialization Boundary",
        "tier": 2,
        "detection": "regex",
        "confidence": 0.4,
        "summary": "json.dumps(...) in a scope that also uses np.array/ndarray — "
                   "numpy arrays are not JSON serializable. Heuristic.",
    },
    "SB2": {
        "category": "Serialization Boundary",
        "tier": 1,
        "detection": "ast",
        "confidence": 0.65,
        "summary": "datetime.utcnow() produces a naive datetime; mixing with "
                   "tz-aware records breaks comparisons. (Shares signal with H6.)",
    },
    # --- AC — Async / Concurrency ------------------------------------------
    "AC1": {
        "category": "Async/Concurrency",
        "tier": 1,
        "detection": "ast",
        "confidence": 0.7,
        "summary": "asyncio.get_event_loop() (deprecated 3.10+; wrong loop in "
                   "executor threads). Use get_running_loop().",
    },
    "AC2": {
        "category": "Async/Concurrency",
        "tier": 2,
        "detection": "ast",
        "confidence": 0.45,
        "summary": "`while True` retry loop whose only body is try/except + "
                   "asyncio.sleep — unbounded retry, no circuit breaker.",
    },
    "AC3": {
        "category": "Async/Concurrency",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "No graceful shutdown drains in-flight queue. Absence-of-code "
                   "pattern; not reliably detectable statically.",
    },
    "AC4": {
        "category": "Async/Concurrency",
        "tier": 2,
        "detection": "ast",
        "confidence": 0.5,
        "summary": "asyncio.create_task(...) used as a bare statement with no "
                   "reference kept and no done-callback — exception silently lost.",
    },
    # --- TEC — Test Environment Contamination ------------------------------
    "TEC1": {
        "category": "Test Contamination",
        "tier": 2,
        "detection": "regex",
        "confidence": 0.4,
        "summary": "Test file constructs a prometheus metric / reads ._value but "
                   "has no autouse reset fixture — registry state leaks across tests.",
    },
    "TEC2": {
        "category": "Test Contamination",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "git stash baseline-count is misleading. A workflow/process "
                   "concern; nothing in .py source to detect.",
    },
    # --- AT — Agent Tooling ------------------------------------------------
    # All AT patterns describe live-agent behavior (read/write ordering,
    # parallel writes, session state, confabulation, compaction). None are
    # detectable by static analysis of a generated .py file.
    "AT1": {
        "category": "Agent Tooling",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Agent wrote a file without reading it first. Agent-runtime "
                   "behavior; not in code.",
    },
    "AT2": {
        "category": "Agent Tooling",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Parallel subagent write conflict. Orchestration-runtime "
                   "behavior; not in code.",
    },
    "AT3": {
        "category": "Agent Tooling",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "execute_code side effect persists across turns. Session-state "
                   "behavior; not in code.",
    },
    "AT4": {
        "category": "Agent Tooling",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Stub-completion fallacy (reported done, never ran). "
                   "Agent-reporting behavior; not in code.",
    },
    "AT5": {
        "category": "Agent Tooling",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Tool-error confabulation. Agent-narration behavior; not in "
                   "code.",
    },
    "AT6": {
        "category": "Agent Tooling",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Compaction-horizon amnesia. Agent-memory behavior; not in "
                   "code.",
    },
    # --- PB — Parse Boundary -----------------------------------------------
    "PB1": {
        "category": "Parse Boundary",
        "tier": 1,
        "detection": "ast",
        "confidence": 0.55,
        "summary": "State mutation (self.x = / self.x += / db insert) BEFORE a "
                   "validation `raise` in the same function — partial mutation "
                   "persists on validation failure.",
    },
    "PB2": {
        "category": "Parse Boundary",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Pydantic validation bypassed via dict round-trip with extra "
                   "keys. Needs model + runtime-data knowledge; no static signal.",
    },
    "PB3": {
        "category": "Parse Boundary",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Optional field defaults to None across a network boundary. "
                   "Cross-service contract; not single-file detectable.",
    },
    "PB4": {
        "category": "Parse Boundary",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "YAML float-vs-string ambiguity. Lives in .yaml config, not "
                   ".py source; out of scope.",
    },
    # --- MF — Metastable Failure -------------------------------------------
    "MF1": {
        "category": "Metastable Failure",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Backpressure queue saturates under load. Runtime dynamics; "
                   "no static signal.",
    },
    "MF2": {
        "category": "Metastable Failure",
        "tier": 2,
        "detection": "ast",
        "confidence": 0.4,
        "summary": "Reconnect loop sleeps a constant (no jitter / no exponential "
                   "backoff) — reconnect storm risk. Heuristic on sleep arg.",
    },
    "MF3": {
        "category": "Metastable Failure",
        "tier": 2,
        "detection": "ast",
        "confidence": 0.4,
        "summary": "logger.warning/error called unconditionally inside a "
                   "`while True` loop — log-volume feedback loop risk.",
    },
    # --- DC — Distributed Consistency --------------------------------------
    "DC1": {
        "category": "Distributed Consistency",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Write-behind cache serves stale read. Cross-component "
                   "runtime behavior; no single-file static signal.",
    },
    "DC2": {
        "category": "Distributed Consistency",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Split singleton: process-global registry diverges across "
                   "workers. Deployment-topology behavior; not statically certain.",
    },
    "DC3": {
        "category": "Distributed Consistency",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "K8s ConfigMap reload lag. Infra behavior; not in .py source.",
    },
    "DC4": {
        "category": "Distributed Consistency",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Lost write under MQTT retain burst. Broker runtime behavior; "
                   "not in source.",
    },
    # --- CD — Configuration Drift ------------------------------------------
    "CD1": {
        "category": "Configuration Drift",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Env var staging/prod mismatch. Deployment-config concern; "
                   "code is identical across envs, nothing to detect in source.",
    },
    "CD2": {
        "category": "Configuration Drift",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "LOG_LEVEL=DEBUG in prod I/O bottleneck. Deployment-config; "
                   "no static signal.",
    },
    "CD3": {
        "category": "Configuration Drift",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Container memory limit too low → OOMKill. Infra config; not "
                   "in source.",
    },
    # --- OG — Observability Gap --------------------------------------------
    "OG1": {
        "category": "Observability Gap",
        "tier": 1,
        "detection": "ast",
        "confidence": 0.6,
        "summary": "except handler whose body is only `pass` or `return None` "
                   "with no logging — exception swallowed without a trace.",
    },
    "OG2": {
        "category": "Observability Gap",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Metric defined but never incremented in the hot path. Needs "
                   "whole-program reachability; no reliable single-file signal.",
    },
    "OG3": {
        "category": "Observability Gap",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "No correlation/trace ID across service boundaries. "
                   "Absence-of-code across services; not statically certain.",
    },
    "OG4": {
        "category": "Observability Gap",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Prometheus scrape interval longer than event duration. Lives "
                   "in prometheus.yml, not .py source.",
    },
    # --- WG — Wiring Gap ----------------------------------------------------
    "WG1": {
        "category": "Wiring Gap",
        "tier": 2,
        "detection": "ast",
        "confidence": 0.45,
        "summary": "Class named *Publisher/*Notifier/*Exporter/*Sender whose "
                   "send/publish method makes no outbound call (requests/httpx/"
                   "socket) — named-for-integration but no-op backend.",
    },
    # --- MC — Metric Misconfiguration --------------------------------------
    "MC1": {
        "category": "Metric Misconfiguration",
        "tier": 3,
        "detection": "advisory",
        "confidence": 0.0,
        "summary": "Gauge sampled at wrong aggregation level (post-flush). Needs "
                   "temporal/semantic intent; no static signal.",
    },
    "MC1b": {
        "category": "Metric Misconfiguration",
        "tier": 2,
        "detection": "regex",
        "confidence": 0.4,
        "summary": "Counter used for a decreasing quantity (name contains "
                   "depth/size/active/count/usage) — should be a Gauge. Heuristic "
                   "on the metric name string.",
    },
}

# Convenience: the canonical full set of taxonomy ids.
ALL_PATTERN_IDS: tuple[str, ...] = tuple(PATTERN_META.keys())


# ---------------------------------------------------------------------------
# Small AST/source helpers
# ---------------------------------------------------------------------------
_METRIC_CTORS = {"Counter", "Gauge", "Histogram", "Summary"}


def _issue(code: str, node: ast.AST, message: str) -> Issue:
    """Build an Issue from an AST node, defaulting line/col safely."""
    return Issue(
        code=code,
        line=getattr(node, "lineno", 0) or 0,
        col=getattr(node, "col_offset", 0) or 0,
        message=message,
    )


def _src_issue(code: str, lineno: int, col: int, message: str) -> Issue:
    return Issue(code=code, line=lineno, col=col, message=message)


def _call_dotted_name(call: ast.Call) -> str | None:
    """Return a dotted name for a call's func, e.g. 'asyncio.get_event_loop'."""
    func = call.func
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
        return ".".join(reversed(parts))
    return None


def _is_test_file(filename: str) -> bool:
    return "test" in filename


# ---------------------------------------------------------------------------
# H2 — Duplicate prometheus metric registration (same literal name in file)
# ---------------------------------------------------------------------------
def check_h2_duplicate_metric(tree: ast.AST) -> list[Issue]:
    seen: dict[str, ast.Call] = {}
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        ctor = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else None
        )
        if ctor not in _METRIC_CTORS:
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        name = first.value
        if name in seen:
            issues.append(_issue(
                "H2", node,
                f"H2 Duplicate prometheus metric name '{name}' registered twice "
                "in this file. Re-registration raises 'Duplicated timeseries in "
                "CollectorRegistry' at import; import and reuse the existing object.",
            ))
        else:
            seen[name] = node
    return issues


# ---------------------------------------------------------------------------
# H6 / SB2 / AC1 — Frozen / deprecated API usage (AST)
# ---------------------------------------------------------------------------
def check_h6_deprecated_api(tree: ast.AST) -> list[Issue]:
    """datetime.utcnow(), asyncio.get_event_loop(), pkg_resources,
    collections.Callable. AC1 and SB2 share signal with two of these and are
    emitted under their own pattern ids for clean per-pattern attribution."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        # pkg_resources import
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pkg_resources" or alias.name.startswith("pkg_resources."):
                    issues.append(_issue(
                        "H6", node,
                        "H6 'pkg_resources' is deprecated; use importlib.metadata / "
                        "importlib.resources (H6: Frozen-Version API).",
                    ))
        if isinstance(node, ast.ImportFrom) and node.module == "pkg_resources":
            issues.append(_issue(
                "H6", node,
                "H6 'pkg_resources' is deprecated; use importlib.metadata / "
                "importlib.resources (H6: Frozen-Version API).",
            ))
        if isinstance(node, ast.Call):
            dotted = _call_dotted_name(node)
            if dotted is None:
                continue
            if dotted.endswith("datetime.utcnow") or dotted == "datetime.utcnow":
                # SB2 + H6 both warn about naive utcnow(); emit under H6 (deprecation)
                # and SB2 (serialization boundary) so both pattern ids are exercised.
                issues.append(_issue(
                    "H6", node,
                    "H6 datetime.utcnow() is deprecated (3.12+) and returns a naive "
                    "datetime. Use datetime.now(tz=timezone.utc).",
                ))
                issues.append(_issue(
                    "SB2", node,
                    "SB2 datetime.utcnow() returns a naive datetime; mixing it with "
                    "tz-aware records breaks comparisons. Use "
                    "datetime.now(tz=timezone.utc).",
                ))
            elif dotted.endswith("asyncio.get_event_loop") or dotted == "asyncio.get_event_loop":
                # AC1 + H6 both cover this; emit AC1 (its primary home).
                issues.append(_issue(
                    "AC1", node,
                    "AC1 asyncio.get_event_loop() is deprecated (3.10+) and may "
                    "return the wrong loop in executor threads. Use "
                    "asyncio.get_running_loop() inside async code.",
                ))
        # collections.Callable attribute access (removed 3.10)
        if isinstance(node, ast.Attribute) and node.attr == "Callable":
            val = node.value
            if isinstance(val, ast.Name) and val.id == "collections":
                issues.append(_issue(
                    "H6", node,
                    "H6 collections.Callable was removed in Python 3.10; use "
                    "collections.abc.Callable.",
                ))
    return issues


# ---------------------------------------------------------------------------
# H9 — async def __init__ (illegal async constructor)
# ---------------------------------------------------------------------------
def check_h9_async_init(tree: ast.AST) -> list[Issue]:
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "__init__":
            issues.append(_issue(
                "H9", node,
                "H9 'async def __init__' is not a valid async constructor — the "
                "awaited work never runs as expected. Use an async factory "
                "classmethod (e.g. `@classmethod async def create(cls, ...)`).",
            ))
    return issues


# ---------------------------------------------------------------------------
# H4 — Confabulated method chain 3+ deep (regex over source)
# ---------------------------------------------------------------------------
_CHAIN_RE = re.compile(r"\.[A-Za-z_][A-Za-z0-9_]*\([^()]*\)"
                       r"\.[A-Za-z_][A-Za-z0-9_]*\([^()]*\)"
                       r"\.[A-Za-z_][A-Za-z0-9_]*\(")


def check_h4_method_chain(source: str) -> list[Issue]:
    issues: list[Issue] = []
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        m = _CHAIN_RE.search(line)
        if m:
            issues.append(_src_issue(
                "H4", i, m.start(),
                "H4 Method chain 3+ deep — if an early link returns None this "
                "raises AttributeError on the next call (H4: Confabulated Method "
                "Chain). Check the intermediate return values before chaining.",
            ))
    return issues


# ---------------------------------------------------------------------------
# H5 — Hardcoded float assertion in a test (advisory heuristic)
# ---------------------------------------------------------------------------
def check_h5_hardcoded_float_assert(tree: ast.AST, filename: str) -> list[Issue]:
    if not _is_test_file(filename):
        return []
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(
            test.ops[0], ast.Eq
        ):
            for side in (test.left, *test.comparators):
                if isinstance(side, ast.Constant) and isinstance(side.value, float):
                    issues.append(_issue(
                        "H5", node,
                        "H5 (advisory) assert compares against a hardcoded float "
                        f"({side.value!r}); if this value was inferred from reading "
                        "logic rather than observed, the test is brittle. Confirm "
                        "the expected value by running the code first.",
                    ))
                    break
    return issues


# ---------------------------------------------------------------------------
# SD1 — Positional integer index on a query-result row in a test (regex)
# ---------------------------------------------------------------------------
_POS_INDEX_RE = re.compile(r"\b(row|rows|values|result|record|fetchone|fetchall)\b"
                           r"[A-Za-z0-9_]*\[\s*\d+\s*\]")


def check_sd1_positional_index(source: str, filename: str) -> list[Issue]:
    if not _is_test_file(filename):
        return []
    issues: list[Issue] = []
    for i, line in enumerate(source.splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        m = _POS_INDEX_RE.search(line)
        if m:
            issues.append(_src_issue(
                "SD1", i, m.start(),
                "SD1 Positional integer index on a query result in a test — "
                "inserting a SQL column shifts every index after it silently "
                "(SD1: Schema Drift). Use named/Row access instead of row[N].",
            ))
    return issues


# ---------------------------------------------------------------------------
# SD2 — Hardcoded column/table name in raw SQL (advisory regex)
# ---------------------------------------------------------------------------
_SQL_RE = re.compile(r"\b(SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM)\b", re.IGNORECASE)


def check_sd2_hardcoded_sql(source: str) -> list[Issue]:
    issues: list[Issue] = []
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if _SQL_RE.search(line) and ('"' in line or "'" in line):
            issues.append(_src_issue(
                "SD2", i, 0,
                "SD2 (advisory) Raw SQL with hardcoded table/column names — "
                "verify against the live schema (PRAGMA table_info) before "
                "trusting these names (SD2: SQLite Schema Drift).",
            ))
    return issues


# ---------------------------------------------------------------------------
# SD3 — Labeled metric called with bare .inc()/.set() (AST)
# ---------------------------------------------------------------------------
def check_sd3_labeled_metric_bare_call(tree: ast.AST) -> list[Issue]:
    """Find metrics constructed WITH a labels list (3rd positional arg or
    labelnames=), then flag bare .inc()/.set()/.observe() on that variable that
    do not first go through .labels(...)."""
    labeled: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            fn = call.func
            ctor = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else None
            )
            if ctor not in _METRIC_CTORS:
                continue
            has_labels = len(call.args) >= 3 or any(
                kw.arg == "labelnames" for kw in call.keywords
            )
            if not has_labels:
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    labeled[tgt.id] = call
    if not labeled:
        return []

    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr in {"inc", "set", "observe"}):
            continue
        # bare metric.inc(): receiver is the labeled Name directly (no .labels())
        if isinstance(fn.value, ast.Name) and fn.value.id in labeled:
            issues.append(_issue(
                "SD3", node,
                f"SD3 '{fn.value.id}' is a labeled metric but '.{fn.attr}()' is "
                "called directly without '.labels(...)'. A labeled "
                "Counter/Gauge requires .labels(...).{m}(); bare .{m}() raises "
                "(SD3: Plain Counter Promoted to LabeledCounter).".format(m=fn.attr),
            ))
    return issues


# ---------------------------------------------------------------------------
# SS3 — Validation raise after an early-return guard (AST)
# ---------------------------------------------------------------------------
def _function_bodies(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def check_ss3_validation_after_guard(tree: ast.AST) -> list[Issue]:
    """In a function's top-level body: if an `if ...: return ...` guard appears
    BEFORE an `if ...: raise ValueError/...` validation, the validation is
    unreachable when the guard fires."""
    issues: list[Issue] = []
    for fn in _function_bodies(tree):
        saw_return_guard = False
        guard_line = 0
        for stmt in fn.body:
            if isinstance(stmt, ast.If):
                # early-return guard: a return inside this if's body
                has_return = any(isinstance(s, ast.Return) for s in stmt.body)
                has_raise = any(isinstance(s, ast.Raise) for s in stmt.body)
                if has_return and not has_raise:
                    saw_return_guard = True
                    guard_line = stmt.lineno
                elif has_raise and saw_return_guard:
                    issues.append(_issue(
                        "SS3", stmt,
                        "SS3 Input-validation `raise` placed AFTER an early-return "
                        f"guard (line {guard_line}); it is unreachable when the "
                        "guard fires. Validate inputs BEFORE resource/connection "
                        "guards (SS3: Validation After Early-Return Guard).",
                    ))
    return issues


# ---------------------------------------------------------------------------
# SS5 — Per-record try/except in a loop wrapping a DB execute, swallowed (AST)
# ---------------------------------------------------------------------------
def check_ss5_per_record_swallow(tree: ast.AST) -> list[Issue]:
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        for child in node.body:
            if not isinstance(child, ast.Try):
                continue
            # try body must contain an execute()-ish call
            has_execute = any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr in {"execute", "executemany", "insert", "save"}
                for c in ast.walk(child)
            )
            if not has_execute:
                continue
            for handler in child.handlers:
                body = handler.body
                swallowed = all(
                    isinstance(s, ast.Pass)
                    or (isinstance(s, ast.Continue))
                    or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
                    for s in body
                )
                if swallowed:
                    issues.append(_issue(
                        "SS5", handler,
                        "SS5 Per-record try/except inside a loop swallows the "
                        "exception (pass/continue) around a DB write — partial "
                        "writes look like success and break batch atomicity. "
                        "Wrap the whole batch in one try and rollback on failure.",
                    ))
                    break
    return issues


# ---------------------------------------------------------------------------
# SL1 — Variable assigned only inside an if-block, used after it (AST)
# ---------------------------------------------------------------------------
def check_sl1_conditional_binding(tree: ast.AST) -> list[Issue]:
    """Best-effort: a name assigned ONLY inside the body of a top-level `if`
    (with no else binding it) that is then referenced (load) later in the same
    function body and is not a parameter / earlier assignment."""
    issues: list[Issue] = []
    for fn in _function_bodies(tree):
        params = {a.arg for a in fn.args.args}
        params |= {a.arg for a in getattr(fn.args, "posonlyargs", [])}
        params |= {a.arg for a in fn.args.kwonlyargs}
        if fn.args.vararg:
            params.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            params.add(fn.args.kwarg.arg)

        unconditional: set[str] = set(params)
        conditional: dict[str, int] = {}

        for stmt in fn.body:
            if isinstance(stmt, ast.If):
                if_assigned = _assigned_names(stmt.body)
                else_assigned = _assigned_names(stmt.orelse)
                # names bound in if-body but not unconditionally and not in else
                for name in if_assigned - else_assigned:
                    if name not in unconditional:
                        conditional.setdefault(name, stmt.lineno)
                # names bound in both branches become unconditional
                unconditional |= (if_assigned & else_assigned)
            else:
                for name in _assigned_names([stmt]):
                    unconditional.add(name)
                    conditional.pop(name, None)
                # a use of a conditionally-bound name at this top-level stmt
                for used in _loaded_names(stmt):
                    if used in conditional and used not in unconditional:
                        issues.append(_src_issue(
                            "SL1", getattr(stmt, "lineno", 0), 0,
                            f"SL1 '{used}' is assigned only inside a conditional "
                            f"block (line {conditional[used]}) but used here on a "
                            "path where that block may not have run — possible "
                            "UnboundLocalError. Hoist it with a safe default.",
                        ))
                        conditional.pop(used, None)
    return issues


def _assigned_names(stmts: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for s in stmts:
        for node in ast.walk(s):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        names.add(tgt.id)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
    return names


def _loaded_names(stmt: ast.stmt) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
    return names


# ---------------------------------------------------------------------------
# SL2 — pytest fixture using `return` inside a `with mock.patch()` block (AST)
# ---------------------------------------------------------------------------
def _is_fixture(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in fn.decorator_list:
        # @pytest.fixture or @fixture, with or without call
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            return True
        if isinstance(target, ast.Name) and target.id == "fixture":
            return True
    return False


def _with_is_patch(node: ast.With) -> bool:
    for item in node.items:
        ctx = item.context_expr
        if isinstance(ctx, ast.Call):
            name = _call_dotted_name(ctx) or ""
            if name.endswith("patch") or ".patch." in (name + "."):
                return True
    return False


def check_sl2_fixture_return_in_patch(tree: ast.AST) -> list[Issue]:
    issues: list[Issue] = []
    for fn in _function_bodies(tree):
        if not _is_fixture(fn):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.With) and _with_is_patch(node):
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        issues.append(_issue(
                            "SL2", child,
                            "SL2 pytest fixture uses `return` inside a "
                            "`with mock.patch(...)` block — the patch is torn down "
                            "at the return, so the test body hits the real backend. "
                            "Use `yield` to keep the mock alive through the test.",
                        ))
                        break
                break
    return issues


# ---------------------------------------------------------------------------
# SB1 — json.dumps in a scope that also references numpy arrays (regex)
# ---------------------------------------------------------------------------
def check_sb1_ndarray_json(source: str) -> list[Issue]:
    has_numpy = bool(re.search(r"\bnp\.array\b|\bnumpy\b|\bndarray\b", source))
    if not has_numpy:
        return []
    issues: list[Issue] = []
    for i, line in enumerate(source.splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        m = re.search(r"\bjson\.dumps\s*\(", line)
        if m:
            issues.append(_src_issue(
                "SB1", i, m.start(),
                "SB1 json.dumps(...) in a module that uses numpy arrays — numpy "
                "ndarray is not JSON serializable (TypeError at runtime). Call "
                ".tolist() at dict construction (SB1: Serialization Boundary).",
            ))
    return issues


# ---------------------------------------------------------------------------
# AC2 — Unbounded retry while-True loop (try/except + sleep, no cap) (AST)
# ---------------------------------------------------------------------------
def check_ac2_unbounded_retry(tree: ast.AST) -> list[Issue]:
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.While) and _is_true(node.test)):
            continue
        # body is essentially a single try whose except sleeps and loops
        tries = [s for s in node.body if isinstance(s, ast.Try)]
        if not tries:
            continue
        for tnode in tries:
            for handler in tnode.handlers:
                sleeps = [
                    c for c in ast.walk(handler)
                    if isinstance(c, ast.Call)
                    and (_call_dotted_name(c) or "").endswith("sleep")
                ]
                # no break / no raise / no attempt-cap in handler → unbounded
                has_exit = any(
                    isinstance(c, (ast.Break, ast.Raise)) for c in ast.walk(handler)
                )
                if sleeps and not has_exit:
                    issues.append(_issue(
                        "AC2", handler,
                        "AC2 `while True` retry loop sleeps and retries on error "
                        "with no attempt cap, break, or circuit breaker — it can "
                        "hammer a failing dependency forever (AC2: Missing Circuit "
                        "Breaker). Use bounded retries with exponential backoff.",
                    ))
                    break
    return issues


def _is_true(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


# ---------------------------------------------------------------------------
# AC4 — Fire-and-forget create_task with no reference/callback (AST)
# ---------------------------------------------------------------------------
def check_ac4_orphan_task(tree: ast.AST) -> list[Issue]:
    issues: list[Issue] = []
    for node in ast.walk(tree):
        # bare expression statement that is asyncio.create_task(...)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = _call_dotted_name(node.value) or ""
            if name.endswith("create_task"):
                issues.append(_issue(
                    "AC4", node,
                    "AC4 asyncio.create_task(...) result is neither stored nor given "
                    "a done-callback — if the task raises, the exception is silently "
                    "discarded (AC4: Async Task Exception Discarded). Keep a "
                    "reference and add_done_callback, or use asyncio.TaskGroup.",
                ))
    return issues


# ---------------------------------------------------------------------------
# MF2 — Reconnect sleep with no jitter / constant backoff (AST)
# ---------------------------------------------------------------------------
def check_mf2_reconnect_no_jitter(tree: ast.AST) -> list[Issue]:
    """Inside a loop that calls a connect()/reconnect()-like function, an
    asyncio.sleep / time.sleep with a plain constant arg and no random jitter
    is a reconnect-storm risk."""
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
        has_connect = any(
            "connect" in (_call_dotted_name(c) or "").lower() for c in calls
        )
        if not has_connect:
            continue
        uses_random = any("random" in (_call_dotted_name(c) or "") for c in calls)
        if uses_random:
            continue
        for c in calls:
            name = _call_dotted_name(c) or ""
            if name.endswith("sleep") and c.args:
                arg = c.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                    issues.append(_issue(
                        "MF2", c,
                        "MF2 Reconnect loop sleeps a constant with no random jitter "
                        "or exponential backoff — all consumers reconnect in lockstep "
                        "after a restart (MF2: Reconnect Storm). Add randomized "
                        "exponential backoff.",
                    ))
                    break
    return issues


# ---------------------------------------------------------------------------
# MF3 — Unconditional logging inside a while-True loop (AST)
# ---------------------------------------------------------------------------
def check_mf3_log_in_loop(tree: ast.AST) -> list[Issue]:
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.While) and _is_true(node.test)):
            continue
        for stmt in node.body:
            # a logger.warning/error called directly at loop top level (not
            # inside a state-change `if`)
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                name = _call_dotted_name(stmt.value) or ""
                if name.endswith(("logger.warning", "logger.error",
                                  "log.warning", "log.error",
                                  "logging.warning", "logging.error")):
                    issues.append(_issue(
                        "MF3", stmt,
                        "MF3 logger.warning/error called unconditionally on every "
                        "iteration of a `while True` loop — under a sustained error "
                        "condition this floods disk I/O and feeds a metastable "
                        "failure loop (MF3). Log once on state change instead.",
                    ))
    return issues


# ---------------------------------------------------------------------------
# OG1 — Exception swallowed without logging (AST)
# ---------------------------------------------------------------------------
def check_og1_swallowed_exception(tree: ast.AST) -> list[Issue]:
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = node.body
        # handler body is only pass, or only `return None`/bare return
        only_pass = len(body) == 1 and isinstance(body[0], ast.Pass)
        only_return_none = (
            len(body) == 1
            and isinstance(body[0], ast.Return)
            and (body[0].value is None
                 or (isinstance(body[0].value, ast.Constant)
                     and body[0].value.value is None))
        )
        if not (only_pass or only_return_none):
            continue
        # is there any logging anywhere in the handler? (already excluded by len==1)
        issues.append(_issue(
            "OG1", node,
            "OG1 except handler swallows the exception (only "
            f"{'pass' if only_pass else 'return None'}) with no logging — the "
            "failure leaves no trace (OG1: Exception Swallowed Without Logging). "
            "Log with exc_info before returning/continuing.",
        ))
    return issues


# ---------------------------------------------------------------------------
# PB1 — State mutation before a validation raise in the same function (AST)
# ---------------------------------------------------------------------------
def check_pb1_validation_after_mutation(tree: ast.AST) -> list[Issue]:
    """If a function mutates self-state / increments / calls insert BEFORE an
    `if ...: raise`, the mutation persists when validation fails."""
    issues: list[Issue] = []
    for fn in _function_bodies(tree):
        mutated = False
        mutate_line = 0
        for stmt in fn.body:
            if not mutated and _is_mutation(stmt):
                mutated = True
                mutate_line = getattr(stmt, "lineno", 0)
                continue
            # a validation guard that raises
            if mutated and isinstance(stmt, ast.If):
                if any(isinstance(s, ast.Raise) for s in stmt.body):
                    issues.append(_issue(
                        "PB1", stmt,
                        "PB1 Input validation `raise` placed AFTER a state mutation "
                        f"(line {mutate_line}) — the partial mutation (counter/DB "
                        "write/queue push) persists when validation fails (PB1: "
                        "Validation After Mutation). Validate first, then mutate.",
                    ))
                    break
    return issues


def _is_mutation(stmt: ast.stmt) -> bool:
    # self.x = / self.x += ; or a db/cursor insert/execute call statement
    if isinstance(stmt, ast.AugAssign):
        return _is_self_attr(stmt.target)
    if isinstance(stmt, ast.Assign):
        return any(_is_self_attr(t) for t in stmt.targets)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        name = _call_dotted_name(stmt.value) or ""
        return name.endswith(("insert", "execute", "executemany", "append",
                              "write", "commit"))
    return False


def _is_self_attr(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


# ---------------------------------------------------------------------------
# WG1 — *Publisher/*Notifier/*Exporter/*Sender with a no-op send/publish (AST)
# ---------------------------------------------------------------------------
_WIRING_SUFFIXES = ("Publisher", "Notifier", "Exporter", "Sender")
_SEND_METHODS = {"send", "publish", "export", "notify", "emit"}
_OUTBOUND_HINTS = ("post", "get", "put", "request", "send", "sendall",
                   "publish", "connect", "urlopen", "sendto")


def check_wg1_noop_integration(tree: ast.AST) -> list[Issue]:
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not node.name.endswith(_WIRING_SUFFIXES):
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name not in _SEND_METHODS:
                continue
            outbound = False
            for c in ast.walk(item):
                if isinstance(c, ast.Call):
                    name = (_call_dotted_name(c) or "").lower()
                    if any(name.endswith(h) for h in _OUTBOUND_HINTS):
                        # exclude logger.* false positives by ignoring logger.
                        if not name.startswith(("logger.", "log.", "logging.",
                                                "print")):
                            outbound = True
                            break
            if not outbound:
                issues.append(_issue(
                    "WG1", item,
                    f"WG1 '{node.name}.{item.name}()' is named for an integration "
                    "but makes no outbound call (no requests/httpx/socket "
                    "post/send) — likely a no-op backend that only logs (WG1: "
                    "Wiring Gap). Add the real outbound call or raise on misconfig.",
                ))
    return issues


# ---------------------------------------------------------------------------
# TEC1 — Test builds a prometheus metric / reads ._value, no reset fixture (regex)
# ---------------------------------------------------------------------------
def check_tec1_registry_leak(source: str, filename: str) -> list[Issue]:
    if not _is_test_file(filename):
        return []
    uses_metric = bool(
        re.search(r"\b(Counter|Gauge|Histogram|Summary)\s*\(", source)
        or re.search(r"\._value\b", source)
    )
    if not uses_metric:
        return []
    has_reset = bool(
        re.search(r"autouse\s*=\s*True", source)
        or re.search(r"reset|clear_registry|CollectorRegistry|_initialized", source)
    )
    if has_reset:
        return []
    return [_src_issue(
        "TEC1", 1, 0,
        "TEC1 Test uses prometheus metrics (or reads ._value) but has no autouse "
        "reset fixture — the process-global registry leaks values between tests "
        "(TEC1: Shared Registry State Leak). Add an autouse fixture that resets "
        "the registry.",
    )]


# ---------------------------------------------------------------------------
# MC1b — Counter used for a decreasing quantity (regex on metric name)
# ---------------------------------------------------------------------------
_GAUGE_WORDS = ("depth", "size", "active", "current", "usage", "level",
                "count", "inflight", "in_flight", "connections", "queue",
                "utilization", "temperature")


def check_mc1b_counter_should_be_gauge(tree: ast.AST) -> list[Issue]:
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        ctor = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else None
        )
        if ctor != "Counter" or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        name = first.value.lower()
        if any(w in name for w in _GAUGE_WORDS):
            issues.append(_issue(
                "MC1b", node,
                f"MC1b Counter('{first.value}') names a value that can decrease — "
                "a Counter only goes up, producing a meaningless monotonic series. "
                "Use a Gauge for queue depth / active connections / utilization "
                "(MC1b: Counter Used Where Gauge Required).",
            ))
    return issues


# ---------------------------------------------------------------------------
# H1 / H7 light backstops (vendored LH001/LH007 are primary)
# ---------------------------------------------------------------------------
# We intentionally do NOT re-emit H1/H7 here: the vendored LH001/LH007 already
# fire on the same constructs and run_code_checks aggregates both. Re-emitting
# would only create duplicates that the dedupe step removes. PATTERN_META marks
# H1/H7 as ruff_core to record that they are covered.


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
# (check_fn, needs) where needs is "tree", "source", "tree+name", "source+name"
_AST_CHECKS = (
    check_h2_duplicate_metric,
    check_h6_deprecated_api,
    check_h9_async_init,
    check_sd3_labeled_metric_bare_call,
    check_ss3_validation_after_guard,
    check_ss5_per_record_swallow,
    check_sl1_conditional_binding,
    check_sl2_fixture_return_in_patch,
    check_ac2_unbounded_retry,
    check_ac4_orphan_task,
    check_mf2_reconnect_no_jitter,
    check_mf3_log_in_loop,
    check_og1_swallowed_exception,
    check_pb1_validation_after_mutation,
    check_wg1_noop_integration,
    check_mc1b_counter_should_be_gauge,
)

_SOURCE_CHECKS = (
    check_h4_method_chain,
    check_sd2_hardcoded_sql,
    check_sb1_ndarray_json,
)

# checks that also need the filename
_AST_NAME_CHECKS = (
    check_h5_hardcoded_float_assert,
)
_SOURCE_NAME_CHECKS = (
    check_sd1_positional_index,
    check_tec1_registry_leak,
)


def run_extended_checks(path: str | Path) -> list[Issue]:
    """Run all loci-owned extended checks on a single ``.py`` file.

    Parses the source once and dispatches the tree to AST checks and the raw
    text to regex/source checks. Fail-safe: a per-check exception is swallowed
    (that pattern is skipped), and an unreadable file or SyntaxError yields an
    empty list — this function never raises.
    """
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source, filename=str(p))
    except SyntaxError:
        # The vendored check_file already emits LH000 for syntax errors; the
        # extended layer simply contributes nothing here.
        return []

    filename = p.name
    issues: list[Issue] = []

    for check in _AST_CHECKS:
        try:
            issues.extend(check(tree))
        except Exception:  # noqa: BLE001 — a broken check must not break the hook
            continue

    for check in _AST_NAME_CHECKS:
        try:
            issues.extend(check(tree, filename))
        except Exception:  # noqa: BLE001
            continue

    for check in _SOURCE_CHECKS:
        try:
            issues.extend(check(source))
        except Exception:  # noqa: BLE001
            continue

    for check in _SOURCE_NAME_CHECKS:
        try:
            issues.extend(check(source, filename))
        except Exception:  # noqa: BLE001
            continue

    return sorted(issues, key=lambda i: (i.line, i.col, i.code))

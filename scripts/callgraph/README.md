# callgraph

A dependency-light, stdlib-only static call-graph tool for the loci corpus
(`mcp/`, `scripts/`, `a2a_server/`, `mlops/`, `eval/`). No LadybugDB, no
network, no third-party imports — it runs during debugging when nothing
else in the stack is up, and it can read a specific git revision so it
stays correct even while another workflow is mid-editing a file it needs
to look at.

It complements `mcp/graph_tools.py` / `mcp/graph/*` (which need a live
LadybugDB and model symbols + CALLS) by modelling the things this codebase
actually does for dispatch that a symbol graph doesn't capture:
`@mcp.tool()` registration, `register(mcp, deps)` injection, dict-of-callables
dispatch, module-global reads/writes across files, and path/key literal
agreement between producers and consumers. See `docs/DESIGN.md` for the
full node/edge model and the `CALLS` resolution ladder, and `cg limits`
(or `docs/LIMITS.md`) for the honest failure-mode catalogue.

## 60-second tour

```
PYTHONPATH=scripts python3 -m callgraph.cli build --rev HEAD
PYTHONPATH=scripts python3 -m callgraph.cli modules --rev HEAD --scope mcp/
PYTHONPATH=scripts python3 -m callgraph.cli defs --rev HEAD --scope mcp/graph_tools.py --all
PYTHONPATH=scripts python3 -m callgraph.cli imports --rev HEAD --unresolved
PYTHONPATH=scripts python3 -m callgraph.cli imports --rev HEAD --lazy
PYTHONPATH=scripts python3 -m callgraph.cli aliases --rev HEAD mcp/server.py::symbol_impact
PYTHONPATH=scripts python3 -m callgraph.cli callers _get_ladybug --rev HEAD --conf proven
PYTHONPATH=scripts python3 -m callgraph.cli reach _get_ladybug --rev HEAD --depth 2
PYTHONPATH=scripts python3 -m callgraph.cli registry --rev HEAD
PYTHONPATH=scripts python3 -m callgraph.cli name _get_ladybug --rev HEAD
PYTHONPATH=scripts python3 -m callgraph.cli dead --rev HEAD --scope mcp/
PYTHONPATH=scripts python3 -m callgraph.cli holes --rev HEAD
PYTHONPATH=scripts python3 -m callgraph.cli writes-dead --rev HEAD --scope mcp/
PYTHONPATH=scripts python3 -m callgraph.cli literals --rev HEAD --paths --orphans
PYTHONPATH=scripts python3 -m callgraph.cli flags --rev HEAD mcp/grounding.py::ground
PYTHONPATH=scripts python3 -m callgraph.cli reach _get_ladybug --rev HEAD --depth 2 --format dot > reach.dot
PYTHONPATH=scripts python3 -m callgraph.cli selftest
PYTHONPATH=scripts python3 -m callgraph.cli limits
```

Always pass `--rev HEAD` (or any commit-ish) when another workflow might be
editing a file you're about to query — the working tree is the default
source, but a mid-edit file can have lines that don't exist yet. Every
report prints which source it read.

## What's implemented (build_steps 1-13 — the full design, minus `whatchanged`)

- **config.py** — the corpus definition: `mcp/`, `scripts/`, `a2a_server/`,
  `mlops/`, `eval/`, non-test `.py` files only (114 files at HEAD),
  excluding `tests/`, `__pycache__`, `.venv`, `*.egg-info`, and this
  package itself.
- **ingest.py** — `ast.parse` over the corpus, either the working tree or a
  specific git revision (`git ls-tree` + `git show`). A SyntaxError on one
  file degrades that file to a stub and never aborts the build.
- **resolve.py** — the module resolution table: which dotted names each
  file answers to, given (a) the implicit repo-root namespace, (b) a
  literal `sys.path.insert(0, <expr>)` this tool can constant-fold, or (c)
  a directory containing a `__main__` entry point (Python auto-adds a
  script's own directory to `sys.path[0]` when run directly — this is how
  `import harness` resolves from `eval/grounding_gate_eval.py`, purely
  because `eval/harness.py` has a `__main__` guard). Every resolution
  records which insert (or which fact) made it work, preferring a root the
  *importing file itself* established over one some other file happened to
  set up first.
- **model.py** — `Node`/`Edge`/`Confidence`/`GraphStore`, the shared
  representation every later extractor and query reads and writes.
- **scopes.py** — one AST pass per module: qualnames (class/closure/lambda
  nesting), decorator source text, params, module-global `NAME` slots
  (including idents that are `global`-declared but never bound at module
  level — the raw signal behind BUG C), and import statements.
- **extract/defs.py** — MODULE / CLASS / FUNCTION / NAME nodes, `DEFINES`
  edges, and `DECORATED_BY` classification (registering / wrapping /
  unknown), driven by `rules.toml`.
- **extract/imports.py** — `IMPORTS` edges (module-level vs function-local,
  with `resolved_via` provenance) and `ALIASES` edges for `from X import Y`
  re-exports and bare `A = B` module-level aliasing — so
  `mcp/server.py`'s `from graph_tools import symbol_impact` and
  `mcp/graph_tools.py`'s own `def symbol_impact` are the same node, not two.

- **extract/walk.py** — one shared whole-module AST walk (module-level vs
  "inside which FUNCTION") that extract/calls.py, extract/names.py and
  extract/registry.py's REG-FN pass all build their per-node callback on
  top of, instead of each re-deriving enclosing-scope from scratch.
- **extract/calls.py** — `CALLSITE` nodes (one per `ast.Call` in the
  corpus, ~11,600 of them) and rungs 1-3 of the `CALLS` resolution ladder
  (all PROVEN-tier): a plain module-level def/class, an attribute call
  through a module-level *or* function-local import, one-or-more `ALIASES`
  hops, Python builtins, and a nested-function's own bare name. Anything
  else — attribute calls on an unknown-type receiver, calls through
  dict/getattr/subscript results, calls on a bare parameter — lands on the
  single `?` UNRESOLVED sink with a `reason`, never silently dropped
  (rungs 4-6, the PROBABLE tier, are a later slice's job).
- **extract/names.py** — `READS_NAME` / `WRITES_NAME` edges, with a precise
  `via` per occurrence (`module-level-assign` | `global-stmt` |
  `import-binding` | `def-binding` | `augmented`) and cross-module reads
  (`mod.X` through a module-level import). This is the data BUG C's
  diagnosis needs: a `global`-declared write with no module-level binding
  anywhere in the OWNING module.
- **extract/registry.py** — `REGISTRY` / `REGISTERS` / `ENTERS` /
  `ENTRYPOINT` for every registration surface this corpus uses: `DEC`
  (`@mcp.tool()`, `@mcp.resource()`, `@mcp.custom_route()`,
  `@app.get/post/...`), `MAN-LOOP` (`for fn in (a, b, c): mcp.tool()(fn)`),
  `MAN-DICT` (a module-level dict-of-bare-callables, e.g. `_SKILL_MAP`),
  `ROOT-CLI` (`if __name__ == "__main__":`), and `ROOT-EXT` (rules.toml's
  hand-maintained `[[roots]]` table for systemd/cron/shell triggers). Also
  `INJECTS`: `register()`'s parameter-or-`deps["key"]` → `global` binding
  pattern, correlated against every call site of that `register()`
  anywhere in the corpus.
- **analyze/reach.py**, **analyze/deadcode.py** — the closures `cg
  callers`/`cg reach`/`cg dead` run on top of. `cg dead`'s hard-gate
  contract (build_steps step 5): **zero** `@mcp.tool()`/manifest-registered
  function may ever be reported dead — every `ENTRYPOINT` enters its
  registered `FUNCTION` directly, independent of any module-execution
  modelling, specifically so this can't slip. Measured at HEAD: **0 false
  positives across all 92 registered functions**, 66 rows total
  (down from 165 before the validator pass). A callsite
  inside a `lambda` body is additionally attributed to the scope that
  BUILDS the lambda, since a lambda is consumed where it is written.

- **extract/flow.py** — escaping-local detection (`LOCALBINDING`/`ESCAPES`)
  and the lightweight per-function control-flow summary (a linear walk with
  a context stack — if-branch / loop-body / except-handler / with-body —
  deliberately not a real CFG) that `cg flags` ranks over, including the
  ASSIGNMENT FORM (plain rebind vs `+=` vs for-target) each update used.
  **analyze/flags.py** turns that into the partial-assignment ranker:
  `cg flags mcp/grounding.py::ground` finds BUG B's `degraded` flag exactly
  (init constant, 3 guarded reassignments, 11 guard exits that skip it).
  `classify()` splits accumulators (counters, correct by construction) off
  from flags — they were 16 of 23 rows and outranked BUG B itself until the
  validator pass; pass `--include-accumulators` to see them.

- **extract/funcrefs.py** — `REFERENCES` edges for a bare function name in
  any value position (passed as an argument, returned, stored). Generalizes
  the registration-specific REFERENCES `extract/registry.py` already
  emitted. This is what keeps callbacks like
  `_health_check("x", _health_probe_embeddings_sparse)` off `cg dead`.
- **extract/literals.py** — `LITERAL`/`PATHEXPR` nodes and
  `PRODUCES_LITERAL`/`CONSUMES_LITERAL` edges for path- and key-like string
  literals, including composed paths (`MEMORY_DIR / "x"`, `os.path.join`,
  f-strings). **analyze/literalaudit.py** is the producer/consumer table,
  orphan detection, and the fenced `--near-miss` pairing: `cg literals
  --paths --orphans` finds BUG D's `graph.ladybug`/`graph.kuzu` split
  exactly, and `--near-miss` pairs them on the shared `graph` stem.
- **analyze/nameaudit.py** — `cg writes-dead`'s two checks:
  DANGLING-GLOBAL (near-zero false positives — finds BUG C's
  `_symbol_index_cache`/`_symbol_index_count` and names the real slot in
  `mcp/ladybug_ops.py`) and the weaker WRITE-WITH-NO-READ triage list
  (`--include-tests` re-checks against a textual scan of `tests/` source).

CLI subcommands available now: `build`, `modules`, `defs`, `imports`,
`aliases`, `callers`, `reach`, `paths`, `entrypoints`, `registry`, `name`,
`writes-dead`, `literals`, `flags`, `dead`, `holes`, `explain`, `selftest`,
`limits`. `--format` accepts `text` (default), `json` (every command), and
`dot` (a Graphviz export — full graph on `build`, the traversed subgraph on
`reach`/`callers`/`paths`; solid/dashed/dotted edges encode
proven/probable/unproven). `whatchanged` (diff-to-blast-radius) is the one
query from the design not yet built — everything else in `docs/DESIGN.md`'s
query list is live.

- **selftest.py** / **`cg selftest`** — the standalone health check: builds
  the graph once over `fixtures/` (no git, no network — one dedicated
  assertion per dispatch shape the design lists) and once over the real
  corpus at `--rev HEAD` (the hard gate + registration-surface counts),
  and prints PASS/FAIL per check with a nonzero exit on any failure. Same
  checks run under `pytest` via `tests/test_selftest.py`. Full run
  (fixture build + the one real-corpus build + all 15 assertions) finishes
  in ~3.5s on this box, comfortably inside the design's 5s ceiling.
- **`cg limits`** — prints `docs/LIMITS.md` verbatim: the measured
  false-positive rates and orphan/row counts behind
  `writes-dead`/`literals`/`flags`, refreshed at every HEAD this package
  moves to. A new false-positive class discovered in use is meant to be
  appended there in the same commit that finds it.
- **`--rev` under a concurrent edit** — `ingest.py`'s git-rev path now
  reads the whole corpus in ONE `git ls-tree` + ONE `git cat-file --batch`
  subprocess pair (previously one `git show` per file), which is both
  faster and, more importantly, structurally incapable of touching the
  working tree at all when `--rev` is given — verified directly in
  `tests/test_rev_freshness.py` by monkeypatching `Path.read_text` to
  raise during a `--rev` build. That's the property that matters while
  this package was built alongside another workflow mid-editing
  `mcp/server.py`: `--rev HEAD` never sees whatever that workflow has
  sitting in the working tree at the moment it runs. `tests/
  test_rev_freshness.py` also covers the "cache invalidation" half of
  step 13's brief: there is no parse cache (see `ingest.py`'s own
  docstring for why, and `--no-cache`'s documented no-op status), so what
  gets tested instead is the absence of a staleness bug — back-to-back
  builds at different revisions never leak content into each other, and
  rebuilding the same rev twice is byte-identical.

## Validation (does it catch bugs that were REAL?)

`tests/test_regression_real_bugs.py` and `tests/test_dead_false_positives.py`
are the gate that decides whether this tool is worth keeping.

The regression tests do NOT use hand-written miniatures. They run the
analyzers against the ACTUAL pre-fix source of three shipped bugs, copied
verbatim out of git history into `fixtures/regress/`, with a provenance test
that re-checks each fixture against `git show` so it cannot be quietly edited
into something easier to pass.

| bug | query that surfaces it | result on the real pre-fix code |
|---|---|---|
| **B** `grounding.py`'s `degraded` set on only some paths | `cg flags` | sole row for the file; **rank 3 of 7** corpus-wide |
| **C** `graph_tools.py`'s dead `global` | `cg writes-dead` (DANGLING-GLOBAL) | exactly 2 findings, names the real slot in `ladybug_ops` |
| **D** `graph.ladybug` vs `graph.kuzu` | `cg literals --paths --orphans --near-miss` | both orphans found and paired on stem `graph` |
| **A** `node["_id"]` vs `"_ID"` | — | **out of reach, by design.** A data-format mismatch with a value produced inside a driver this tool never parses. No structural signal exists; see `docs/LIMITS.md`. |

`cg dead`'s false-positive rate is documented and gated the same way — see
`docs/LIMITS.md`'s audit section for the hand-verified breakdown of every
row it still reports.

Run the tests:

```
PYTHONPATH=scripts python3 -m pytest scripts/callgraph/tests -q
```

227 tests, full corpus (114 files) parsed repeatedly at several different
`--rev`s along the way — expect roughly two minutes wall time for the
whole suite (dominated by those repeated real-corpus builds, not by
`cg selftest`'s own ~3.5s fixture-plus-one-HEAD-build budget).

## Provenance

`docs/census.txt` is the original hand-written census of dispatch patterns
in this corpus (40 `@mcp.tool()` decorators, ~30 `register()` functions,
288 local imports, ...) that this tool's design was built against.
`docs/legacy/` holds the one-off AST-walking scanners (`census.py`,
`extended_census.py`, `PATTERNS_SUMMARY.json`) that produced it — they are
superseded by `extract/*` and must not be imported by it; keeping two AST
walkers alive is how rule drift starts.

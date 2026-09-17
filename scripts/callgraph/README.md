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

## Command index

CLI subcommands: `build`, `modules`, `defs`, `imports`, `aliases`, `callers`,
`reach`, `paths`, `entrypoints`, `registry`, `name`, `writes-dead`,
`literals`, `flags`, `dead`, `holes`, `explain`, `selftest`, `limits`.
`--format` accepts `text` (default), `json` (every command), and `dot`
(Graphviz export). `whatchanged` (diff-to-blast-radius) is the one query
from the design not yet built; everything else in `docs/DESIGN.md`'s query
list is live.

The full node/edge model and the `CALLS` resolution ladder — what each
node/edge type means and how a callsite gets resolved — live in
[`docs/DESIGN.md`](docs/DESIGN.md); read the source under `extract/*` and
`analyze/*` for the per-module implementation of that model. Measured
false-positive rates,
known failure modes (dynamic dispatch, non-Python entry points, untyped
receivers), and the `--rev`-under-concurrent-edit freshness guarantee are
catalogued in [`docs/LIMITS.md`](docs/LIMITS.md) — read that before trusting
or gating on any single query's output.

## Validation

`tests/test_regression_real_bugs.py` and `tests/test_dead_false_positives.py`
are the gate that decides whether this tool is worth keeping — they run the
analyzers against the actual pre-fix source of shipped bugs, copied verbatim
out of git history, with a provenance check against `git show`. Full
results and the bug-by-bug table are in
[`docs/LIMITS.md`](docs/LIMITS.md#validating-against-real-bugs-not-shapes).

Run the tests:

```
PYTHONPATH=scripts python3 -m pytest scripts/callgraph/tests -q
```

As of rev `989942b5` (2026-08-10): 249 tests, full corpus (115 files at that
revision) parsed repeatedly at several different `--rev`s — expect roughly
two minutes wall time for the whole suite.

## Provenance

`docs/census.txt` is the original hand-written census of dispatch patterns
in this corpus that this tool's design was built against. `docs/legacy/`
holds the superseded one-off AST-walking scanners that produced it — they
must not be imported by `extract/*`; keeping two AST walkers alive is how
rule drift starts.

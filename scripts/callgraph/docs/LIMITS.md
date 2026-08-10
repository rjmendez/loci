# Known limits

The honest failure-mode catalogue for `callgraph`. A new false-positive
class discovered in use gets appended here in the same commit that finds
it. Numbers below were measured at HEAD (`rev 989942b5`, 2026-08-10) with
`PYTHONPATH=scripts python3 -m callgraph.cli <cmd> --rev HEAD`, whole
corpus (114 files) unless a `--scope` is noted.

## `cg writes-dead`

Two checks, reported separately because their false-positive rates are not
remotely comparable.

- **DANGLING-GLOBAL** (the strong check, worth a CI gate): near-zero false
  positives — it is almost always either a typo or a cross-module cache
  invalidation that silently no-ops. At HEAD: **0** findings corpus-wide.
  At `c1c40a9^` (BUG C): **2** findings, both in `mcp/graph_tools.py`
  (`_symbol_index_cache`, `_symbol_index_count`), each correctly naming the
  real module-level slot at `mcp/ladybug_ops.py:106-107`.

- **WRITE-WITH-NO-READ** (the weak check, triage list only): **297**
  findings corpus-wide at HEAD. Re-running with `--include-tests` (a
  textual, whole-word scan of `tests/` source for each candidate ident)
  drops that to **104** — i.e. **123 of 297 (41%)** of the raw findings are
  read only from test code and are false positives of the default (tests
  excluded) view. The remaining 104 are dominated by two further known
  false-positive shapes this check cannot distinguish from a real orphaned
  write: (a) module-level bindings meant purely as a public re-export
  surface (`from __future__ import annotations`, `__all__` entries, and
  plain function/class defs that are never referenced by bare name because
  every caller goes through `from mod import symbol` — which correctly
  creates an ALIASES edge, not a READS_NAME, so the origin slot reads as
  "written, never read" even though it's very much in use); (b) a handful
  of genuinely internal helpers that exist only for interactive/REPL use.
  **Conclusion: never gate on WRITE-WITH-NO-READ; always read its output
  with a file open next to it, and prefer `--include-tests` as a first
  triage pass.**

## `cg literals`

- **Orphan counts at HEAD** (the haystack size `--orphans`/`--near-miss`
  work against): path-like — **0** producer-only, **1** consumer-only.
  Key-like — **83** producer-only, **549** consumer-only. The remaining
  path-like consumer-only orphan at HEAD is legitimate: a wildcard
  `glob.glob(".../settings.json")` pattern with no single producer site
  this tool can attribute to it.

- **BUG D regression**: at `08c9198^`, `cg literals --paths --orphans`
  reports `graph.ladybug` (produced at `mcp/server.py:261`, consumed by
  nobody) and `graph.kuzu` (consumed at `scripts/graph_facts.py:29`,
  produced by nobody) as two separate orphan groups; `--near-miss` pairs
  them on the shared stem `graph` (distance 0.609). At HEAD both literals
  are `graph.ladybug`, matched producer+consumer, zero orphans for either
  text.

- **FALSE POSITIVE — regex over-triggers on short/ambiguous key-like
  strings.** `looks_key_like`'s identifier-shaped regex matches `_`
  (Python's throwaway-variable convention) as a KEY-LIKE literal wherever
  it appears in a lookup position (e.g. an f-string built key like
  `f"_{name.upper()}"` in `mcp/graph/queries.py`) — it now correctly sorts
  into the KEY-LIKE bucket (a PATHEXPR's flavour is stamped from its call
  site, not guessed from node kind — see extract/literals.py) rather than
  polluting `--paths`, but it is still present as noise inside
  `--keys --orphans`. A future pass should raise the minimum length or
  blocklist single-underscore/single-letter strings.

- **FALSE POSITIVE/NEGATIVE — role inference loses track across
  variables.** A path or key built in one function, stored in a variable,
  and used three functions later has no PRODUCES/CONSUMES edge at all
  (silently dropped, not miscounted) — see the design's own note on this;
  not separately re-measured here beyond the orphan counts above, which
  are therefore an UNDER-count of true orphans, not an over-count.

- **KEY-LIKE literal counts are dominated by dict-get/dict-subscript
  noise.** 83+549 = 632 key-like orphan groups out of 732 total LITERAL
  nodes at HEAD is a strong hint that most `.get("some_word")` calls in
  this corpus are looking up per-call-site-unique or genuinely
  producer/consumer-decoupled keys (e.g. reading fields out of a JSON
  response body) rather than a coordinated cross-module contract —
  `--keys --orphans` is not a useful lead-generator on its own for this
  corpus; `--paths --orphans` is where the signal is.

## `cg flags`

- **Row counts at HEAD**: `mcp/` alone — **5** rows. Whole corpus —
  **7** rows (**28** with `--include-accumulators`; 21 of the 28 are
  counters, see the next bullet). Given ~558 escaping LOCALBINDINGs
  corpus-wide, only constant-initialized ones are ranked at all, and the
  design's own precision statement ("expect roughly one real finding per
  dozen rows") should be read against these row counts, not against the
  558 total LOCALBINDINGs — most escaping locals are never
  constant-initialized flags in the first place and never appear in
  `cg flags` output at all.

- **FALSE POSITIVE CLASS, now suppressed — accumulators.** A counter
  (`n = 0` ... `n += 1`) matches "constant init, updated on only some
  paths" perfectly, because a counter is SUPPOSED to be updated on only
  some paths. Measured at `d359e9a` (where BUG B was live) this class
  filled **16 of 23** corpus-wide rows and pushed the real bug down to
  **rank 7**. `analyze/flags.classify()` now separates them: any local
  whose reassignments include an augmented assignment, or include no plain
  rebind-to-a-constant at all, is `pattern="accumulator"` and is excluded
  unless `--include-accumulators` is passed. Same corpus, same revision,
  after: **7** rows, BUG B at **rank 3**. The two rows still above it
  (`canary.py::monitor_live`'s `rollback_recommended`,
  `server.py::_process_reflection_item`'s `sampling_mode`) are the same
  shape in CORRECT code — irreducible without knowing intent.

- **BUG B regression**: at `69adfa4^`, `cg flags mcp/grounding.py::ground`
  reports exactly one row — `degraded` — init `False` (constant) at
  `:179`, escaping as a dict-value at `:316`, reassigned `True` at 3
  guarded sites (2 `if-branch`, 1 `except-handler`), and **10** guard exits
  (`break`/`continue` inside `if-branch`/`except-handler` blocks) that
  never touch it, for a score of 3.67 — `degraded` is the only (hence
  top-ranked) row for the function, matching the acceptance criteria. At
  `69adfa4` (the actual fix, commit message: "grounding: report degraded
  when the server module fails to import"), a fourth assignment
  (`degraded = True` inside the `except Exception: S = None` handler that
  every server-backed lane short-circuits on) is added; re-running the
  same command against HEAD confirms the fix is visible to this tool as a
  DROP in score (**3.67 -> 2.75**) from the new guarded assignment, while
  the same 10 unrelated `if not S: break`/`continue` guard exits remain
  (they are not about `degraded` at all — they are the lanes' own
  fail-open behavior, unrelated to the flag). `cg flags` does not, and is
  not meant to, assert "this is now fixed"; it only ever reports a ratio.

- **RANKER, NOT A DETECTOR.** Every conditional flag in Python matches the
  shape this check ranks on. Never wire `cg flags` to a non-zero exit
  code or a CI gate; always scope it to one file or function before
  reading the output as anything but a lead list.

## `cg dead` — false-positive audit (validator pass, HEAD `989942b5`)

The number that decides whether this query is worth reading. Measured by
`tests/test_dead_false_positives.py`, which is the permanent gate.

- **HARD GATE: 0 false positives on the registration surface.** All 92
  registered functions — 40 `@mcp.tool()`, 31 `register()` manifest-tuple
  members, 13 `_SKILL_MAP` entries, 6 FastAPI routes, 1 `mcp.resource`,
  1 `mcp.custom_route` — are reachable. None is reported dead. This is an
  absolute, not a ratio.

- **Total rows: 66** (was **165** before this pass). Five resolution bugs
  plus one noise class, found by hand-verifying the 165, each now fixed and
  regression-tested:

  | shape | real site | rows recovered |
  |---|---|---|
  | bare name passed as a call argument | `_health_check("x", _health_probe_*)` | ~62 |
  | call inside a `lambda` body | `lambda: _health_probe_qdrant_reachable(...)` | (same cluster) |
  | function-local import used from a nested scope | `loci_health`'s `import backends`, called in a lambda | 4 |
  | nested def called from a SIBLING nested def | `code_parse.py::parse_source` | 2 |
  | `from . import X` of a sibling submodule | `analytics.py`'s `from . import queries as Q` | 4 |
  | protocol dunders invoked by language machinery | `with _Mutex(MUTEX_FLAG):` calls `__enter__`/`__exit__` | 5 |

- **Hand-verified breakdown of the remaining 66:**

  - **53 are methods** reached only through an attribute call on a receiver
    whose type this tool does not infer (`ks.symbol_findings(...)` where
    `ks` is a parameter or an untyped module global). These are FALSE
    POSITIVES and they are **out of reach without type inference** — the
    honest limit, not a bug to be fixed by loosening the gate. Some are
    additionally reached only via `getattr(obj, "name")` on an untyped
    receiver (`readable_probe`, `lock_holder_pid`), which is the same
    problem wearing a different hat.
  - **1 is a nested closure inside one of those methods**
    (`related_investigations.<locals>._bump`) — dead purely by
    transitivity from a false positive above it, not an independent claim.
  - **12 are module-level functions.** Spot-checked: `queries.py`'s
    `symbol_findings` / `related_findings_via_code` really are called from
    nowhere in the non-test corpus — they are `__all__` exports of a library
    module, exercised only by `mcp/tests/`.

  Cross-checking all 66 rows against a whole-tree `git grep` of each name:
  **3** are reachable only from test code (`backends.py::_reset_cache`,
  `ladybug_store.py::writable_probe`, `critic.py::record_label`), which this
  corpus definition excludes on purpose; **3** are mentioned nowhere in the
  tree at all and are the rows a reader should open first
  (`memcheck/daemon.py::MemcheckDaemon.handle_error`,
  `memcheck/engine.py::VerdictEngine.in_memory`, `::recall_decision`); the
  remaining 60 are mentioned elsewhere in the corpus, overwhelmingly as the
  untyped-receiver method calls described above.

- **The trade that bought this.** `extract/funcrefs.py` treats ANY bare
  function name in a value position as an escape, so a function mentioned
  once in code that itself never runs will not be reported. `cg dead` is a
  lead generator, and recall was spent to keep its rows worth reading.
  `tests/test_dead_false_positives.py::test_genuinely_unreferenced_function_is_still_reported`
  is the counter-test that keeps this from degenerating into a query that
  never says anything.

## Validating against real bugs, not shapes

`tests/test_regression_real_bugs.py` runs the analyzers against the actual
pre-fix source of BUGs B, C and D, copied verbatim out of git history into
`fixtures/regress/` (a provenance test re-checks each fixture against
`git show` so it cannot be quietly edited into a toy). This is a different
and stronger claim than `cg selftest`'s "BUG x shape" checks, which run
against hand-written miniatures and can keep passing after the analyzer has
stopped working on real input.

Result on the real code: **B, C and D are all surfaced.** C and D are
surfaced exactly and with zero noise in their fixtures (2 findings and 1
orphan pair respectively). B is surfaced as the sole row for its file, but
only as **rank 3 of 7** corpus-wide — `cg flags` is a ranker, and its BUG B
result depends on the user already suspecting `mcp/grounding.py`.

None of the three was found by this tool first; all three were already
fixed when it was pointed at them. What is demonstrated here is that the
queries surface them from the real source, not that they would have been
noticed unprompted.

## Carried over from earlier slices (see also `cg holes`, `cg registry
--unmatched`, `cg dead`'s footer)

- Dynamic dispatch this tool cannot see (`getattr(obj, <var>)`,
  `importlib.import_module(<var>)`, `globals()[...]`, `exec`/`eval`,
  monkeypatching) becomes an edge to `?` with a reason, never a fabricated
  edge.
- Non-Python entry points (systemd/cron/shell/docker-compose/CI) are only
  covered by the hand-maintained `[[roots]]` table in `rules.toml`, which
  WILL go stale — `cg dead`'s footer names the table size for this reason.
- Method calls on values of unknown type resolve only via a
  unique-method-name-in-corpus heuristic; common names (`run`, `get`,
  `predict`, `record`, `close`) fan out or fall to `?`.
- An unrecognized registrar (a new decorator/manifest shape rules.toml
  doesn't know) is the single most damaging failure mode this tool can
  have; `cg registry --unmatched` and `cg selftest`'s known-count
  assertions are mitigations, not a guarantee.
- `node["_id"]` vs a graph backend's `"_ID"` key-casing mismatch (BUG A)
  is explicitly OUT OF SCOPE: the string `"_ID"` is not a literal anywhere
  in this corpus's own `.py` files (the case mangling happens inside a
  driver this tool never parses), so there is no producer node for the
  key-literal audit to pair the consumer against, and the near-miss check
  cannot fire on a single-sided key. A structural call/reference graph
  should not be expected to catch this; it needs an executed backend or a
  typed payload contract instead.

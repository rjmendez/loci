# Design

The node/edge model this package implements, and the confidence contract
every query is built on. This is documentation of what `extract/*` and
`analyze/*` actually produce — see `docs/LIMITS.md` for where it falls
short, and `docs/census.txt` for the dispatch-pattern census the model was
built against.

## Confidence

Every `CALLS`, `REGISTERS`, `INJECTS`, `DISPATCHES`, `IMPORTS`, and
`ALIASES` edge carries a `Confidence`: `PROVEN` > `PROBABLE` > `UNPROVEN`,
ordered so `Confidence.combine(...)` over a chain's hops (see `cg paths`)
always yields the WEAKEST link, never an average. `UNPROVEN` is the
confidence stamped on an edge to the `?` sink — a hole made visible, not a
silent omission (`cg holes` counts and groups these by reason).

## Nodes

- **MODULE** (`mod:<repo-rel-path>`) — one per non-test `.py` file in the
  corpus. Attrs: `path`, `kind`, `importable_as` (every dotted name the
  module answers to), `has_main`.
- **FUNCTION** (`fn:<mod-path>::<qualname>`) — qualname carries class and
  closure nesting (`register.<locals>.<lambda>@<line>`, `Widget.method`).
  Lambdas handed to a registering call are real, addressable FUNCTION
  nodes. Attrs: `lineno`, `end_lineno`, `is_async`, `is_lambda`,
  `is_method`, `is_nested`, `decorators` (verbatim source), docstring
  first line.
- **CLASS** (`cls:<mod-path>::<Name>`) — thin: `bases` (source text) and
  method ids, so method defs have a parent. No type inference.
- **NAME** (`name:<mod-path>::<ident>`) — a module-global BINDING SLOT,
  distinct from any value in it. Created for module-level assigns,
  `def`/`class` bindings, import bindings, and idents that appear ONLY in a
  `global` statement with no module-level binding (`binding_kind` includes
  `"global-only"` — the raw signal BUG C's diagnosis is a lookup over, not
  a heuristic).
- **CALLSITE** (`call:<mod-path>:<line>:<col>:<end_line>:<end_col>`) — one
  per `ast.Call`, first-class because a single syntactic call can resolve
  to several candidates at different confidences. Attrs: `form`, `callee`,
  `enclosing_fn`.
- **REGISTRY** (`reg:<mod-path>::<label>`) — a registration surface:
  `mechanism` in `{decorator, manifest-tuple, manifest-dict, route-table,
  register-fn}`. The node that stops registered functions from being
  reported dead.
- **ENTRYPOINT** (`entry:<kind>:<key>`) — synthetic reachability roots,
  the only nodes with no inbound edges by construction. `trust` is
  `declared-in-source` or `declared-in-roots.toml`.
- **EXTERNAL** (`ext:<dotted>`) — terminal sink for every call resolving
  outside the corpus (stdlib, third-party, or the venv). Without this node
  thousands of calls would vanish silently instead of landing somewhere
  countable.
- **UNRESOLVED** — the single sink node `id="?"`. Every callsite this tool
  genuinely cannot resolve lands here with a `reason` on its inbound edge.
- **LOCALBINDING** (`local:<fn-id>::<ident>`) — materialized only for
  locals that ESCAPE the function (returned, packed into a returned
  container, or captured by an escaping closure). `cg flags`'s ranker
  reads `init_is_constant`, `assign_lines`/`assign_contexts`, and
  `guard_exits` off this node.
- **LITERAL** / **PATHEXPR** — string literals and composed paths
  (`Path / str`, `os.path.join`, f-strings, `+` concat), tagged
  `path`-like or `key`-like, with an occurrence list rather than one site.

## Edges

| Edge | Meaning | Confidence |
|---|---|---|
| `DEFINES` | this source physically contains that def | always PROVEN |
| `CALLS` | executing this callsite may transfer control there | 3-tier ladder, see below |
| `REFERENCES` | the function OBJECT is named here, NOT invoked (passed by reference into a manifest/registrar) | PROVEN |
| `REGISTERS` | this function becomes callable from outside the Python corpus through this surface | PROVEN if the manifest/decorator is literal, else PROBABLE |
| `ENTERS` | reachability starts here | trust-tagged, not confidence-tagged |
| `IMPORTS` | at this statement the target module loads and binds names | PROVEN/UNPROVEN by resolution |
| `ALIASES` | this NAME slot is another name for that object (re-exports, bare `A = B`) — may chain NAME→NAME→FUNCTION | PROVEN |
| `READS_NAME` / `WRITES_NAME` | this body loads/rebinds that module-global slot, cross-module included | attrs carry `via`/`in_call_position` |
| `INJECTS` | an argument at this callsite is written into that module's global slot when the callee runs (`register()`'s `global X = param` / `deps["key"]` shape) | PROBABLE |
| `DISPATCHES` | this callsite invokes SOME registry member, selected at runtime by a value this tool cannot evaluate — fans out to every candidate, never guesses one | PROBABLE |
| `DECORATED_BY` | this decorator expression was applied verbatim | classification: registering / wrapping / transparent / unknown |
| `PRODUCES_LITERAL` / `CONSUMES_LITERAL` | this code constructs / looks up that path or key | role inferred from syntactic context |
| `NEAR_MISS` | two literals, different modules, one producer-only + one consumer-only, similar text (derived, never syntactic) | capped at PROBABLE, fenced out of reachability/dead-code/exit codes |
| `ESCAPES` | this local's value leaves the function | `escape_form`: direct-return / dict-value / tuple-element / closure |

### The `CALLS` resolution ladder

Seven rungs, each stamping its own confidence and (for rungs past the
first three) a `rung`/`because`/`alternatives` attribute so `--why` and
`cg explain` can show exactly which rule fired:

1. plain module-level `Name` bound by a def/class or module-level import — **PROVEN**
2. `Attribute` call through a module-level *or* function-local import — **PROVEN**
3. one-or-more `ALIASES` hops, Python builtins, a nested function's own bare name — **PROVEN**
4. a module-global written only by an injection site — **PROBABLE**, `because=injected at <file:line>` (fans out to `DISPATCHES` if more than one distinct value was ever injected — the re-entrant-`register()` case)
5. a value fetched from a dict-of-callables (`.get(...)`/`[...]`) — **PROBABLE**, fans out to every registry member via `DISPATCHES`
6. an attribute on a value of unknown type, resolved by unique-method-name-in-corpus — **PROBABLE**, `alternatives=N`; ambiguous names stay unresolved with the same `alternatives` count
7. anything else (param-call, computed getattr, computed dict key, `importlib` with a variable, star-import) — **UNPROVEN**, edge to `?` with a `reason`

## Queries

See `README.md`'s 60-second tour for runnable examples and `cli.py --help`
for the full flag surface. `cg selftest` exercises one representative
finding per dispatch shape above, plus the corpus-wide hard gate, and is
the fastest way to confirm a checkout is healthy without reading this
whole document.

## Out of scope

BUG A (`node["_id"]` vs a backend's `"_ID"` key-casing mismatch) is
explicitly not something this tool can catch — see `docs/LIMITS.md`'s
final section for why, stated once and not re-litigated here.

# investigation_store decomposition — grounded workflow output

_Produced by the loci-native grounded/tiered workflow (wf_427beb21-313). All agents shared one injected grounding block and verified against live code._



---

# `investigation_store` call-site inventory

Repo: `/home/rjmendez/development/loci`. Enumerated via `grep -rn` over `*.py`, `*.js`, `*.md`.

## Public signature (source of truth)
`mcp/server.py:3026` — 15 params:
`investigation_id, finding_type, text, source, confidence="medium", tags=None, derived_from=None, numeric_confidence=None, procedure_preconditions=None, procedure_steps=None, procedure_postconditions=None, valid_from=None, valid_until=None, authored_by=None, tier="warm"`

## A. Production call sites (real invocations, non-test)
All in `mcp/server.py`, all **all-keyword**, all using the SAME 6-arg subset (`investigation_id, finding_type, text, source, confidence, tags`):

| file:line | caller | finding_type | notes |
|---|---|---|---|
| server.py:4034 | `reflection_loop_tick` | `observed` | kwargs: id, type, text, source, confidence, tags |
| server.py:4059 | `reflection_loop_tick` (batched low-signal) | `observed` | same 6 kwargs |
| server.py:4076 | `reflection_loop_tick` (error-cluster) | `inferred` | same 6 kwargs |
| server.py:4107 | `reflection_loop_tick` (dropped-item gap) | `gap` | same 6 kwargs |
| server.py:7925 | `investigation_reason` | `inferred` | same 6 kwargs |

No production caller ever passes `derived_from`, `numeric_confidence`, `procedure_*`, `valid_from/until`, `authored_by`, or `tier`. All 5 pass by keyword only.

## B. Test call sites (exercise the full signature)
- `mcp/tests/test_graph_integration.py:23` — helper `_store()`. **Only positional call shape in the repo**: `investigation_store(inv, ftype, text, source, conf, derived_from=...)` → first 5 params passed positionally, `derived_from` by keyword.
- `mcp/tests/test_mcp_integration.py` — **41 call expressions**. Collectively the only exerciser of the tail params, all by keyword: `tier=` (warm/cold/hot/invalid, lines 1451/1479/1491/1498), `numeric_confidence=` (531/568/591/603), `derived_from=` (604), `valid_from=`/`valid_until=` (195/196/246), `procedure_preconditions/steps/postconditions=` (653-655/684), `authored_by=` (909/934/943).
- `mcp/tests/test_reflection_loop.py:61,111,150` — **not calls**; monkeypatches `investigation_store` (side_effect/return_value).

## C. Non-call references (not invocations)
- `.vulture_whitelist.py:9` — dead-code whitelist alias.
- `scripts/event_log.py:16` — docstring/comment.
- Docstrings/help text in server.py (lines 23, 1814, 2766, 3020, 3094-example, 3211, 3905, 4593, 7961, 8010, 8142, 8147).
- Docs: `README.md`, `mcp/README.md:94`, `docs/CONCEPTS.md`, `docs/memory-and-code-review-tools.md` (multiple).
- `deep_think_loci/workflows/*.js` + CHANGELOG/README — ~25 references, but these are **prompt-string instructions to sub-agents** to call the MCP tool `mcp__loci__investigation_store`, not in-process Python calls. Confirms the MCP-tool-boundary shape agents actually use: `investigation_id, finding_type, text, source, confidence, tags` (+ occasional `derived_from`).

## Distinct call shapes → what's load-bearing for the refactor
1. **All-keyword 6-arg** (`investigation_id, finding_type, text, source, confidence, tags`): every production caller + every MCP-agent invocation. This is the dominant real-world shape.
2. **Positional first-5** (`investigation_id, finding_type, text, source, confidence`) then kw: only `test_graph_integration.py` `_store`. → **Positional order of the first 5 params is load-bearing** (a caller relies on it). Everything from param 6 (`tags`) onward is only ever passed by keyword, so their positional slots are NOT load-bearing — only their names are.
3. **Tail kwargs** (`derived_from, numeric_confidence, procedure_*, valid_from/until, authored_by, tier`): exercised only by `test_mcp_integration.py`, always by keyword. Load-bearing by **name**, not position; a signature refactor may reorder/regroup them freely as long as keyword names are preserved.

No module other than `server.py` defines or invokes the function (`graph_tools.py` does not call it).

## Flag on the grounding
`[case:loci-code-review-2026-07-07:finding]` states fan-in of **35 callers**. I could not reproduce 35 literal Python call expressions — actual in-process call sites are **5 production + ~42 test = ~47** including tests, or **5** if you count only non-test invocations. The "35" is almost certainly the Kuzu code-graph's symbol-reference fan-in (docstrings, doc mentions, workflow prompt-strings, whitelist alias all counted as edges), not literal call expressions. The grounding is **silent on how the 35 was computed**, so I'm flagging the discrepancy rather than asserting either number is the "caller" count the refactor should trust. For a signature change, the surfaces that actually break are the 5 server.py internal callers + the test suite (esp. the positional `_store` helper), plus the MCP-tool JSON schema consumed by `deep_think_loci` workflows.

---

Full function read (lines 3025-3239) and all referenced helpers located. Here is the decomposition plan.

---

# Decomposition plan: `investigation_store` (mcp/server.py)

Grounds P2 [case:loci-code-review-2026-07-07:finding] — investigation_store is the god-function (fan-out 17 / fan-in 35). This plan makes it a thin orchestrator while keeping its public MCP signature (lines 3026-3042) and return JSON (lines 3227-3239) byte-identical.

**Live line refs (verified now, 2026-07-08; the [rag] June numbers are stale):**
- Definition + `@mcp.tool()`: 3025-3042
- Docstring: 3043-3095
- Region A — validation + manifest load: 3096-3105
- Region B — numeric_confidence resolve: 3107-3115
- Region C — finding-dict build (ts, tags, valid_from, derived_from validation, entities, procedure_meta): 3117-3153
- Region D — locked persist (append + manifest counts + hot-notes + save, under `.lock` flock): 3155-3168
- Region E — mnemosyne mirror: 3170-3181
- Region F — tiered Qdrant embed (cold-skip): 3183-3185
- Region G — event-log append: 3186-3193
- Region H — Kuzu graph mirror + autolink: 3194-3197
- Region I — conflict detect + write (fail-open): 3199-3211
- Region J — session-hints ring push: 3213-3222
- Region K — entities.jsonl update: 3224-3225
- Region L — result assembly: 3227-3239

**Key finding about the current state:** Regions E/F/G/H/I/J/K are *already* thin calls into extracted helpers (`_mnemo_remember` 477, `_qdrant_upsert` 934, `_event_log_append` 103, `_mirror_finding_to_kuzu` 277, `_autolink_finding_to_kuzu` 346, `_detect_conflicts` 2760, `_write_conflict` 2850, `_session_hints_push` 1823, `_update_entities_jsonl` 2948). So the decomposition is NOT "extract seven side-effects" — most side-effects are already leaf helpers. The remaining monolith is the **inline glue**: validation (A), finding construction (B+C), the locked persist (D), and the orchestration sequencing (E-K) plus conflict-result plumbing (I→L). Four extractions, not seven.

---

## Proposed helpers

### 1. `_validate_store_args`
```python
def _validate_store_args(finding_type: str, confidence: str, tier: str) -> str | None:
    """Return an error-JSON string if any enum arg is invalid, else None."""
```
**Moves in:** the three enum guards at 3100-3105. Orchestrator: `err = _validate_store_args(...); if err: return err`. (Manifest existence check at 3096-3098 stays inline — it needs the loaded `manifest` object the orchestrator uses later, so extracting it would just re-load.)

### 2. `_build_finding`
```python
def _build_finding(
    investigation_id: str, finding_type: str, text: str, source: str,
    confidence: str, tags, numeric_confidence: float | None,
    valid_from: str | None, valid_until: str | None, authored_by: str | None,
    tier: str, procedure_preconditions: str | None,
    procedure_steps: str | None, procedure_postconditions: str | None,
) -> dict | tuple[None, str]:
    """Construct the finding dict (numeric_confidence resolution, tags parse,
    entities, procedure_meta). derived_from is validated separately by the
    caller because it needs the manifest lock context. Returns the finding
    dict, or (None, error_json) on bad derived_from — see note."""
```
**Moves in:** Region B (3107-3115) + Region C construction (3117-3136), `_extract_entities` call (3144), procedure_meta block (3146-3153).

**Deliberately NOT moved:** the `derived_from` parent-existence check (3137-3143) reads `findings.jsonl` and can early-return an error. Two safe options:
- (a) Keep derived_from validation inline in the orchestrator (simplest; preserves the exact early-return-before-any-write ordering), and have `_build_finding` take the already-normalized `derived` list as a param, OR
- (b) Move it in and return the `(None, error)` tuple form. Prefer **(a)** — it keeps the "validate-then-write" boundary visible in the orchestrator and avoids changing when the JSONL read happens.

### 3. `_persist_finding_locked`
```python
def _persist_finding_locked(investigation_id: str, finding: dict, manifest: dict) -> None:
    """Under the per-investigation .lock: append finding to findings.jsonl,
    bump manifest['finding_counts'][type], update hot-tier notes, save manifest.
    Mutates `manifest` in place."""
```
**Moves in:** the entire flock block 3155-3168 verbatim. This is the correctness-critical region — the [rag] June audit flagged the append+manifest-save pair as unlocked; the live code shows it IS now guarded (flock acquired 3156-3157, released 3168, both `_append_jsonl` and `_save_manifest` inside). Extracting it as one unit **preserves that fix** and makes the atomicity boundary a named, testable seam. Do not split append from save.

### 4. `_index_finding`
```python
def _index_finding(finding: dict, investigation_id: str, text: str,
                   source: str, finding_type: str, confidence: str, tier: str) -> bool:
    """Post-persist fan-out to the secondary stores (all fail-open, none gate
    the store): mnemosyne mirror, tiered Qdrant embed (cold-skip), event-log
    append, Kuzu mirror + autolink. Returns mnemo_stored."""
```
**Moves in:** Regions E (3170-3181), F (3183-3185), G (3186-3193), H (3194-3197). These four run unconditionally post-lock, share no early-return, and only produce `mnemo_stored` for the result. Bundling them is safe because ordering among them is not externally observable (each is an independent side-effect to a different store). Keep the *relative* order to minimize diff risk.

### Conflict detection (Region I)
Leave the `try/except` orchestration inline at 3199-3211 **as-is**, or extract to:
```python
def _run_conflict_detection(investigation_id: str, finding: dict) -> tuple[bool, str | None, str | None]:
    """Fail-open. Returns (conflict_detected, conflicting_finding_id, conflict_id)."""
```
`_detect_conflicts`/`_write_conflict` are already leaves; this just moves the try/except + tuple assembly. Low value but clean — optional. Regions J (session hints) and K (entities.jsonl) are already single helper calls; leave them inline.

---

## Resulting orchestrator (shape, not final code)
```python
manifest = _load_manifest(investigation_id)
if not manifest: return <not found>
err = _validate_store_args(finding_type, confidence, tier)
if err: return err
derived = _normalize_derived_from(derived_from)
if derived:
    <inline parent-existence check → early-return on unknown>   # 3137-3143 stays
finding = _build_finding(...)          # +derived injected
_persist_finding_locked(investigation_id, finding, manifest)
mnemo_stored = _index_finding(finding, investigation_id, text, source, finding_type, confidence, tier)
conflict_detected, conflicting_finding_id, conflict_id = _run_conflict_detection(...)  # or inline
_session_hints_push(...)               # 3215-3222 stays
_update_entities_jsonl(...)            # 3225 stays
return json.dumps(<result>, indent=2)  # 3227-3239 unchanged
```

---

## Safe leaf-first extraction order + per-step proof

Extract in dependency-leaf order so each step is a pure move with no reordering. After **every** step: run the existing suite (grounding says 276 pass on the RAG-injection branch; investigation_store has direct coverage via investigation tests) and confirm the returned JSON is unchanged.

1. **`_validate_store_args`** (pure function, no I/O). *Proof:* unit-test all invalid enum combos return the same 3 error strings as lines 3101/3103/3105; valid combo returns None. Zero behavior change — it only relocates string literals.

2. **`_index_finding`** (side-effects, but each already fail-open and non-gating). *Proof:* it produces exactly one observable output, `mnemo_stored`, which flows into the result. Assert the store still returns the identical result dict for a warm finding (mnemo mirrored, qdrant upserted) and a cold finding (qdrant skipped — the `tier != "cold"` guard at 3184 must move in intact). Spy/mock each of the four helpers to confirm they're still called once, in order, with the same args.

3. **`_persist_finding_locked`** (the atomicity seam — do this deliberately, alone). *Proof:* assert findings.jsonl gains exactly one line, `finding_counts[type]` increments by exactly one, hot-tier notes append only when `tier=="hot"`, and — critically — that the flock is still held across append+save (test by asserting the `.lock` file is created and that a concurrent writer blocks, or at minimum that the code path still enters the `with open(_lock_path...)` context). This preserves the June [rag] fix; regression here is the highest risk.

4. **`_build_finding`** (largest move; do last so the orchestrator is already thin). *Proof:* golden-value test — call with a fixed input and a frozen `_now()`/`uuid` and assert the full finding dict equals the pre-refactor dict field-for-field (id, ts, record_type+type duality at 3123-3124, tags parse from both str and list forms per 3129-3131, valid_from default at 3132, entities, procedure_meta only for procedures). Keep `derived_from` validation inline per option (a) so this step never touches JSONL.

5. *(optional)* **`_run_conflict_detection`**. *Proof:* assert the result's `conflict_detected`/`conflicting_finding_id`/`conflict_id` keys are populated identically, and that a raised `_detect_conflicts` still fails open (logged, store succeeds).

**Cross-cutting proof at each step:** the MCP tool signature (3026-3042) and docstring-declared return contract (3087-3090) must not change — diff the `@mcp.tool()` schema before/after. A characterization test that stores one finding of each `finding_type` (observed/inferred/assumed/gap/procedure) across all three tiers and snapshots the returned JSON is the strongest single guard; run it unchanged through all five steps.

---

## Flags (grounding is silent — verify before acting, do not treat as settled)
- The [rag] June concern about the unlocked append+manifest pair is **resolved in live code** (flock now wraps both, 3156-3168). `_persist_finding_locked` should preserve, not re-introduce, this.
- Separately observed while reading (out of scope for this decomposition, grounding silent): `_update_entities_jsonl` (Region K, 3225) and `_detect_conflicts` (Region I) both read/write investigation files *outside* the `.lock` held only in Region D. That's a pre-existing property, unchanged by this refactor — but if a future step tries to "consolidate locking," that's a behavior change and must not ride along on a pure-extraction commit.
- Cosmetic: line 3224 comment is over-indented (8 spaces inside a 4-space block). Harmless; a formatter pass would fix it — leave out of the behavior-preserving extraction commits.

No files written. Source of truth: `/home/rjmendez/development/loci/mcp/server.py` lines 3025-3239.

---

I have everything I need. I verified the implementation, the chokepoint helpers, and the existing test style (`server.MEMORY_DIR` swapped to a `TemporaryDirectory` in `setUp`/`tearDown`, `_new_id()` for unique ids, `_json()` parse helper, tests run without Qdrant/Ollama which "degrade gracefully").

Two grounding-relevant facts I confirmed against live code before proposing:
- The `[rag]` June concern (manifest save unguarded) is **RESOLVED on main**: `server.py:3155-3168` wraps BOTH `_append_jsonl(findings.jsonl)` and `_save_manifest(...)` inside one `.lock` `fcntl.flock`. So `finding_counts` consistency is now lockable behavior worth pinning.
- `conflict-detect` ([case P2]) is **Qdrant+embedding dependent** (`_detect_conflicts`, server.py:2760 — returns `[]` when `_get_qdrant()` is None). Since the test env runs without Qdrant, `conflict_detected` is *always False* on the happy path. To characterize the conflict-detect side effect deterministically you MUST monkeypatch `server._detect_conflicts` / `server._write_conflict` — a test that relies on live Qdrant would silently assert nothing. Same for `_autolink_finding_to_kuzu` (server.py:3197): spy on it, don't depend on a live Kuzu graph.

---

# Characterization test proposal — `investigation_store`

**File:** `mcp/tests/test_investigation_store_characterization.py` (new; mirrors `test_mcp_integration.py`)

**Shared fixture** (identical style to `TestInvestigationLifecycle`):
```python
def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self._orig = server.MEMORY_DIR
    server.MEMORY_DIR = Path(self._tmp.name)
def tearDown(self):
    server.MEMORY_DIR = self._orig
    self._tmp.cleanup()
```
Helpers reused from that module: `_json(result)`, `_new_id(prefix)`. Add one local reader:
`_findings(inv_id) -> list[dict]` = `server._read_jsonl(server._inv_dir(inv_id) / "findings.jsonl")`, and `_manifest(inv_id)` = `server._load_manifest(inv_id)`.

Each test does `server.investigation_start(...)` first, then acts.

## A. Return-value shape (locks the JSON contract in the docstring)

1. **`test_return_shape_happy_path_keys`** — a single `observed`/warm store returns exactly the keys `{stored, finding_id, type, mnemo_stored, conflict_detected, tier}`; asserts `stored is True`, `type == "observed"`, `tier == "warm"`, `conflict_detected is False`, `finding_id` is a valid UUID (`uuid.UUID(...)` parses), and `conflicting_finding_id`/`conflict_id` are **absent** when no conflict. Also assert the raw return is `indent=2`-pretty JSON (`"\n"` in the string) to pin the `json.dumps(..., indent=2)` format.

2. **`test_return_finding_id_matches_persisted_record`** — the returned `finding_id` equals the `id` of the single row in `findings.jsonl` (ties the return value to the side effect).

## B. `findings.jsonl` contents (locks the persisted record schema)

3. **`test_persisted_finding_full_field_set`** — after one store, the JSONL row has exactly these keys: `{id, investigation_id, ts, created_at_ts, record_type, type, text, source, confidence, numeric_confidence, tags, valid_from, valid_until, authored_by, tier, entities}`. Assert the **dual type fields** both equal the input (`record_type == type == "observed"` — the backwards-compat duplication at server.py:3123-3124), `authored_by == ""` default, `valid_until is None` default, and `valid_from == ts` when `valid_from` not supplied (server.py:3132).

4. **`test_tags_normalized_to_stripped_list`** — passing `tags="a, b ,,c"` persists `tags == ["a","b","c"]` (empty segments dropped, whitespace stripped — server.py:3129-3131); and passing `tags=["x","y"]` (list form) persists `["x","y"]`. Pins both accepted input forms.

5. **`test_numeric_confidence_derivation_and_clamp`** — three sub-asserts on the persisted `numeric_confidence`: omitted + `confidence="high"` → `0.9`, `"medium"` → `0.6`, `"low"` → `0.3` (server.py:3108-3110); explicit `numeric_confidence=1.7` → clamped to `1.0`; explicit `-0.5` → `0.0` (server.py:3113). (Overlaps existing tests at line 518+ but pins it as a *store* invariant, not a search one.)

6. **`test_procedure_meta_written_for_procedure_type`** — `finding_type="procedure"` persists a `procedure_meta` dict with `success_count == 0`, `attempt_count == 0`, and the passed `preconditions/steps/postconditions`; and a non-procedure finding has **no** `procedure_meta` key (server.py:3146-3153).

7. **`test_derived_from_valid_parent_links_and_unknown_rejected`** — storing with `derived_from=<real parent id>` persists `derived_from == [parent_id]` and returns no error; storing with `derived_from="nonexistent-id"` returns `{"error": ...}` containing `"unknown parent id"` **and writes no new JSONL row** (assert findings count unchanged — pins the pre-write validation at server.py:3139-3142).

8. **`test_append_only_multiple_findings_preserve_order`** — three sequential stores yield three JSONL rows in insertion order with distinct `id`s (pins append-only semantics that the decomposition must not reorder/dedupe).

## C. Manifest `finding_counts` (the June/[rag] concurrency-adjacent invariant)

9. **`test_finding_counts_increment_per_type`** — after 2×`observed` + 1×`gap`, `manifest["finding_counts"]["observed"] == 2` and `["gap"] == 1` (server.py:3160).

10. **`test_finding_counts_equals_jsonl_length`** — the key invariant the lock protects: `sum(manifest["finding_counts"].values()) == len(findings.jsonl rows)` after N mixed stores. This is the exact consistency the flock (server.py:3155-3168) guarantees; the refactor must preserve it.

11. **`test_hot_tier_appends_manifest_notes_warm_does_not`** — `tier="hot"` appends `text[:200]` to `manifest["notes"]` (server.py:3162-3165); `tier="warm"` leaves `notes` unchanged; a second hot store joins with `"; "`. Pins the hot-tier notes side effect.

## D. Autolink + conflict-detect side effects (must monkeypatch — see grounding note)

12. **`test_autolink_invoked_with_persisted_finding`** — monkeypatch `server._autolink_finding_to_kuzu` (and `server._mirror_finding_to_kuzu`) with a spy capturing its arg; assert it's called **exactly once** and the captured finding's `id` equals the returned `finding_id`. Locks that decomposition still routes the finding to the graph layer (server.py:3196-3197) regardless of Kuzu availability.

13. **`test_conflict_detect_writes_conflicts_jsonl_and_return_fields`** — monkeypatch `server._detect_conflicts` to return `[{"neighbor_id": "<a prior finding id>", "neighbor_type": "gap", "score": 0.9}]`. Assert: (a) return now has `conflict_detected is True`, `conflicting_finding_id == "<that id>"`, and a non-null `conflict_id`; (b) `conflicts.jsonl` gains exactly one row whose `finding_id_a == this finding's id`, `finding_id_b == neighbor_id`, `status == "open"`, `resolution is None` (server.py:3204-3209, 2850-2862).

14. **`test_conflict_detect_failure_is_fail_open`** — monkeypatch `server._detect_conflicts` to raise; assert the store still returns `stored is True`, `conflict_detected is False`, the finding is persisted, and `conflicts.jsonl` is absent/empty. Pins the fail-open contract (server.py:3210-3211) so the refactor can't turn a conflict-path exception into a store failure.

## E. Error paths (pre-write guards — no partial writes)

15. **`test_error_paths_write_nothing`** — parametrized over the three rejections: unknown `investigation_id` (server.py:3097), bad `finding_type` (3100), bad `confidence` (3102), bad `tier` (3104). Each returns `{"error": ...}` and — for the existing investigation cases — leaves `findings.jsonl` empty and `finding_counts` all-zero. Locks that validation precedes any side effect.

---

**Coverage map to the 5 requested surfaces:** return shape → tests 1,2,5(return),13; `findings.jsonl` → 3–8; manifest `finding_counts` → 9–11,15; autolink → 12; conflict-detect → 13,14.

**Minimality:** 15 tests, one new file, zero external services required (Qdrant/mnemo/Kuzu are monkeypatched-or-degraded, matching the existing suite's stated design). I deliberately did **not** propose tests for `mnemo_stored` truthiness (nondeterministic — depends on live mnemosyne; existing suite already treats it as best-effort), for `_qdrant_upsert`, `_session_hints_push`, or `_update_entities_jsonl` — the grounding names only the four observable surfaces in the task, and those three are either untestable without a service or out of the stated scope. Flagging them as **grounding-silent** rather than inventing assertions.

**Key files:** implementation `/home/rjmendez/development/loci/mcp/server.py:3026-3239`; helpers `_detect_conflicts:2760`, `_write_conflict:2850`, `_inv_dir:1700`, `_load_manifest:1709`, `_read_jsonl:1755`; style reference `/home/rjmendez/development/loci/mcp/tests/test_mcp_integration.py:48-58` (fixture), `:30-45` (`_json`/`_new_id`).

---

VERDICT: The claim HOLDS. Every load-bearing assertion checks out against live code.

Verified against `/home/rjmendez/development/loci/mcp/server.py` (8607 lines total):

**Line refs — all accurate:**
- Signature 3025-3042, docstring 3043-3095, Regions A(3096-3105) B(3107-3115) C(3117-3153) D(3155-3168) E(3170-3181) F(3183-3185) G(3186-3193) H(3194-3197) I(3199-3211) J(3213-3222) K(3224-3225) L(3227-3239) — all confirmed exact.
- All 9 cited helper definition lines are correct: `_event_log_append`103, `_extract_entities`129, `_mirror_finding_to_kuzu`277, `_autolink_finding_to_kuzu`346, `_mnemo_remember`477, `_qdrant_upsert`934, `_session_hints_push`1823, `_normalize_derived_from`2133, `_detect_conflicts`2760, `_write_conflict`2850, `_update_entities_jsonl`2948.

**Key substantive claims — all hold:**
- The [rag] June "unlocked append+manifest-save" concern IS resolved in live code: `_append_jsonl` (3159) and `_save_manifest` (3166) are both inside the `flock(LOCK_EX)` context (3156-3168). The plan's insistence that `_persist_finding_locked` move this block verbatim as one unit correctly preserves the fix.
- E/F/G/H/I/J/K really are already thin helper calls; the "four extractions, not seven" framing is accurate.
- `mnemo_stored` is genuinely the *sole* observable output of Regions E-H flowing into the result dict (F/G/H return None), so bundling them into `_index_finding` is safe as claimed.
- Cold-skip guard `if tier != "cold"` is at 3184 as stated (3185 does the upsert).
- `record_type`/`type` duality (3123-3124), dual-form tags parse (3129-3131), `valid_from` default (3132), procedure_meta block (3146-3153) all located as described.
- The derived_from parent-existence check (3138-3142) does read `findings.jsonl` via `_read_jsonl(_inv_dir(...)/"findings.jsonl")` and can early-return — the plan's option-(a) to keep it inline is sound.
- Conflict detect (I, 3204-3211) and `_update_entities_jsonl` (K, 3225) run *outside* the `.lock` (which wraps only D) — the "pre-existing property, don't consolidate locking on a pure-extraction commit" flag is correct.
- Cosmetic: line 3224 comment is confirmed 8-space indented inside the 4-space block (`        # Background entity extraction…`).
- Test-coverage claim holds: `investigation_store` is exercised by 4 test files (test_grounding, test_mcp_integration, test_graph_integration, test_reflection_loop), 42 call sites — so "direct coverage via investigation tests" is real, not assumed. (The grounding's "276 pass" figure is correctly cited as branch-specific / to-be-reverified, not asserted as current.)

**Unsupported / contradicted points:** none material. One minor nuance the plan itself already flags but worth naming: its proposed orchestrator hoists the `derived_from` check to *before* `_build_finding`, whereas live code runs it mid-construction (after the base dict at 3137, before `_extract_entities` at 3144). This reorders the findings.jsonl read relative to entities extraction — but entities extraction is side-effect-free and touches no investigation files, so the "read-before-any-write" boundary is preserved and the reorder is non-observable. The claim explicitly argues this, so it is not a defect, just the one place where "pure move, no reordering" is technically a benign micro-reorder rather than literally zero movement.
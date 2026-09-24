> Archived / retired from active navigation
>
> This page is kept only for historical continuity and searchability. It is no longer part of the default FlyBrain reading path.
>
> Canonical entry point: [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md)
> Archive home: [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md)
> For current guidance, use the guide and the evidence/provenance docs before digging into historical detail.
>
# FlyBrain Research Path Prioritization

**Purpose**: Rank and sequence the remaining fly-brain comparative research paths
by payoff for Loci and reasoning-system design.

**Audience**: Engineers picking the next FlyBrain-informed PR, reviewers assessing
the research program's return, and anyone asking "what should come next?"

**Related docs**:
- [`FLYBRAIN_GUIDE.md`](./FLYBRAIN_GUIDE.md) — canonical entry point and core message
- [`FLYBRAIN_INSPIRATION_MAP.md`](./FLYBRAIN_INSPIRATION_MAP.md) — placement map for new ideas
- [`FLYBRAIN_IO_TO_LOCI_MAPPING.md`](./FLYBRAIN_IO_TO_LOCI_MAPPING.md) — architecture translation reference
- [`FLYBRAIN_CROSS_DATASET_LESSONS.md`](./FLYBRAIN_CROSS_DATASET_LESSONS.md) — comparative design lessons
- [`FLYBRAIN_DATASET_PROVENANCE_MATRIX.md`](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md) — dataset bounds

---

## Scoring framework

Each research path is scored on six dimensions (1–5 each, max composite 30).
Higher composite = higher priority.

| Dimension | What it measures | 5 = best means |
|---|---|---|
| **Impact** | How much does this unlock for Loci reasoning design? | Foundational; unblocks multiple subsystems |
| **Feasibility** | Are tools, datasets, and hooks already in place? | Everything needed is already wired |
| **Time-to-value (T2V)** | How quickly can we derive usable implementation patterns? | Useful within one sprint |
| **Evidence quality (EQ)** | How behaviorally validated and cross-replicated is the biology? | Validated across ≥3 datasets with direct behavior link |
| **Integration effort (IE)** | How much new code is needed? (inverted: low effort = high score) | Hooks exist; only configuration or minor extension needed |
| **Risk** | How uncertain is the payoff? (inverted: low risk = high score) | Path is well-understood with predictable outcome |

---

## Tier A — Execute now (composite ≥ 24)

These paths have strong biological evidence, existing Loci hooks, and near-term
implementation payoff. They should be scheduled before any Tier B or C work.

### A1 · Complete `flybrain-provenance-audit-hook` (in progress)

**Composite score: 29/30**

| Dimension | Score | Rationale |
|---|---|---|
| Impact | 4 | Audit hooks make every other FlyBrain-derived finding verifiable and replayable |
| Feasibility | 5 | Work is in progress; `replay_fingerprint.py`, `flybrain_scope_engine.py`, `provenance_firewall.py` are all present |
| T2V | 5 | Already started; closing it is hours of work |
| EQ | 5 | Provenance correctness is engineering, not biological uncertainty |
| IE | 5 | Extend existing `audit_log` and `investigation_export` MCP surface |
| Risk | 5 | No new unknowns; completing in-progress work is low risk |

**Loci hooks to use**:
- `mcp/replay_fingerprint.py` — already stores access-path fingerprints
- `mcp/provenance_firewall.py` — enforces scope envelope at write time
- `mcp/flybrain_scope_engine.py` — `ComparativeClaim`, `DEFAULT_SCOPE_KEYS`
- MCP tools: `loci-audit_log`, `loci-investigation_export`, `loci-investigation_finding_provenance`

**Blocking/unblocking note**: Until this is done, audit reviewers cannot replay
any FlyBrain-derived claim chain. All downstream Tier A/B integration work depends
on trust in the claim lineage this hook exposes.

---

### A2 · MBON valence-gating → `memory_route` confidence weighting

**Composite score: 28/30**

**What this is**: Mushroom body output neurons (MBONs, `FBbt_00047953`) receive input
from Kenyon cells encoding sensory history and output behavioral valence signals
(approach / avoid). The MBON population vote is the functional analog of a
confidence-weighted retrieval policy: individual cells are unreliable; the
population average is stable.

**Evidence base**:
- Owald et al. 2015, *Neuron* 86(2): 417–427 — MBON valence output gating; behavioral validation
- Eichler et al. 2017, *Nature* 548(7666): 175–182 — larval MB connectome (`l1em`); structural validation
- Saumweber et al. 2018, *Nat. Commun.* 9(1): 1104 — cross-dataset confirmation; larval + adult
- Verified via VFB: `FBbt_00047953` tagged `has_neuron_connectivity`; instances confirmed in `l1em`, `fw`, `hb`

| Dimension | Score | Rationale |
|---|---|---|
| Impact | 5 | MB learning circuits are the best-understood memory system in any connectome; direct model for retrieval confidence |
| Feasibility | 5 | `memory_route`, `_homeostatic_route_policy`, `slow_neuromod.py::routing_policy` are all in place |
| T2V | 5 | Add a `valence_weight` parameter to `_homeostatic_route_policy` using existing confidence thresholds |
| EQ | 5 | Validated in larval + adult across multiple labs; one of the most replicated connectome results |
| IE | 5 | Add one scoring path to `slow_neuromod.py`; no new subsystem required |
| Risk | 3 | Valence → retrieval mapping is conceptually clean; some interpretive translation risk |

**Loci hooks to use**:
- `mcp/slow_neuromod.py::routing_policy`, `consolidation_policy`
- `mcp/server.py::memory_route`, `_homeostatic_route_policy`, `_derive_homeostatic_drives`
- Memory retrieval: `memory_hints`, `memory_surface`
- Scoring engine: `mcp/flybrain_scope_engine.py::ComparativeClaim.confidence`

**Concrete implementation step**: Add a `valence_weight` scaling factor in
`_homeostatic_route_policy` that reduces the `candidate_limit` and tightens
`similarity_threshold` when `homeostatic_drives["retrieval_confidence"]` is low —
directly mirroring how low MBON signal leads to conservative behavior.

---

### A3 · Fan-shaped body persistent state → homeostatic drive as working-memory vector

**Composite score: 25/30**

**What this is**: The fan-shaped body (`FBbt_00003679`) of the central complex
maintains a multi-dimensional persistent state vector used for navigation, context
gating, and sleep-wake control. It is the closest fly-brain analog to a
working-memory register — distributed, not pointlike, but durable across a task
episode. This maps to `_derive_homeostatic_drives` as a typed state dictionary
rather than a scalar.

**Evidence base**:
- Wolff et al. 2015, *eLife* 4: e10726 — FSB anatomical organization; adult hemibrain
- Hulse et al. 2021, *eLife* 10: e66039 — full central complex wiring; FlyWire + hemibrain
- Cross-verified via VFB: `FBbt_00003679` is a `Synaptic_neuropil` entity with rich
  hierarchy under the central complex (`part_of`)

| Dimension | Score | Rationale |
|---|---|---|
| Impact | 5 | Working-memory model shapes how Loci accumulates context across a session |
| Feasibility | 4 | `_derive_homeostatic_drives` exists; state dict is already typed |
| T2V | 3 | Requires extending the state key taxonomy and adding durability semantics |
| EQ | 5 | CX/FSB function is one of the most replicated central-complex results; both structural and behavioral validation |
| IE | 4 | Extend drive derivation; no new subsystem |
| Risk | 4 | Mapping is a useful design guide; specific parameter values need tuning |

**Loci hooks to use**:
- `mcp/server.py::_derive_homeostatic_drives` — extend drive keys with `context_load`, `task_horizon`
- `mcp/slow_neuromod.py::routing_policy` — consume new drive keys
- `mcp/server.py::memory_route`, `investigation_reason` — context injection point
- Docs: `docs/REASONING_POLICY_SPEC.md` — add CX-inspired working-memory note

**Concrete implementation step**: Add `context_load` (0..1, tracks how many
open investigation branches remain unresolved) and `task_horizon` (0..1,
tracks whether the current task is exploratory vs. convergent) to the homeostatic
drive dict; route them as priority-slot modifiers in `_homeostatic_route_policy`.

---

### A4 · Efference copy / corollary discharge → `investigation_pre_answer_check` self-prediction

**Composite score: 24/30**

**What this is**: The fly brain generates a corollary discharge — an internal
prediction of what self-generated actions will produce — to suppress false alarms
from self-motion. The reasoning equivalent is the pre-answer self-check: before
an agent emits a claim, it should predict whether that claim would survive
challenge and flag inconsistencies before they escape.

**Evidence base**:
- Kim et al. 2015, *eLife* 4: e13273 — efference copy in visual system; FAFB-supported
- Tuthill & Bhatt 2020, *Curr. Biol.* 30: R1002–R1012 — mechanosensory corollary discharge
- `flybrain-efference-copy-research` todo is done; research findings are in the repo

| Dimension | Score | Rationale |
|---|---|---|
| Impact | 4 | Pre-answer gating prevents confident errors from leaking into stored memory |
| Feasibility | 5 | `investigation_pre_answer_check`, `memory_self_check`, `verify_finding` are all live |
| T2V | 4 | Extend the self-check with an explicit prediction step; one function addition |
| EQ | 4 | Efference copy is well-documented in fly; some interpretive translation still required |
| IE | 5 | Extend existing `pre_answer_entailment.py` pathway |
| Risk | 2 | Conceptual translation from sensory suppression to claim gating requires careful scoping |

**Loci hooks to use**:
- `mcp/server.py::investigation_pre_answer_check`
- `mcp/pre_answer_entailment.py` — add prediction step before entailment check
- `mcp/server.py::memory_self_check`, `verify_finding`
- `mcp/server.py::retrieval_selftest`

**Concrete implementation step**: Before the lexical + semantic corroboration
steps in `investigation_pre_answer_check`, add a lightweight prediction call:
"given the current investigation state, what claim am I _expected_ to make?"
then flag deviation from the expected claim as a corollary mismatch advisory.

---

## Tier B — Schedule next (composite 18–23)

These paths have good evidence and moderate integration potential but need more
design work before implementation can start.

### B1 · Descending neuron action gate → `swarm_reason` orchestration cutoff

**Composite score: 23/30**

**What this is**: Descending neurons (DNs) in the fly brain translate a
population-level decision signal from the central brain into specific motor
command sequences in the VNC. Their threshold-gating behavior — many neurons
provide weak input; a few cross a threshold and trigger a committed action — is a
direct model for how `swarm_reason` should decide when to commit to a synthesis
vs. keep gathering evidence.

**Evidence base**:
- Namiki et al. 2018, *J. Comp. Neurol.* 526(3): 534–566 — MANC-era descending neuron map
- Schnell et al. 2017, *Curr. Biol.* 27(15): 2300–2315 — DN threshold gating for escape
- 483 DN classes available in VFB (`FBbt_00047665` and descendants)
- `flybrain-descending-control-research` todo is done

| Dimension | Score | Rationale |
|---|---|---|
| Impact | 4 | Swarm commit threshold directly affects quality and cost of reasoning |
| Feasibility | 4 | `swarm_reason` is live; orchestration policy is extensible |
| T2V | 3 | Requires a commit-threshold policy abstraction in `swarm_reason` |
| EQ | 4 | DN threshold gating is well-validated; dataset coverage in MANC + male-CNS |
| IE | 4 | Add threshold parameter to `swarm_reason` configuration |
| Risk | 4 | Principled threshold design reduces overspend and premature commitment risk |

**Loci hooks to use**:
- `mcp/server.py::swarm_reason`
- `docs/API_SWARM_AND_REASONING.md` — document commit threshold semantics
- `docs/swarm-agent-ensemble-spec.md` — add DN-inspired threshold policy
- `mcp/slow_neuromod.py::routing_policy` — configure min evidence before commit

---

### B2 · Kenyon cell sparse coding → investigation claim density cap

**Composite score: 22/30**

**What this is**: Kenyon cells encode odors and other stimuli via extremely sparse
activity (~5% of cells active per odor). This sparseness prevents overlap and
makes representations orthogonal. The equivalent for Loci is a claim density cap:
investigations should not store redundant near-duplicate findings; sparse,
non-overlapping claims are more retrievable and less likely to interfere.

**Evidence base**:
- Perez-Orive et al. 2002, *Science* 297(5580): 359–365 — sparse coding in MB; foundational
- Honegger et al. 2011, *Nat. Neurosci.* 14: 1057–1063 — cross-dataset KC sparseness confirmation
- `flybrain-mushroom-body-comparison` todo is done; research findings in repo

| Dimension | Score | Rationale |
|---|---|---|
| Impact | 4 | Sparse claim stores reduce retrieval noise; directly improves `investigation_search` quality |
| Feasibility | 4 | `semantic_dedup` already exists; density cap is a policy parameter |
| T2V | 4 | `semantic_dedup` + `investigation_store` integration is one configuration change |
| EQ | 5 | KC sparseness is one of the most replicated connectome findings |
| IE | 4 | Use `semantic_dedup` before `investigation_store`; no new code path |
| Risk | 1 | Semantic dedup must be calibrated carefully to avoid over-deduplication |

**Loci hooks to use**:
- `mcp/server.py::semantic_dedup`
- `mcp/server.py::investigation_store` — add pre-store dedup gate
- `mcp/qdrant_ops.py` — similarity threshold for near-duplicate detection
- `mcp/server.py::memory_consolidate` — periodic sparsification pass

---

### B3 · Naturalistic task research → robust controller validation patterns

**Composite score: 20/30**

**Todo**: `flybrain-naturalistic-task-research` (**done**)

**What this is**: Lab-only fly experiments miss behaviors that emerge only under
naturalistic conditions (free flight, foraging, navigation in darkness, predator
evasion). Circuits that appear sufficient in constrained tasks often reveal new
architecture under ecologically valid conditions. This research path synthesized
the naturalistic-task literature and extracted four robust controller principles
that translate directly into Loci subsystem design.

| Dimension | Score | Rationale |
|---|---|---|
| Impact | 4 | Naturalistic validation de-risks over-engineered circuits; adds continuous-update and multi-source fallback patterns |
| Feasibility | 3 | Curation complete; findings integrated below and in `FLYBRAIN_INSPIRATION_MAP.md` |
| T2V | 3 | Findings inform test case design and two concrete implementation extensions |
| EQ | 4 | Evidence spans ring attractor physiology, foraging imaging, free-flight assays, and circuit flexibility reviews |
| IE | 3 | Primarily documentation, test-case work, and targeted implementation notes |
| Risk | 3 | Lab-to-naturalistic reconciliation complete; conflicts noted where present |

---

#### Synthesized findings

**N1 · Ring attractor path integration → context persistence across modality failure**

The central complex ellipsoid body EPG neurons (`FBbt_00003670`) implement a ring
attractor that encodes the fly's heading as a localized activity bump. The key
naturalistic result: the bump persists in *complete darkness* using only self-motion
(angular velocity) integration — path integration without external anchors. When
visual landmarks reappear they *re-anchor* the bump without resetting it. This
circuit degeneracy (visual lane OR self-motion lane both capable of maintaining
state) is invisible in head-fixed lab preparations.

- Seelig & Jayaraman 2015, *Nature* 521: 186–191 — EPG compass persists in darkness via idiothetic path integration
- Kim et al. 2017, *Science* 356: 849–853 — ring attractor dynamics validated; activity bump tracks turning
- Green et al. 2017, *Nature* 543: 757–762 — EPG + PEN circuit integrates angular velocity; architecture confirmed in CX
- Hulse et al. 2021, *eLife* 10: e66039 — CX connectome; ring attractor motif confirmed structurally across all adult datasets

**Loci translation**: Session working context must persist through evidence-source
failures. If `qdrant` retrieval returns empty or errors, the working hypothesis and
open-branch state should not reset — instead, use accumulated session state (path
integration analog) to maintain context. This is an extension of A3
(`_derive_homeostatic_drives` + `context_load`) where the state must survive partial
source blackout, not just normal operation. Visual re-anchoring maps to re-engagement
after source recovery: the existing state is updated, not replaced.

**Concrete step**: In `mcp/server.py::_homeostatic_route_policy`, when a retrieval
lane returns an error or zero results, apply a small bounded decay to `context_load`
(not a reset), continue routing on remaining lanes, and re-anchor `context_load` when
the source recovers on a subsequent call.

---

**N2 · Continuous DAN valence recalibration → per-call homeostatic drive update**

Under naturalistic foraging (continuous environment, not discrete trials), the PAM
and PPL1 subsets of dopaminergic neurons continuously recalibrate reward valence based
on satiation state, food quality, and prediction error — without waiting for explicit
trial boundaries. This continuous-update architecture only becomes apparent in
free-moving or free-foraging assays; head-fixed preparations with discrete conditioning
events conceal it.

- Aso & Rubin 2016, *eLife* 5: e21074 — MB-DAN compartmental architecture; each compartment an independent valence learner
- Siju et al. 2021, *Curr. Biol.* 31(3): 543–555 — hunger-state modulates PAM-DAN activity continuously; no trial structure required
- Devineni & Scaplen 2022, *Front. Behav. Neurosci.* 15: 821680 (PMC8770416) — review of context-driven circuit flexibility including DAN-MB

**Loci translation**: `_derive_homeostatic_drives` should be recalculated on a
per-call cadence, not only at explicit task or session boundaries. Specifically,
recent retrieval success/failure patterns should incrementally update
`retrieval_confidence` and `context_load` — mimicking how PAM-DANs recalibrate
valence based on continuous foraging outcomes.

**Concrete step**: In `mcp/slow_neuromod.py::routing_policy`, add a lightweight
per-call update to `retrieval_confidence` driven by a rolling ratio of successful
vs. failed retrieval calls within the current session window (last N=10 calls).
Do not reset this ratio on each call — accumulate it as a sliding window.

---

**N3 · Stride-gated sensory processing → execution-phase evidence gating**

Fujiwara et al. 2022 (*Neuron* 110: 2124–2138) showed that during walking, specific
visual circuit neurons are gated on/off by locomotor stride phase — motor state
actively shapes which sensory inputs are processed. This effect was invisible in
head-fixed preparations and was only discovered under naturalistic (freely walking)
conditions. The principle: *efferent motor state shapes sensory intake bandwidth*.

- Fujiwara et al. 2022, *Neuron* 110: 2124–2138 — stride-gated visual neuron recruitment in freely walking Drosophila

**Loci translation**: When Loci is in an "active tool-call" phase (tool calls are
outstanding, results pending), expensive semantic retrieval and cross-validation
should be gated or deferred. When Loci enters a "consolidation" phase (all calls
complete, synthesizing evidence), full retrieval depth should engage. This parallels
stride-gating: swing phase (action in flight, sensory input reduced) vs. stance phase
(settled, full sensory intake).

**Concrete step**: In `mcp/server.py::investigation_reason`, pass a `phase`
context hint (`active` | `settling`). In `active` phase, limit `qdrant` retrieval
depth and skip `semantic_dedup`; in `settling` phase, engage full retrieval and dedup.

---

**N4 · Combinatorial circuit recruitment → shared-hook reasoning modes**

Devineni & Scaplen 2022 synthesize evidence that under naturalistic conditions the
same neurons participate in multiple behaviors depending on context — there are no
dedicated single-behavior command neurons in the fly brain. Walking, grooming, and
courtship song all recruit overlapping neuron populations with different parameter
configurations (neuromodulatory state, sensory context). This is only measurable in
naturalistic multi-behavior assays.

- Devineni & Scaplen 2022, *Front. Behav. Neurosci.* 15: 821680 (PMC8770416)
- Wosniack et al. 2022, *eLife* 11: e75826 — larval foraging adaptation; same circuit, different drive state

**Loci translation**: Reasoning modes (investigate, synthesize, monitor) must share
the same MCP tool hooks with different parameter configurations. Do not create
mode-specific tool instances; modes are parameter configurations over shared circuits,
not new circuits.

**Concrete step**: Audit `scripts/self_model_trigger_eval.py` trigger tiers (T1/T2/T3)
to confirm all tiers route through the same core tools (`investigation_reason`,
`memory_surface`, `reflection_loop_tick`). Add a regression test verifying no trigger
tier uses a tool instance unavailable to other tiers.

---

#### Naturalistic validation test scenarios

Four investigation scenarios (naturalistic test tier) to add to the test suite.
They stress conditions that only reveal failures under ecologically realistic
workloads — not toy single-turn queries.

| Scenario | FlyBrain analog | Target Loci behavior | Failure mode it catches |
|---|---|---|---|
| Multi-turn investigation where one retrieval lane goes down mid-session | EPG compass persists in darkness (N1) | Context and state persist; fallback lane activates; state decays, not resets | State wiped on source blackout |
| Long investigation (>20 turns) with continuously shifting sub-topics | DAN continuous valence recalibration (N2) | Homeostatic drives update per-call; routing adapts without explicit reset | Drives frozen until explicit boundary |
| Rapid alternation between tool-call phases and consolidation phases | Stride-gated visual recruitment (N3) | Evidence intake gated by execution phase; deep retrieval deferred during active phase | Over-retrieval during active phase; under-retrieval during consolidation |
| Two competing reasoning chains both partially supported by evidence | Combinatorial circuit recruitment (N4) | Both chains share tools; highest-evidence chain commits; shared hooks confirm | Premature commitment; siloed mode-specific tool instances |

---

**Loci hooks to use**:
- `mcp/server.py::_derive_homeostatic_drives`, `_homeostatic_route_policy` — N1 decay-not-reset, N2 continuous update
- `mcp/slow_neuromod.py::routing_policy` — N2 per-call rolling window recalibration
- `mcp/server.py::investigation_reason` — N3 execution-phase gating hint
- `scripts/self_model_trigger_eval.py` — N4 shared-hook audit + all four scenario test cases
- `docs/REASONING_POLICY_SPEC.md` — add naturalistic-validity test tier
- Test suite: four naturalistic investigation scenarios (table above)

---

### B4 · Evolutionary comparative research → conserved vs. fly-specific claims

**Composite score: 18/30**

**Todo**: `flybrain-evolutionary-comparative-research` (**done** — see [`FLYBRAIN_EVOLUTIONARY_COMPARATIVE.md`](./FLYBRAIN_EVOLUTIONARY_COMPARATIVE.md))

**What this is**: Comparing fly-brain principles with other insects (bee, locust,
moth) and C. elegans identifies which patterns are evolutionarily conserved
(and therefore more likely to generalize to AI reasoning design) versus
fly-specific.

| Dimension | Score | Rationale |
|---|---|---|
| Impact | 3 | Separates universal from fly-specific design lessons |
| Feasibility | 3 | Requires cross-species literature review; VFB is Drosophila-only |
| T2V | 3 | Value is in claim-scope annotation, not direct code change |
| EQ | 4 | Comparative insect neuroscience literature is mature (Farris 2013; Strausfeld 2012) |
| IE | 2 | Requires new scope keys (`species`, `phylum_context`) in `flybrain_scope_engine.py` |
| Risk | 3 | Risk of over-generalizing from small-sample comparative data |

**Loci hooks to use**:
- `mcp/flybrain_scope_engine.py::ComparativeClaim` — add `species` scope key
- `docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md` — add non-Drosophila scope boundary notes
- VFB: use `search_terms` + `get_term_info` for homologous structure queries

---

## Tier C — Unblock then schedule (currently blocked)

These paths are blocked pending resolution of specific technical or data issues.
Do not schedule them ahead of Tier A/B work.

### C1 · Hemibrain vs. FlyWire cell-type reassignment

**Todo**: `flybrain-hemibrain-vs-flywire` (blocked)

**Blocker**: Cell-type reassignment between hemibrain `hb` and FlyWire `fw` is
non-trivial; naming conventions differ and some hemibrain morphological classes
are not direct FlyWire equivalents. The compat layer (`loci-hemibrain-flywire-compat-layer`
todo) is done, but cross-dataset annotation drift resolution is unresolved.

**When to unblock**: After `flybrain-coverage-drift-research` lessons are
fully encoded in `FLYBRAIN_CROSS_DATASET_LESSONS.md` and the provenance
audit hook (A1) is closed.

**Loci hooks when unblocked**:
- `mcp/server.py::conflict_list`, `conflict_resolve`
- `mcp/flybrain_scope_engine.py::DATASET_SYMBOLS` — add cross-dataset alias map
- `docs/FLYBRAIN_CROSS_DATASET_LESSONS.md` — add hemibrain↔FlyWire translation table

---

### C2 · Neuromodulatory circuit comparison across datasets

**Todo**: `flybrain-neuromodulatory-comparison` (blocked)

**Blocker**: Dopaminergic, serotonergic, and octopaminergic cell predictions are
inconsistent across datasets; predicted neurotransmitter confidence varies by
dataset and annotation stage. VFB `get_predicted_neurotransmitters` returns
dataset-dependent confidence; direct cross-dataset comparison is unreliable
until annotation methods are reconciled.

**When to unblock**: After provenance audit hook (A1) is closed and
`flybrain-receptor-expression-map` lessons are integrated into claim scoping.

**Loci hooks when unblocked**:
- `mcp/server.py::investigation_pre_answer_check` — add neuromodulator state flag
- `mcp/slow_neuromod.py::routing_policy` — add modulator-typed state gates
- VFB: `get_predicted_neurotransmitters` with `split_by_dataset=true` for DA/5HT/OA

---

## Tier D — Documentation enablement cluster

These are not research paths but documentation enablers. They are prerequisites
for FlyBrain work to be adopted by engineers who are not already familiar with
the background material. Schedule these in parallel with Tier A/B implementation
work; do not block implementation on them.

| Todo ID | Priority order | Why |
|---|---|---|
| `docs-flybrain-plain-language-rewrite` | D1 | Existing docs are dense; new contributors cannot start without plain language |
| `docs-flybrain-glossary` | D2 | Technical terms (MBON, DAN, CX, VNC, etc.) need definitions |
| `docs-flybrain-why-page` | D3 | Engineers ask "why does fly-brain matter to Loci?"; this page answers that |
| `docs-flybrain-user-story` | D4 | Adds outcome-focused framing for PMs and non-neuroscience reviewers |
| `docs-flybrain-prune-plan` | D5 | Plan which dense files can be archived; execute after new simplified docs land |
| `docs-flybrain-redirect-map` | D6 | Required for safe pruning; maps dense files to simplified replacements |
| `docs-flybrain-dense-content-extraction` | D7 | Extract reusable claims before retiring dense files |
| `docs-flybrain-content-validation` | D8 | Verify simplified docs preserve key claims |
| `docs-flybrain-archive-index` | D9 | Final archive index after pruning |
| `docs-flybrain-file-retirement` | D10 | Retire dense files only after D6–D8 are done |
| `docs-flybrain-evidence-map` | D11 | Maps simplified docs back to evidence; good for auditors |
| `docs-flybrain-onboarding-update` | D12 | Final step: update onboarding to point at simplified guide |

---

## Recommended execution order

The following order minimizes blocking, maximizes near-term value, and avoids
work that becomes obsolete when later paths land.

```
Sprint 1 (now):
  1. A1  · Close flybrain-provenance-audit-hook         [in progress; 1–2 days]
  2. D1  · Plain-language rewrite (parallel with A1)    [enabler; 1 day]

Sprint 2:
  3. A2  · MBON → memory_route valence weighting        [core implementation; 2–3 days]
  4. D2  · Glossary                                     [parallel; 0.5 days]

Sprint 3:
  5. A3  · FSB → homeostatic drive working-memory keys  [extension; 2 days]
  6. A4  · Efference copy → pre_answer_check prediction [extension; 1 day]

Sprint 4:
  7. B1  · DN → swarm_reason commit threshold           [policy change; 1–2 days]
  8. B2  · KC sparse coding → claim density cap         [config + test; 1 day]
  9. D3–D4 · Why-page + user story                      [1 day each]

Sprint 5:
  10. B3  · Naturalistic task research                  [DONE — findings in B3 section above]
  11. B4  · Evolutionary comparative research           [research; 3–5 days]
  12. D5–D10 · Doc pruning and archiving                [1–2 days total]

After unblocking (no fixed sprint):
  13. C1  · Hemibrain ↔ FlyWire compat
  14. C2  · Neuromodulatory comparison
  15. D11–D12 · Evidence map + onboarding update
```

### Why each step comes next

| Step | Why it's next |
|---|---|
| A1 first | In-progress work should close before new work starts; it's also a trust prerequisite for all FlyBrain claims downstream |
| D1 parallel | No dependency; reduces onboarding friction while implementation lands |
| A2 second | Highest composite score among new implementation paths; existing hooks are ready; adds measurable retrieval quality improvement |
| A3 + A4 third | Both extend existing hooks with minimal new code; A3 adds a useful state-dimension, A4 adds a safety property |
| B1 + B2 fourth | Lower IE than A-tier but still hook-ready; improves reasoning quality and cost efficiency |
| B3 + B4 fifth | New research needed; takes longer; value lands as test improvements and scope annotations |
| C1 + C2 last | Unblocking these requires A1 and prior research outputs to be complete |

---

## Dataset evidence quality summary

This table summarizes which datasets support the highest-priority implementation
paths. Use this when setting `claim_scope` on new findings.

| Research path | Primary datasets | Evidence type | Claim confidence tier |
|---|---|---|---|
| MBON valence gating (A2) | `l1em`, `fw`, `hb` | Structural + behavioral | HIGH — cross-dataset, behavioral validation |
| Fan-shaped body state (A3) | `fw`, `hb` | Structural + behavioral | HIGH — adult brain validated |
| Efference copy (A4) | `fafb`, `fw` | Structural inferred | MEDIUM — behavioral validation indirect |
| Descending neuron gate (B1) | `mv`, `mc` | Structural | MEDIUM — male VNC; transfer to female brain requires annotation |
| Kenyon cell sparseness (B2) | `l1em`, `hb`, `fw` | Structural + physiological | HIGH — most replicated MB result |
| Naturalistic validation (B3) | Cross-dataset + behavioral assays | Behavioral + structural | MEDIUM — lab-to-naturalistic confirmed; CX ring attractor HIGH; DAN-foraging MEDIUM |
| Evolutionary comparison (B4) | Cross-species | Comparative | LOW — limited to conserved homologs |

---

## Loci subsystem coverage matrix

This maps each priority path to the specific Loci subsystem it improves and the
files that need to change.

| Research path | Loci subsystem | Files to change |
|---|---|---|
| A1 Provenance audit hook | Provenance / audit | `mcp/replay_fingerprint.py`, `mcp/provenance_firewall.py`, MCP `audit_log` |
| A2 MBON valence → memory_route | Retrieval routing | `mcp/slow_neuromod.py`, `mcp/server.py::_homeostatic_route_policy` |
| A3 FSB → homeostatic drives | Homeostatic routing | `mcp/server.py::_derive_homeostatic_drives`, `mcp/slow_neuromod.py` |
| A4 Efference copy → pre_answer | Safety gating | `mcp/pre_answer_entailment.py`, `mcp/server.py::investigation_pre_answer_check` |
| B1 DN → swarm commit threshold | Swarm orchestration | `mcp/server.py::swarm_reason`, `docs/swarm-agent-ensemble-spec.md` |
| B2 KC sparse → claim density cap | Memory store | `mcp/server.py::semantic_dedup`, `mcp/server.py::investigation_store` |
| B3 Naturalistic tasks | Homeostatic routing + test design | `mcp/server.py::_homeostatic_route_policy`, `mcp/slow_neuromod.py::routing_policy`, `mcp/server.py::investigation_reason`, `scripts/self_model_trigger_eval.py`, test suite |
| B4 Evolutionary comparison | Scope annotation | `mcp/flybrain_scope_engine.py`, `FLYBRAIN_DATASET_PROVENANCE_MATRIX.md` |
| C1 Hemibrain↔FlyWire | Conflict resolution | `mcp/server.py::conflict_list`, `mcp/flybrain_scope_engine.py` |
| C2 Neuromodulatory comparison | State modulation | `mcp/slow_neuromod.py::routing_policy`, VFB neurotransmitter queries |

---

## Evidence citations

All biological claims in this document should be traced to:

- Owald et al. 2015, *Neuron* 86(2): 417–427 (MBON valence output; `FBbt_00047953`)
- Eichler et al. 2017, *Nature* 548(7666): 175–182 (larval MBON; `l1em`)
- Saumweber et al. 2018, *Nat. Commun.* 9(1): 1104 (adult + larval MBON)
- Wolff et al. 2015, *eLife* 4: e10726 (fan-shaped body anatomy; `FBbt_00003679`)
- Hulse et al. 2021, *eLife* 10: e66039 (central complex full wiring; ring attractor motif confirmed)
- Kim et al. 2015, *eLife* 4: e13273 (efference copy / corollary discharge)
- Tuthill & Bhatt 2020, *Curr. Biol.* 30: R1002–R1012 (mechanosensory corollary discharge)
- Perez-Orive et al. 2002, *Science* 297(5580): 359–365 (KC sparse coding)
- Namiki et al. 2018, *J. Comp. Neurol.* 526(3): 534–566 (descending neurons; MANC-era)
- Schnell et al. 2017, *Curr. Biol.* 27(15): 2300–2315 (DN threshold gating)
- Seelig & Jayaraman 2015, *Nature* 521: 186–191 (EPG ring attractor compass; path integration in darkness; N1)
- Kim et al. 2017, *Science* 356: 849–853 (ring attractor dynamics; bump-tracking validated; N1)
- Green et al. 2017, *Nature* 543: 757–762 (EPG + PEN circuit; angular velocity integration; N1)
- Aso & Rubin 2016, *eLife* 5: e21074 (MB-DAN compartmental architecture; N2)
- Siju et al. 2021, *Curr. Biol.* 31(3): 543–555 (PAM-DAN continuous recalibration under hunger state; N2)
- Fujiwara et al. 2022, *Neuron* 110: 2124–2138 (stride-gated visual circuit recruitment; N3)
- Devineni & Scaplen 2022, *Front. Behav. Neurosci.* 15: 821680 (behavioral flexibility review; N2, N4)
- Wosniack et al. 2022, *eLife* 11: e75826 (larval foraging adaptation; combinatorial drive state; N4)

Dataset-level citations follow the provenance matrix in
[`FLYBRAIN_DATASET_PROVENANCE_MATRIX.md`](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md).

---

*This document is kept in-sync with the active todo database. When a path moves
from pending → in_progress → done, update the tier table and execution order above.
Canonical entry point for discovery: [`FLYBRAIN_GUIDE.md`](./FLYBRAIN_GUIDE.md).*

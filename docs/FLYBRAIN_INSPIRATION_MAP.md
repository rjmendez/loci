> Archived / retired from active navigation
>
> This page is kept only for historical continuity and searchability. It is no longer part of the default FlyBrain reading path.
>
> Canonical entry point: [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md)
> Archive home: [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md)
> For current guidance, use the guide and the evidence/provenance docs before digging into historical detail.
>
# FlyBrain-Inspired Idea Map for Loci

This is a placement guide for new FlyBrain-inspired ideas so they land in the right Loci subsystem and reuse existing hooks.

Use this as a first stop before adding new modules.

## Quick placement rubric

| If your idea is mainly about... | Put it first in... | Reuse these hooks before adding anything new |
|---|---|---|
| Timing, cadence, burst vs background work | **Temporal control** | `mcp/slow_neuromod.py::consolidation_policy`, `mcp/server.py::memory_consolidate`, `mcp/server.py::memory_hints`, `docs/sleep-consolidation-scheduler-spec.md` |
| Resource pressure, urgency, explore-vs-exploit balance | **Homeostatic routing** | `mcp/server.py::_derive_homeostatic_drives`, `_homeostatic_route_policy`, `memory_route`, `mcp/slow_neuromod.py::routing_policy` |
| Self-directed probing and "what should I check next?" | **Intrinsic motivation** | `mcp/server.py::reflection_loop_seed`, `reflection_loop_tick`, `memory_surface`, `investigation_reason`, `scripts/self_model_trigger_eval.py` |
| Fast correction after bad assumptions or drift | **Embodied correction** | `mcp/server.py::investigation_pre_answer_check`, `memory_self_check`, `verify_finding`, `memory_route_counterfactual_simulate`, `retrieval_selftest` |
| Redundant pathways that keep behavior robust if one lane degrades | **Degeneracy** | `mcp/server.py::investigation_search` (mnemo -> qdrant -> keyword), `investigation_pre_answer_check` (lexical + semantic + entailment advisory), `docs/REASONING_POLICY_SPEC.md` fail-open lanes |
| Bootstrapping better behavior from repeated execution | **Developmental bootstrap** | `mcp/server.py::procedure_search`, `procedure_attempt`, `reflection_loop_*`, `memory_consolidate`, `scripts/self_model_trigger_eval.py` trigger loops |

## Cross-cutting concept map

### 1) Temporal control

**FlyBrain inspiration**: fast reflex loops + slower consolidation rhythms.

**Loci subsystem fit**:
- Memory lifecycle and scheduler/orchestration policy.
- Slow state modulation that tunes thresholds over time rather than per-event.

**Primary hooks**:
- `mcp/slow_neuromod.py::consolidation_policy`
- `mcp/server.py::memory_consolidate`
- `mcp/server.py::memory_hints`
- `docs/sleep-consolidation-scheduler-spec.md`
- `docs/memory-consolidation-spec.md`

**Implementation pattern to reuse**:
- Keep fast-path checks cheap and bounded.
- Defer expensive synthesis to consolidation windows.
- Apply deterministic thresholds first; keep model-heavy checks advisory.

### 2) Homeostatic routing

**FlyBrain inspiration**: internal drives (hunger/fatigue/urgency) bias behavior without fully overriding reflexes.

**Loci subsystem fit**:
- Retrieval/routing control plane and candidate selection.
- Cross-session modulation layer that nudges limits, not hard rewrites.

**Primary hooks**:
- `mcp/server.py::_derive_homeostatic_drives`
- `mcp/server.py::_homeostatic_route_policy`
- `mcp/server.py::memory_route`
- `mcp/server.py::memory_route_counterfactual_simulate`
- `mcp/slow_neuromod.py::routing_policy`
- `mcp/slow_neuromod.py::assert_routing_policy_invariants`

**Implementation pattern to reuse**:
- Express drive signals as bounded numeric state (`0..1`).
- Convert drives into deterministic policy parameters (`candidate_limit`, `priority_slots`).
- Preserve replayability with route traces and counterfactual simulation.
- **Naturalistic-task extension (N2)**: Recalibrate drives on a per-call cadence using a rolling window of retrieval success/failure — do not freeze drives between explicit task boundaries. See B3 in `FLYBRAIN_RESEARCH_PRIORITIES.md`.

### 3) Intrinsic motivation

**FlyBrain inspiration**: exploratory behavior is not random noise; it is constrained novelty-seeking under state.

**Loci subsystem fit**:
- Reflection loop intake, backlog shaping, and next-step generation.
- Proactive "what to inspect next" selection when uncertainty remains.

**Primary hooks**:
- `mcp/server.py::reflection_loop_seed`
- `mcp/server.py::reflection_loop_tick`
- `mcp/server.py::memory_surface`
- `mcp/server.py::investigation_reason`
- `scripts/self_model_trigger_eval.py` (`_build_triggers`, `run_eval`, `render_introspection_report`)

**Implementation pattern to reuse**:
- Mine recent artifacts with bounded queue budgets.
- Convert low-signal observations into explicit `gap` findings.
- Trigger deeper reasoning only when trigger conditions are met.

### 4) Embodied correction

**FlyBrain inspiration**: behavior is continuously corrected by feedback from the environment/body.

**Loci subsystem fit**:
- Claim verification, contradiction pressure handling, and route replay checks.
- Pre-answer safety rails that prevent unsupported assertions from escaping.

**Primary hooks**:
- `mcp/server.py::investigation_pre_answer_check`
- `mcp/server.py::memory_self_check`
- `mcp/server.py::verify_finding`
- `mcp/server.py::memory_route_counterfactual_simulate`
- `mcp/server.py::memory_health`
- `mcp/server.py::retrieval_selftest`

**Implementation pattern to reuse**:
- Keep deterministic corroboration as the acceptance gate.
- Use local-model checks as additive advisory evidence only.
- Prefer reversible actions (`memory_retract`/`memory_restore`) over destructive correction.

### 5) Degeneracy (robust redundancy)

**FlyBrain inspiration**: multiple circuit paths can realize similar behavior under perturbation.

**Loci subsystem fit**:
- Multi-lane retrieval and validation.
- Fail-open reasoning/escalation architecture.

**Primary hooks**:
- `mcp/server.py::investigation_search` (mnemo recall, qdrant enrichment, keyword fallback)
- `mcp/server.py::investigation_pre_answer_check` (lexical, semantic-corroborated, entailment advisory)
- `docs/REASONING_POLICY_SPEC.md` (decompose -> cheap -> escalate -> synthesize fallbacks)
- `docs/API_SWARM_AND_REASONING.md` (`swarm_reason` behavior and degradation modes)

**Implementation pattern to reuse**:
- Add alternate lanes only when they are independently useful.
- Make lane disagreement visible (`degraded`, contradiction markers) instead of silently averaged.
- Ensure every lane can fail without collapsing the whole result shape.
- **Naturalistic-task extension (N1)**: Test degeneracy under source blackout, not just under normal operation. A context that survives one retrieval lane going dark (state decays, does not reset) validates the degeneracy property under realistic conditions. See B3 in `FLYBRAIN_RESEARCH_PRIORITIES.md`.

### 6) Developmental bootstrap

**FlyBrain inspiration**: competence emerges through staged adaptation, not one-shot optimization.

**Loci subsystem fit**:
- Procedure memory, reflection-fed improvement loops, and consolidation.
- Triggered self-model updates that escalate from T1/T2/T3 conditions.

**Primary hooks**:
- `mcp/server.py::procedure_search`
- `mcp/server.py::procedure_attempt`
- `mcp/server.py::reflection_loop_seed`
- `mcp/server.py::reflection_loop_tick`
- `mcp/server.py::memory_consolidate`
- `scripts/self_model_trigger_eval.py` (proactive trigger tiers and cooldowns)

**Implementation pattern to reuse**:
- Record procedures as first-class findings, then track success/attempt counts.
- Promote useful patterns gradually (cold/warm/hot and procedure confidence trends).
- Use scheduler cadence to avoid overfitting to short-term noise.

## Reuse-first decision path for new ideas

1. **Classify the idea** into one dominant concept above (secondary concept optional).
2. **Bind to one existing hook** in that concept before writing any new subsystem code.
3. **Define invariants** next to the hook (for example routing bounds, confidence bounds, consolidation thresholds).
4. **Add traceability** (`include_trace`, provenance metadata, or deterministic replay fingerprints).
5. **Add policy/docs linkage** by updating the nearest spec (`REASONING_POLICY_SPEC`, sleep scheduler spec, or memory API docs).

## Common misplacement traps

- Putting temporal behavior directly into claim-verification logic instead of the scheduler/consolidation layer.
- Adding a new motivation/routing module when `memory_route` + homeostatic drives already provide policy knobs.
- Treating degeneracy as "more models" rather than "independent, inspectable fallback lanes."
- Using consolidation as truth promotion; promotion belongs to verification/policy gates, not sleep passes.

## Related references

- `docs/FLYBRAIN_IO_TO_LOCI_MAPPING.md`
- `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md`
- `docs/REASONING_POLICY_SPEC.md`
- `docs/API_MEMORY.md`
- `docs/API_SWARM_AND_REASONING.md`
- `docs/sleep-consolidation-scheduler-spec.md`

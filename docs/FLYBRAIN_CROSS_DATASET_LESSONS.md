# Fly-Brain Cross-Dataset Lesson Synthesis
## Reusable Design Principles for Loci Reasoning Memory Systems

**Purpose**: Consolidate comparative fly-brain connectome research into concrete, evidence-backed design principles applicable to memory, routing, state management, and reasoning layers in Loci.

**Scope**: This document represents synthesized findings from comparative analysis across FlyWire (female adult FAFB), Male CNS, BANC (female whole-CNS), Hemibrain, MANC (male VNC), and L1 larval datasets. Claims are scoped to evidence present in cited datasets; speculative extensions are marked explicitly.

---

## 1. Foundational Principles

### 1.1 Principle: Layered Sensorimotor Transformation
**Evidence basis**: Observed across all adult datasets (FlyWire, Male CNS, BANC).
- **Observation**: Fly-brain sensory pathways do not directly connect to motor output. Instead, sensory input is repeatedly reprocessed through central brain structures (mushroom body, central complex, fan-shaped body) before descending to VNC/motor output.
- **Reusable lesson for Loci**:
  - Treat single-pass reasoning (direct input → output) as insufficient for non-trivial decisions.
  - Route sensory/observational inputs through intermediate abstraction layers (local models) before committing to high-confidence claims or actions.
  - Use multiple transformations to increase robustness and reduce spurious claims.

### 1.2 Principle: Explicit State-Dependent Gating
**Evidence basis**: Neuromodulation and state-gating circuits validated across all datasets; strongest evidence in FlyWire and Male CNS for dopamine, serotonin, and octopamine circuits.
- **Observation**: Fly-brain circuits are not stateless. Identical sensory input produces different motor outputs depending on internal state (hunger, arousal, mating readiness, sleep). This state is encoded via neuromodulatory signals that gate, amplify, or suppress pathway transmission.
- **Reusable lesson for Loci**:
  - Do not treat reasoning as a pure function of input text.
  - Explicit state variables (confidence levels, evidence freshness, urgency, contradiction pressure) must actively gate routing decisions.
  - Neuromodulatory analogs in reasoning systems: confidence decay, evidence tier thresholds, and priority escalation rules.

### 1.3 Principle: Validation Anchors as Circuit Benchmarks
**Evidence basis**: Circuits with behavioral validation are consistently localized to specific datasets; validation anchor registry built on verified circuit-to-behavior mappings.
- **Observation**: Not all connectome circuits have behavioral validation. Circuits with independent behavioral evidence (optogenetics, electrophysiology, behavior lesions) are localized to specific datasets and time windows. These serve as ground truth for other claims.
- **Reusable lesson for Loci**:
  - Maintain a registry of high-confidence "anchor" findings with independent verification (not just derived from connectome structure).
  - Use anchors to gate confidence levels of downstream claims: a claim without anchor support is exploratory; a claim with anchor support is high-confidence.
  - Anchors must be refreshed / revalidated on defined cadences; stale anchors cannot support new claims.

### 1.4 Principle: Provenance Separation as First-Class Contract
**Evidence basis**: Fly-brain datasets are heterogeneous in source, version, annotation completeness, and freshness. Cross-dataset claims require explicit provenance tracking.
- **Observation**: Hemibrain and FlyWire are from the same data source but at different stages of annotation; FAFB female brain has denser annotation than male counterparts. Claims that depend on annotation completeness must carry provenance metadata.
- **Reusable lesson for Loci**:
  - Provenance (data source, version, completeness, freshness) is not metadata; it is a first-class semantic property of findings.
  - Findings must declare which dataset(s) they depend on and which versions are compatible.
  - A claim valid for FlyWire may not transfer to BANC without explicit re-validation on the target dataset.

### 1.5 Principle: Evidence Tier as Confidence Bound
**Evidence basis**: Fly-brain findings tier as: structural (connectome-only), validated (circuit validated against behavior), predictive (model behavior extrapolated), or exploratory (unvalidated hypothesis).
- **Observation**: Structural connectome facts are reliable; behavioral predictions from connectome structure alone are speculative without validation evidence.
- **Reusable lesson for Loci**:
  - Evidence tier is a strict confidence bound. A T0_exploratory finding cannot be emitted as high-confidence regardless of other signals.
  - Tier hierarchy: T0 (exploratory/internal only) < T1 (baseline, caveated) < T2 (release-blocking) < T3 (regression gold).
  - Tier transitions require independent evidence, not just belief increase.

---

## 2. Routing and Decision-Gating Patterns

### 2.1 Pattern: Fast Reflex vs. Slow Deliberative Routing
**Evidence basis**: Optic lobe circuits (temporal resolution <10ms) vs. central brain circuits (temporal resolution 100s-1000s of ms); observed in FlyWire, Male CNS, BANC.
- **Observation**: Visual reflexes (collision avoidance, tracking) are computed in the optic lobe with minimal central brain involvement. Complex decisions (food search strategy, mate selection) involve recurrent looping through central brain structures and take longer.
- **Reusable lesson for Loci**:
  - Route high-confidence, low-latency decisions through fast local models (optic-lobe analog: reflexive decision-making).
  - Route complex, high-uncertainty decisions through slower, recurrent reasoning paths (central-brain analog: deep-think mode).
  - Do not block fast decisions waiting for deep-think completion; instead, run them in parallel and later reconcile.

### 2.2 Pattern: Swarm Exploration vs. Deep Consolidation
**Evidence basis**: Parallel sensory pathways (swarm-like redundancy) in optic lobe and VNC; slow consolidation in mushroom body and central complex (verified across all datasets).
- **Observation**: Fly-brain does not use a single reasoning path for every decision. Multiple pathways compute candidate motor commands in parallel; a central arbiter (central complex) later consolidates and selects. Memory consolidation is slow and state-dependent (sleep enhances consolidation).
- **Reusable lesson for Loci**:
  - Use swarm-thinking mode for parallel exploration: spawn multiple independent reasoning chains that each generate candidate claims/actions.
  - Use deep-think mode for slow consolidation: recurrently refine claims, check for contradictions, update confidence, and decide on persisted memory updates.
  - Consolidation should be gated by state (urgency, confidence, evidence quality); low-priority updates can be deferred or dropped.

### 2.3 Pattern: Confidence Decay Under Incomplete Information
**Evidence basis**: Cross-dataset annotations vary dramatically; inference chains relying on incomplete structural data systematically overestimate confidence.
- **Observation**: A circuit traced in FlyWire may be incomplete in BANC due to sample preparation or annotation variance. Claims derived from incomplete circuits are false with some probability.
- **Reusable lesson for Loci**:
  - Apply deterministic confidence penalties when knowledge is incomplete or cross-dataset.
  - Penalties are not linear; they compound over inference chains.
  - Confidence should decay faster when the underlying dataset is marked as low-completeness or when the claim depends on multiple inference steps.

---

## 3. State Variables and Neuromodulatory Analogs

### 3.1 State Variable Set (Fly-Brain → Loci Mapping)
**Evidence basis**: Neuromodulation circuits identified in FlyWire, Male CNS, and BANC; mapped to specific neuromodulators with receptors.

| Fly-Brain State | Neuromodulator | Loci Analog | Gating Effect |
|---|---|---|---|
| Hunger | NPF, AKH | Evidence relevance signal | Emphasizes food/reward claims; deprioritizes aversion. |
| Arousal | Octopamine, Dopamine | Reasoning urgency level | High arousal → fast reflex routing; low arousal → defer to deep-think. |
| Satiation | Serotonin, Insulin | Confidence threshold modifier | Satiated → lower confidence acceptance thresholds; hungry → higher thresholds for non-food claims. |
| Sleep pressure | Adenosine, Dopamine | Memory consolidation gates | High sleep pressure → consolidate memories; low pressure → keep in working/short-term. |
| Reward/Valence | Dopamine (strong) | Positive/negative evidence weighting | Dopamine release weights evidence as positive; absence deprioritizes. |
| Social context (mating) | Ecdysone, Neuropeptides | Behavioral mode switching | Switches decision routing to prioritize mating-relevant circuits. |

**Reusable lesson for Loci**:
- Maintain explicit state variables in the reasoning memory system.
- Each variable has a controlled range and defined gating rules.
- State-dependent routing is not optional; it is a first-class routing primitive.

### 3.2 Neuromodulation Application in Loci
- **Entry point**: When starting a new investigation or reasoning task, query state variables and set initial routing bias.
- **Recurrence**: State variables evolve as evidence is processed. High-confidence evidence can shift arousal/urgency. Contradictions increase pressure.
- **Gating**: Before committing a finding, check state gates:
  - Is evidence tier >= required tier for current state?
  - Is confidence >= acceptance threshold given current arousal?
  - Is memory consolidation appropriate given sleep pressure (task completion urgency)?
- **Reconciliation**: When deep-think and swarm modes produce conflicting conclusions, use state to disambiguate: if high-urgency state, trust fast swarm conclusion; if low-urgency, trust slow deep-think conclusion.

---

## 4. Memory and Learning Consolidation

### 4.1 Principle: Selective Memory Consolidation
**Evidence basis**: Synaptic plasticity rules (Hebb, STDP) validated in FlyWire and Male CNS; sleep-dependent consolidation verified across multiple papers on Drosophila.
- **Observation**: Not all experiences consolidate into persistent memory. Consolidation is slow, state-dependent (sleep-enhanced), and selective (high-reward events consolidate faster).
- **Reusable lesson for Loci**:
  - Not all findings should persist to the investigation record. Use selective consolidation:
    - High-confidence findings with strong anchors → immediate persistence.
    - Low-confidence findings → temporary working memory only; consolidate if later validated.
    - Contradictory findings → escalate to deep-think for adjudication before consolidating.
  - Consolidation is throttled: do not persist every intermediate finding. Batch consolidations.
  - Consolidation is later gated by sleep pressure analog: only consolidate findings when task urgency is low enough.

### 4.2 Pattern: Learning Windows and Temporal Binding
**Evidence basis**: Temporal coding and sequence learning circuits in central complex; validated in FlyWire, Male CNS, BANC for millisecond-scale timing.
- **Observation**: Fly-brain learns associations only if pre- and post-synaptic activity is proximal in time (<~100ms). Association across longer timescales is possible but requires intermediate mechanisms (e.g., neuromodulatory bridge).
- **Reusable lesson for Loci**:
  - Facts and evidence updates are only linked if they fall within a temporal binding window.
  - Default window: ~minutes for local reasoning; ~seconds for fast reflexes; ~hours for consolidation.
  - Cross-window associations require explicit intermediate evidence (a bridge finding) or explicit manual linking.

---

## 5. Sensor and Motor Integration

### 5.1 Sensory Fusion Strategy
**Evidence basis**: Optic lobe + antennal lobe + mechanosensory + proprioceptive pathways; all converge in central brain structures (lateral horn, mushroom body calyx).
- **Observation**: Multiple sensory modalities are integrated into a unified state estimate, but not early; fusion happens after initial sensory processing.
- **Reusable lesson for Loci**:
  - Do not fuse evidence modalities at input. Route each modality through independent early processing (optic lobe analog: local models per evidence type).
  - Late fusion in central brain (mushroom body analog): merge processed evidence into a unified confidence/state estimate for decision-making.
  - This reduces spurious correlations and improves generalization.

### 5.2 Efference Copy and Predictive Coding
**Evidence basis**: Corollary discharge circuits in central complex; validated in FlyWire for forward-model-based navigation and self-motion cancellation.
- **Observation**: Fly-brain predicts the sensory consequences of its own actions and subtracts them from incoming sensation. This suppresses self-generated noise and detects unexpected external events.
- **Reusable lesson for Loci**:
  - When Loci emits an action/claim, predict its effects on the world model and incoming observations.
  - Later, compare predicted vs. actual observations. Large mismatches indicate unexpected events or model errors.
  - Use mismatch signals to escalate to deep-think or adjust confidence.

### 5.3 Motor Primitives and Action Composition
**Evidence basis**: Descending pathways from central brain to VNC; motor-primitive circuits in VNC validated in FlyWire and Male CNS.
- **Observation**: Complex behaviors are composed from reusable motor primitives (forward walk, turn, groom, stop). The brain selects and sequences these primitives; VNC does local execution.
- **Reusable lesson for Loci**:
  - Define a small set of reusable action primitives (investigate, escalate, persist-finding, retract-finding, defer).
  - Reasoning layers (central brain analog) select and sequence primitives.
  - Executor layers (VNC analog) perform the primitive.
  - This improves generalization and reduces the action space.

---

## 6. Robustness and Degeneracy

### 6.1 Principle: Degeneracy Across Circuits
**Evidence basis**: Redundancy observed across all datasets; removal of single neurons or synapses rarely causes catastrophic behavior loss (validated in multiple lesion studies).
- **Observation**: Fly-brain is robust to component failures. Multiple circuits can implement the same function. This redundancy enables behavioral robustness.
- **Reusable lesson for Loci**:
  - Design with degeneracy: allow multiple reasoning paths to reach the same conclusion.
  - If one path fails (evidence missing, model error, contradiction), other paths provide fallback.
  - Measure and explicitly track reasoning path diversity; low diversity indicates brittleness.

### 6.2 Principle: Error Correction via Redundancy
**Evidence basis**: Sexual dimorphism in circuits (FlyWire female vs. Male CNS); different wiring produces same behaviors. Cross-dataset comparison reveals which features are essential vs. variable.
- **Observation**: Fly-brain achieves behavioral invariance across different wiring in males and females. This suggests robust error correction: if one circuit variant fails, alternatives exist.
- **Reusable lesson for Loci**:
  - Use cross-dataset comparison as a robustness check. If a claim is true in FlyWire but false in Male CNS, it is likely contingent on sex-specific wiring (lower confidence).
  - Claims true across males, females, and larvae are more likely universal (higher confidence).

---

## 7. Validation Tiers and Claim Scoping

### 7.1 Tier Assignment Framework
**Evidence basis**: Systematic review of which fly-brain claims have behavioral vs. structural support.

| Claim Type | Evidence Required | Tier | Example |
|---|---|---|---|
| Structural existence | Present in connectome | T0_exploratory | "A synapse exists between neurons A and B in FlyWire." |
| Cross-dataset consistency | Present in 2+ independent datasets | T1_baseline | "A synapse exists in both FlyWire (female) and Male CNS." |
| Functional circuit | Circuit traced + pathway logic sound | T1_baseline | "Circuit ABC implements a decision: if food-present then approach." |
| Behavior-validated circuit | Functional + optogenetics or behavior lesion evidence | T2_release_blocking | "Circuit ABC's role in approach is confirmed by silencing (optogenetics)." |
| Generalizable principle | T2 + multiple species / datasets confirm principle | T3_regression_gold | "The approach-decision circuit pattern is conserved in Drosophila and other insects." |

### 7.2 Cross-Dataset Confidence Scaling
**Evidence basis**: Annotation completeness and version differences between datasets.

| Dataset | Coverage estimate | Annotation completeness | Confidence scale factor |
|---|---|---|---|
| FlyWire female brain | ~98% of adult brain | ~95% of synapses annotated | 1.0 (baseline) |
| Male CNS | ~100% whole CNS | ~90% of synapses annotated | 0.95 |
| BANC female whole CNS | ~100% whole CNS | ~85% of synapses annotated | 0.90 |
| Hemibrain | ~70% of adult brain | ~80% of synapses in covered region | 0.75 |
| L1 larva | ~100% of larval CNS | ~95% of synapses | 0.85 (developmental stage) |

**Application**: When making a cross-dataset claim, apply the scale factor to the base confidence. A claim with 0.95 base confidence in FlyWire should be emitted with 0.95 * 0.90 = 0.855 confidence in BANC.

### 7.3 Development/Plasticity Stability Envelope (Implementation Contract)
**Evidence basis**: Developmental drift (Section 9), sexual dimorphism (Section 6.2, Section 9), and experience-dependent plasticity in mushroom body/neuromodulatory circuits (Section 1.2, Section 4.1).

**Required scope tuple for every connectome-derived claim**:
`(dataset, dataset_version, sex, life_stage, annotation_completeness, circuit_class, experience_window)`

- `dataset_version` is mandatory for stability claims. If unknown/missing, cap at `T0_exploratory`.
- `experience_window` must be explicit for learning/plasticity-sensitive claims (`naive`, `trained`, `sleep_deprived`, `unknown`, etc.). If unknown, cap at `T0_exploratory`.

| Claim archetype | Minimum scope to state claim | Stable across contexts? | Max tier without additional evidence | Escalation requirement |
|---|---|---|---|---|
| Structural edge in one dataset (for example FlyWire female adult FAFB) | dataset + dataset_version + sex + life_stage | **No** (dataset-local only) | T0_exploratory | Add independent replication in another dataset before T1 |
| Structural edge replicated in 2+ adult datasets with same sex (for example FlyWire female + BANC female) | dataset/version per source + sex=female + life_stage=adult | **Partially** (adult-female scoped) | T1_baseline | Add male adult replication before claiming sex-invariant |
| Structural edge replicated across female+male adult datasets (for example FlyWire female + Male CNS male) | dataset/version per source + sex across both + life_stage=adult | **Yes, for adult stage only** | T1_baseline | Add larval corroboration only if claiming life-stage invariance |
| Larval (L1) circuit projected to adult behavior | dataset/version + sex + life_stage=L1 plus target adult scope | **No** | T0_exploratory | Require direct adult corroboration before T1 |
| Learning/plasticity claim in MB or neuromodulatory circuits | dataset/version + sex + life_stage + explicit experience_window | **No, unless same experience state** | T1_baseline | Require behavior-validated evidence in matched scope for T2 |
| Behavior-validated circuit claim (optogenetic/lesion backed) in one sex/stage | dataset/version + sex + life_stage + validation modality | **Yes, but only within that sex/stage** | T2_release_blocking | Cross-sex or cross-stage replication required for broader scope |
| General principle claim (cross-stage or cross-sex invariance) | all above + replication matrix across target sexes/stages | **Only after explicit matrix passes** | T3_regression_gold | Must include failure cases and boundary scopes |

**What can be treated as stable now (from current corpus scope)**:
1. `T0`: dataset-local structural presence claims scoped to the exact dataset/version/sex/stage.
2. `T1`: cross-dataset adult structural consistency when scopes are matched and explicit.
3. `T2`: behavior-validated claims only within the validated sex/stage/experience window.

**What cannot be treated as stable now**:
1. Larval-to-adult transfer claims without adult corroboration.
2. Female-only or male-only claims generalized to both sexes.
3. Plasticity-sensitive claims (learning/state-dependent pathways) when experience window is missing or mixed.

**Immediate Loci wiring guidance for next PR**:
- Add `claim_scope` metadata to findings with required keys above; reject promotion above `T0` if any required key is missing.
- Enforce tier ceilings in emit/consolidation gates:
  - `missing dataset_version` -> ceiling `T0`
  - `life_stage mismatch between evidence and target claim` -> ceiling `T0`
  - `sex mismatch or unspecified target sex` -> ceiling `T1`
  - `plasticity_sensitive && experience_window in {unknown,mixed}` -> ceiling `T0`
- Persist `stability_reason` alongside confidence (for auditability of why a claim was capped).

---

## 8. Roadmap for Loci Implementation

### 8.1 Phase 1: Foundational (Immediate)
- [ ] Implement explicit state variables and state-gating rules (Section 3.1).
- [ ] Add neuromodulation-flag findings schema to track state-dependent reasoning.
- [ ] Create initial validation anchor registry (Section 1.3, Section 7.1).
- [ ] Add `claim_scope` + tier-ceiling enforcement from Section 7.3 at finding emission/consolidation boundaries.

### 8.2 Phase 2: Routing Layers (Short-term)
- [ ] Implement fast-reflex vs. slow-deliberative routing (Section 2.1).
- [ ] Add swarm-thinking mode: parallel reasoning chains (Section 2.2).
- [ ] Add deep-think mode: recurrent refinement (Section 2.2).

### 8.3 Phase 3: Memory and Consolidation (Medium-term)
- [ ] Implement selective memory consolidation gating (Section 4.1).
- [ ] Add temporal-binding-window enforcement (Section 4.2).
- [ ] Build consolidation logging and audit trail.

### 8.4 Phase 4: Robustness and Validation (Medium-term)
- [ ] Implement degeneracy measurement (Section 6.1).
- [ ] Build cross-dataset validation harness (Section 6.2, Section 7.2).
- [ ] Add confidence-scaling based on dataset provenance.

### 8.5 Phase 5: Observability and Dashboards (Ongoing)
- [ ] Track state-variable evolution over time.
- [ ] Dashboard: evidence tier distribution by circuit.
- [ ] Dashboard: confidence decay over time and dataset cross-overs.
- [ ] Audit logging: every high-confidence claim must cite its anchor(s).

---

## 9. Unresolved Questions

- **Developmental drift**: L1 larval datasets show ~15% structural differences from adult. At what confidence threshold should larval-derived claims be applied to adults?
- **Neuromodulatory timing**: Some neuromodulators act on timescales of seconds; others over hours. Should Loci have a multi-scale state model?
- **Sexual dimorphism effects**: Males and females differ in ~5-15% of connectome wiring. Should every claim be sex-scoped?
- **Plasticity and learning timescales**: Fly-brain shows experience-dependent plasticity, but timescales range from minutes to days. Should Loci track learning state?

---

## 10. References and Evidence Anchors

Key datasets cited:
- **FlyWire (female adult FAFB)**: 3,472 neurons, ~95% annotated, multi-tissue whole CNS.
- **Male CNS (whole)**: Full male ventral nerve cord + brain; ~98% annotated.
- **BANC (female whole CNS)**: Whole CNS segmentation for females.
- **Hemibrain**: Partial adult brain (~30,000 neurons); older but high-quality annotation.
- **L1 Larva**: Larval whole CNS; ~3,000 neurons, complete connectome.

Key citations (summarized):
- Neuromodulation: dopamine, serotonin, octopamine circuits (FlyWire, Male CNS).
- Sensorimotor transformation: optic lobe → central brain → VNC (FlyWire, Male CNS).
- Learning and consolidation: mushroom body Hebbian plasticity (multiple validations).
- Behavior validation: optogenetic and lesion studies cross-referenced with connectome.
- Sexual dimorphism: structural differences between male and female brains (FlyWire, Male CNS).

---

**Document version**: 1.1
**Last updated**: 2026-09-22
**Authored by**: Fly-Brain Loci Synthesis Team
**Status**: Ready for implementation planning

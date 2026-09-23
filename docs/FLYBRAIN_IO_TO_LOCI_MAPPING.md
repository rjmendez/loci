# Fly-Brain Input/Output to Loci Architecture Mapping
## Concrete Translation of Connectome Organization to Memory System Design

**Purpose**: Provide a precise mapping from fly-brain sensory ingress, internal routing, state management, and motor output to concrete Loci architecture patterns.

**Scope**: Translates structural and functional organization observed in FlyWire, Male CNS, and other datasets into Loci investigation lifecycle, finding types, state routing, and action primitives.

---

## 1. Fly-Brain Architecture Layers

### 1.1 Sensory Input Pathways (Ingress Layer)

**Fly-Brain Structure**:
- **Sensory periphery**: Eyes, antennae, legs, wings, proboscis, taste receptors.
- **Primary sensory neuropils**:
  - Optic lobe (vision): 6 layers (photoreceptors → lamina → medulla → lobula).
  - Antennal lobe (olfaction): glomerular structure; ~50 types of olfactory receptors.
  - Mechanosensory (touch, proprioception): distributed in VNC + some central processing.
  - Gustatory: taste receptors on legs, proboscis; limited central processing.

**Key Properties**:
- Multiple modalities (vision, olfaction, touch, proprioception) are processed independently in initial layers.
- Fusion is late and selective: only relevant modalities are combined downstream.
- Temporal filtering: sensory coding emphasizes transients and novelty (adapt-and-modulate).
- Confidence varies by modality: vision is high-confidence for distance/motion; olfaction is high-confidence for identity but low for distance.

### 1.2 Central Processing Layers

**Fly-Brain Structure**:
- **Mushroom body (MB)**: 4 main types (αβ, α'β', γ, dorsal paired medial). Inputs from antennal lobe and other sensory; outputs to central complex and descending neurons.
- **Central complex (CX)**: Fan-shaped body, ellipsoid body, protocerebral bridge, noduli. Processes spatial information, navigation, state integration.
- **Lateral horn (LH)**: Processes odors; projects to mushroom body and central complex.
- **Lateral accessory lobe (LAL)**: Integrates motor feedback and state information.

**Key Properties**:
- Recurrent loops: central complex has strong recurrent connections; mushroom body has feedback loops.
- State integration: neuromodulatory inputs gate and modify routing.
- Temporal integration: central complex neurons show sustained/persistent firing; mushroom body shows coincidence detection.
- Decision bottleneck: output converges to ~100-200 descending neurons (DNs) that control behavior.

### 1.3 Motor Output Pathways (Egress Layer)

**Fly-Brain Structure**:
- **Descending neurons (DNs)**: ~100-200 neurons from central brain to VNC; roughly organized by behavior (locomotion, turning, grooming, feeding, courtship, escape).
- **Ventral nerve cord (VNC)**: Motor circuits for leg control, wing control, abdomen, mouthparts.
- **Neuromuscular junctions**: Direct control of ~700 muscle cells.

**Key Properties**:
- Motor primitives: forward walk, turn, stop, groom, feed, freeze, etc.
- Sequencing: complex behaviors sequence motor primitives; sequencing is done in VNC and descending neurons.
- Parallel execution: multiple motor primitives can overlap (e.g., walking while grooming).
- Feedback loops: proprioceptive feedback from muscles/joints closes loops in VNC and sends signals back to central brain.

---

## 2. Loci Mapping: Data Ingress Architecture

### 2.1 Investigation Input Stage (Optic Lobe Analog)

**Fly-Brain analog**: Initial sensory processing in optic lobe (vision) and antennal lobe (olfaction).

**Loci Mapping**:

| Fly-Brain Layer | Loci Component | Function |
|---|---|---|
| Photoreceptors (R1-R9 cells) | Raw evidence input | Observations, code snippets, external signals, tool outputs. |
| Lamina (early filtering) | Input validation and tokenization | Classify evidence by modality (code/text/telemetry/user-supplied). Check format; reject malformed input. |
| Medulla (motion/contrast detection) | Evidence novelty detection | Identify new vs. seen-before evidence; flag contradictions. |
| Lobula (object tracking, depth) | Evidence importance scoring | Score evidence by relevance to investigation hypothesis; rank by urgency. |

**Implementation**:
`
investigation.ingest(raw_evidence):
  1. Validate format and schema (lamina analog).
  2. Extract modality (code / finding / contradiction / external telemetry).
  3. Compute novelty score (compare to known_findings; flag if contradicts).
  4. Score relevance to hypothesis (TF-IDF + semantic similarity to open questions).
  5. Enqueue to processing queue with (modality, novelty, relevance) scores.
`

**Confidence properties**:
- Initial confidence from ingress: low (C0 = 0.3 for unvalidated new evidence).
- Novelty/contradiction flags immediately trigger investigation.investigate_reason() escalation.

### 2.2 Antennal Lobe Analog: Modality-Specific Processing

**Fly-Brain analog**: Olfactory processing in antennal lobe; independent from vision.

**Loci Mapping**:

| Evidence Modality | Loci Local Model | Confidence Baseline |
|---|---|---|
| Code snapshot (static analysis) | Linter/parser (fast, deterministic) | 0.7 (structure is reliable; behavior is inferred) |
| Code execution output (telemetry) | Telemetry parser + anomaly detector | 0.8 (direct observation; may be noisy) |
| Text finding (user-supplied or model) | Text classifier + entity linker | 0.5 (unvalidated; requires evidence) |
| Contradiction signal | Diff/merge checker | 0.6 (structural mismatch detected; resolution pending) |
| External tool output (VCS, test, profiler) | Tool-specific parser + validation | 0.7-0.9 (high if tool is deterministic; lower if heuristic) |

**Implementation**:
`
process_evidence_by_modality(evidence, modality):
  if modality == "code":
    return process_code_snapshot(evidence)  # static analysis
  elif modality == "telemetry":
    return process_telemetry(evidence)  # anomaly detection
  elif modality == "finding":
    return process_text_finding(evidence)  # entity linking + classification
  elif modality == "contradiction":
    return process_contradiction(evidence)  # diff + merge logic
  elif modality == "external":
    return process_external_tool(evidence)  # tool-specific parser
  return (evidence, confidence_baseline=0.5)
`

**Confidence property**: Modality-specific baseline is applied at ingress; later found-evidence and anchor validation can increase or decrease.

### 2.3 Multiple-Modality Fusion (Early Stage)

**Fly-Brain analog**: Antennal lobe + optic lobe → lateral horn fusion.

**Loci Mapping**: Do NOT fuse modalities early. Route each through independent processing (separate local models) first. Fusion happens later in central processing (Section 2.4).

**Rationale**: In fly-brain, olfaction and vision are processed independently for ~10-20ms before central fusion. This independence prevents spurious correlations and allows each modality to contribute confidence independently.

**Implementation**:
`
# Anti-pattern (wrong):
fused_evidence = combine_all_modalities(evidence_list)
confidence = estimate(fused_evidence)

# Pattern (correct):
per_modality_findings = {}
for modality, evidence in evidence_list.group_by_modality():
  per_modality_findings[modality] = process_by_modality(evidence)
# Fusion deferred to Section 2.4 (central processing).
`

---

## 3. Loci Mapping: Internal Routing and State Management

### 3.1 Mushroom Body Analog: Associative Memory and Evidence Binding

**Fly-Brain analog**: Mushroom body (MB) receives multiple input types and learns associations through Hebbian plasticity.

**Loci Mapping**:

| MB Component | Loci Analog | Function |
|---|---|---|
| Kenyon cells (learning neurons) | Finding association matrix | Store relationships between evidence and hypothesis. |
| MB lobe output (learned associations) | Consolidated memory record | Persist associations after validation. |
| Neuromodulation input (dopamine) | Evidence valence and reward signal | Mark evidence as positive (supports hypothesis) or negative (contradicts). |
| Concurrent activation (Hebb rule) | Temporal binding window | Associate evidence only if temporally proximal (within ~minutes). |

**Implementation**:
`
associate_evidence(finding_a, finding_b, window_seconds=600):
  if time_delta(finding_a.timestamp, finding_b.timestamp) > window_seconds:
    raise TemporalBindingViolation("Findings outside binding window; no association.")
  
  # Only create association if both findings are in working memory.
  if finding_a in working_memory and finding_b in working_memory:
    association = Association(
      finding_a_id=finding_a.id,
      finding_b_id=finding_b.id,
      temporal_proximity=time_delta(...),
      valence=reward_signal,  # dopamine analog
      confidence_boost=0.1 * reward_signal  # Hebbian reinforcement
    )
    store(association)
  else:
    # Evidence outside binding window is stored but not immediately associated.
    log(f"Deferred association: findings not concurrent in working memory")
`

**State properties**:
- Working memory: ~10-20 most recent findings (fly-brain MB has ~100K neurons; active firing ~10K).
- Binding window: 10 minutes default (tunable based on investigation pace).
- Valence signal: derived from contradiction detection (negative) or anchor validation (positive).

### 3.2 Central Complex Analog: State Variables and Navigation

**Fly-Brain analog**: Central complex (CX) maintains spatial state (heading, position estimate) and integrates neuromodulation signals.

**Loci Mapping**:

| CX Component | Loci State Variable | Range | Update Frequency |
|---|---|---|---|
| Fan-shaped body (heading) | Investigation direction / hypothesis confidence | 0.0–1.0 | Per new high-confidence finding. |
| Ellipsoid body (position) | Investigation progress (% of open questions resolved) | 0.0–1.0 | Per closed question or escalation. |
| Protocerebral bridge (spatial memory) | Visited investigation branches (circuit history) | Set of branch_ids | Per decision point; pruned after depth-5. |
| Noduli (vestibular integration) | Contradiction pressure / urgency | 0.0–1.0 (high=urgent) | Incremented on contradiction; decayed per resolution. |
| Neuromodulatory input | Arousal, hunger, sleep pressure analogs | Multiple state dimensions (Section 1.3 of cross-dataset doc). | Per user input or timeout. |

**Implementation**:
`
investigation_state:
  hypothesis_confidence: float [0.0, 1.0]
  progress: float [0.0, 1.0]  # open_questions_resolved / total_open_questions
  contradiction_pressure: float [0.0, 1.0]
  investigation_urgency: float [0.0, 1.0]
  visited_branches: Set[str]
  time_since_update: float  # seconds

  neuromodulation:
    arousal: float [0.0, 1.0]  # 1.0 = high urgency, use fast paths
    fatigue: float [0.0, 1.0]  # 1.0 = defer consolidation, prepare halt
    confidence_threshold: float [0.5, 0.95]  # Varies with arousal/fatigue
`

**Update rules**:
`
on_finding_received(finding):
  # Update hypothesis confidence
  investigation_state.hypothesis_confidence *= 0.95
  investigation_state.hypothesis_confidence += 0.05 * finding.confidence
  
  # Check for contradiction
  if finding.contradicts_known:
    investigation_state.contradiction_pressure = min(1.0, pressure + 0.2)
  
  # Update progress
  if finding.closes_question:
    investigation_state.progress += (1.0 / len(open_questions))

  # If contradiction_pressure > 0.7, escalate to deep-think
  if investigation_state.contradiction_pressure > 0.7:
    escalate_to_deep_think()
`

### 3.3 Lateral Horn / LAL Analog: Sensorimotor Integration

**Fly-Brain analog**: Lateral horn processes gustatory/olfactory information; LAL integrates proprioceptive feedback and state.

**Loci Mapping**: Proprioceptive feedback (efference copy) is the key; Loci predicts the effects of its emitted actions and monitors for surprises.

**Implementation**:
`
on_action_emitted(action):
  # Predict consequences of action (efference copy).
  predicted_next_state = predict_world_state_after(action)
  predicted_observations = predict_observations(predicted_next_state)
  
  # Store prediction for later reconciliation.
  store(ActionPrediction(
    action_id=action.id,
    predicted_state=predicted_next_state,
    predicted_observations=predicted_observations,
    emitted_at=now()
  ))
  
  # Execute action (descending neuron → VNC analog).
  execute_action(action)

on_new_observation_received(observation):
  # Compare actual vs. predicted.
  for pred in recent_predictions:
    mismatch = compare(observation, pred.predicted_observations)
    if mismatch > threshold:
      # Unexpected result; escalate to deep-think.
      log(f"Action {pred.action_id} produced unexpected result: {mismatch}")
      escalate_to_deep_think(context={
        "action": pred.action_id,
        "expected": pred.predicted_observations,
        "actual": observation
      })
`

---

## 4. Loci Mapping: Routing Decisions and Mode Selection

### 4.1 Fast Reflex Routing (Optic Lobe Latency ~ 10ms)

**Fly-Brain analog**: Optic lobe local circuits; direct visual reflexes (collision avoidance, tracking).

**Loci Mapping**: Fast-path routing for high-confidence, low-uncertainty decisions.

**Conditions for fast-path**:
- Evidence confidence >= 0.85.
- Evidence is recent (< 1 minute old) or comes from a fast-updating source (telemetry, real-time tool).
- No contradictions with recent findings.
- Decision is low-stakes (doesn't modify memory; doesn't require user decision).

**Implementation**:
`
decide_action(evidence):
  # Optic lobe analog: fast reflex
  if (evidence.confidence >= 0.85 and
      time_since_evidence < 60 and
      no_contradictions(evidence) and
      not evidence.requires_memory_persist and
      not evidence.requires_user_input):
    
    # Fast-path: use local model + heuristic
    action = local_model_decide(evidence)  # Fast, deterministic
    return Action(
      type="fast_reflex",
      confidence=evidence.confidence * 0.95,
      decision_path="optic_lobe_analog"
    )
  
  # Fallthrough: escalate to deep-think
  return escalate_to_deep_think(evidence)
`

**Latency SLA**: < 500ms decision time (Loci local models).

### 4.2 Slow Deliberative Routing (Central Complex Latency ~ 100-1000ms)

**Fly-Brain analog**: Central complex recurrent processing; complex decisions involving memory and state integration.

**Loci Mapping**: Deep-think mode for high-uncertainty or high-stakes decisions.

**Conditions for deep-think**:
- Evidence confidence < 0.85 OR evidence is old.
- Contradictions present (requires adjudication).
- Decision requires persistent memory update.
- Investigation is in low-urgency state (time available for recurrence).
- User requires explanation / audit trail.

**Implementation**:
`
escalate_to_deep_think(evidence):
  # Central complex analog: slow deliberative routing
  deep_think_context = {
    "evidence": evidence,
    "existing_findings": query_by_relevance(evidence, limit=20),
    "contradictions": find_contradictions(evidence),
    "state": investigation_state.to_dict(),
    "open_questions": investigation.open_questions,
  }
  
  # Recurrent reasoning: multiple passes
  for iteration in range(max_iterations):
    refined = refine_claim(deep_think_context)
    deep_think_context["findings"].append(refined)
    
    if refined.confidence >= target_confidence:
      break
    
    # Check for contradictions; if present, continue refining
    if contradictions_reduced(refined, deep_think_context):
      continue
    else:
      # Cannot resolve; escalate to investigation_reason()
      log(f"Deep-think failed to resolve contradictions; escalating.")
      break
  
  return refined
`

**Latency SLA**: < 10s decision time (Loci OpenRouter models with recurrence).

### 4.3 Swarm Thinking Mode (Parallel Exploration)

**Fly-Brain analog**: Multiple descending neurons compute candidate motor commands in parallel; central arbiter selects.

**Loci Mapping**: Parallel reasoning chains for diverse hypothesis exploration.

**Implementation**:
`
swarm_think(evidence):
  # Spawn parallel reasoning chains (different model temperatures, prompts, reasoning paths).
  chains = []
  for reasoning_strategy in ["conservative", "exploratory", "evidence_focused"]:
    chain = spawn_reasoning_chain(
      evidence=evidence,
      strategy=reasoning_strategy,
      model="local"  # Fast models for parallelism
    )
    chains.append(chain)
  
  # Collect results
  results = parallel_wait_all(chains)
  
  # Central arbiter (central complex analog): select best result
  selected = arbiter_select(results, criteria=[
    "confidence",
    "anchor_support",
    "consistency_with_state"
  ])
  
  return selected
`

**Latency SLA**: < 2s total (parallel execution).

---

## 5. Loci Mapping: Memory Consolidation and Persistence

### 5.1 Selective Consolidation (Sleep Analog)

**Fly-Brain analog**: Sleep-dependent memory consolidation in mushroom body and central complex.

**Loci Mapping**: Findings consolidation is gated by task urgency and evidence quality.

**Consolidation gates**:
`
should_consolidate(finding):
  # Gate 1: Evidence tier
  if finding.evidence_tier < T1_baseline:
    return False  # Exploratory findings don't consolidate
  
  # Gate 2: Confidence
  if finding.confidence < consolidation_threshold:
    return False  # Low-confidence findings stay in working memory
  
  # Gate 3: Urgency / sleep pressure
  if investigation_state.urgency > 0.8:
    return False  # High-urgency mode: defer consolidation
  
  # Gate 4: Memory load
  if working_memory.size() > 50:
    # Consolidate oldest findings first (forced consolidation under pressure).
    return True
  
  # Gate 5: Anchor support
  if finding.has_anchor_support:
    return True  # Anchored findings consolidate immediately.
  
  return finding.confidence > high_threshold
`

**Consolidation timing**:
- Immediate: findings with T2/T3 tier and anchor support.
- Deferred: findings with T1 tier; consolidate if investigation goes idle.
- Never: findings with T0 tier; kept in working memory for debugging but not persisted.

### 5.2 Temporal Binding Window (Consolidation Batching)

**Fly-Brain analog**: Synaptic tagging and capture; time-dependent consolidation.

**Loci Mapping**: Findings are consolidated in temporal batches (e.g., every 10 minutes or when batch reaches size limit).

**Implementation**:
`
consolidation_batch:
  findings: List[Finding]
  opened_at: Timestamp
  closed_at: Timestamp = None
  
  size_limit: int = 50
  time_limit_seconds: int = 600  # 10 minutes
  
  def add_finding(finding):
    if should_consolidate(finding):
      findings.append(finding)
      if len(findings) >= size_limit or time_since_opened >= time_limit_seconds:
        self.flush()
  
  def flush():
    for finding in findings:
      persist_finding(finding)
    findings.clear()
    closed_at = now()
`

---

## 6. Loci Mapping: Motor Output and Action Primitives

### 6.1 Action Primitives (Motor Command Set)

**Fly-Brain analog**: Motor primitives (forward walk, turn, stop, groom, feed) sequenced by central brain.

**Loci Mapping**:

| Action Primitive | Loci Equivalent | Preconditions | Postconditions |
|---|---|---|---|
| Forward walk | xecute_investigation_step | Hypothesis confidence > 0.5 | Progress += 1 step; investigate_reason() if blocked. |
| Turn | edirect_investigation | Contradiction detected OR hypothesis confidence < 0.3 | New hypothesis branch opened; progress reset. |
| Stop/Freeze | scalate_to_deep_think | Urgency too high OR contradictions irreconcilable | Investigation paused; awaiting deep-think completion. |
| Groom | udit_findings | Periodic (every 20 findings) or on request | Check consistency, remove duplicates, merge related findings. |
| Feed/Consolidate | consolidate_findings | Findings ready for persistence AND low urgency | Persist batch to investigation record. |
| Escape/Abort | close_investigation | User request OR deadline exceeded | Investigation closed; summary generated. |

**Action sequencing**:
`
investigation_loop():
  while investigation.is_active():
    # 1. Ingest new evidence (optic lobe analog)
    evidence = ingest_evidence()
    
    # 2. Process by modality (antennal lobe analog)
    findings = process_by_modality(evidence)
    
    # 3. Route decision (central complex analog)
    if urgency_high and confidence_low:
      action = fast_reflex_route(findings)  # Walk forward
    elif contradictions_present:
      action = redirect_investigation(findings)  # Turn
    elif urgency_low and consolidation_ready:
      action = consolidate_findings()  # Groom + Feed
    else:
      action = escalate_to_deep_think()  # Deep deliberation
    
    # 4. Execute and predict consequences (VNC + LAL analog)
    execute_action(action)
    predict_consequences(action)
    
    # 5. Monitor for surprises (efference copy mismatch)
    check_for_action_mismatch()
`

### 6.2 Motor Output: Finding Emission

**Fly-Brain analog**: Descending neurons deliver motor commands to VNC; VNC executes via neuromuscular junctions.

**Loci Mapping**: Findings are emitted (persisted and visible to other agents) only after passing gates.

**Emission gates**:
`
emit_finding(finding):
  # Gate 1: Confidence threshold (depends on urgency/state)
  if finding.confidence < state.confidence_threshold:
    return False  # Keep in working memory; don't emit.
  
  # Gate 2: Evidence tier
  if finding.evidence_tier < T1_baseline:
    return False  # Exploratory findings stay internal.
  
  # Gate 3: Anchor support (if required by operational mode)
  if operational_mode == "production" and finding.evidence_tier == T1:
    if not finding.has_anchor_support:
      finding.confidence *= 0.9  # Downgrade
  
  # Gate 4: No active contradictions
  if find_contradictions(finding):
    return False  # Defer until contradictions resolved.
  
  # Gate 5: Audit trail
  create_audit_trail(
    finding=finding,
    decision="emit",
    timestamp=now(),
    decision_path=investigation_state.current_decision_path
  )
  
  persist_finding(finding)
  return True
`

---

## 7. State-Dependent Routing Table

**Purpose**: Summarize routing decisions based on investigation state.

| Investigation State | Routing Mode | Evidence Confidence Threshold | Consolidation Gate | Action Emphasis |
|---|---|---|---|---|
| High urgency + low contradiction | Fast reflex | >= 0.75 | Deferred | Execute → predict |
| High urgency + high contradiction | Escalate to deep-think | >= 0.65 | Deferred | Resolve contradiction first |
| Low urgency + low contradiction | Swarm + consolidate | >= 0.60 | Immediate | Audit → consolidate → emit |
| Low urgency + high contradiction | Deep-think | >= 0.70 | Deferred | Refine claim; resolve contradiction |
| Idle / waiting | Standby | N/A | Batch consolidation | Groom → audit |

---

## 8. Implementation Roadmap

### Phase 1: Ingress and Modality Processing
- [ ] Implement evidence ingestion pipeline (Section 2.1).
- [ ] Build modality-specific processors (Section 2.2).
- [ ] Integrate local models for per-modality confidence baseline.

### Phase 2: Central Routing and State Management
- [ ] Implement investigation state machine (Section 3.2).
- [ ] Build state-dependent routing table (Section 7).
- [ ] Integrate contradiction detection and escalation logic.

### Phase 3: Memory and Consolidation
- [ ] Implement associative memory / evidence binding (Section 3.1).
- [ ] Build temporal binding window (Section 5.2).
- [ ] Implement selective consolidation gates (Section 5.1).

### Phase 4: Action Execution and Feedback
- [ ] Implement action primitives (Section 6.1).
- [ ] Build efference-copy prediction and mismatch detection (Section 3.3).
- [ ] Implement finding emission gates (Section 6.2).

### Phase 5: Observability and Validation
- [ ] Dashboard: state variables over time.
- [ ] Audit logging: every routing decision with context.
- [ ] Cross-dataset validation harness.

---

## 9. Example Workflow: Investigation with State Transitions

**Scenario**: Investigating a code performance regression.

`
1. INGRESS
   - User submits: "Performance regression in AuthService.validate() after commit 12345"
   - Novelty score: 0.9 (new finding)
   - Relevance: high (relates to ongoing stability investigation)
   - Evidence confidence: 0.5 (user-supplied; unvalidated)
   - Queue to investigation.

2. MODALITY PROCESSING
   - Modality: finding (text) + external (code snapshot at commit 12345)
   - Local model: "AuthService.validate()" exists in codebase; structure unchanged.
   - Confidence after processing: 0.6 (structure intact; regression is behavioral)

3. STATE-DEPENDENT ROUTING
   - Investigation state:
     - hypothesis_confidence: 0.4 (not yet validated)
     - urgency: 0.8 (performance regression is urgent)
     - contradiction_pressure: 0.0 (no prior contradictions)
   - Routing decision: Fast-reflex (urgency high) + escalate to telemetry check.

4. PARALLEL EXECUTION
   - Swarm chain 1: Check recent commits for behavioral changes.
   - Swarm chain 2: Query performance metrics before/after commit 12345.
   - Swarm chain 3: Compare CPU/memory profiles (static analysis).

5. DEEP-THINK (if swarm inconclusive)
   - Collect top 3 findings from swarm chains.
   - Refine using OpenRouter: "Which commit introduced the regression?"
   - Iterate until confidence >= 0.85.

6. CONTRADICTION RESOLUTION
   - If findings conflict (e.g., commit A blamed by chain 1, commit B by chain 2):
     - Escalate to deep-think with explicit reconciliation prompt.
     - If still unresolved, mark as "needs_manual_review" and emit caveat finding.

7. ACTION: EMIT FINDING
   - Finding: "Performance regression likely in commit 12345; CPU increased 15%."
   - Confidence: 0.82 (high enough after deep-think refinement).
   - Evidence tier: T1_baseline (validated by telemetry, not yet optogenetics/lesion analog).
   - Audit trail: decision_path="deep_think", iterations=3, timestamp=...

8. EFFERENCE COPY: Predict Investigation Next Step
   - Prediction: "Follow-up investigation should examine commit 12345 diff."
   - Store prediction; monitor next evidence for match.

9. CONSOLIDATION
   - Urgency still high (production regression), but finding is high-confidence.
   - Consolidate immediately (don't defer to batch).
   - Emit finding to investigation record.

10. FEEDBACK
    - Other agents see emitted finding; can build on it.
    - Investigation state updated: hypothesis_confidence += 0.2 → 0.6.
`

---

## 10. Testing and Validation Checklist

- [ ] **Ingress**: Raw evidence correctly classified by modality.
- [ ] **Confidence**: Per-modality baseline applied correctly; evolves with evidence.
- [ ] **State tracking**: Investigation state variables update deterministically.
- [ ] **Routing**: Fast-reflex path taken for high-confidence, low-urgency evidence.
- [ ] **Deep-think**: Recurrent mode activated for contradictions; iterations converge.
- [ ] **Consolidation**: Findings persist deterministically based on confidence/tier/urgency.
- [ ] **Efference copy**: Predictions emitted; mismatch detection triggers escalation.
- [ ] **Edge cases**:
  - Evidence arriving out-of-order (old finding after new finding).
  - Contradiction between T3_gold and new T1 finding (T3 should override).
  - Consolidation batch timeout (batch should flush even if not full).
  - Action emission in high-urgency mode (emit with caveat even if confidence slightly below threshold).

---

**Document version**: 1.0
**Last updated**: 2026-09-22
**Authored by**: Fly-Brain Loci Synthesis Team
**Status**: Ready for implementation and integration testing
